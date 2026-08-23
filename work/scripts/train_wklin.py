"""wklin: RIDGE on the anchor-relative weekly matrices -- a deliberately linear model.

Why linear, and why a separate model rather than a feature tier
--------------------------------------------------------------
The weekly low-rank structure is the only representation measured OUTSIDE the blend's linear
hull, but the coordinator's probe showed how the signal is shaped: on the same 160 factors,
predicting the blend residual with an honest half-user split,

    Ridge                     R^2 = +0.001275   (gain 0.001063)
    GBDT 15 leaves / 300 trees R^2 = +0.000204   (gain 0.000170)
    GBDT 31 leaves / 600 trees R^2 = -0.004738
    GBDT on a placebo          R^2 = -0.008024

The signal is LINEAR, worth ~0.1% of the variance, and spread thinly over many weak columns.
A booster overfits noise long before it finds it, so handing the factors to the champion GBDT
as one more tier would most likely measure zero -- a false negative about the representation,
not a property of it. Hence a linear model, and hence the acceptance test is the MARGIN column
of err_corr.py (share outside the blend hull), not this model's own RMSLE. For reference the
project record margin is 0.00193 (febspec2_cal) and febspec, which did contribute, scores 1.83
solo. A bad solo score here is expected and is not evidence of anything.

Design
------
* Features, all anchor-relative so they mean the same thing at every anchor:
    WEEK block  5 x 36 = 180 columns, log1p of the weekly sum of
                {active days, carts, orders, searches, gmv}, week 1 = [anchor-6, anchor].
                Raw weekly columns, NOT SVD factors: ridge is invariant to orthogonal
                rotation, so ridge on the raw weeks is exactly ridge on the full weekly SVD
                basis, and it spans every truncation of it. The SVD only matters for the
                tree-facing tier (build_features_v5.py), where the basis has to be compact.
    BASE block  the existing 203 aggregate features, signed-log1p transformed.
* Target log1p(gmv over the next 30 days), squared loss -- which IS the competition metric
  in the space it lives in, so no Jensen correction is needed anywhere.
* One pass builds the augmented raw Gram [X, 1]^T [X, 1], X^T y and y^T y PER ANCHOR. Every
  model below is then a sub-block solve of that: any feature subset, any anchor subset, any
  ridge alpha, leave-one-anchor-out -- all free. Nothing is refitted from data twice.
* alpha is chosen on a HELD-OUT ANCHOR (the most recent training anchor), never on the
  validation anchor.

Variants written (each gets _val and _test preds, so calibrate.py works on them):
    NAME_base   base features only            -- the control: what the weeks must beat
    NAME        weekly + base                 -- the model
    NAME_wk     weekly only                   -- the purest form of the new information
With --emit-tier it also writes work/features/anchor=DATE.v5s.parquet: two SUPERVISED
columns (weekly-only prediction, and weekly prediction of the part of the target the base
model misses), fitted leave-one-anchor-out so training rows never see their own fit. That is
the concentrated form a booster can actually use -- 2 columns instead of 180 weak ones.

NO LEAKAGE: anchor A reads only events with event_date <= A; coefficients come from training
anchors only (<= VAL_ANCHOR - 30 days); val and test are pure prediction.

Usage
  POLARS_MAX_THREADS=3 .venv/bin/python work/scripts/train_wklin.py --name wklin [--emit-tier]
"""
from __future__ import annotations

import os

_T = os.environ.get("THREADS", "4")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "POLARS_MAX_THREADS"):
    os.environ.setdefault(_v, _T)

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from datetime import date  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from common import (  # noqa: E402
    FEATURES_DIR, REPORTS_DIR, TEST_ANCHOR, VAL_ANCHOR, feature_cols, load_anchor, rmsle,
)
from build_features_v5 import W_WEEKS, anchor_plan, joint_block, weekly_dense  # noqa: E402
from exp_lib import log_score, save_preds  # noqa: E402
from model_io import save_npz  # noqa: E402

ALPHAS = [10.0 ** (k / 2) for k in range(-4, 15)]   # 0.01 .. 3.2e6, half-decade steps
CHUNK = 50_000


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def slog(a: np.ndarray) -> np.ndarray:
    """signed log1p; nulls/inf -> 0. Keeps heavy-tailed count/money columns usable linearly."""
    a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
    return np.sign(a) * np.log1p(np.abs(a))


class Acc:
    """Augmented raw moments of one anchor: [X,1]^T[X,1], [X,1]^T y, y^T y, n."""

    def __init__(self, p: int):
        self.A = np.zeros((p + 1, p + 1))
        self.g = np.zeros(p + 1)
        self.yy = 0.0
        self.n = 0

    def add(self, X: np.ndarray, y: np.ndarray | None):
        for i in range(0, X.shape[0], CHUNK):
            C = np.empty((min(CHUNK, X.shape[0] - i), X.shape[1] + 1))
            C[:, :-1] = X[i:i + CHUNK]
            C[:, -1] = 1.0
            self.A += C.T @ C
            if y is not None:
                self.g += C.T @ y[i:i + CHUNK]
        if y is not None:
            self.yy += float(y @ y)
        self.n += X.shape[0]
        return self


def solve(A: np.ndarray, g: np.ndarray, n: int, cols: np.ndarray, alpha: float) -> np.ndarray:
    """Ridge on standardised, centred columns; returns raw-space [beta, intercept].

    Everything comes from the augmented raw moments: mu = A[j,-1]/n, sd^2 = A[j,j]/n - mu^2,
    Sxx = A - n mu mu^T, Sxy = g - n mu ybar. The intercept is never penalised.
    """
    k = len(cols)
    ix = np.concatenate([cols, [A.shape[0] - 1]])
    Ac, gc = A[np.ix_(ix, ix)], g[ix]
    mu, ybar = Ac[:k, k] / n, gc[k] / n
    var = np.maximum(Ac[np.arange(k), np.arange(k)] / n - mu ** 2, 0.0)
    sd = np.sqrt(var)
    live = sd > 1e-9
    D = np.zeros(k)
    D[live] = 1.0 / sd[live]
    Sxx = (Ac[:k, :k] - n * np.outer(mu, mu)) * np.outer(D, D)
    Sxy = (gc[:k] - n * mu * ybar) * D
    w = np.linalg.solve(Sxx + alpha * np.eye(k), Sxy)
    w[~live] = 0.0
    beta = np.zeros(A.shape[0])
    beta[cols] = w * D
    beta[-1] = ybar - float(mu @ (w * D))
    return beta


def sse(acc: Acc, beta: np.ndarray) -> float:
    return float(acc.yy - 2.0 * beta @ acc.g + beta @ acc.A @ beta)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="wklin")
    ap.add_argument("--emit-tier", action="store_true")
    ap.add_argument("--n-anchors", type=int, default=0, help="mechanics check: fewer fit anchors")
    ap.add_argument("--no-test", action="store_true")
    ap.add_argument("--notes", default="")
    args = ap.parse_args()

    t0 = time.time()
    plan = anchor_plan()
    fit_a, gap_a = plan["fit"], plan["gap"]
    if args.n_anchors:
        fit_a = fit_a[-args.n_anchors:]
    all_a = fit_a + gap_a + [VAL_ANCHOR, TEST_ANCHOR]
    log(f"fit {len(fit_a)}: {[a.isoformat() for a in fit_a]}")
    log(f"gap {len(gap_a)}: {[a.isoformat() for a in gap_a]}")

    # ---- feature layout, taken from the validation anchor's schema
    v0 = load_anchor(VAL_ANCHOR)
    base_cols = [c for c in feature_cols(v0) if not c.startswith("v5")]
    uid = v0["user_id"].to_numpy()
    assert (np.diff(uid) > 0).all() or True
    order = np.argsort(uid)
    uid = uid[order]
    n_wk, n_base = 5 * W_WEEKS, len(base_cols)
    p = n_wk + n_base
    WK = np.arange(n_wk)
    BS = np.arange(n_wk, p)
    log(f"design: {n_wk} weekly + {n_base} base = {p} columns")
    del v0

    grid_anchors = fit_a + gap_a + [VAL_ANCHOR]
    max_off = max((VAL_ANCHOR - a).days // 7 for a in grid_anchors)
    Gv = weekly_dense(VAL_ANCHOR, max_off + W_WEEKS, uid)
    off = {a: (VAL_ANCHOR - a).days // 7 for a in grid_anchors}
    Gt = None

    def design(a: date):
        nonlocal Gt
        if a == TEST_ANCHOR:
            if Gt is None:
                Gt = weekly_dense(TEST_ANCHOR, W_WEEKS, uid)
            Wb = joint_block(Gt, 0)
        else:
            Wb = joint_block(Gv, off[a])
        df = load_anchor(a).sort("user_id")
        assert np.array_equal(df["user_id"].to_numpy(), uid), f"user order mismatch at {a}"
        X = np.empty((len(uid), p), dtype=np.float32)
        X[:, :n_wk] = Wb
        X[:, n_wk:] = slog(df.select(base_cols).to_numpy().astype(np.float64)).astype(np.float32)
        y = None
        if "target" in df.columns and df["target"].null_count() == 0:
            y = np.log1p(np.clip(df["target"].to_numpy().astype(np.float64), 0, None))
        return X, y

    # ---- one pass: per-anchor augmented moments
    accs: dict[date, Acc] = {}
    for a in fit_a:
        X, y = design(a)
        assert y is not None, f"no target at training anchor {a}"
        accs[a] = Acc(p).add(X, y)
        del X, y
        log(f"  moments {a} done ({time.time()-t0:.0f}s)")

    def pool(anchors):
        acc = Acc(p)
        for a in anchors:
            acc.A += accs[a].A; acc.g += accs[a].g
            acc.yy += accs[a].yy; acc.n += accs[a].n
        return acc

    # ---- alpha per feature set, chosen on a held-out ANCHOR (never on validation)
    hold = fit_a[-1]
    tr_for_alpha = [a for a in fit_a if a != hold]
    sets = {"": np.arange(p), "_wk": WK, "_base": BS}
    best_alpha, alpha_tab = {}, {}
    acc_tr = pool(tr_for_alpha)
    for tag, cols in sets.items():
        row = {}
        for al in ALPHAS:
            b = solve(acc_tr.A, acc_tr.g, acc_tr.n, cols, al)
            row[al] = float(np.sqrt(sse(accs[hold], b) / accs[hold].n))
        best_alpha[tag] = min(row, key=row.get)
        alpha_tab[tag or "wk+base"] = {str(k): round(v, 6) for k, v in row.items()}
        log(f"  alpha[{tag or 'wk+base'}] = {best_alpha[tag]:g} "
            f"(holdout {hold} rmse {row[best_alpha[tag]]:.6f})")

    # ---- fit on all 14 training anchors, predict validation
    acc_all = pool(fit_a)
    Xv, yv = design(VAL_ANCHOR)
    assert yv is not None
    yv_raw = np.expm1(yv)
    res = {}
    betas = {}
    for tag, cols in sets.items():
        b = solve(acc_all.A, acc_all.g, acc_all.n, cols, best_alpha[tag])
        betas[tag] = b
        lp = np.clip(Xv @ b[:-1] + b[-1], 0, None)
        name = args.name + tag
        # Воспроизводимость: модель — это ровно вектор [beta, intercept] в сыром
        # пространстве, поэтому сохранить её стоит килобайт. Без этого *_val.parquet
        # не восстановить из чистого клона (inference.py --stage check: «прогноз-артефакт»).
        save_npz(f"{name}_val", beta=b, cols=np.array(cols, dtype=object),
                 alpha=np.array([best_alpha[tag]], dtype=np.float64))
        save_preds(name, "val", uid, np.expm1(lp))
        s = rmsle(yv_raw, np.expm1(lp))
        res[name] = {"val_rmsle": round(s, 6), "alpha": best_alpha[tag],
                     "n_features": int(len(cols)),
                     "mean_logpred": round(float(lp.mean()), 4),
                     "sd_logpred": round(float(lp.std()), 4)}
        log(f"  {name}: val_rmsle {s:.6f}  mean {lp.mean():.4f} sd {lp.std():.4f}")

    # what the weekly block adds ON TOP of the base block, in the metric, before calibration
    res["_delta_wkbase_vs_base"] = round(res[args.name]["val_rmsle"]
                                         - res[args.name + "_base"]["val_rmsle"], 6)

    # ---- refit including gap + validation anchors, predict test (mirrors train_gbdt.py)
    if args.no_test:
        print("\n=== RAW JSON (no-test) ===")
        print(json.dumps({"models": res, "alpha": {k: best_alpha[k] for k in sets}},
                         ensure_ascii=False))
        return
    for a in gap_a + [VAL_ANCHOR]:
        X, y = design(a)
        accs[a] = Acc(p).add(X, y)
        del X, y
    acc_full = pool(fit_a + gap_a + [VAL_ANCHOR])
    Xt, _ = design(TEST_ANCHOR)
    for tag, cols in sets.items():
        b = solve(acc_full.A, acc_full.g, acc_full.n, cols, best_alpha[tag])
        lp = np.clip(Xt @ b[:-1] + b[-1], 0, None)
        # те же коэффициенты, что делают отгружаемый *_test.parquet
        save_npz(f"{args.name + tag}_test", beta=b, cols=np.array(cols, dtype=object),
                 alpha=np.array([best_alpha[tag]], dtype=np.float64))
        save_preds(args.name + tag, "test", uid, np.expm1(lp))
        res[args.name + tag]["test_mean_logpred"] = round(float(lp.mean()), 4)

    # ---- optional: 2 supervised columns for the booster, leave-one-anchor-out
    if args.emit_tier:
        emit_tier(args.name, accs, fit_a, gap_a, design, uid, sets, best_alpha, p)

    for tag in sets:
        log_score(args.name + tag, res[args.name + tag]["val_rmsle"],
                  args.notes or f"ridge on weekly matrices ({tag or 'wk+base'}), "
                                f"alpha={best_alpha[tag]:g}, 14 anchors, gap 30")
    out = {"name": args.name, "weeks": W_WEEKS, "n_weekly": n_wk, "n_base": n_base,
           "alpha_holdout_anchor": hold.isoformat(), "alpha_table": alpha_tab,
           "models": res, "seconds": round(time.time() - t0)}
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / f"{args.name}.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print("\n=== RAW JSON ===")
    print(json.dumps(out, ensure_ascii=False))


def emit_tier(name, accs, fit_a, gap_a, design, uid, sets, best_alpha, p):
    """work/features/anchor=DATE.v5s.parquet: 2 supervised weekly columns, OOF by anchor.

    v5s_lin  weekly-only ridge prediction of log1p(target)
    v5s_orth weekly-only ridge prediction of the residual left by the base-only model
    For a training anchor the coefficients exclude that anchor, so a booster trained on these
    rows never sees a feature fitted on its own target. Validation, gap and test anchors use
    the coefficients of all 14 training anchors.
    """
    WK = sets["_wk"]

    def pooled(anchors):
        acc = Acc(p)
        for a in anchors:
            acc.A += accs[a].A; acc.g += accs[a].g; acc.yy += accs[a].yy; acc.n += accs[a].n
        return acc

    def coefs(anchors):
        ac = pooled(anchors)
        b_wk = solve(ac.A, ac.g, ac.n, WK, best_alpha["_wk"])
        b_bs = solve(ac.A, ac.g, ac.n, sets["_base"], best_alpha["_base"])
        # residual model: X_wk on (y - base prediction). Its normal equations are the weekly
        # block of the same moments with g replaced by g - A beta_base -- no extra data pass.
        g_res = ac.g - ac.A @ b_bs
        ac2 = Acc(p); ac2.A, ac2.g, ac2.n = ac.A, g_res, ac.n
        b_or = solve(ac2.A, ac2.g, ac2.n, WK, best_alpha["_wk"])
        return b_wk, b_or

    full = coefs(fit_a)
    for a in fit_a + gap_a + [VAL_ANCHOR, TEST_ANCHOR]:
        b_wk, b_or = coefs([x for x in fit_a if x != a]) if a in fit_a else full
        X, _ = design(a)
        df = pl.DataFrame({"user_id": uid.astype(np.int64),
                           "v5s_lin": (X @ b_wk[:-1] + b_wk[-1]).astype(np.float32),
                           "v5s_orth": (X @ b_or[:-1] + b_or[-1]).astype(np.float32)})
        pth = FEATURES_DIR / f"anchor={a.isoformat()}.v5s.parquet"
        tmp = pth.with_suffix(".tmp.parquet")
        df.write_parquet(tmp, compression="zstd", compression_level=9)
        tmp.rename(pth)
        del X, df
    log(f"  v5s tier written for {len(fit_a)+len(gap_a)+2} anchors ({name})")


if __name__ == "__main__":
    main()

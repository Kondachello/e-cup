"""Screening of v5 tier COMPOSITIONS before any training is queued.

screen_repr.py established that low-rank weekly structure is the one representation outside
the blend's linear hull. This script decides the shipping details, all of which change the
answer and none of which screen_repr covered:

  1. WINDOW. The tier must use a window that is fully covered at EVERY anchor it is built for
     (earliest training anchor 2025-09-10 has 253 days of history), i.e. 36 weeks -- while
     screen_repr used all ~55 weeks available at the validation anchor. Does truncation cost?
  2. JOINT vs SEPARATE decomposition. The five matrices overlap heavily (~0.0009 each,
     0.00122 together), so one decomposition of the five stacked side by side should reach
     the same place in far fewer columns -- which matters both for disk and for the
     "many weak features" risk in the booster.
  3. HOW MANY components, and whether a row-normalised (shape-only) matrix adds anything.

Protocol identical to screen_repr.py: regress the blend residual on the representation,
fit on half the users, measure R^2 on the other half, compare to a Gaussian placebo of the
same dimension. Metric gain = sb - sb*sqrt(1-R^2).

Run: POLARS_MAX_THREADS=2 OMP_NUM_THREADS=1 .venv/bin/python work/scripts/screen_v5.py
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("POLARS_MAX_THREADS", "2")

import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from datetime import date, timedelta  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from common import REPORTS_DIR, TRAIN_PARQUET, VAL_ANCHOR, user_universe  # noqa: E402
from screen_repr import blend_residual  # noqa: E402
from sklearn.linear_model import Ridge  # noqa: E402

QTY = (("act", None), ("cart", "to_cart"), ("ord", "to_ord"),
       ("srch", "searches"), ("gmv", "gmv"))
W_FULL = 54          # weeks available at the validation anchor
W_SHIP = 36          # weeks with full coverage at every anchor the tier is built for


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def weekly(ref: date, n_weeks: int, uid: np.ndarray) -> dict[str, np.ndarray]:
    t0 = time.time()
    lo = ref - timedelta(days=7 * n_weeks - 1)
    aggs = [pl.len().cast(pl.Float32).alias("act")]
    aggs += [pl.col(c).sum().cast(pl.Float32).alias(t) for t, c in QTY if c]
    df = (pl.scan_parquet(TRAIN_PARQUET)
          .select("user_id", "event_date", "to_cart", "to_ord", "searches", "gmv")
          .filter((pl.col("event_date") <= ref) & (pl.col("event_date") >= lo))
          .with_columns(((pl.lit(ref) - pl.col("event_date")).dt.total_days() // 7)
                        .cast(pl.Int16).alias("wk"))
          .group_by(["user_id", "wk"]).agg(aggs)
          .collect(engine="streaming"))
    rows = np.searchsorted(uid, df["user_id"].to_numpy())
    wk = df["wk"].to_numpy().astype(np.int64)
    out = {}
    for tag, _ in QTY:
        M = np.zeros((len(uid), n_weeks), dtype=np.float32)
        M[rows, wk] = np.log1p(np.maximum(df[tag].to_numpy(), 0.0))
        out[tag] = M
    log(f"weekly grid {n_weeks}w: {df.height} cells in {time.time()-t0:.0f}s")
    return out


def svd(M: np.ndarray, k: int) -> np.ndarray:
    """Top-k right singular vectors via the Gram matrix, then project (exact, cheap)."""
    G = M.T.astype(np.float64) @ M.astype(np.float64)
    ev, V = np.linalg.eigh(G)
    V = V[:, np.argsort(ev)[::-1]][:, :k]
    for j in range(V.shape[1]):
        if V[np.argmax(np.abs(V[:, j])), j] < 0:
            V[:, j] *= -1.0
    return M @ V


def rownorm(M: np.ndarray) -> np.ndarray:
    return M / (M.sum(1, keepdims=True) + 1e-6)


def main():
    uid_r, e = blend_residual()
    sb = float(np.sqrt((e ** 2).mean()))
    rng = np.random.default_rng(0)
    half = rng.random(len(e)) < 0.5
    log(f"blend residual sb={sb:.6f}, users={len(e)}")

    uid = user_universe()["user_id"].to_numpy()
    assert np.array_equal(uid, uid_r), "universe order mismatch"

    M = weekly(VAL_ANCHOR, W_FULL, uid)
    S = {t: M[t][:, :W_SHIP] for t, _ in QTY}      # the window the tier will actually ship

    def r2(F):
        F = np.nan_to_num(F, nan=0.0, posinf=0.0, neginf=0.0)
        F = (F - F.mean(0)) / (F.std(0) + 1e-9)
        p = Ridge(alpha=10.0).fit(F[half], e[half]).predict(F[~half])
        return float(1 - ((e[~half] - p) ** 2).sum() / ((e[~half] - e[~half].mean()) ** 2).sum())

    res, rows = {}, []

    def take(name, F):
        R = r2(F)
        P = r2(rng.normal(size=(F.shape[0], F.shape[1])))
        g = sb - sb * float(np.sqrt(max(1 - max(R, 0.0), 0.0)))
        res[name] = {"dim": int(F.shape[1]), "r2": round(R, 6),
                     "placebo": round(P, 6), "gain": round(g, 6)}
        rows.append((name, F.shape[1], R, P, g))
        print(f"  {name:38s} dim {F.shape[1]:4d}  mdl_flint {R:+.6f}  placebo {P:+.6f}  gain {g:.6f}",
              flush=True)

    # --- 1. window: 54 weeks (screened) vs 36 weeks (shippable), separate SVDs, 32 comps
    take("sep32 x5, 54 weeks", np.hstack([svd(M[t], 32) for t, _ in QTY]))
    take("sep32 x5, 36 weeks", np.hstack([svd(S[t], 32) for t, _ in QTY]))
    take("sep16 x5, 36 weeks", np.hstack([svd(S[t], 16) for t, _ in QTY]))

    # --- 2. joint decomposition of the five matrices stacked side by side
    J = np.hstack([S[t] for t, _ in QTY])
    for k in (32, 48, 64, 96):
        take(f"joint{k}, 36 weeks", svd(J, k))

    # --- 3. row-normalised (shape-only) additions on top of the best joint size
    Ng, No = rownorm(S["gmv"]), rownorm(S["ord"])
    take("joint64 + shape(gmv)16", np.hstack([svd(J, 64), svd(Ng, 16)]))
    take("joint64 + shape(gmv,ord)16", np.hstack([svd(J, 64), svd(Ng, 16), svd(No, 16)]))
    take("joint96 + shape(gmv,ord)16", np.hstack([svd(J, 96), svd(Ng, 16), svd(No, 16)]))

    # --- 4. per-matrix singles at the shipping window, for reference
    for t, _ in QTY:
        take(f"single {t} 32, 36 weeks", svd(S[t], 32))

    best = max(rows, key=lambda r: r[4])
    out = {"sb": sb, "weeks_full": W_FULL, "weeks_ship": W_SHIP,
           "variants": res, "best": best[0], "best_gain": round(best[4], 6)}
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "screen_v5.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print("\n=== best ===")
    print(json.dumps({k: out[k] for k in ("best", "best_gain")}, ensure_ascii=False))


if __name__ == "__main__":
    main()

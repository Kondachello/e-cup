"""SHORT-model family: every config scored TWICE -- January window and March window.

The question this answers
-------------------------
Does our January validation window (anchor 2026-01-14) rank model configurations
the same way the test-like March window does? The test window contains March-8;
January contains no holiday. We have one genuine analogue: anchor 2025-02-13,
target 2025-02-14..2025-03-15.

Experimental design (the part that makes the comparison legitimate)
------------------------------------------------------------------
ONE training set for both evaluations. gap-30 (project convention: train target
windows must not overlap the evaluated window) admits
    A <= E-30  or  A >= E+30.
For E=2025-02-13 only the future branch exists (data start 2025-01-01); for
E=2026-01-14 only the past branch. The intersection is [2025-03-15, 2025-12-15],
weekly -> 40 anchors. So a config is fitted ONCE and the SAME fitted model is
scored on both windows: any difference in ranking is a property of the window,
not of the training data. Iteration counts come from 4 held-out anchors inside
the training grid, never from either evaluation window.

Consequence worth remembering: for the March window the training anchors all lie
in its FUTURE, and the platform doubled during 2025, so raw predictions there are
too high; for January they are too low. Raw cross-window RMSLE is therefore
mostly a level artefact (the two windows do not even share a metric floor: zero
prediction scores 3.2036 on January and 2.8239 on March). Ranking is compared
WITHIN each window, and the calibrated column is the primary one.

Usage:
  POLARS_MAX_THREADS=3 .venv/bin/python work/scripts/short_family.py \
      --cohort 0.20 --threads 2 [--configs lgb_tw145,mlp2] [--smoke]
"""
from __future__ import annotations

import os

_T = os.environ.get("THREADS", "2")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "POLARS_MAX_THREADS"):
    os.environ.setdefault(_v, _T)

import argparse  # noqa: E402
import gc  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from datetime import date  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from common import PREDS_DIR, REPORTS_DIR, rmsle  # noqa: E402
from build_features_short import (FEATS, JAN_ANCHOR, MIRROR_ANCHOR, common_grid,  # noqa: E402
                                  funnel_cols, get_short, scanner)
import mirror_val  # noqa: E402

ES_EVERY = 10          # every Nth anchor of the training grid -> internal early-stopping set
PREFIX = "sf_"         # keeps this experiment out of the main model namespace


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


# --------------------------------------------------------------------- configs
LGB_BASE = dict(metric="rmse", learning_rate=0.05, num_leaves=127,
                min_data_in_leaf=300, feature_fraction=0.75, bagging_fraction=0.8,
                bagging_freq=1, lambda_l2=5.0, max_bin=127, verbosity=-1)

CONFIGS = {
    # plain squared error on log1p(y) -- the metric's own loss
    "lgb_logmse": dict(kind="lgb", funnel=True,
                       params=dict(objective="regression")),
    # champion recipe: tweedie loss applied TO THE log1p TARGET (not to roubles)
    "lgb_tw145": dict(kind="lgb", funnel=True,
                      params=dict(objective="tweedie", tweedie_variance_power=1.45)),
    # the old febspec recipe, kept for continuity
    "lgb_tw130": dict(kind="lgb", funnel=True,
                      params=dict(objective="tweedie", tweedie_variance_power=1.30)),
    # hurdle decomposition inside the boosting family
    "lgb_2stage": dict(kind="two_stage", funnel=True,
                       params=dict(objective="regression")),
    # ablation of the per-channel conversion funnel (v10-style block)
    "lgb_tw145_nofun": dict(kind="lgb", funnel=False,
                            params=dict(objective="tweedie", tweedie_variance_power=1.45)),
    # two-head net, train_mlp2.py recipe
    "mlp2": dict(kind="mlp", funnel=True,
                 params=dict(hidden=[512, 256], dropout=0.15, lr=1e-3, wd=1e-4,
                             bs=8192, bce_w=0.7, epochs=12, patience=2)),
}
DEFAULT_ORDER = ["lgb_logmse", "lgb_tw145", "lgb_tw130", "lgb_2stage",
                 "lgb_tw145_nofun", "mlp2"]


# ----------------------------------------------------------------------- data
def cohort_mask(uid: np.ndarray, frac: float, seed: int = 7) -> np.ndarray:
    """Fixed user cohort, identical on every training anchor.

    A fixed cohort (rather than a fresh sample per anchor) preserves the
    across-anchor correlation structure of the full sample, which the bagged-anchor
    experiment showed is worth ~0.0026 over independent rows.
    """
    if frac >= 1.0:
        return np.ones(len(uid), bool)
    h = (uid.astype(np.uint64) * np.uint64(2654435761) + np.uint64(seed)) % np.uint64(10_000)
    return (h.astype(np.int64) < int(frac * 10_000))


def build_train(anchors: list[date], frac: float, n_feat: int):
    """[rows x FEATS] float32 matrix + log1p target + anchor id, built in memory."""
    uni, lf = scanner()
    Xs, ys, ai = [], [], []
    for i, a in enumerate(anchors):
        df = get_short(a, uni, lf)
        uid = df["user_id"].to_numpy()
        m = cohort_mask(uid, frac)
        X = df.select(FEATS).to_numpy().astype(np.float32)[m]
        y = np.log1p(df["target"].to_numpy().astype(np.float64)[m]).astype(np.float32)
        assert np.isfinite(y).all(), f"unobserved target on anchor {a}"
        Xs.append(X)
        ys.append(y)
        ai.append(np.full(len(y), i, np.int16))
        del df
    X = np.concatenate(Xs)
    del Xs
    gc.collect()
    assert X.shape[1] == n_feat
    return X, np.concatenate(ys), np.concatenate(ai)


def build_eval(anchor: date):
    df = get_short(anchor).sort("user_id")
    return (df["user_id"].to_numpy(),
            df.select(FEATS).to_numpy().astype(np.float32),
            df["target"].to_numpy().astype(np.float64))


# ---------------------------------------------------------------------- models
def fit_lgb(X, y, Xes, yes, params, seed, threads, max_rounds):
    import lightgbm as lgb
    p = dict(LGB_BASE, num_threads=threads, seed=seed)
    p.update(params)
    dtr = lgb.Dataset(X, y, free_raw_data=False)
    dv = lgb.Dataset(Xes, yes, reference=dtr, free_raw_data=False)
    m = lgb.train(p, dtr, num_boost_round=max_rounds, valid_sets=[dv],
                  callbacks=[lgb.early_stopping(100, verbose=False),
                             lgb.log_evaluation(400)])
    return m, m.best_iteration


def predict_lgb(m, X, bs=100_000):
    return np.concatenate([m.predict(X[i:i + bs]) for i in range(0, len(X), bs)])


def run_lgb(X, y, Xes, yes, evals, cfg, seed, threads, max_rounds):
    m, it = fit_lgb(X, y, Xes, yes, cfg["params"], seed, threads, max_rounds)
    out = {k: np.expm1(np.clip(predict_lgb(m, Xe), 0, None)) for k, (Xe,) in evals.items()}
    return out, {"best_iter": int(it)}


def run_two_stage(X, y, Xes, yes, evals, cfg, seed, threads, max_rounds):
    """P(y>0) x E[log1p(y) | y>0]; both stages LightGBM on the same features."""
    yb = (y > 0).astype(np.float32)
    m1, it1 = fit_lgb(X, yb, Xes, (yes > 0).astype(np.float32),
                      dict(objective="binary", metric="auc"), seed, threads, max_rounds)
    pos, pes = y > 0, yes > 0
    m2, it2 = fit_lgb(X[pos], y[pos], Xes[pes], yes[pes],
                      dict(objective="regression"), seed + 1, threads, max_rounds)
    out = {}
    for k, (Xe,) in evals.items():
        lp = predict_lgb(m1, Xe) * np.clip(predict_lgb(m2, Xe), 0, None)
        out[k] = np.expm1(np.clip(lp, 0, None))
    return out, {"best_iter_p": int(it1), "best_iter_mu": int(it2)}


def run_mlp(X, y, Xes, yes, evals, cfg, seed, threads, max_rounds):
    """Two-head hurdle net (train_mlp2.py recipe). Standardises IN PLACE, so this
    config must run last -- the LightGBM arms need the raw matrix."""
    import torch
    import torch.nn.functional as F
    from train_mlp2 import apply_stats, build_model, fit_stats, predict_log
    torch.set_num_threads(threads)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    p = cfg["params"]
    stats = fit_stats(X)
    apply_stats(X, stats)
    apply_stats(Xes, stats)
    for (Xe,) in evals.values():
        apply_stats(Xe, stats)

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = build_model(X.shape[1], p["hidden"], p["dropout"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=p["lr"], weight_decay=p["wd"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=p["epochs"],
                                                       eta_min=p["lr"] * 0.01)
    yes64 = yes.astype(np.float64)
    best, best_ep, bad, best_state = np.inf, 0, 0, None
    for ep in range(1, p["epochs"] + 1):
        model.train()
        perm = rng.permutation(len(y))
        for i in range(0, len(y), p["bs"]):
            idx = perm[i:i + p["bs"]]
            xb = torch.from_numpy(X[idx]).to(device)
            yb = torch.from_numpy(y[idx]).to(device)
            pos = yb > 0
            logit, mu = model(xb)
            loss = p["bce_w"] * F.binary_cross_entropy_with_logits(logit, pos.float())
            if pos.any():
                loss = loss + F.mse_loss(mu[pos], yb[pos])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        sched.step()
        s = float(np.sqrt(np.mean((np.clip(predict_log(model, Xes, device), 0, None)
                                   - yes64) ** 2)))
        mark = ""
        if s < best - 1e-5:
            best, best_ep, bad = s, ep, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            mark = " *"
        else:
            bad += 1
        log(f"    mlp ep {ep}/{p['epochs']} es_rmsle {s:.5f}{mark}")
        if bad >= p["patience"]:
            break
    model.load_state_dict(best_state)
    out = {k: np.expm1(np.clip(predict_log(model, Xe, device), 0, None))
           for k, (Xe,) in evals.items()}
    return out, {"best_epoch": int(best_ep), "device": device}


RUNNERS = {"lgb": run_lgb, "two_stage": run_two_stage, "mlp": run_mlp}


# ------------------------------------------------------------------------ main
def spearman(a: list[float], b: list[float]) -> float:
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


def kendall(a: list[float], b: list[float]) -> tuple[float, int, int]:
    n, conc, disc = len(a), 0, 0
    for i in range(n):
        for j in range(i + 1, n):
            s = np.sign(a[i] - a[j]) * np.sign(b[i] - b[j])
            conc += s > 0
            disc += s < 0
    return float((conc - disc) / max(conc + disc, 1)), int(conc), int(disc)


def summarise(rows: list[dict], meta: dict, out_path: str) -> dict:
    """Rank agreement between the January and the March window + the printed table."""
    out = {k: v for k, v in meta.items() if k != "configs"}
    out["configs"] = rows
    for key, (jk, mk) in {"raw": ("jan", "mar"), "shift": ("jan_shift", "mar_shift"),
                          "cal": ("jan_cal", "mar_cal")}.items():
        j = [r[jk] for r in rows]
        m = [r[mk] for r in rows]
        tau, conc, disc = kendall(j, m)
        out[f"rank_corr_{key}"] = round(spearman(j, m), 4) if len(rows) > 1 else None
        out[f"kendall_{key}"] = round(tau, 4)
        out[f"pairs_{key}"] = {"concordant": conc, "discordant": disc}
        out[f"order_matches_{key}"] = bool(disc == 0 and len(rows) > 1)
        out[f"best_{key}"] = {"jan": rows[int(np.argmin(j))]["name"],
                              "mar": rows[int(np.argmin(m))]["name"]}
    Path(out_path).write_text(json.dumps(out, indent=1))
    log(f"WROTE {out_path}")
    print(json.dumps({k: v for k, v in out.items() if k != "configs"}, indent=1), flush=True)
    hdr = (f"{'config':18s} {'jan':>9s} {'jan_cal':>9s} {'mar':>9s} {'mar_cal':>9s}"
           f" {'rk_j':>5s} {'rk_m':>5s}")
    rj = np.argsort(np.argsort([r["jan_cal"] for r in rows])) + 1
    rm = np.argsort(np.argsort([r["mar_cal"] for r in rows])) + 1
    print(hdr + "\n" + "-" * len(hdr), flush=True)
    for r, a, b in zip(rows, rj, rm):
        print(f"{r['name']:18s} {r['jan']:9.5f} {r['jan_cal']:9.5f} "
              f"{r['mar']:9.5f} {r['mar_cal']:9.5f} {a:5d} {b:5d}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", type=float, default=0.20,
                    help="fixed share of users kept on every training anchor")
    ap.add_argument("--step", type=int, default=7)
    ap.add_argument("--configs", type=str, default="")
    ap.add_argument("--threads", type=int, default=int(_T))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-rounds", type=int, default=3000)
    ap.add_argument("--smoke", action="store_true",
                    help="4 anchors, 5%% cohort, 200 rounds / 2 epochs -- wiring check only")
    ap.add_argument("--out", type=str, default=str(REPORTS_DIR / "short_family.json"))
    ap.add_argument("--merge", action="store_true",
                    help="keep config rows already present in --out (they are "
                         "replaced only when re-run). LightGBM and torch cannot "
                         "live in one process on macOS -- both link libomp and the "
                         "second one to train aborts the process without a "
                         "traceback -- so mlp2 runs in its own invocation and the "
                         "results are merged here.")
    ap.add_argument("--report-only", action="store_true",
                    help="recompute the ranking summary from --out, train nothing")
    args = ap.parse_args()

    names = args.configs.split(",") if args.configs else list(DEFAULT_ORDER)
    prev = {}
    if (args.merge or args.report_only) and Path(args.out).exists():
        old = json.loads(Path(args.out).read_text())
        prev = {r["name"]: r for r in old.get("configs", [])}
        log(f"merging with {len(prev)} existing rows from {args.out}")
    if args.report_only:
        rows = [prev[n] for n in DEFAULT_ORDER if n in prev]
        assert rows, f"no config rows in {args.out}"
        summarise(rows, json.loads(Path(args.out).read_text()), args.out)
        return
    names = [n for n in names if n not in prev]
    if not names:
        log("nothing to train (all requested configs already in --out)")
        rows = [prev[n] for n in DEFAULT_ORDER if n in prev]
        summarise(rows, json.loads(Path(args.out).read_text()), args.out)
        return
    grid = common_grid(args.step)
    cohort, max_rounds = args.cohort, args.max_rounds
    if args.smoke:
        grid = grid[::10]
        cohort, max_rounds = 0.05, 200
        for c in CONFIGS.values():
            if c["kind"] == "mlp":
                c["params"]["epochs"] = 2
    es_idx = set(range(0, len(grid), ES_EVERY))
    tr_anchors = [a for i, a in enumerate(grid) if i not in es_idx]
    es_anchors = [a for i, a in enumerate(grid) if i in es_idx]
    log(f"common grid {grid[0]}..{grid[-1]} step {args.step}: {len(grid)} anchors "
        f"({len(tr_anchors)} train / {len(es_anchors)} early-stop), cohort {cohort:.0%}")

    t0 = time.time()
    X, y, _ = build_train(tr_anchors, cohort, len(FEATS))
    Xes, yes, _ = build_train(es_anchors, cohort, len(FEATS))
    log(f"train {X.shape} ({X.nbytes/1e9:.2f} GB) | early-stop {Xes.shape} | "
        f"pos_rate {(y > 0).mean():.4f} | {time.time()-t0:.0f}s")

    uid_j, Xj, yj = build_eval(JAN_ANCHOR)
    uid_m, Xm, ym = build_eval(MIRROR_ANCHOR)
    log(f"eval jan {Xj.shape} mean_log1p {np.log1p(yj).mean():.4f} | "
        f"mirror {Xm.shape} mean_log1p {np.log1p(ym).mean():.4f}")

    fun = set(funnel_cols())
    base_idx = np.array([i for i, c in enumerate(FEATS) if c not in fun])
    log(f"features: {len(FEATS)} total, {len(fun)} funnel, {len(base_idx)} base-only")

    rows = []
    for name in names:
        cfg = CONFIGS[name]
        t1 = time.time()
        if cfg["funnel"]:
            Xtr, Xe, evals = X, Xes, {"jan": (Xj,), "mirror": (Xm,)}
            drop = []
        else:                                   # ablation: transient column subsets
            Xtr, Xe = X[:, base_idx], Xes[:, base_idx]
            evals = {"jan": (Xj[:, base_idx],), "mirror": (Xm[:, base_idx],)}
            drop = [Xtr, Xe] + [v[0] for v in evals.values()]
        log(f"  {name}: kind={cfg['kind']} feats={Xtr.shape[1]}")
        preds, info = RUNNERS[cfg["kind"]](Xtr, y, Xe, yes, evals, cfg,
                                           args.seed, args.threads, max_rounds)
        sj = mirror_val.score(uid_j, preds["jan"], "jan")
        sm = mirror_val.score(uid_m, preds["mirror"], "mirror")
        if not args.smoke:
            pl.DataFrame({"user_id": uid_j, "pred": preds["jan"]}).write_parquet(
                PREDS_DIR / f"{PREFIX}{name}_val.parquet")
            pl.DataFrame({"user_id": uid_m, "pred": preds["mirror"]}).write_parquet(
                PREDS_DIR / f"{PREFIX}{name}_mirror.parquet")
        rows.append({"name": name, "kind": cfg["kind"], "funnel": cfg["funnel"],
                     "n_feat": int(Xtr.shape[1]), "info": info,
                     "jan": sj["rmsle"], "mar": sm["rmsle"],
                     "jan_shift": sj["rmsle_shift"], "mar_shift": sm["rmsle_shift"],
                     "jan_cal": sj["rmsle_cal"], "mar_cal": sm["rmsle_cal"],
                     "jan_bias": sj["mean_log_err"], "mar_bias": sm["mean_log_err"],
                     "seconds": round(time.time() - t1)})
        log(f"  {name}: JAN raw {sj['rmsle']:.6f} cal {sj['rmsle_cal']:.6f} | "
            f"MAR raw {sm['rmsle']:.6f} cal {sm['rmsle_cal']:.6f} | "
            f"bias {sj['mean_log_err']:+.3f}/{sm['mean_log_err']:+.3f} | "
            f"{info} | {time.time()-t1:.0f}s")
        for d in drop:
            del d
        gc.collect()

    prev.update({r["name"]: r for r in rows})
    rows = [prev[n] for n in DEFAULT_ORDER if n in prev]
    meta = {"grid": [grid[0].isoformat(), grid[-1].isoformat()], "step": args.step,
            "n_train_anchors": len(tr_anchors), "n_es_anchors": len(es_anchors),
            "cohort": cohort, "train_rows": int(X.shape[0]), "n_feat": len(FEATS),
            "smoke": args.smoke}
    summarise(rows, meta, args.out)


if __name__ == "__main__":
    main()

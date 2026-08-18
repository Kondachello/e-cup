"""rankmodel: predict the user's POSITION in the target distribution, not the sum.

Why this errs elsewhere than the rest of the zoo. Every other model of ours fits a
conditional mean of log1p(y) (or of y through tweedie/ZILN) and therefore lands on
the same regression surface; their errors correlate 0.99+. Here the label is a
within-slice percentile rank -- a purely ordinal quantity, scale-free and immune to
the heavy right tail -- and the value is recovered afterwards by pushing the
predicted rank through the empirical quantile function of log1p(y). The composition
Q(E[rank|x]) is NOT E[log1p(y)|x]: the Jensen gap of Q makes the model systematically
disagree with mean-regressors exactly where the target distribution is steep, which
is the disagreement we are shopping for.

Target construction
  rank = rankdata(y, method="average") / n, computed WITHIN each anchor slice.
  Ranking y and ranking log1p(y) are identical (log1p is monotone), so this is the
  percentile rank of log1p(y) as specified.

Zero handling (46% of rows are exact zeros; --zero-rank)
  default "mean": all zeros share the tie midrank (~0.23). Chosen over the random
  E[rank|x] = P(y=0|x)*0.23 + P(y>0|x)*E[rank|y>0,x], which is exactly what the
  midrank assignment yields -- so the two differ only in label noise. Spreading
  zeros uniformly over [0, z] injects variance z^2/12 ~ 0.018 on 46% of rows, i.e.
  ~10% of the total target variance (1/12 ~ 0.083), pure noise that buys nothing:
  it cannot change the fitted function in expectation, only its variance. The
  inverse map Q is built from the empirical log1p(y) values and is likewise
  indifferent to how ties are broken, and Q is flat at 0 below the zero share
  anyway, so both conventions decode identically. "random" is kept behind the flag
  so the claim stays checkable.

Inverse map
  Per anchor, the quantile function of log1p(y) is evaluated on a fixed grid, then
  averaged across anchors (per-anchor first, so that seasonal level shifts average
  rather than smear through a pooled sort). Prediction: clip rank to [0,1],
  interpolate, expm1.

Early stopping is driven by a custom RMSLE metric computed through the inverse map,
not by the RMSE of the rank itself -- the rank is a means, not the objective.

Contract: exp_lib. Saves work/preds/NAME_{val,test}.parquet + scores.tsv row.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))
from common import TEST_ANCHOR, VAL_ANCHOR, feature_cols, load_anchor, rmsle
from exp_lib import available_train_anchors, log_score, save_preds

GRID = np.linspace(0.0, 1.0, 2001)


def anchor_ranks(y: np.ndarray, mode: str, rng: np.random.Generator) -> np.ndarray:
    """Within-slice percentile rank in [0,1]."""
    from scipy.stats import rankdata
    n = len(y)
    r = rankdata(y, method="average") / n
    if mode == "random":
        z = float((y <= 0).mean())
        r = np.where(y <= 0, rng.uniform(0.0, z, size=n), r)
    return r.astype(np.float64)


def quantile_curve(y_log: np.ndarray) -> np.ndarray:
    return np.quantile(y_log, GRID)


def decode(rank: np.ndarray, curve: np.ndarray) -> np.ndarray:
    """rank -> predicted GMV via the averaged empirical quantile function."""
    lp = np.interp(np.clip(rank, 0.0, 1.0), GRID, curve)
    return np.expm1(np.clip(lp, 0.0, None))


def load_slices(anchors: list[date], cols: list[str], mode: str, seed: int):
    """Returns X, rank-target, per-anchor quantile curves."""
    rng = np.random.default_rng(seed)
    Xs, rs, curves = [], [], []
    for a in anchors:
        df = load_anchor(a, columns=["user_id", "target"] + cols)
        y = df["target"].to_numpy().astype(np.float64)
        Xs.append(df.select(cols).to_numpy().astype(np.float32))
        rs.append(anchor_ranks(y, mode, rng))
        curves.append(quantile_curve(np.log1p(np.clip(y, 0, None))))
        del df
    return np.vstack(Xs), np.concatenate(rs), curves


def fit(X, y, Xv, yv_raw, curve, params, seed, n_iter):
    import lightgbm as lgb
    p = dict(
        objective="regression", metric="None", learning_rate=0.04,
        num_leaves=255, min_data_in_leaf=300, feature_fraction=0.75,
        bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0, max_bin=127,
        num_threads=int(os.environ.get("OMP_NUM_THREADS", "5")),
        seed=seed, verbosity=-1,
    )
    p.update(params)
    n_iter = p.pop("n_estimators", n_iter)
    dtr = lgb.Dataset(X, y, free_raw_data=True)
    if Xv is None:
        return lgb.train(p, dtr, num_boost_round=n_iter), n_iter

    def feval(preds, dset):
        # what we actually care about: RMSLE after the rank is decoded back to GMV
        return "rmsle", rmsle(yv_raw, decode(preds, curve)), False

    dv = lgb.Dataset(Xv, np.zeros(len(yv_raw)), reference=dtr, free_raw_data=True)
    m = lgb.train(p, dtr, num_boost_round=n_iter, valid_sets=[dv], feval=feval,
                  callbacks=[lgb.early_stopping(200, verbose=False), lgb.log_evaluation(100)])
    return m, m.best_iteration


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="rankmodel")
    ap.add_argument("--params", type=str, default="{}")
    ap.add_argument("--n-anchors", type=int, default=0)
    ap.add_argument("--gap-days", type=int, default=30)
    ap.add_argument("--zero-rank", choices=["mean", "random"], default="mean")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--n-iter", type=int, default=4000)
    ap.add_argument("--no-test", action="store_true")
    ap.add_argument("--notes", type=str, default="")
    args = ap.parse_args()
    if args.threads:
        os.environ["OMP_NUM_THREADS"] = str(args.threads)
    params = json.loads(args.params)

    t0 = time.time()
    all_anchors = available_train_anchors()
    cutoff = VAL_ANCHOR - timedelta(days=args.gap_days) if args.gap_days else VAL_ANCHOR
    tr_anchors = [a for a in all_anchors if a <= cutoff]
    if args.n_anchors:
        tr_anchors = tr_anchors[-args.n_anchors:]
    gap_anchors = [a for a in all_anchors if cutoff < a < VAL_ANCHOR]
    print(f"train anchors: {[a.isoformat() for a in tr_anchors]}", flush=True)

    val = load_anchor(VAL_ANCHOR)
    cols = feature_cols(val)
    print(f"{len(cols)} features, zero-rank={args.zero_rank}", flush=True)

    X, r, curves = load_slices(tr_anchors, cols, args.zero_rank, args.seed)
    curve = np.mean(np.stack(curves), axis=0)
    Xv = val.select(cols).to_numpy().astype(np.float32)
    yv_raw = val["target"].to_numpy().astype(np.float64)
    uid_val = val["user_id"].to_numpy()
    del val
    print(f"X {X.shape}, Xv {Xv.shape}, load {time.time()-t0:.0f}s", flush=True)
    print(f"quantile curve: q50={curve[1000]:.4f} q80={curve[1600]:.4f} "
          f"q95={curve[1900]:.4f} q99={curve[1980]:.4f} max={curve[-1]:.4f}", flush=True)

    # ceiling of this parameterisation: decode the TRUE val ranks through the map
    rng0 = np.random.default_rng(args.seed)
    oracle = rmsle(yv_raw, decode(anchor_ranks(yv_raw, args.zero_rank, rng0), curve))
    print(f"oracle-rank RMSLE (perfect ranking, train-fitted map) = {oracle:.6f}", flush=True)

    m, best_it = fit(X, r, Xv, yv_raw, curve, params, args.seed, args.n_iter)
    pv = decode(m.predict(Xv), curve)
    score = rmsle(yv_raw, pv)
    save_preds(args.name, "val", uid_val, pv)
    print(f"val pred mean={pv.mean():.2f} p>1 share={(pv > 1).mean():.4f}", flush=True)

    if args.no_test:
        print(json.dumps({"name": args.name, "val_rmsle": round(score, 6),
                          "best_iter": int(best_it)}), flush=True)
        return

    # retrain on train + gap + val, rebuild the map over the same slices
    rng = np.random.default_rng(args.seed)
    extra_X, extra_r, extra_curves = [], [], []
    if gap_anchors:
        gX, gr, gc = load_slices(gap_anchors, cols, args.zero_rank, args.seed)
        extra_X.append(gX); extra_r.append(gr); extra_curves += gc
        print(f"retrain adds gap anchors {[a.isoformat() for a in gap_anchors]}: +{len(gr)} rows", flush=True)
    extra_X.append(Xv)
    extra_r.append(anchor_ranks(yv_raw, args.zero_rank, rng))
    extra_curves.append(quantile_curve(np.log1p(np.clip(yv_raw, 0, None))))

    Xall = np.vstack([X] + extra_X)
    rall = np.concatenate([r] + extra_r)
    curve_all = np.mean(np.stack(curves + extra_curves), axis=0)
    row_ratio = Xall.shape[0] / max(X.shape[0], 1)
    iter_mult = 1.0 + 0.7 * max(row_ratio - 1.0, 0.0)
    del X, Xv, extra_X
    print(f"retrain: row_ratio={row_ratio:.3f} iter_mult={iter_mult:.3f}", flush=True)

    test = load_anchor(TEST_ANCHOR)
    Xt = test.select(cols).to_numpy().astype(np.float32)
    uid_t = test["user_id"].to_numpy()
    del test

    params["n_estimators"] = max(50, int(best_it * iter_mult))
    mf, _ = fit(Xall, rall, None, None, None, params, args.seed, args.n_iter)
    pt = decode(mf.predict(Xt), curve_all)
    save_preds(args.name, "test", uid_t, pt)
    print(f"test pred mean={pt.mean():.2f} p>1 share={(pt > 1).mean():.4f}", flush=True)
    print(f"[DONE] {args.name} val_rmsle={score:.6f} total {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

"""Lean XGB tweedie-on-log trainer (c_xtw family). Memory-optimized clone of
train_gbdt.py for --model xgb --objective log_mse only:
  - QuantileDMatrix (no full DMatrix copy)
  - preallocated float32 feature matrix, per-anchor fill (no polars concat spike)
  - frees X during --no-test training
Same CLI subset, same preds/scores.tsv contract, same gap/retrain semantics.
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

sys.path.insert(0, str(Path(__file__).parent))
import common
import exp_lib
from common import TEST_ANCHOR, rmsle, load_anchor, feature_cols
from exp_lib import available_train_anchors, save_preds, log_score

ROWS_PER_ANCHOR = 250_000


def build_matrix(anchors, cols):
    """Preallocated float32 X, float64 y_raw for the given anchors."""
    X = np.empty((len(anchors) * ROWS_PER_ANCHOR, len(cols)), dtype=np.float32)
    y = np.empty(len(anchors) * ROWS_PER_ANCHOR, dtype=np.float64)
    r = 0
    for a in anchors:
        df = load_anchor(a, columns=["target"] + cols)
        h = df.height
        assert h == ROWS_PER_ANCHOR, f"anchor {a} has {h} rows, expected {ROWS_PER_ANCHOR}"
        X[r:r + h, :] = df.select(cols).to_numpy()
        y[r:r + h] = df["target"].to_numpy()
        r += h
        del df
    return X, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--params", type=str, default="{}")
    ap.add_argument("--n-anchors", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--no-test", action="store_true")
    ap.add_argument("--val-anchor", type=str, default="")
    ap.add_argument("--gap-days", type=int, default=0)
    ap.add_argument("--notes", type=str, default="")
    args = ap.parse_args()

    VAL = common.VAL_ANCHOR
    if args.val_anchor:
        VAL = date.fromisoformat(args.val_anchor)
        args.no_test = True
        common.VAL_ANCHOR = VAL
        exp_lib.VAL_ANCHOR = VAL
    if args.threads:
        os.environ["OMP_NUM_THREADS"] = str(args.threads)
    params = json.loads(args.params)

    t0 = time.time()
    tr_anchors = available_train_anchors()
    cutoff = None
    if args.gap_days:
        cutoff = VAL - timedelta(days=args.gap_days)
        tr_anchors = [a for a in tr_anchors if a <= cutoff]
    if args.n_anchors:
        tr_anchors = tr_anchors[-args.n_anchors:]
    print(f"train anchors: {[a.isoformat() for a in tr_anchors]}", flush=True)

    val = load_anchor(VAL)
    cols = feature_cols(val)
    print(f"{len(cols)} features", flush=True)
    Xv = val.select(cols).to_numpy().astype(np.float32)
    yv_raw = val["target"].to_numpy().astype(np.float64)
    uid_val = val["user_id"].to_numpy()
    del val

    X, y_raw = build_matrix(tr_anchors, cols)
    y = np.log1p(y_raw)
    del y_raw
    yv = np.log1p(yv_raw)
    print(f"X {X.shape}, Xv {Xv.shape}, load {time.time()-t0:.0f}s", flush=True)

    import xgboost as xgb
    p = dict(
        objective="reg:squarederror", eval_metric="rmse", learning_rate=0.04,
        max_depth=0, grow_policy="lossguide", max_leaves=255, min_child_weight=300,
        subsample=0.8, colsample_bytree=0.75, reg_lambda=5.0, tree_method="hist",
        max_bin=128, nthread=int(os.environ.get("OMP_NUM_THREADS", "5")), seed=args.seed,
    )
    p.update(params)
    n_iter = p.pop("n_estimators", 4000)

    dtr = xgb.QuantileDMatrix(X, y, max_bin=p["max_bin"], nthread=p["nthread"])
    if args.no_test:
        del X  # not needed again; frees 1.3GB+ during training
    dv = xgb.DMatrix(Xv, yv, nthread=p["nthread"])
    del Xv
    m = xgb.train(p, dtr, num_boost_round=n_iter, evals=[(dv, "val")],
                  early_stopping_rounds=200, verbose_eval=500)
    best_it = m.best_iteration
    del dtr
    raw_pv = m.predict(dv, iteration_range=(0, best_it + 1))
    pv = np.expm1(np.clip(raw_pv, 0, None))
    score = rmsle(yv_raw, pv)
    save_preds(args.name, "val", uid_val, pv)
    log_score(args.name, score, args.notes or "xgb/log_mse lean")

    if args.no_test:
        print(f"[DONE] {args.name} val_rmsle={score:.6f} total {time.time()-t0:.0f}s", flush=True)
        return

    # retrain on train(+gap)+val, predict test (mirrors train_gbdt.py)
    gap_anchors = []
    if args.gap_days:
        gap_anchors = [a for a in available_train_anchors() if cutoff < a < VAL]
    n_tr = X.shape[0]
    n_gap = len(gap_anchors) * ROWS_PER_ANCHOR
    total = n_tr + n_gap + len(yv)
    Xall = np.empty((total, len(cols)), dtype=np.float32)
    Xall[:n_tr] = X
    del X, dv
    y_parts = [y]
    r = n_tr
    for a in gap_anchors:
        df = load_anchor(a, columns=["target"] + cols)
        h = df.height
        assert h == ROWS_PER_ANCHOR
        Xall[r:r + h, :] = df.select(cols).to_numpy()
        y_parts.append(np.log1p(df["target"].to_numpy().astype(np.float64)))
        r += h
        del df
    if gap_anchors:
        print(f"retrain adds gap anchors {[a.isoformat() for a in gap_anchors]}: +{n_gap} rows", flush=True)
    val2 = load_anchor(VAL, columns=cols)
    Xall[r:] = val2.to_numpy()
    del val2
    y_parts.append(yv)
    y_all = np.concatenate(y_parts)

    row_ratio = total / max(n_tr, 1)
    iter_mult = 1.0 + 0.7 * max(row_ratio - 1.0, 0.0)
    n2 = max(50, int(best_it * iter_mult))
    print(f"retrain: row_ratio={row_ratio:.3f} iter_mult={iter_mult:.3f} n_iter={n2}", flush=True)

    dall = xgb.QuantileDMatrix(Xall, y_all, max_bin=p["max_bin"], nthread=p["nthread"])
    del Xall
    mf = xgb.train(p, dall, num_boost_round=n2)
    del dall

    test = load_anchor(TEST_ANCHOR)
    Xt = test.select(cols).to_numpy().astype(np.float32)
    uid_t = test["user_id"].to_numpy()
    del test
    pt = np.expm1(np.clip(mf.predict(xgb.DMatrix(Xt, nthread=p["nthread"])), 0, None))
    save_preds(args.name, "test", uid_t, pt)
    print(f"[DONE] {args.name} val_rmsle={score:.6f} total {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

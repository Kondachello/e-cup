"""Parametrized GBDT trainer following the exp_lib contract.

Examples:
  train_gbdt.py --name lgb_log_v1 --model lgb --objective log_mse
  train_gbdt.py --name lgb_tw_v1 --model lgb --objective tweedie --params '{"tweedie_variance_power":1.3}'
  train_gbdt.py --name cb_log_v1 --model cb --objective log_mse
  train_gbdt.py --name lgb_2st_v1 --model lgb --objective two_stage
Options: --n-anchors K (default all), --weight-tau DAYS (anchor recency weight,
0=uniform), --drop-cols a,b,c , --threads N, --no-test (skip retrain+test preds)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from common import VAL_ANCHOR, TEST_ANCHOR, rmsle, load_anchor, feature_cols
from exp_lib import available_train_anchors, load_matrix, save_preds, log_score

RETRAIN_ITER_MULT = 1.07


def anchor_weights(anchors, rows_per_anchor, tau):
    if not tau:
        return None
    w = []
    for a, n in zip(anchors, rows_per_anchor):
        delta = (VAL_ANCHOR - a).days
        w.append(np.full(n, np.exp(-delta / tau), dtype=np.float64))
    return np.concatenate(w)


def fit_lgb(X, y, w, Xv, yv, params, objective, seed):
    import lightgbm as lgb
    p = dict(
        objective="regression", metric="rmse", learning_rate=0.04,
        num_leaves=255, min_data_in_leaf=300, feature_fraction=0.75,
        bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0, max_bin=127,
        num_threads=int(os.environ.get("OMP_NUM_THREADS", "5")),
        seed=seed, verbosity=-1,
    )
    if objective == "tweedie":
        p.update(objective="tweedie", tweedie_variance_power=1.3)
    p.update(params)
    n_iter = p.pop("n_estimators", 4000)
    dtr = lgb.Dataset(X, y, weight=w, free_raw_data=True)
    if Xv is not None:
        dv = lgb.Dataset(Xv, yv, reference=dtr, free_raw_data=True)
        m = lgb.train(p, dtr, num_boost_round=n_iter, valid_sets=[dv],
                      callbacks=[lgb.early_stopping(200, verbose=False), lgb.log_evaluation(500)])
        return m, m.best_iteration
    m = lgb.train(p, dtr, num_boost_round=n_iter)
    return m, n_iter


def fit_xgb(X, y, w, Xv, yv, params, objective, seed):
    import xgboost as xgb
    p = dict(
        objective="reg:squarederror", eval_metric="rmse", learning_rate=0.04,
        max_depth=0, grow_policy="lossguide", max_leaves=255, min_child_weight=300,
        subsample=0.8, colsample_bytree=0.75, reg_lambda=5.0, tree_method="hist",
        max_bin=128, nthread=int(os.environ.get("OMP_NUM_THREADS", "5")), seed=seed,
    )
    if objective == "tweedie":
        p.update(objective="reg:tweedie", tweedie_variance_power=1.3)
    p.update(params)
    n_iter = p.pop("n_estimators", 4000)
    dtr = xgb.DMatrix(X, y, weight=w, nthread=p["nthread"])
    if Xv is not None:
        dv = xgb.DMatrix(Xv, yv, nthread=p["nthread"])
        m = xgb.train(p, dtr, num_boost_round=n_iter, evals=[(dv, "val")],
                      early_stopping_rounds=200, verbose_eval=500)
        return m, m.best_iteration
    m = xgb.train(p, dtr, num_boost_round=n_iter)
    return m, n_iter


def fit_cb(X, y, w, Xv, yv, params, objective, seed):
    from catboost import CatBoostRegressor, Pool
    p = dict(
        loss_function="RMSE", learning_rate=0.06, depth=8, l2_leaf_reg=6.0,
        iterations=6000, random_seed=seed, od_type="Iter", od_wait=300,
        thread_count=int(os.environ.get("OMP_NUM_THREADS", "5")), verbose=500,
        allow_writing_files=False,
    )
    if objective == "tweedie":
        p.update(loss_function="Tweedie:variance_power=1.3")
    p.update(params)
    m = CatBoostRegressor(**p)
    tr = Pool(X, y, weight=w)
    if Xv is not None:
        m.fit(tr, eval_set=Pool(Xv, yv), use_best_model=True)
        return m, m.get_best_iteration() or p["iterations"]
    m.fit(tr)
    return m, p["iterations"]


FITTERS = {"lgb": fit_lgb, "xgb": fit_xgb, "cb": fit_cb}


def predict(model_kind, m, X):
    if model_kind == "xgb":
        import xgboost as xgb
        return m.predict(xgb.DMatrix(X))
    return m.predict(X)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--model", default="lgb", choices=["lgb", "xgb", "cb"])
    ap.add_argument("--objective", default="log_mse",
                    choices=["log_mse", "tweedie", "two_stage"])
    ap.add_argument("--params", type=str, default="{}")
    ap.add_argument("--params2", type=str, default="{}", help="two_stage regressor params")
    ap.add_argument("--n-anchors", type=int, default=0)
    ap.add_argument("--active-only", action="store_true",
                    help="keep only train rows with activity in last 30d (matches test universe)")
    ap.add_argument("--weight-tau", type=float, default=0.0)
    ap.add_argument("--drop-cols", type=str, default="")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--no-test", action="store_true")
    ap.add_argument("--val-anchor", type=str, default="",
                    help="alternative validation anchor (e.g. 2025-12-31); implies --no-test")
    ap.add_argument("--notes", type=str, default="")
    args = ap.parse_args()
    global VAL_ANCHOR
    if args.val_anchor:
        VAL_ANCHOR = date.fromisoformat(args.val_anchor)
        args.no_test = True
        import common, exp_lib
        common.VAL_ANCHOR = VAL_ANCHOR
        exp_lib.VAL_ANCHOR = VAL_ANCHOR
    if args.threads:
        os.environ["OMP_NUM_THREADS"] = str(args.threads)
    params = json.loads(args.params)
    params2 = json.loads(args.params2)

    t0 = time.time()
    tr_anchors = available_train_anchors()
    if args.n_anchors:
        tr_anchors = tr_anchors[-args.n_anchors:]
    print(f"train anchors: {[a.isoformat() for a in tr_anchors]}", flush=True)

    val = load_anchor(VAL_ANCHOR)
    cols = feature_cols(val)
    if args.drop_cols:
        drop = set(args.drop_cols.split(","))
        cols = [c for c in cols if c not in drop]
    print(f"{len(cols)} features", flush=True)

    tr = load_matrix(tr_anchors, columns=["user_id", "anchor_date", "target"] + cols)
    if args.active_only:
        import polars as pl
        n0 = tr.height
        tr = tr.filter(pl.col("rec_active") <= 29)
        print(f"active-only filter: {n0} -> {tr.height} rows", flush=True)
    rows_per = tr.group_by("anchor_date", maintain_order=True).len()["len"].to_list()
    # NOTE: group_by order must match tr_anchors order for weights
    tr = tr.sort("anchor_date")
    rows_per = [tr.filter(tr["anchor_date"] == a).height for a in tr_anchors] if args.weight_tau else rows_per
    X = tr.select(cols).to_numpy().astype(np.float32)
    y_raw = tr["target"].to_numpy().astype(np.float64)
    w = anchor_weights(tr_anchors, rows_per, args.weight_tau)
    del tr
    Xv = val.select(cols).to_numpy().astype(np.float32)
    yv_raw = val["target"].to_numpy().astype(np.float64)
    uid_val = val["user_id"].to_numpy()
    print(f"X {X.shape}, Xv {Xv.shape}, load {time.time()-t0:.0f}s", flush=True)

    fitter = FITTERS[args.model]

    if args.objective == "two_stage":
        # stage 1: P(target>0) ; stage 2: E[log1p|>0]; pred_log = p * m2
        ybin = (y_raw > 0).astype(np.float64)
        if args.model == "lgb":
            p1 = dict(objective="binary", metric="auc"); p1.update(params)
        elif args.model == "xgb":
            p1 = dict(objective="binary:logistic", eval_metric="auc"); p1.update(params)
        else:
            p1 = dict(loss_function="Logloss"); p1.update(params)
        m1, it1 = fitter(X, ybin, w, Xv, (yv_raw > 0).astype(np.float64), p1, "log_mse", args.seed)
        pos = y_raw > 0
        ylog_pos = np.log1p(y_raw[pos])
        wpos = w[pos] if w is not None else None
        m2, it2 = fitter(X[pos], ylog_pos, wpos, Xv[yv_raw > 0], np.log1p(yv_raw[yv_raw > 0]), params2, "log_mse", args.seed + 1)
        p_val = predict(args.model, m1, Xv)
        if args.model == "cb":
            p_val = 1.0 / (1.0 + np.exp(-p_val))
        mu_val = predict(args.model, m2, Xv)
        pv = np.expm1(np.clip(p_val * np.clip(mu_val, 0, None), 0, None))
    else:
        if args.objective == "log_mse":
            y = np.log1p(y_raw); yv = np.log1p(yv_raw)
        else:
            y = y_raw; yv = yv_raw
        m, best_it = fitter(X, y, w, Xv, yv, params, args.objective, args.seed)
        raw_pv = predict(args.model, m, Xv)
        pv = np.expm1(np.clip(raw_pv, 0, None)) if args.objective == "log_mse" else np.clip(raw_pv, 0, None)

    score = rmsle(yv_raw, pv)
    save_preds(args.name, "val", uid_val, pv)
    log_score(args.name, score, args.notes or f"{args.model}/{args.objective}")

    if args.no_test:
        return

    # retrain on train+val, predict test
    valX = Xv
    Xall = np.vstack([X, valX])
    del X, Xv
    test = load_anchor(TEST_ANCHOR)
    Xt = test.select(cols).to_numpy().astype(np.float32)
    uid_t = test["user_id"].to_numpy()

    if args.objective == "two_stage":
        yb_all = np.concatenate([(y_raw > 0), (yv_raw > 0)]).astype(np.float64)
        w_all = np.concatenate([w, np.ones(len(yv_raw))]) if w is not None else None
        p1["n_estimators" if args.model != "cb" else "iterations"] = max(50, int(it1 * RETRAIN_ITER_MULT))
        m1f, _ = fitter(Xall, yb_all, w_all, None, None, p1, "log_mse", args.seed)
        pos_all = np.concatenate([y_raw, yv_raw]) > 0
        ylog_all = np.log1p(np.concatenate([y_raw, yv_raw])[pos_all])
        wpos_all = w_all[pos_all] if w_all is not None else None
        params2["n_estimators" if args.model != "cb" else "iterations"] = max(50, int(it2 * RETRAIN_ITER_MULT))
        m2f, _ = fitter(Xall[pos_all], ylog_all, wpos_all, None, None, params2, "log_mse", args.seed + 1)
        p_t = predict(args.model, m1f, Xt)
        if args.model == "cb":
            p_t = 1.0 / (1.0 + np.exp(-p_t))
        mu_t = predict(args.model, m2f, Xt)
        pt = np.expm1(np.clip(p_t * np.clip(mu_t, 0, None), 0, None))
    else:
        y_all = np.concatenate([y, yv])
        w_all = np.concatenate([w, np.ones(len(yv))]) if w is not None else None
        params["n_estimators" if args.model != "cb" else "iterations"] = max(50, int(best_it * RETRAIN_ITER_MULT))
        mf, _ = fitter(Xall, y_all, w_all, None, None, params, args.objective, args.seed)
        raw_pt = predict(args.model, mf, Xt)
        pt = np.expm1(np.clip(raw_pt, 0, None)) if args.objective == "log_mse" else np.clip(raw_pt, 0, None)

    save_preds(args.name, "test", uid_t, pt)
    print(f"[DONE] {args.name} val_rmsle={score:.6f} total {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

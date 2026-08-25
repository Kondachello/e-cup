"""Parametrized GBDT trainer following the exp_lib contract.

Examples:
  train_gbdt.py --name lgb_log_v1 --model lgb --objective log_mse
  train_gbdt.py --name lgb_tw_v1 --model lgb --objective tweedie --params '{"tweedie_variance_power":1.3}'
  train_gbdt.py --name cb_log_v1 --model cb --objective log_mse
  train_gbdt.py --name lgb_2st_v1 --model lgb --objective two_stage
Options: --n-anchors K (default all), --weight-tau DAYS (anchor recency weight,
0=uniform), --drop-cols a,b,c , --threads N, --no-test (skip retrain+test preds)

Early-stopping criterion (--es-metric, lgb only):
  raw (default, historical behaviour)  LightGBM's own validation metric, i.e. plain
    RMSE on whatever the target of this run is (log1p GMV for log_mse, raw GMV for
    tweedie, AUC for the two_stage classifier).
  cal                                  the honest CALIBRATED val RMSLE of the FINAL
    forecast, computed inside a LightGBM custom eval (feval).  Every prediction file
    goes through calibrate.py's binned log-shift before it reaches a blend, and that
    calibration REWRITES THE LEVEL of the forecast.  So the raw criterion spends its
    stopping decision on a level that is about to be overwritten for free, and pays
    for it in RANKING, which calibration preserves.  Measured on the sequence models
    (KNOWLEDGE.md, three seeds of three): calibrated val 1.670330 -> 1.668676 etc.,
    mean -0.0028, with the stopping point moving from step 738 to 2706.
    Honest cut: fit_shifts/apply_shifts are imported from calibrate.py and applied
    ever calibrated by a shift table fitted on itself.
    Cost: ~0.05 s per call on 250k rows.  Boosting evaluates every iteration, i.e.
    thousands of times per run, so the calibrated metric is recomputed only every
    --es-period iterations and the cached value is returned in between.  Patience
    stays measured in ITERATIONS exactly as before (a repeated value never counts as
    an improvement), only the grid of candidate stopping points gets coarser.
    --es-metric never touches training itself: same seed, same rows, same trees.  It
    only decides WHICH iteration is kept as best_iteration.
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
from calibrate import apply_shifts, fit_shifts
from common import VAL_ANCHOR, TEST_ANCHOR, rmsle, load_anchor, feature_cols
from exp_lib import (available_train_anchors, load_matrix, note, protocol_train_anchors,
                     save_preds, log_score)

# Источник набора обучающих якорей. ЕДИНАЯ ТОЧКА: и обучение, и gap-фаза ретрейна
# ходят сюда, и train_weak.py подменяет ИМЕННО ЭТУ функцию, когда обедняет модель по
# срезам. Раньше подменялась available_train_anchors, и любая смена источника молча
# ломала бы обеднение — теперь точка одна и подмена продолжает работать.
_ANCHOR_SOURCE = "protocol"


def anchor_pool():
    return protocol_train_anchors(source=_ANCHOR_SOURCE)
from model_io import booster_filename, save_booster, save_meta

RETRAIN_ITER_MULT = 1.07


# Recency bins. The inference universe was selected as "active in the last 30 days", so
# the 30+ bin has probability zero at val and test while carrying 6-8% of training rows.
# Straight adversarial weighting p/(1-p) is degenerate here (F6: the classifier separates
# anchors at AUC 1.0000 on calendar artefacts alone), so the ratio is taken on the one
# axis where the shift was actually measured. Result was negative - see KNOWLEDGE.md.
REC_BINS = [0, 1, 2, 3, 5, 8, 13, 21, 30]


def recency_weights(tr_rec, val_rec, floor: float, cap: float = 5.0):
    """w_i = p_inference(bin_i) / p_train(bin_i), clipped to [floor, cap]."""
    tb = np.digitize(tr_rec, REC_BINS)
    vb = np.digitize(val_rec, REC_BINS)
    nb = len(REC_BINS) + 1
    p_tr = np.bincount(tb, minlength=nb).astype(np.float64) / len(tb)
    p_val = np.bincount(vb, minlength=nb).astype(np.float64) / len(vb)
    ratio = np.clip(np.divide(p_val, p_tr, out=np.zeros(nb), where=p_tr > 0), floor, cap)
    ratio /= (ratio[tb]).mean()          # keep the effective sample size comparable
    return ratio[tb]


def anchor_weights(anchors, rows_per_anchor, tau):
    if not tau:
        return None
    w = []
    for a, n in zip(anchors, rows_per_anchor):
        delta = (VAL_ANCHOR - a).days
        w.append(np.full(n, np.exp(-delta / tau), dtype=np.float64))
    return np.concatenate(w)


def cal_rmsle_2fold(pred_log, ly, y_raw, half, bins):
    """Honest calibrated val RMSLE of a checkpoint -> (pooled, single_fold).

    Same transform as calibrate.py (fit_shifts / apply_shifts imported from it), but
    the shift table is never applied to the rows it was fitted on: half A fits the
    shifts that score half B and vice versa.  `pooled` scores all rows this way (the
    criterion actually used); `single_fold` is only the B half, i.e. exactly the
    number calibrate.py prints as `holdout`.  Both folds are honest, so using both is
    the same estimator with half the variance — which matters here, because the
    differences being resolved are ~1e-4.
    """
    lp = np.clip(np.asarray(pred_log, dtype=np.float64), 0, None)
    out = np.empty_like(lp)
    c_a, s_a = fit_shifts(lp[half], ly[half], bins)
    out[~half] = apply_shifts(lp[~half], c_a, s_a)
    c_b, s_b = fit_shifts(lp[~half], ly[~half], bins)
    out[half] = apply_shifts(lp[half], c_b, s_b)
    return (rmsle(y_raw, np.expm1(out)),
            rmsle(y_raw[~half], np.expm1(out[~half])))


def make_cal_feval(to_log1p, ly, y_raw, half, bins, period, stats):
    """LightGBM custom eval returning the honest calibrated RMSLE of the FINAL forecast.

    `to_log1p` maps this booster's validation output to log1p(predicted GMV) of the
    WHOLE model (for two_stage that includes the frozen stage-1 probability), so the
    criterion always scores the thing calibrate.py will later see, not a stage of it.

    Throttling: recomputed on iterations 1, 1+period, 1+2*period, ...; in between the
    cached value is returned.  A repeated value is never an improvement for
    lgb.early_stopping, so patience keeps its old meaning (iterations without
    improvement) and only the grid of candidate stopping points becomes coarser.
    """
    state = {"n": 0, "last": None}

    def feval(preds, eval_data):
        if state["n"] % period == 0:
            t = time.time()
            v, _ = cal_rmsle_2fold(to_log1p(np.asarray(preds, dtype=np.float64)),
                                   ly, y_raw, half, bins)
            stats.append(time.time() - t)
            state["last"] = float(v)
        state["n"] += 1
        return "cal_rmsle", state["last"], False

    return feval


def fit_lgb(X, y, w, Xv, yv, params, objective, seed, feval=None):
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
        if feval is None:
            m = lgb.train(p, dtr, num_boost_round=n_iter, valid_sets=[dv],
                          callbacks=[lgb.early_stopping(200, verbose=False), lgb.log_evaluation(500)])
        else:
            # only the custom metric may drive early stopping: lgb.early_stopping
            # stops on whichever metric runs out of patience first, so the built-in
            # one has to go away entirely.
            p["metric"] = "None"
            m = lgb.train(p, dtr, num_boost_round=n_iter, valid_sets=[dv], feval=feval,
                          callbacks=[lgb.early_stopping(200, verbose=False), lgb.log_evaluation(500)])
        return m, m.best_iteration
    m = lgb.train(p, dtr, num_boost_round=n_iter)
    return m, n_iter


def fit_xgb(X, y, w, Xv, yv, params, objective, seed, feval=None):
    assert feval is None, "--es-metric cal is implemented for --model lgb only"
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


def fit_cb(X, y, w, Xv, yv, params, objective, seed, feval=None):
    assert feval is None, "--es-metric cal is implemented for --model lgb only"
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
    ap.add_argument("--reweight-recency", type=float, default=0.0, metavar="FLOOR",
                    help="density-ratio row weights matching train recency to inference "
                         "(O6, the soft version of --active-only); FLOOR is the smallest "
                         "weight a row may get, e.g. 0.05. 0 disables. MEASURED: hurts "
                         "(1.6894 control -> 1.6909), kept for the record.")
    ap.add_argument("--drop-cols", type=str, default="")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--no-test", action="store_true")
    ap.add_argument("--val-anchor", type=str, default="",
                    help="alternative validation anchor (e.g. 2025-12-31); implies --no-test")
    ap.add_argument("--detrend", action="store_true",
                    help="log_mse only: train on log1p(y) minus per-anchor mean; add back last-2-anchor mean level at predict")
    ap.add_argument("--anchor-source", choices=["protocol", "disk"], default="protocol",
                    help="откуда брать обучающие якоря: protocol — train_anchors(14), не "
                         "зависит от каталога (умолчание); disk — историческое поведение, "
                         "нужно для воспроизведения артефактов до 25.08")
    ap.add_argument("--gap-days", type=int, default=30,
                    help="exclude train anchors within GAP days before the val anchor "
                         "(30 = no target-window overlap with val; ЭТО УМОЛЧАНИЕ). "
                         "Было 0 — единственный трейнер проекта с небезопасным "
                         "умолчанием при том, что правило №1 звучит «обучение только с "
                         "зазором 30, иначе скор завышается на 0.05-0.10». Именно так "
                         "зоопарк work/preds оказался отравлен pre-gap эпохой. "
                         "0 задавать можно, но теперь только явно.")
    ap.add_argument("--es-metric", choices=["raw", "cal"], default="raw",
                    help="early-stopping criterion: raw = LightGBM's own val metric "
                         "(default, keeps historical behaviour bit-for-bit), cal = the "
                         "honest calibrated val RMSLE of the final forecast (lgb only)")
    ap.add_argument("--es-bins", type=int, default=24,
                    help="quantile bins of the --es-metric cal calibration "
                         "(calibrate.py default is 24; keep them equal)")
    ap.add_argument("--es-period", type=int, default=10,
                    help="recompute the calibrated criterion every N boosting "
                         "iterations (cached in between); patience stays measured in "
                         "iterations, only the stopping grid gets coarser")
    ap.add_argument("--notes", type=str, default="")
    args = ap.parse_args()
    global _ANCHOR_SOURCE
    _ANCHOR_SOURCE = args.anchor_source
    es_cal = args.es_metric == "cal"
    assert not es_cal or args.model == "lgb", "--es-metric cal is implemented for --model lgb only"
    assert args.es_period >= 1, "--es-period must be >= 1"
    # The stopping point is what a paired raw-vs-cal measurement needs to see, so it is
    # printed by BOTH arms — but only when --es-metric is given explicitly, which keeps
    # the stdout of every pre-existing command line byte-identical to before.
    es_verbose = "--es-metric" in sys.argv
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
    tr_anchors = anchor_pool()
    if args.gap_days:
        from datetime import timedelta
        cutoff = VAL_ANCHOR - timedelta(days=args.gap_days)
        tr_anchors = [a for a in tr_anchors if a <= cutoff]
    if args.n_anchors:
        tr_anchors = tr_anchors[-args.n_anchors:]
    print(f"gap_days={args.gap_days}"
          + ("  ВНИМАНИЕ: зазор выключен явно, val-скор будет завышен" if not args.gap_days else ""),
          flush=True)
    print(f"train anchors: {[a.isoformat() for a in tr_anchors]}", flush=True)

    val = load_anchor(VAL_ANCHOR)
    cols = feature_cols(val)
    if args.drop_cols:
        drop = set(args.drop_cols.split(","))
        cols = [c for c in cols if c not in drop]
    print(f"{len(cols)} features", flush=True)
    # ЭФФЕКТИВНЫЕ параметры в отпечаток: умолчание --gap-days сменилось с 0 на 30,
    # поэтому одна и та же архивная команда до и после обучает РАЗНОЕ, а argv этого
    # не покажет — там только явно переданное.
    note(gap_days=args.gap_days, n_train_anchors=len(tr_anchors), n_features=len(cols),
         model=args.model, objective=args.objective, seed=args.seed,
         es_metric=args.es_metric, n_anchors_flag=args.n_anchors or None,
         active_only=bool(args.active_only) or None)

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
    if args.reweight_recency:
        rw = recency_weights(tr["rec_active"].to_numpy(), val["rec_active"].to_numpy(),
                             args.reweight_recency)
        print(f"reweight-recency: floor={args.reweight_recency}, "
              f"вес спящих {rw[tr['rec_active'].to_numpy() > 29].mean() if (tr['rec_active'].to_numpy() > 29).any() else float('nan'):.3f}",
              flush=True)
        w = rw if w is None else w * rw
    del tr
    Xv = val.select(cols).to_numpy().astype(np.float32)
    yv_raw = val["target"].to_numpy().astype(np.float64)
    uid_val = val["user_id"].to_numpy()
    print(f"X {X.shape}, Xv {Xv.shape}, load {time.time()-t0:.0f}s", flush=True)

    fitter = FITTERS[args.model]

    # Calibrated criterion: prepared once, and ONLY when asked for.  default_rng(0)
    # is its own stream, so the raw path draws exactly what it always drew.
    cal_secs, es_ly, es_half = [], None, None
    if es_cal:
        es_ly = np.log1p(np.clip(yv_raw, 0, None))
        es_half = np.random.default_rng(0).permutation(len(yv_raw)) < len(yv_raw) // 2
        print(f"es_metric=cal: bins={args.es_bins} period={args.es_period} "
              f"halves {int(es_half.sum())}/{int((~es_half).sum())}", flush=True)

    if args.objective == "two_stage":
        # stage 1: P(target>0) ; stage 2: E[log1p|>0]; pred_log = p * m2
        ybin = (y_raw > 0).astype(np.float64)
        if args.model == "lgb":
            p1 = dict(objective="binary", metric="auc"); p1.update(params)
        elif args.model == "xgb":
            p1 = dict(objective="binary:logistic", eval_metric="auc"); p1.update(params)
        else:
            p1 = dict(loss_function="Logloss"); p1.update(params)
        # Stage 1 keeps its own criterion (AUC) even under --es-metric cal, on purpose:
        # when it trains there is no stage 2 yet, so the FINAL forecast does not exist;
        # and AUC is a pure RANKING metric, so it is immune to the defect being fixed
        # here (the defect is that a raw metric spends the stopping decision on the
        # LEVEL, and AUC ignores level entirely).  The level of a two_stage model lives
        # in stage 2 (E[log1p|y>0]) — that is where the calibrated criterion is applied,
        # and it is applied to p_1 * mu, the whole forecast, not to mu alone.
        m1, it1 = fitter(X, ybin, w, Xv, (yv_raw > 0).astype(np.float64), p1, "log_mse", args.seed)
        pos = y_raw > 0
        ylog_pos = np.log1p(y_raw[pos])
        wpos = w[pos] if w is not None else None
        p_val = predict(args.model, m1, Xv)
        if args.model == "cb":
            p_val = 1.0 / (1.0 + np.exp(-p_val))
        if es_cal:
            # stage 2 is validated on ALL val rows (not just the positives) so that the
            # criterion sees exactly the forecast calibrate.py will see later.  Only the
            # metric changes; stage 2 is still TRAINED on positives alone.
            fev = make_cal_feval(
                lambda mu, p=p_val: np.clip(p * np.clip(mu, 0, None), 0, None),
                es_ly, yv_raw, es_half, args.es_bins, args.es_period, cal_secs)
            m2, it2 = fitter(X[pos], ylog_pos, wpos, Xv, np.log1p(yv_raw), params2,
                             "log_mse", args.seed + 1, fev)
        else:
            m2, it2 = fitter(X[pos], ylog_pos, wpos, Xv[yv_raw > 0], np.log1p(yv_raw[yv_raw > 0]), params2, "log_mse", args.seed + 1)
        mu_val = predict(args.model, m2, Xv)
        pv = np.expm1(np.clip(p_val * np.clip(mu_val, 0, None), 0, None))
    else:
        anchor_arr = None
        m_hat = 0.0
        if args.objective == "log_mse":
            y = np.log1p(y_raw); yv = np.log1p(yv_raw)
            if args.detrend:
                anchor_arr = load_matrix(tr_anchors, columns=["anchor_date"])["anchor_date"].to_numpy()
                means = {}
                for a in tr_anchors:
                    mask = anchor_arr == np.datetime64(a)
                    means[a] = float(y[mask].mean())
                    y[mask] -= means[a]
                m_hat = float(np.mean([means[a] for a in tr_anchors[-2:]]))
                yv = yv - m_hat
                print(f"detrend: anchor means {[round(means[a],3) for a in tr_anchors]}, m_hat={m_hat:.4f}", flush=True)
        else:
            y = y_raw; yv = yv_raw
        if es_cal:
            # single-stage: the booster's own val output IS the forecast, up to the
            # same transform applied below (log_mse predicts log1p GMV, everything
            # else predicts GMV directly).
            to_lp = ((lambda r, mh=m_hat: np.clip(r + mh, 0, None)) if args.objective == "log_mse"
                     else (lambda r: np.log1p(np.clip(r, 0, None))))
            fev = make_cal_feval(to_lp, es_ly, yv_raw, es_half, args.es_bins,
                                 args.es_period, cal_secs)
            m, best_it = fitter(X, y, w, Xv, yv, params, args.objective, args.seed, fev)
        else:
            m, best_it = fitter(X, y, w, Xv, yv, params, args.objective, args.seed)
        raw_pv = predict(args.model, m, Xv)
        if args.objective == "log_mse":
            pv = np.expm1(np.clip(raw_pv + m_hat, 0, None))
        else:
            pv = np.clip(raw_pv, 0, None)

    score = rmsle(yv_raw, pv)
    if es_verbose:
        stop = (f"stage1 {it1} stage2 {it2}" if args.objective == "two_stage"
                else f"{best_it}")
        cal_now, cal_hold = cal_rmsle_2fold(
            np.log1p(np.clip(pv, 0, None)), es_ly if es_ly is not None
            else np.log1p(np.clip(yv_raw, 0, None)), yv_raw,
            es_half if es_half is not None
            else np.random.default_rng(0).permutation(len(yv_raw)) < len(yv_raw) // 2,
            args.es_bins)
        print(f"es_metric={args.es_metric}: stop_iter {stop}; "
              f"val raw {score:.6f} CAL {cal_now:.6f} (holdout {cal_hold:.6f})"
              + (f"; {len(cal_secs)} calibrations, {sum(cal_secs):.1f}s total, "
                 f"{np.mean(cal_secs) if cal_secs else 0:.3f}s each "
                 f"(period {args.es_period})" if es_cal else ""), flush=True)
    save_preds(args.name, "val", uid_val, pv)
    log_score(args.name, score, args.notes or f"{args.model}/{args.objective}")

    if args.no_test:
        return

    # retrain on train(+gap)+val, predict test
    Xg, yg_raw, wg = None, None, None
    if args.gap_days:
        from datetime import timedelta
        gap_anchors = [a for a in anchor_pool()
                       if VAL_ANCHOR - timedelta(days=args.gap_days) < a < VAL_ANCHOR]
        if gap_anchors:
            import polars as pl
            gtr = load_matrix(gap_anchors, columns=["user_id", "anchor_date", "target"] + cols)
            if args.active_only:
                gtr = gtr.filter(pl.col("rec_active") <= 29)
            Xg = gtr.select(cols).to_numpy().astype(np.float32)
            yg_raw = gtr["target"].to_numpy().astype(np.float64)
            if args.weight_tau:
                gaw = []
                for a in gap_anchors:
                    n = gtr.filter(gtr["anchor_date"] == a).height
                    gaw.append(np.full(n, np.exp(-(VAL_ANCHOR - a).days / args.weight_tau)))
                wg = np.concatenate(gaw)
            del gtr
            print(f"retrain adds gap anchors {[a.isoformat() for a in gap_anchors]}: +{len(yg_raw)} rows", flush=True)
    parts = [X] + ([Xg] if Xg is not None else []) + [Xv]
    Xall = np.vstack(parts)
    row_ratio = Xall.shape[0] / max(X.shape[0], 1)
    # parts держал ссылки на X/Xg/Xv, поэтому прежний `del X, Xv` память не отдавал:
    # копия жила до конца функции. На 16 ГБ это заметно (у проекта два OOM в истории).
    del parts, X, Xv, Xg
    test = load_anchor(TEST_ANCHOR)
    Xt = test.select(cols).to_numpy().astype(np.float32)
    uid_t = test["user_id"].to_numpy()

    # iterations scale with data growth (old no-gap behavior: ~x1.07 for 14 anchors)
    iter_mult = 1.0 + 0.7 * max(row_ratio - 1.0, 0.0)
    print(f"retrain: row_ratio={row_ratio:.3f} iter_mult={iter_mult:.3f}", flush=True)

    n_gap = len(yg_raw) if yg_raw is not None else 0

    def cat_y(a, b):
        return np.concatenate([a] + ([yg_raw] if yg_raw is not None else []) + [b])

    def cat_w(base_w):
        if base_w is None:
            return None
        segs = [base_w]
        if n_gap:
            segs.append(wg if wg is not None else np.ones(n_gap))
        segs.append(np.ones(len(yv_raw)))
        return np.concatenate(segs)

    m_hat_test = 0.0   # detrend level added back at predict (log_mse+--detrend only)
    if args.objective == "two_stage":
        raw_all = cat_y(y_raw, yv_raw)
        yb_all = (raw_all > 0).astype(np.float64)
        w_all = cat_w(w)
        p1["n_estimators" if args.model != "cb" else "iterations"] = max(50, int(it1 * iter_mult))
        m1f, _ = fitter(Xall, yb_all, w_all, None, None, p1, "log_mse", args.seed)
        pos_all = raw_all > 0
        ylog_all = np.log1p(raw_all[pos_all])
        wpos_all = w_all[pos_all] if w_all is not None else None
        params2["n_estimators" if args.model != "cb" else "iterations"] = max(50, int(it2 * iter_mult))
        m2f, _ = fitter(Xall[pos_all], ylog_all, wpos_all, None, None, params2, "log_mse", args.seed + 1)
        # freeze: retrained boosters -> work/models/ (stage1 = P(y>0), stage2 = E[log1p|>0])
        save_booster(args.model, args.name, m1f, tag="stage1")
        save_booster(args.model, args.name, m2f, tag="stage2")
        p_t = predict(args.model, m1f, Xt)
        if args.model == "cb":
            p_t = 1.0 / (1.0 + np.exp(-p_t))
        mu_t = predict(args.model, m2f, Xt)
        pt = np.expm1(np.clip(p_t * np.clip(mu_t, 0, None), 0, None))
    else:
        if args.objective == "log_mse" and args.detrend:
            assert not args.gap_days, "--detrend with --gap-days is not supported"
            yv_own_mean = float(np.log1p(yv_raw).mean())
            yv_det = np.log1p(yv_raw) - yv_own_mean
            y_all = np.concatenate([y, yv_det])
            m_hat_test = float(np.mean([means[tr_anchors[-1]], yv_own_mean]))
            print(f"detrend retrain: m_hat_test={m_hat_test:.4f}", flush=True)
        elif args.objective == "log_mse":
            y_all = np.concatenate([y] + ([np.log1p(yg_raw)] if yg_raw is not None else [])
                                   + [np.log1p(yv_raw)])
        else:
            y_all = cat_y(y, yv_raw)
        w_all = cat_w(w)
        params["n_estimators" if args.model != "cb" else "iterations"] = max(50, int(best_it * iter_mult))
        mf, _ = fitter(Xall, y_all, w_all, None, None, params, args.objective, args.seed)
        save_booster(args.model, args.name, mf)   # freeze: retrained booster
        raw_pt = predict(args.model, mf, Xt)
        pt = np.expm1(np.clip(raw_pt + m_hat_test, 0, None)) if args.objective == "log_mse" else np.clip(raw_pt, 0, None)

    # freeze: what inference needs to reuse the boosters above
    two = args.objective == "two_stage"
    save_meta(args.name, kind="gbdt", model=args.model, objective=args.objective,
              feature_cols=cols, params=params, params2=params2 if two else None,
              seed=args.seed, gap_days=args.gap_days, detrend=bool(args.detrend),
              **({"es_metric": "cal", "es_bins": args.es_bins,
                  "es_period": args.es_period} if es_cal else {}),
              m_hat_test=float(m_hat_test), active_only=bool(args.active_only),
              weight_tau=args.weight_tau, n_anchors=len(tr_anchors),
              val_rmsle=float(score),
              weights=[booster_filename(args.model, args.name, t)
                       for t in (("stage1", "stage2") if two else (None,))])
    save_preds(args.name, "test", uid_t, pt)
    print(f"[DONE] {args.name} val_rmsle={score:.6f} total {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

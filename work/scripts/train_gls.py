"""GLS-weighted GBDT: empirical conditional-variance weights instead of a postulated one.

Idea
----
The metric is UNWEIGHTED MSE in log1p space, so we need an accurate estimate of the
conditional MEAN E[log1p(y)|x].  Under heteroscedasticity the efficient (min-variance)
estimator of a conditional mean weights observations by the inverse of their conditional
variance (GLS / Gauss-Markov).  Our champion tweedie-on-log1p loss already does something
like this IMPLICITLY -- a tweedie deviance imposes V(mu) ~ mu^p, i.e. a postulated,
mean-dependent variance function -- which is probably why it beats plain MSE.  Here the
postulated power law is replaced by an EMPIRICALLY ESTIMATED variance v(x):

  stage 1  m1  : fit log1p(y) the usual way, CROSS-FIT over K user folds -> OOF residuals
  stage 2  v   : fit (y_log - m1_oof)^2  -> estimate of the conditional variance v(x)
  stage 3  m2  : refit log1p(y) with sample_weight = 1/max(v(x), floor), still predicting
                 the conditional mean.  Weights are normalised to mean 1.

The residuals MUST be out-of-fold: in-sample residuals are shrunk by the fit, v(x) comes
out biased low exactly where the model memorised, and 1/v blows those rows up.  Folds are
assigned BY USER (a user appears in every anchor slice, windows overlap by 23d).

No custom objective is needed -- this is pure sample_weight, so it composes with any loss;
--params carries the base loss, e.g. tweedie-on-log1p for the gls_tw variant.

Examples
--------
  # smoke: 2 slices, 400 trees, paired unweighted control in the same process
  train_gls.py --name gls_smoke --n-anchors 2 --gap-days 30 --threads 2 \
      --params '{"n_estimators":400}' --direct-baseline --no-test
  # full champion-loss variant
  train_gls.py --name gls_tw --n-anchors 14 --gap-days 30 --threads 6 \
      --params '{"objective":"tweedie","tweedie_variance_power":1.45,"n_estimators":6000}'
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import timedelta
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from common import VAL_ANCHOR, TEST_ANCHOR, rmsle, load_anchor, feature_cols
from exp_lib import save_preds, log_score
from train_gbdt import fit_lgb
from train_bagged import cal_holdout, eligible_anchors, stack_anchors


def lgb_defaults(params: dict, seed: int) -> dict:
    """Mirror of fit_lgb()'s parameter block (train_gbdt.py) -- keep in sync.

    Used only to build the shared Dataset for the cross-fit; the baseline and the final
    stage-3 model go through fit_lgb itself so the control is literally the repo's path.
    """
    p = dict(
        objective="regression", metric="rmse", learning_rate=0.04,
        num_leaves=255, min_data_in_leaf=300, feature_fraction=0.75,
        bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0, max_bin=127,
        num_threads=int(os.environ.get("OMP_NUM_THREADS", "5")),
        seed=seed, verbosity=-1,
    )
    p.update(params)
    return p


def crossfit_oof(X, ylog, fold_row, k, params, seed, Xv, yv_log, cap_iter=0):
    """OOF predictions of the base model over K user folds (memory-lean).

    Uses one constructed lgb.Dataset + .subset() per fold instead of X[mask] copies
    (3.5M x 203 float32 = 2.8GB; a boolean copy per fold would be +1.9GB peak), and
    predicts the WHOLE matrix with each fold model (3.5M float64 = 28MB) instead of
    slicing rows out.
    """
    import lightgbm as lgb
    p = lgb_defaults(params, seed)
    n_iter = p.pop("n_estimators", 4000)
    if cap_iter:
        n_iter = min(n_iter, cap_iter)
    dfull = lgb.Dataset(X, label=ylog, params=p, free_raw_data=False)
    dfull.construct()
    dv = lgb.Dataset(Xv, yv_log, reference=dfull, free_raw_data=False)
    oof = np.empty(len(ylog), np.float64)
    ins = np.empty(len(ylog), np.float64)  # in-sample twin, for the bias diagnostic
    iters = []
    for j in range(k):
        idx = np.where(fold_row != j)[0]
        dtr = dfull.subset(idx)
        pj = dict(p); pj["seed"] = seed + j
        m = lgb.train(pj, dtr, num_boost_round=n_iter, valid_sets=[dv],
                      callbacks=[lgb.early_stopping(200, verbose=False)])
        iters.append(int(m.best_iteration or n_iter))
        pred = m.predict(X)
        hold = fold_row == j
        oof[hold] = pred[hold]
        ins[~hold] = pred[~hold]  # last writer wins; only used for a mean-level diagnostic
        print(f"  [cf] fold {j}: n={len(idx)} it={iters[-1]}", flush=True)
        del m, pred, dtr
    del dfull, dv
    return oof, ins, iters


def fit_var(X, r2, params, seed, objective="gamma", threads=None):
    """Conditional-variance model.  Squared residuals of a ~normal error are Gamma
    (shape 1/2), so a gamma-deviance fit with log link is the ML-correct estimator of
    E[r^2|x]; l2 on raw r^2 is dominated by the heavy tail.  Deliberately low capacity:
    an overfitted v(x) turns the weights into noise."""
    import lightgbm as lgb
    p = dict(
        objective="gamma", metric="gamma", learning_rate=0.06,
        num_leaves=63, min_data_in_leaf=2000, feature_fraction=0.7,
        bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0, max_bin=63,
        num_threads=threads or int(os.environ.get("OMP_NUM_THREADS", "5")),
        seed=seed, verbosity=-1,
    )
    y = np.clip(r2, 1e-6, None)
    if objective == "l2":
        p.update(objective="regression", metric="rmse")
    elif objective == "l2log":
        p.update(objective="regression", metric="rmse")
        y = np.log(y)
    p.update(params)
    n_iter = p.pop("n_estimators", 400)
    m = lgb.train(p, lgb.Dataset(X, y, params=p), num_boost_round=n_iter)
    return m, objective


def predict_var(m, objective, X, chunk=1_000_000):
    out = np.empty(X.shape[0], np.float64)
    for i in range(0, X.shape[0], chunk):
        out[i:i + chunk] = m.predict(X[i:i + chunk])
    if objective == "l2log":
        out = np.exp(np.clip(out, -20, 20))
    return np.clip(out, 1e-9, None)


def gls_weights(v, floor_pct: float, wcap: float = 0.0, power: float = 1.0):
    """w = max(v, floor)^-power.  power=1 is textbook GLS; power<1 damps the reweighting,
    the standard hedge when the variance model itself is noisy."""
    floor = float(np.percentile(v, floor_pct))
    w = np.maximum(v, floor) ** (-power)
    if wcap:
        w = np.minimum(w, wcap * float(np.median(w)))
    w *= len(w) / w.sum()  # mean 1
    ess = float(w.sum() ** 2 / np.square(w).sum())
    stats = dict(floor=floor, w_min=float(w.min()), w_p1=float(np.percentile(w, 1)),
                 w_p50=float(np.percentile(w, 50)), w_p99=float(np.percentile(w, 99)),
                 w_max=float(w.max()), ess_frac=ess / len(w))
    return w, stats


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    return float(np.corrcoef(ra, rb)[0, 1])


def diagnose(v_val, yv_raw, err2, val_df, cols, pred_log=None):
    """Is v(x) meaningful?  Correlate it with the zero rate / recency / realised error.

    sp_pred matters for interpretation: if v(x) is essentially a function of the predicted
    mean, GLS is only a re-derivation of what a tweedie variance function already does.
    """
    d = dict(sp_err2=spearman(v_val, err2))
    if pred_log is not None:
        d["sp_pred"] = spearman(v_val, pred_log)
    for f in ("rec_order", "rec_active", "ord_cnt_30", "gmv_sum_365"):
        if f in cols:
            x = val_df[f].to_numpy().astype(np.float64)
            ok = np.isfinite(x)
            d[f"sp_{f}"] = spearman(v_val[ok], x[ok])
    q = np.quantile(v_val, np.linspace(0, 1, 11))
    q[0] -= 1e-9; q[-1] += 1e-9
    rows = []
    for i in range(10):
        m = (v_val > q[i]) & (v_val <= q[i + 1])
        if m.sum() == 0:
            continue
        row = dict(dec=i, n=int(m.sum()), v_mean=float(v_val[m].mean()),
                   zero_rate=float((yv_raw[m] == 0).mean()),
                   mean_logy=float(np.log1p(yv_raw[m]).mean()),
                   mse=float(err2[m].mean()))
        if "rec_order" in cols:
            row["rec_order"] = float(np.nanmean(val_df["rec_order"].to_numpy()[m]))
        rows.append(row)
    d["deciles"] = rows
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--n-anchors", type=int, default=14)
    ap.add_argument("--gap-days", type=int, default=30)
    ap.add_argument("--params", type=str, default='{"n_estimators":6000}',
                    help="base LightGBM params (carries the loss; empty = MSE on log1p)")
    ap.add_argument("--var-params", type=str, default='{"n_estimators":400}')
    ap.add_argument("--var-objective", default="gamma", choices=["gamma", "l2", "l2log"])
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--floor-pct", type=float, default=5.0)
    ap.add_argument("--wcap", type=float, default=0.0,
                    help="cap weights at WCAP * median(w) before normalisation (0 = off)")
    ap.add_argument("--w-power", type=float, default=1.0,
                    help="w = v^-POWER; 1.0 = textbook GLS, 0.5 = damped")
    ap.add_argument("--cf-cap-iter", type=int, default=0,
                    help="cap cross-fit trees (smoke: keep = --params n_estimators)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--direct-baseline", action="store_true",
                    help="train the identical UNWEIGHTED config in the same process (paired control)")
    ap.add_argument("--no-test", action="store_true")
    ap.add_argument("--notes", type=str, default="")
    args = ap.parse_args()

    if args.threads:
        os.environ["OMP_NUM_THREADS"] = str(args.threads)
    params = json.loads(args.params)
    var_params = json.loads(args.var_params)
    t0 = time.time()

    pool = eligible_anchors(args.gap_days)
    anchors = pool[-args.n_anchors:] if args.n_anchors else pool
    val = load_anchor(VAL_ANCHOR)
    cols = feature_cols(val)
    n_users = val.height
    Xv = val.select(cols).to_numpy().astype(np.float32)
    yv_raw = val["target"].to_numpy().astype(np.float64)
    uid_val = val["user_id"].to_numpy()
    yv_log = np.log1p(yv_raw)
    half = np.random.default_rng(0).permutation(n_users) < n_users // 2  # same split as calibrate.py
    print(f"{len(cols)} features, {n_users} users; train anchors "
          f"{[a.isoformat() for a in anchors]}", flush=True)

    X, y_raw = stack_anchors(anchors, cols, n_users)
    ylog = np.log1p(y_raw)
    # stack_anchors fills anchor i at rows [i*n_users, (i+1)*n_users) and every anchor file
    # holds the same sorted user_ids -> row % n_users identifies the user.
    a0 = load_anchor(anchors[0], ["user_id"])["user_id"].to_numpy()
    assert (a0 == uid_val).all(), "anchor user order differs from val -- positional folds unsafe"
    user_fold = (np.random.default_rng(args.seed).permutation(n_users) % args.folds).astype(np.int8)
    fold_row = np.tile(user_fold, len(anchors))
    print(f"X {X.shape}, folds by user {np.bincount(user_fold).tolist()}, "
          f"load {time.time()-t0:.0f}s", flush=True)

    res = {}

    # ---- stage 1: cross-fit OOF residuals -------------------------------------------
    t1 = time.time()
    oof, ins, cf_iters = crossfit_oof(X, ylog, fold_row, args.folds, params, args.seed,
                                      Xv, yv_log, args.cf_cap_iter)
    r2_oof = (ylog - oof) ** 2
    r2_ins = (ylog - ins) ** 2
    res["stage1"] = dict(iters=cf_iters, oof_mse=float(r2_oof.mean()),
                         insample_mse=float(r2_ins.mean()),
                         bias_ratio=float(r2_ins.mean() / r2_oof.mean()),
                         secs=round(time.time() - t1))
    print(f"[stage1] OOF mse={r2_oof.mean():.4f} in-sample={r2_ins.mean():.4f} "
          f"(ratio {r2_ins.mean()/r2_oof.mean():.3f}) [{time.time()-t1:.0f}s]", flush=True)

    # ---- stage 2: variance model ----------------------------------------------------
    t2 = time.time()
    vm, vobj = fit_var(X, r2_oof, var_params, args.seed + 100, args.var_objective, args.threads)
    v_tr = predict_var(vm, vobj, X)
    v_val = predict_var(vm, vobj, Xv)
    res["stage2"] = dict(objective=args.var_objective, params=var_params,
                         v_p5=float(np.percentile(v_tr, 5)), v_p50=float(np.percentile(v_tr, 50)),
                         v_p95=float(np.percentile(v_tr, 95)),
                         sp_v_r2=spearman(v_tr[:1_000_000], r2_oof[:1_000_000]),
                         secs=round(time.time() - t2))
    print(f"[stage2] v: p5={res['stage2']['v_p5']:.3f} p50={res['stage2']['v_p50']:.3f} "
          f"p95={res['stage2']['v_p95']:.3f} spearman(v, r2_oof)={res['stage2']['sp_v_r2']:.3f} "
          f"[{time.time()-t2:.0f}s]", flush=True)

    # ---- stage 3: weighted refit ----------------------------------------------------
    w, wstats = gls_weights(v_tr, args.floor_pct, args.wcap, args.w_power)
    wstats["power"] = args.w_power
    res["weights"] = wstats
    print(f"[weights] floor={wstats['floor']:.4f} p1={wstats['w_p1']:.3f} "
          f"p50={wstats['w_p50']:.3f} p99={wstats['w_p99']:.3f} max={wstats['w_max']:.3f} "
          f"ess={wstats['ess_frac']:.3f}", flush=True)
    t3 = time.time()
    m_gls, it_gls = fit_lgb(X, ylog, w, Xv, yv_log, dict(params), "log_mse", args.seed)
    lp_gls = np.clip(m_gls.predict(Xv), 0, None)
    pv_gls = np.expm1(lp_gls)
    raw_gls = rmsle(yv_raw, pv_gls)
    hraw_gls, hcal_gls = cal_holdout(lp_gls, yv_log, yv_raw, half)
    res["gls"] = dict(raw=raw_gls, holdout_raw=hraw_gls, holdout_cal=hcal_gls,
                      iters=int(it_gls), secs=round(time.time() - t3))
    print(f"[gls] raw={raw_gls:.6f} holdout raw={hraw_gls:.6f} cal={hcal_gls:.6f} "
          f"it={it_gls} [{time.time()-t3:.0f}s]", flush=True)

    # ---- paired control: identical config, no weights --------------------------------
    lp_base = it_base = None
    if args.direct_baseline:
        t4 = time.time()
        m_b, it_base = fit_lgb(X, ylog, None, Xv, yv_log, dict(params), "log_mse", args.seed)
        lp_base = np.clip(m_b.predict(Xv), 0, None)
        raw_b = rmsle(yv_raw, np.expm1(lp_base))
        hraw_b, hcal_b = cal_holdout(lp_base, yv_log, yv_raw, half)
        res["base"] = dict(raw=raw_b, holdout_raw=hraw_b, holdout_cal=hcal_b,
                           iters=int(it_base), secs=round(time.time() - t4))
        res["delta"] = dict(raw=raw_gls - raw_b, holdout_raw=hraw_gls - hraw_b,
                            cal=hcal_gls - hcal_b,
                            err_corr=float(np.corrcoef(lp_gls - yv_log, lp_base - yv_log)[0, 1]))
        print(f"[base] raw={raw_b:.6f} holdout raw={hraw_b:.6f} cal={hcal_b:.6f} it={it_base}\n"
              f"[DELTA] raw={raw_gls-raw_b:+.6f} cal={hcal_gls-hcal_b:+.6f} "
              f"(negative = GLS better) [{time.time()-t4:.0f}s]", flush=True)
        save_preds(args.name + "_direct", "val", uid_val, np.expm1(lp_base))
        del m_b

    # ---- diagnostics: does v(x) mean anything? ---------------------------------------
    ref_lp = lp_base if lp_base is not None else lp_gls
    err2_val = (yv_log - ref_lp) ** 2
    res["diag"] = diagnose(v_val, yv_raw, err2_val, val, cols, ref_lp)
    print(f"[diag] spearman(v_val, realised err^2)={res['diag']['sp_err2']:.3f} "
          + " ".join(f"{k}={v:+.3f}" for k, v in res["diag"].items()
                     if k.startswith("sp_") and k != "sp_err2"), flush=True)
    for r in res["diag"]["deciles"]:
        print(f"   v-dec {r['dec']}: v={r['v_mean']:.2f} zero_rate={r['zero_rate']:.3f} "
              f"mean_logy={r['mean_logy']:.2f} realised_mse={r['mse']:.2f}"
              + (f" rec_order={r['rec_order']:.0f}" if "rec_order" in r else ""), flush=True)

    save_preds(args.name, "val", uid_val, pv_gls)
    note = args.notes or (f"GLS 1/v weights, {len(anchors)} anchors gap{args.gap_days} "
                          f"folds={args.folds} floor_p{args.floor_pct} var={args.var_objective} "
                          f"it={it_gls}")
    if "delta" in res:
        note += (f"; base={res['base']['raw']:.6f} d_raw={res['delta']['raw']:+.6f}"
                 f"; cal gls={hcal_gls:.6f} base={res['base']['holdout_cal']:.6f} "
                 f"d_cal={res['delta']['cal']:+.6f}")
    log_score(args.name, raw_gls, note)
    out = Path(__file__).resolve().parents[1] / "reports" / f"{args.name}.json"
    out.write_text(json.dumps({"results": res, "params": params, "var_params": var_params,
                               "anchors": [a.isoformat() for a in anchors],
                               "n_anchors": len(anchors), "folds": args.folds,
                               "floor_pct": args.floor_pct, "seed": args.seed},
                              indent=1, default=str))
    print(f"[JSON] {out}", flush=True)

    if args.no_test:
        print(f"[DONE] {args.name} {time.time()-t0:.0f}s", flush=True)
        return

    # ---- retrain on train+gap+val, predict test --------------------------------------
    gap_anchors = [a for a in eligible_anchors(0)
                   if VAL_ANCHOR - timedelta(days=args.gap_days) < a < VAL_ANCHOR]
    extra = gap_anchors + [VAL_ANCHOR]
    Xe, ye = stack_anchors(extra, cols, n_users)
    print(f"[retrain] adds {[a.isoformat() for a in extra]}: +{len(ye)} rows", flush=True)
    Xall = np.vstack([X, Xe])
    yall = np.concatenate([ylog, np.log1p(ye)])
    row_ratio = Xall.shape[0] / X.shape[0]
    iter_mult = 1.0 + 0.7 * max(row_ratio - 1.0, 0.0)
    del X, ylog, Xe, ye
    # v(x) for the new rows comes from the same stage-2 model (out-of-sample for them)
    v_all = np.concatenate([v_tr, predict_var(vm, vobj, Xall[len(v_tr):])])
    w_all, _ = gls_weights(v_all, args.floor_pct, args.wcap, args.w_power)
    test = load_anchor(TEST_ANCHOR)
    Xt = test.select(cols).to_numpy().astype(np.float32)
    uid_t = test["user_id"].to_numpy()
    del test
    print(f"[retrain] row_ratio={row_ratio:.3f} iter_mult={iter_mult:.3f}", flush=True)

    pf = dict(params); pf["n_estimators"] = max(50, int(it_gls * iter_mult))
    mf, _ = fit_lgb(Xall, yall, w_all, None, None, pf, "log_mse", args.seed)
    save_preds(args.name, "test", uid_t, np.expm1(np.clip(mf.predict(Xt), 0, None)))
    del mf
    if args.direct_baseline:
        pb = dict(params); pb["n_estimators"] = max(50, int(it_base * iter_mult))
        mb, _ = fit_lgb(Xall, yall, None, None, None, pb, "log_mse", args.seed)
        save_preds(args.name + "_direct", "test", uid_t, np.expm1(np.clip(mb.predict(Xt), 0, None)))
        del mb
    print(f"[DONE] {args.name} val_rmsle={raw_gls:.6f} total {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

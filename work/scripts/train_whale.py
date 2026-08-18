"""Whale-specialist gated blend (Elo Merchant 1st-place pattern) over a frozen base.

Motivation (error_analysis.md): squared-log error concentrates in upper GMV deciles
and target>0 users are underpredicted by the base blend; Elo-1st recipe = whale
classifier + big-value specialist + probability-gated mix with the main model.

Heads (protocol: gap-30, USE_V2/V3/V4; anchors = last K with anchor <= VAL-30d):
  clf  -- LGB binary,   label y >= q_top(train y)      (default top-5%: q=0.95)
  spec -- LGB reg on log1p(y), ONLY rows y > q_spec(train y) (default q=0.80)
Base -- work/preds/<base>_{val,test}.parquet joined per-user (log1p space).
Gate -- final_log = (1-g)*base_log + g*spec_log,  g = clip(a*(P - t), 0, gmax),
        t = quantile(P over the scored universe, q_gate).
(a, q_gate) tuned on val HONESTLY: 2-fold by users, params fit on one half and
applied to the other; the saved val preds and the logged score are cross-fitted.
Test: heads retrained on train+gap+val (iterations scaled by row growth like
train_gbdt), thresholds recomputed on the pooled rows, (a, q_gate) refit on full
val, t taken as the q_gate-quantile of P over the test universe.

Smoke: --smoke   (1 anchor, <=200k train rows, <=200 trees, 2 threads, no test)
Full:  --name whale_final --n-anchors 14 --threads 6   (run via work/queue only)
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import timedelta
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))
from common import VAL_ANCHOR, TEST_ANCHOR, PREDS_DIR, load_anchor, feature_cols  # noqa: E402
from exp_lib import available_train_anchors, load_matrix, save_preds, log_score  # noqa: E402

A_GRID = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0]
Q_GRID = [0.80, 0.85, 0.90, 0.93, 0.95, 0.97, 0.98, 0.99]


def lgb_fit(X, y, Xv, yv, params, es_rounds):
    import lightgbm as lgb
    p = dict(params)
    n_iter = p.pop("n_estimators")
    dtr = lgb.Dataset(X, y, free_raw_data=True)
    if Xv is None:
        return lgb.train(p, dtr, num_boost_round=n_iter), n_iter
    dv = lgb.Dataset(Xv, yv, reference=dtr, free_raw_data=True)
    m = lgb.train(p, dtr, num_boost_round=n_iter, valid_sets=[dv],
                  callbacks=[lgb.early_stopping(es_rounds, verbose=False),
                             lgb.log_evaluation(500)])
    return m, (m.best_iteration or n_iter)


def auc(label, score):
    order = np.argsort(score)
    r = np.empty(len(score), dtype=np.float64)
    r[order] = np.arange(1, len(score) + 1)
    pos = label > 0.5
    n1, n0 = int(pos.sum()), int((~pos).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float((r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def gate_apply(P, spec_log, base_log, a, t, gmax):
    g = np.clip(a * (P - t), 0.0, gmax)
    f = np.clip((1.0 - g) * base_log + g * spec_log, 0.0, None)
    return f, g


def tune_gate(P, spec_log, base_log, ylog, gmax):
    """Grid-search (a, q_gate) minimizing log-RMSE on the given rows.

    a=0 (pure base) is in the grid, so in-sample best is never worse than base.
    Returns (score_in_sample, a, q_gate).
    """
    best = None
    for q in Q_GRID:
        t = float(np.quantile(P, q))
        for a in A_GRID:
            f, _ = gate_apply(P, spec_log, base_log, a, t, gmax)
            s = float(np.sqrt(np.mean((f - ylog) ** 2)))
            if best is None or s < best[0] - 1e-12:
                best = (s, a, q)
    return best


def load_base(split, base_name, uids):
    df = pl.read_parquet(PREDS_DIR / f"{base_name}_{split}.parquet")
    j = (pl.DataFrame({"user_id": uids.astype(np.int64),
                       "_ord": np.arange(len(uids), dtype=np.int64)})
         .join(df, on="user_id", how="left").sort("_ord"))
    n_null = j["pred"].null_count()
    assert n_null == 0, f"base {base_name}_{split}: {n_null} users without base pred"
    return np.log1p(j["pred"].to_numpy().astype(np.float64))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="")
    ap.add_argument("--base", default="my26")
    ap.add_argument("--n-anchors", type=int, default=14)
    ap.add_argument("--top-q", type=float, default=0.95)
    ap.add_argument("--spec-q", type=float, default=0.80)
    ap.add_argument("--gmax", type=float, default=0.6)
    ap.add_argument("--gap-days", type=int, default=30)
    ap.add_argument("--max-rows", type=int, default=0)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--no-test", action="store_true")
    ap.add_argument("--notes", type=str, default="")
    args = ap.parse_args()

    if args.smoke:
        args.n_anchors = 1
        args.max_rows = args.max_rows or 200_000
        args.threads = args.threads or 2
        args.no_test = True
    name = args.name or ("whale_smoke" if args.smoke else "whale_final")
    threads = args.threads or int(os.environ.get("OMP_NUM_THREADS", "5"))
    os.environ["OMP_NUM_THREADS"] = str(threads)
    rng = np.random.default_rng(args.seed)
    t0 = time.time()

    # ---- anchors (gap protocol, mirrors train_gbdt) ----
    cutoff = VAL_ANCHOR - timedelta(days=args.gap_days)
    tr_anchors = [a for a in available_train_anchors() if a <= cutoff]
    if args.n_anchors:
        tr_anchors = tr_anchors[-args.n_anchors:]
    print(f"train anchors: {[a.isoformat() for a in tr_anchors]}", flush=True)

    val = load_anchor(VAL_ANCHOR)
    cols = feature_cols(val)
    print(f"{len(cols)} features", flush=True)

    tr = load_matrix(tr_anchors, columns=["user_id", "anchor_date", "target"] + cols)
    X = tr.select(cols).to_numpy().astype(np.float32)
    y_raw = tr["target"].to_numpy().astype(np.float64)
    del tr
    if args.max_rows and len(y_raw) > args.max_rows:
        keep = rng.choice(len(y_raw), args.max_rows, replace=False)
        X, y_raw = X[keep], y_raw[keep]
        print(f"subsampled train to {args.max_rows} rows", flush=True)

    Xv = val.select(cols).to_numpy().astype(np.float32)
    yv_raw = val["target"].to_numpy().astype(np.float64)
    uid_val = val["user_id"].to_numpy()
    del val
    ylog_v = np.log1p(yv_raw)
    print(f"X {X.shape}, Xv {Xv.shape}, load {time.time() - t0:.0f}s", flush=True)

    # ---- thresholds from train rows (pooled quantiles) ----
    thr_top = float(np.quantile(y_raw, args.top_q))
    thr_spec = float(np.quantile(y_raw, args.spec_q))
    lab = (y_raw >= thr_top).astype(np.float64)
    lab_v = (yv_raw >= thr_top).astype(np.float64)
    spec_mask = y_raw > thr_spec
    spec_mask_v = yv_raw > thr_spec
    print(f"thr_top(q={args.top_q})={thr_top:.1f} pos_rate={lab.mean():.4f} | "
          f"thr_spec(q={args.spec_q})={thr_spec:.1f} spec_rows={int(spec_mask.sum())}",
          flush=True)

    es = 50 if args.smoke else 200
    clf_p = dict(objective="binary", metric="auc", learning_rate=0.04, num_leaves=127,
                 min_data_in_leaf=500, feature_fraction=0.75, bagging_fraction=0.8,
                 bagging_freq=1, lambda_l2=5.0, max_bin=127, num_threads=threads,
                 seed=args.seed, verbosity=-1, n_estimators=4000)
    spec_p = dict(objective="regression", metric="rmse", learning_rate=0.04,
                  num_leaves=255, min_data_in_leaf=100, feature_fraction=0.75,
                  bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0, max_bin=127,
                  num_threads=threads, seed=args.seed + 1, verbosity=-1,
                  n_estimators=4000)
    if args.smoke:  # <=200 trees, tiny leaves
        for p in (clf_p, spec_p):
            p.update(learning_rate=0.1, num_leaves=31, min_data_in_leaf=100,
                     n_estimators=200)

    # early-stop eval sets (smoke: subsample to keep row counts tiny)
    if args.smoke:
        es_idx = rng.choice(len(yv_raw), min(100_000, len(yv_raw)), replace=False)
        dom_idx = np.where(spec_mask_v)[0]
        if len(dom_idx) > 50_000:
            dom_idx = rng.choice(dom_idx, 50_000, replace=False)
    else:
        es_idx = np.arange(len(yv_raw))
        dom_idx = np.where(spec_mask_v)[0]

    clf, it1 = lgb_fit(X, lab, Xv[es_idx], lab_v[es_idx], clf_p, es)
    P_v = clf.predict(Xv)
    clf_auc = auc(lab_v, P_v)
    print(f"clf: best_it={it1} val_auc={clf_auc:.4f} "
          f"P[q90,q95,q99]={np.quantile(P_v, [0.90, 0.95, 0.99]).round(4).tolist()}",
          flush=True)

    Xs, ys = X[spec_mask], np.log1p(y_raw[spec_mask])
    spec, it2 = lgb_fit(Xs, ys, Xv[dom_idx], ylog_v[dom_idx], spec_p, es)
    del Xs, ys
    spec_v = spec.predict(Xv)
    dom_rmse = float(np.sqrt(np.mean((spec_v[spec_mask_v] - ylog_v[spec_mask_v]) ** 2)))
    print(f"spec: best_it={it2} val_domain_rmse={dom_rmse:.4f}", flush=True)

    base_v = load_base("val", args.base, uid_val)
    base_rmsle = float(np.sqrt(np.mean((base_v - ylog_v) ** 2)))

    # ---- honest 2-fold-by-users gate tuning on val ----
    half = rng.permutation(len(uid_val)) % 2
    fin = np.empty_like(base_v)
    g_fin = np.empty_like(base_v)
    fold_params = []
    for f in (0, 1):
        tr_m = half == f
        ap_m = ~tr_m
        _, a_f, q_f = tune_gate(P_v[tr_m], spec_v[tr_m], base_v[tr_m],
                                ylog_v[tr_m], args.gmax)
        t_ap = float(np.quantile(P_v[ap_m], q_f))
        fin[ap_m], g_fin[ap_m] = gate_apply(P_v[ap_m], spec_v[ap_m], base_v[ap_m],
                                            a_f, t_ap, args.gmax)
        fold_params.append((a_f, q_f))
    honest = float(np.sqrt(np.mean((fin - ylog_v) ** 2)))
    s_full, a_star, q_star = tune_gate(P_v, spec_v, base_v, ylog_v, args.gmax)
    gated = g_fin > 0
    mean_g = float(g_fin[gated].mean()) if gated.any() else 0.0
    print(f"[GATE] base={base_rmsle:.6f} honest={honest:.6f} "
          f"delta={base_rmsle - honest:+.6f} fold_params={fold_params} "
          f"full=(a={a_star}, q={q_star}, insample={s_full:.6f}) "
          f"gated_share={gated.mean():.4f} mean_g_gated={mean_g:.3f}", flush=True)
    if a_star == 0.0:
        print("[WARN] full-val tuner picked a=0 -> gate off, test == base", flush=True)

    save_preds(name, "val", uid_val, np.expm1(fin))
    aux_dir = PREDS_DIR / "aux_whale"
    aux_dir.mkdir(exist_ok=True)
    pl.DataFrame({"user_id": uid_val.astype(np.int64), "p_top5": P_v,
                  "spec_log": spec_v, "base_log": base_v, "g": g_fin,
                  "target": yv_raw}).write_parquet(aux_dir / f"{name}_heads_val.parquet")
    notes = (args.notes or
             f"whale gate over {args.base} (base={base_rmsle:.6f}); honest 2-fold; "
             f"folds={fold_params} full=(a={a_star},q={q_star}); "
             f"gated={gated.mean():.3f} mean_g={mean_g:.3f}; auc={clf_auc:.4f}; "
             f"topq={args.top_q} specq={args.spec_q} gmax={args.gmax} "
             f"anchors={len(tr_anchors)} gap{args.gap_days}")
    log_score(name, honest, notes)
    if args.smoke:
        ok = honest <= base_rmsle + 5e-4
        print(f"[SMOKE] ok={ok} base={base_rmsle:.6f} honest={honest:.6f}", flush=True)
    if args.no_test:
        print(f"[DONE] {name} total {time.time() - t0:.0f}s", flush=True)
        return

    # ---- test: retrain heads on train+gap+val, gate with full-val params ----
    gap_anchors = [a for a in available_train_anchors() if cutoff < a < VAL_ANCHOR]
    Xg, yg_raw = None, None
    if gap_anchors:
        gtr = load_matrix(gap_anchors, columns=["user_id", "anchor_date", "target"] + cols)
        Xg = gtr.select(cols).to_numpy().astype(np.float32)
        yg_raw = gtr["target"].to_numpy().astype(np.float64)
        del gtr
        print(f"retrain adds gap anchors {[a.isoformat() for a in gap_anchors]}: "
              f"+{len(yg_raw)} rows", flush=True)
    n_stage1, n_spec1 = len(y_raw), int(spec_mask.sum())
    parts = [X] + ([Xg] if Xg is not None else []) + [Xv]
    Xall = np.vstack(parts)
    del parts, X, Xg, Xv
    y_all = np.concatenate([y_raw] + ([yg_raw] if yg_raw is not None else []) + [yv_raw])

    thr_top2 = float(np.quantile(y_all, args.top_q))
    thr_spec2 = float(np.quantile(y_all, args.spec_q))
    lab_all = (y_all >= thr_top2).astype(np.float64)
    spec_all = y_all > thr_spec2
    mult_clf = 1.0 + 0.7 * max(len(y_all) / n_stage1 - 1.0, 0.0)
    mult_spec = 1.0 + 0.7 * max(int(spec_all.sum()) / max(n_spec1, 1) - 1.0, 0.0)
    print(f"retrain: rows {n_stage1}->{len(y_all)} (iter_mult {mult_clf:.3f}), "
          f"spec {n_spec1}->{int(spec_all.sum())} (iter_mult {mult_spec:.3f}), "
          f"thr_top={thr_top2:.1f} thr_spec={thr_spec2:.1f}", flush=True)

    clf_p["n_estimators"] = max(50, int(it1 * mult_clf))
    clf_f, _ = lgb_fit(Xall, lab_all, None, None, clf_p, es)
    spec_p["n_estimators"] = max(50, int(it2 * mult_spec))
    Xs = Xall[spec_all]
    spec_f, _ = lgb_fit(Xs, np.log1p(y_all[spec_all]), None, None, spec_p, es)
    del Xs, Xall, y_all

    test = load_anchor(TEST_ANCHOR)
    Xt = test.select(cols).to_numpy().astype(np.float32)
    uid_t = test["user_id"].to_numpy()
    del test
    P_t = clf_f.predict(Xt)
    spec_t = spec_f.predict(Xt)
    del Xt
    base_t = load_base("test", args.base, uid_t)
    t_test = float(np.quantile(P_t, q_star))
    fin_t, g_t = gate_apply(P_t, spec_t, base_t, a_star, t_test, args.gmax)
    gt = g_t > 0
    print(f"test gate: a={a_star} q={q_star} t={t_test:.4f} "
          f"gated_share={gt.mean():.4f} "
          f"mean_g_gated={(float(g_t[gt].mean()) if gt.any() else 0.0):.3f}", flush=True)
    save_preds(name, "test", uid_t, np.expm1(fin_t))
    pl.DataFrame({"user_id": uid_t.astype(np.int64), "p_top5": P_t,
                  "spec_log": spec_t, "base_log": base_t, "g": g_t}).write_parquet(
        aux_dir / f"{name}_heads_test.parquet")
    print(f"[DONE] {name} val_honest={honest:.6f} total {time.time() - t0:.0f}s",
          flush=True)


if __name__ == "__main__":
    main()

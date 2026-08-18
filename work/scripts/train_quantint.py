"""Quantile-integration estimator (quantint): E[log1p(y)|x] via the quantile function.

RMSLE is RMSE in log space, so the optimal prediction is E[Z|x] with Z = log1p(y).
Every other model in the project estimates that expectation DIRECTLY (regression).
This one estimates the whole conditional DISTRIBUTION of Z with K LightGBM
quantile models and integrates it back:

    E[Z|x] = \int_0^1 Q_Z(u|x) du   (mean = area under the quantile function)

Same target, different route -> different error structure (each alpha is fitted
with its own pinball loss, so the estimate is a weighted vote of K independent
the conditional spread (Q90-Q10) and the implied zero mass p0(x).

Integration details (all justified by the data, see notes below):
  * Q(0) = 0 EXACTLY, because Z = log1p(y) >= 0. That is a free left anchor,
    so the segment [0, alpha_1] is a triangle 0.5*alpha_1*Q(alpha_1) and costs
    nothing for the (many) users whose Q(alpha_1) is already 0.
  * The alpha grid must reach well BELOW the median: the zero mass is
    conditional, not global. Val zero rate is 0.459 overall but only 0.115 for
    users who ordered today and 0.147 for rec_order in 1-2 days (measured on the
    val anchor, deciles of rec_order). For such users Q(u) > 0 from u ~ 0.12, so
    a grid starting at 0.5 would throw away the whole [0.12, 0.5] band of their
    mean. Hence the full grid starts at 0.05.
  * Tail [alpha_K, 1] is closed explicitly (--tail): 'lin' extrapolates with the
    last (clipped) slope, 'rect' holds Q(alpha_K) flat, 'none' drops it. With
    alpha_K = 0.99 the tail is worth ~0.07 log units on average - not optional.
  * Quantile models cross (Q_a1 > Q_a2 for a1 < a2). Repaired per user before
    integrating (--mono): 'sort' = Chernozhukov-Fernandez-Val-Galichon
    rearrangement (default), 'cummax' = running max, 'none' = raw.

Protocol: exp_lib contract, gap-30, USE_V2/V3/V4 features, val preds ->
NAME_val.parquet, retrain (train + gap + val anchors) -> NAME_test.parquet, one
line in scores.tsv. Also writes NAME_aux_{val,test}.parquet (q_spread, p0_hat,
q50, q90) and, unless --no-cal, the binned log-shift calibrated NAME_cal_*.

Smoke (2 anchors, 300 trees, 8 quantiles, + same-protocol direct baseline):
  USE_V2=1 USE_V3=1 USE_V4=1 work/scripts/train_quantint.py --name quantint_smoke \
    --alphas smoke8 --n-anchors 2 --threads 2 --params '{"n_estimators":300}' \
    --no-test --no-cal --direct-baseline
Full run:
  USE_V2=1 USE_V3=1 USE_V4=1 OMP_NUM_THREADS=6 work/scripts/train_quantint.py \
    --name quantint --alphas full21 --n-anchors 14 --threads 6
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
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))
from common import PREDS_DIR, TEST_ANCHOR, VAL_ANCHOR, feature_cols, load_anchor, rmsle
from exp_lib import available_train_anchors, load_matrix, log_score, save_preds
from train_gbdt import fit_lgb

DIRECT_CHAMPION = 1.6927  # twl_repair_ab (lgb tweedie1.45 on log1p, gap30, 14 anchors)

# champion topology, quantile loss instead of tweedie
QUANT_BASE = dict(
    objective="quantile", metric="quantile",
    num_leaves=255, min_data_in_leaf=300, learning_rate=0.05,
    feature_fraction=0.75, n_estimators=6000,
)
DIRECT_BASE = dict(
    objective="tweedie", tweedie_variance_power=1.45, metric="rmse",
    num_leaves=255, min_data_in_leaf=300, learning_rate=0.05,
    feature_fraction=0.75, n_estimators=6000,
)
LOSS_KEYS = ("objective", "metric", "alpha", "tweedie_variance_power")

ALPHA_PRESETS = {
    # dense uniform body + tail refinement; lower end at 0.05 covers the
    # conditional zero mass of the most active users (0.115 measured)
    "full21": [round(0.05 * i, 3) for i in range(1, 20)] + [0.975, 0.99],
    "full16": [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60,
               0.70, 0.775, 0.85, 0.90, 0.94, 0.97, 0.985, 0.995],
    "smoke8": [0.10, 0.25, 0.40, 0.55, 0.70, 0.85, 0.95, 0.99],
    "smoke6": [0.15, 0.35, 0.55, 0.75, 0.90, 0.99],
}


def parse_alphas(spec: str) -> np.ndarray:
    if spec in ALPHA_PRESETS:
        a = ALPHA_PRESETS[spec]
    else:
        a = [float(x) for x in spec.split(",") if x.strip()]
    a = np.asarray(sorted(a), dtype=np.float64)
    assert a.ndim == 1 and len(a) >= 2, "need >= 2 alphas"
    assert a[0] > 0 and a[-1] < 1, "alphas must be strictly inside (0,1)"
    return a


def mono_repair(Q: np.ndarray, mode: str) -> tuple[np.ndarray, float]:
    """Clip to >=0 and make each row non-decreasing in u. Returns (Q, cross_rate)."""
    Q = np.clip(Q, 0.0, None)
    cross = float((np.diff(Q, axis=1) < -1e-9).mean())
    if mode == "sort":
        Q = np.sort(Q, axis=1)          # rearrangement (CFG 2010)
    elif mode == "cummax":
        Q = np.maximum.accumulate(Q, axis=1)
    elif mode != "none":
        raise ValueError(mode)
    return Q, cross


def integrate(Q: np.ndarray, a: np.ndarray, rule: str = "trap",
              tail: str = "lin") -> np.ndarray:
    """E[Z|x] ~= int_0^1 Q(u|x) du. Q is (n, K) with columns at alphas a."""
    if rule == "mid":
        # rectangle rule on Voronoi cells of the grid; tail cell included
        edges = np.concatenate([[0.0], 0.5 * (a[:-1] + a[1:]), [1.0]])
        return Q @ np.diff(edges)
    if rule != "trap":
        raise ValueError(rule)
    E = 0.5 * a[0] * Q[:, 0]                       # [0, a1] with Q(0)=0 exactly
    E = E + 0.5 * (Q[:, :-1] + Q[:, 1:]) @ np.diff(a)
    rem = 1.0 - a[-1]
    if tail == "rect":
        E = E + rem * Q[:, -1]
    elif tail == "lin":
        slope = np.clip((Q[:, -1] - Q[:, -2]) / (a[-1] - a[-2]), 0.0, None)
        E = E + rem * (Q[:, -1] + 0.5 * slope * rem)
    elif tail != "none":
        raise ValueError(tail)
    return E


def to_pred(E_log: np.ndarray) -> np.ndarray:
    return np.expm1(np.clip(E_log, 0.0, None))


def aux_frame(uid: np.ndarray, Q: np.ndarray, a: np.ndarray) -> pl.DataFrame:
    """Distribution by-products a point model cannot give."""
    def at(u):
        return Q[:, int(np.argmin(np.abs(a - u)))]
    return pl.DataFrame({
        "user_id": uid.astype(np.int64),
        "q_spread": (at(0.90) - at(0.10)).astype(np.float32),
        # implied zero mass: largest alpha whose quantile is still ~0
        "p0_hat": np.where((Q > 1e-6).any(axis=1),
                           a[np.argmax(Q > 1e-6, axis=1)],
                           1.0).astype(np.float32),
        "q50": at(0.50).astype(np.float32),
        "q90": at(0.90).astype(np.float32),
    })


def pinball(z: np.ndarray, q: np.ndarray, alpha: float) -> float:
    d = z - q
    return float(np.mean(np.maximum(alpha * d, (alpha - 1.0) * d)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="quantint")
    ap.add_argument("--alphas", default="full21",
                    help=f"preset ({'|'.join(ALPHA_PRESETS)}) or comma list")
    ap.add_argument("--params", type=str, default="{}",
                    help="JSON overrides on top of the champion topology")
    ap.add_argument("--rule", default="trap", choices=["trap", "mid"])
    ap.add_argument("--tail", default="lin", choices=["lin", "rect", "none"])
    ap.add_argument("--mono", default="sort", choices=["sort", "cummax", "none"])
    ap.add_argument("--n-anchors", type=int, default=14)
    ap.add_argument("--gap-days", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--no-test", action="store_true")
    ap.add_argument("--no-cal", action="store_true")
    ap.add_argument("--cal-bins", type=int, default=24)
    ap.add_argument("--no-aux", action="store_true")
    ap.add_argument("--direct-baseline", action="store_true",
                    help="also train the direct champion (tweedie1.45-on-log1p) "
                         "on the same anchors/params; smoke accept criterion")
    ap.add_argument("--notes", type=str, default="")
    args = ap.parse_args()
    if args.threads:
        os.environ["OMP_NUM_THREADS"] = str(args.threads)

    over = json.loads(args.params)
    qparams = dict(QUANT_BASE); qparams.update(over)
    qparams.update({k: v for k, v in QUANT_BASE.items() if k in LOSS_KEYS})
    dparams = dict(DIRECT_BASE); dparams.update(over)
    dparams.update({k: v for k, v in DIRECT_BASE.items() if k in LOSS_KEYS})

    alphas = parse_alphas(args.alphas)
    K = len(alphas)
    t0 = time.time()

    cutoff = VAL_ANCHOR - timedelta(days=args.gap_days)
    avail = available_train_anchors()
    tr_anchors = [a for a in avail if a <= cutoff][-args.n_anchors:]
    gap_anchors = [a for a in avail if cutoff < a < VAL_ANCHOR]
    print(f"alphas ({K}): {alphas.tolist()}", flush=True)
    print(f"train anchors ({len(tr_anchors)}): {[a.isoformat() for a in tr_anchors]}",
          flush=True)
    print(f"gap anchors for retrain ({len(gap_anchors)}): "
          f"{[a.isoformat() for a in gap_anchors]}", flush=True)

    val = load_anchor(VAL_ANCHOR)
    cols = feature_cols(val)
    print(f"{len(cols)} features", flush=True)
    Xv = val.select(cols).to_numpy().astype(np.float32)
    yv_raw = val["target"].to_numpy().astype(np.float64)
    uid_val = val["user_id"].to_numpy()
    zv = np.log1p(yv_raw)
    del val

    tr = load_matrix(tr_anchors, columns=["user_id", "anchor_date", "target"] + cols)
    X = tr.select(cols).to_numpy().astype(np.float32)
    y_raw = tr["target"].to_numpy().astype(np.float64)
    z = np.log1p(y_raw)
    del tr
    print(f"X {X.shape}, Xv {Xv.shape}, train zero_rate={float((y_raw == 0).mean()):.4f} "
          f"val zero_rate={float((yv_raw == 0).mean()):.4f}, load {time.time()-t0:.0f}s",
          flush=True)

    # --- K quantile models, early stop per alpha on VAL pinball loss ---
    Qv = np.empty((len(zv), K), dtype=np.float64)
    best_it, diag = {}, []
    for i, al in enumerate(alphas):
        p = dict(qparams); p["alpha"] = float(al)
        m, it = fit_lgb(X, z, None, Xv, zv, p, "quantile", args.seed + i)
        q = m.predict(Xv)
        del m
        Qv[:, i] = q
        best_it[f"{al:g}"] = int(it)
        qc = np.clip(q, 0, None)
        diag.append(dict(alpha=float(al), it=int(it),
                         cover=float((zv <= qc).mean()),
                         pinball=round(pinball(zv, qc, float(al)), 5),
                         zero_share=float((qc <= 1e-6).mean())))
        print(f"  a={al:<6g} it={it:<5} cover={diag[-1]['cover']:.3f} "
              f"(nom {al:.3f})  pin={diag[-1]['pinball']:.4f}  "
              f"zeroQ={diag[-1]['zero_share']:.3f}  [{time.time()-t0:.0f}s]", flush=True)

    Qv, cross = mono_repair(Qv, args.mono)
    print(f"quantile crossing rate before repair: {cross:.4f} "
          f"(repair='{args.mono}')", flush=True)

    variants = {}
    for rule in ("trap", "mid"):
        for tail in (("lin", "rect", "none") if rule == "trap" else ("na",)):
            key = f"{rule}/{tail}"
            E = integrate(Qv, alphas, rule, tail if tail != "na" else "none")
            variants[key] = dict(rmsle=round(rmsle(yv_raw, to_pred(E)), 6),
                                 mean_E=round(float(E.mean()), 4))
    for k, v in sorted(variants.items(), key=lambda kv: kv[1]["rmsle"]):
        print(f"  integ {k:<12} rmsle={v['rmsle']:.6f} mean_E={v['mean_E']:.4f}",
              flush=True)

    E_val = integrate(Qv, alphas, args.rule, args.tail)
    pv = to_pred(E_val)
    score = rmsle(yv_raw, pv)
    print(f"[MAIN] rule={args.rule} tail={args.tail} mono={args.mono} "
          f"val_rmsle={score:.6f} mean_E={float(E_val.mean()):.4f} "
          f"(true mean log1p={float(zv.mean()):.4f})", flush=True)

    direct = None
    if args.direct_baseline:
        print("--- direct champion baseline (tweedie1.45 on log1p, same anchors)",
              flush=True)
        md, itd = fit_lgb(X, z, None, Xv, zv, dict(dparams), "direct", args.seed)
        pdir = to_pred(md.predict(Xv))
        del md
        direct = {"score": rmsle(yv_raw, pdir), "it": int(itd)}
        e_q = np.log1p(pv) - zv
        e_d = np.log1p(pdir) - zv
        direct["err_corr"] = float(np.corrcoef(e_q, e_d)[0, 1])
        print(f"direct baseline: {direct['score']:.6f} it={itd} "
              f"delta={score - direct['score']:+.6f} "
              f"err_corr(quantint,direct)={direct['err_corr']:.4f}", flush=True)

    save_preds(args.name, "val", uid_val, pv)
    if not args.no_aux:
        aux_frame(uid_val, Qv, alphas).write_parquet(PREDS_DIR / f"{args.name}_aux_val.parquet")
    notes = (args.notes or
             f"quantile-integration K={K} ({args.alphas}) {args.rule}/{args.tail} "
             f"mono={args.mono}; lgb quantile nl{qparams['num_leaves']} "
             f"mdl{qparams['min_data_in_leaf']} lr{qparams['learning_rate']} "
             f"gap{args.gap_days} n{len(tr_anchors)}") + (
             f"; cross={cross:.4f}; direct_champ={DIRECT_CHAMPION} "
             f"d={score - DIRECT_CHAMPION:+.4f}" +
             (f"; same_slice_direct={direct['score']:.6f} "
              f"d={score - direct['score']:+.4f} "
              f"err_corr_direct={direct['err_corr']:.4f}" if direct else ""))
    log_score(args.name, score, notes)

    cal = None
    if not args.no_cal:
        from calibrate import apply_shifts, fit_shifts
        lp = np.log1p(pv)
        ly = zv
        half = np.random.default_rng(0).permutation(len(uid_val)) < len(uid_val) // 2
        c1, s1 = fit_shifts(lp[half], ly[half], args.cal_bins)
        base_h = rmsle(yv_raw[~half], pv[~half])
        hold = rmsle(yv_raw[~half], np.expm1(apply_shifts(lp[~half], c1, s1)))
        ctr, sh = fit_shifts(lp, ly, args.cal_bins)
        val_cal = np.expm1(apply_shifts(lp, ctr, sh))
        cal = dict(centers=ctr, shifts=sh, holdout=hold, base_holdout=base_h,
                   full=rmsle(yv_raw, val_cal))
        print(f"[CAL] holdout {base_h:.6f} -> {hold:.6f}; full-val {cal['full']:.6f}",
              flush=True)
        save_preds(f"{args.name}_cal", "val", uid_val, val_cal)
        log_score(f"{args.name}_cal", cal["full"],
                  f"binned log-shift calibration of {args.name} (bins={args.cal_bins}); "
                  f"honest holdout {base_h:.6f}->{hold:.6f}")

    print("RESULT " + json.dumps({
        "name": args.name, "val_rmsle": round(score, 6),
        "direct_baseline": None if direct is None else round(direct["score"], 6),
        "delta_vs_direct": None if direct is None else round(score - direct["score"], 6),
        "err_corr_vs_direct": None if direct is None else round(direct["err_corr"], 4),
        "delta_vs_champion": round(score - DIRECT_CHAMPION, 6),
        "integration_variants": variants, "crossing_rate": round(cross, 5),
        "cal_holdout": None if cal is None else round(cal["holdout"], 6),
        "n_alphas": K, "n_anchors": len(tr_anchors),
        "per_alpha": diag, "best_it": best_it,
        "seconds": round(time.time() - t0),
    }), flush=True)

    if args.no_test:
        return

    # --- retrain on train + gap + val, predict test (exp_lib contract) ---
    parts, z_parts = [X], [z]
    if gap_anchors:
        g = load_matrix(gap_anchors, columns=["user_id", "anchor_date", "target"] + cols)
        parts.append(g.select(cols).to_numpy().astype(np.float32))
        z_parts.append(np.log1p(g["target"].to_numpy().astype(np.float64)))
        print(f"retrain adds gap anchors: +{parts[-1].shape[0]} rows", flush=True)
        del g
    parts.append(Xv)
    z_parts.append(zv)
    Xall = np.vstack(parts)
    z_all = np.concatenate(z_parts)
    row_ratio = Xall.shape[0] / max(X.shape[0], 1)
    iter_mult = 1.0 + 0.7 * max(row_ratio - 1.0, 0.0)
    print(f"retrain: row_ratio={row_ratio:.3f} iter_mult={iter_mult:.3f}", flush=True)
    del X, Xv, parts, z_parts

    test = load_anchor(TEST_ANCHOR)
    Xt = test.select(cols).to_numpy().astype(np.float32)
    uid_t = test["user_id"].to_numpy()
    del test

    Qt = np.empty((Xt.shape[0], K), dtype=np.float64)
    for i, al in enumerate(alphas):
        p = dict(qparams)
        p["alpha"] = float(al)
        p["n_estimators"] = max(50, int(best_it[f"{al:g}"] * iter_mult))
        mf, _ = fit_lgb(Xall, z_all, None, None, None, p, "quantile", args.seed + i)
        Qt[:, i] = mf.predict(Xt)
        del mf
        print(f"  retrain a={al:g}: {p['n_estimators']} iters [{time.time()-t0:.0f}s]",
              flush=True)

    Qt, cross_t = mono_repair(Qt, args.mono)
    E_t = integrate(Qt, alphas, args.rule, args.tail)
    pt = to_pred(E_t)
    save_preds(args.name, "test", uid_t, pt)
    if not args.no_aux:
        aux_frame(uid_t, Qt, alphas).write_parquet(PREDS_DIR / f"{args.name}_aux_test.parquet")
    if cal is not None:
        from calibrate import apply_shifts
        save_preds(f"{args.name}_cal", "test", uid_t,
                   np.expm1(apply_shifts(np.log1p(pt), cal["centers"], cal["shifts"])))
    print(f"test crossing rate {cross_t:.4f}, mean_E={float(E_t.mean()):.4f}", flush=True)
    print(f"[DONE] {args.name} val_rmsle={score:.6f} total {time.time()-t0:.0f}s",
          flush=True)


if __name__ == "__main__":
    main()

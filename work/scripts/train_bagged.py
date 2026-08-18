"""Anchor-bagged GBDT: ONE random anchor slice per user per bag.

Motivation
----------
The standard trainer stacks K anchor slices, so every user appears K times with
heavily overlapping 30d target windows (7d stride -> 23d overlap between
neighbours). Rows are strongly correlated: nominal 3.5M rows, effective sample
size much smaller. Here each bag draws ONE anchor per user (uniformly, fixed
seed) from the gap-eligible pool -> 250k rows that are independent across users
and use ALL available slices, not just the last 14. N bags with different draws
cover the pool; their predictions are averaged in log1p space.

gap-days is respected: only anchors whose 30d target window ended >= gap days

Examples
--------
  train_bagged.py --name bagged_smoke --n-bags 3 --pool-anchors 24,14 \
      --params '{"objective":"tweedie","tweedie_variance_power":1.45,"n_estimators":1500}' \
      --gap-days 30 --threads 6 --direct-baseline --no-test
  train_bagged.py --name bagged_champ --n-bags 8 --pool-anchors 24 \
      --params '{"objective":"tweedie","tweedie_variance_power":1.45,"n_estimators":6000}' \
      --gap-days 30 --threads 6
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
from common import VAL_ANCHOR, TEST_ANCHOR, FEATURES_DIR, rmsle, load_anchor, feature_cols
from exp_lib import available_train_anchors, save_preds, log_score
from train_gbdt import fit_lgb
from calibrate import apply_shifts, fit_shifts

SIDE = {"USE_V2": "extra", "USE_V3": "v3", "USE_V4": "v4", "USE_V6": "v6"}


def cal_holdout(lp, ly, y_raw, half, bins: int = 24):
    """Honest binned log-shift calibration: fit on half the users, score the rest.

    The pipeline calibrates EVERY model before blending (KNOWLEDGE.md), so the
    calibrated number is the decision-relevant one; raw scores mostly reflect
    the ~0.25 log level bias that calibration removes anyway.
    """
    lp = np.clip(lp, 0, None)
    c, s = fit_shifts(lp[half], ly[half], bins)
    return (float(rmsle(y_raw[~half], np.expm1(lp[~half]))),
            float(rmsle(y_raw[~half], np.expm1(apply_shifts(lp[~half], c, s)))))


def has_side_files(a: date) -> bool:
    """load_anchor silently skips a missing side file -> column would vanish."""
    for env, suf in SIDE.items():
        if os.environ.get(env) and not (FEATURES_DIR / f"anchor={a.isoformat()}.{suf}.parquet").exists():
            return False
    return True


def eligible_anchors(gap_days: int, include_val: bool = False) -> list[date]:
    out = [a for a in available_train_anchors() if has_side_files(a)]
    if gap_days:
        cutoff = VAL_ANCHOR - timedelta(days=gap_days)
        out = [a for a in out if a <= cutoff]
    if include_val and has_side_files(VAL_ANCHOR):
        out = out + [VAL_ANCHOR]
    return out


def draw_assignments(n_bags: int, n_pool: int, n_users: int, sample_seed: int) -> list[np.ndarray]:
    return [np.random.default_rng(sample_seed + 1000 * b).integers(0, n_pool, size=n_users)
            for b in range(n_bags)]


def build_bags(pool: list[date], assigns: list[np.ndarray], cols: list[str], n_users: int):
    """One pass over the pool; scatter each anchor's rows into every bag that wants them.

    All anchor files hold the same 250k user_ids in the same sorted order
    (verified), so rows can be scattered positionally.
    """
    Xb = [np.empty((n_users, len(cols)), np.float32) for _ in assigns]
    yb = [np.empty(n_users, np.float64) for _ in assigns]
    for j, a in enumerate(pool):
        masks = [asg == j for asg in assigns]
        if not any(m.any() for m in masks):
            continue
        df = load_anchor(a, ["user_id", "target"] + cols)
        assert df.height == n_users, f"{a}: {df.height} rows != {n_users}"
        Xa = df.select(cols).to_numpy().astype(np.float32)
        ya = df["target"].to_numpy().astype(np.float64)
        del df
        for b, m in enumerate(masks):
            if m.any():
                Xb[b][m] = Xa[m]
                yb[b][m] = ya[m]
        del Xa, ya
    return Xb, yb


def stack_anchors(anchors: list[date], cols: list[str], n_users: int):
    """Standard stacked matrix, filled into a preallocated array (peak RAM ~1 copy)."""
    X = np.empty((len(anchors) * n_users, len(cols)), np.float32)
    y = np.empty(len(anchors) * n_users, np.float64)
    for i, a in enumerate(anchors):
        df = load_anchor(a, ["user_id", "target"] + cols)
        assert df.height == n_users, f"{a}: {df.height} rows != {n_users}"
        X[i * n_users:(i + 1) * n_users] = df.select(cols).to_numpy().astype(np.float32)
        y[i * n_users:(i + 1) * n_users] = df["target"].to_numpy().astype(np.float64)
        del df
    return X, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--n-bags", type=int, default=3)
    ap.add_argument("--pool-anchors", type=str, default="0",
                    help="comma list of pool sizes (last N eligible anchors); 0 = all eligible")
    ap.add_argument("--params", type=str, default='{"objective":"tweedie","tweedie_variance_power":1.45,"n_estimators":6000}')
    ap.add_argument("--gap-days", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42, help="base LightGBM seed (bag b uses seed+b)")
    ap.add_argument("--sample-seed", type=int, default=777, help="anchor-draw seed")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--direct-baseline", action="store_true",
                    help="also train the standard stacked model on the last --baseline-anchors slices")
    ap.add_argument("--baseline-anchors", type=int, default=14)
    ap.add_argument("--baseline-seeds", type=int, default=1,
                    help=">1 averages that many stacked models in log space (ensembling control)")
    ap.add_argument("--no-test", action="store_true")
    ap.add_argument("--notes", type=str, default="")
    args = ap.parse_args()

    if args.threads:
        os.environ["OMP_NUM_THREADS"] = str(args.threads)
    params = json.loads(args.params)
    t0 = time.time()

    pool_full = eligible_anchors(args.gap_days)
    pool_sizes = [int(s) for s in args.pool_anchors.split(",") if s.strip()]
    val = load_anchor(VAL_ANCHOR)
    cols = feature_cols(val)
    n_users = val.height
    Xv = val.select(cols).to_numpy().astype(np.float32)
    yv_raw = val["target"].to_numpy().astype(np.float64)
    uid_val = val["user_id"].to_numpy()
    del val
    yv_log = np.log1p(yv_raw)
    # same user split as calibrate.py so holdout numbers are comparable repo-wide
    half = np.random.default_rng(0).permutation(n_users) < n_users // 2
    print(f"{len(cols)} features, {n_users} users, {len(pool_full)} eligible anchors "
          f"({pool_full[0]} .. {pool_full[-1]})", flush=True)

    results = {}
    best = None  # (score, pool_size, pred_log, best_iters, pool)

    for ps in pool_sizes:
        pool = pool_full if ps <= 0 else pool_full[-ps:]
        assigns = draw_assignments(args.n_bags, len(pool), n_users, args.sample_seed)
        tb = time.time()
        Xb, yb = build_bags(pool, assigns, cols, n_users)
        print(f"[pool={len(pool)}] {pool[0]}..{pool[-1]} built {args.n_bags} bags "
              f"of {n_users} rows in {time.time()-tb:.0f}s", flush=True)
        preds, iters, solo = [], [], []
        for b in range(args.n_bags):
            p = dict(params)
            m, it = fit_lgb(Xb[b], np.log1p(yb[b]), None, Xv, yv_log, p, "log_mse", args.seed + b)
            pl_ = m.predict(Xv)
            preds.append(pl_)
            iters.append(int(it))
            solo.append(rmsle(yv_raw, np.expm1(np.clip(pl_, 0, None))))
            print(f"  bag {b}: it={it} solo={solo[-1]:.6f} [{time.time()-tb:.0f}s]", flush=True)
            del m
        Xb.clear(); yb.clear()
        mean_log = np.mean(preds, axis=0)
        pv = np.expm1(np.clip(mean_log, 0, None))
        sc = rmsle(yv_raw, pv)
        raw_h, cal_h = cal_holdout(mean_log, yv_log, yv_raw, half)
        solo_raw_h, solo_cal_h = cal_holdout(preds[0], yv_log, yv_raw, half)
        results[f"bagged_pool{len(pool)}"] = dict(score=sc, iters=iters, solo=solo,
                                                  solo_mean=float(np.mean(solo)),
                                                  holdout_raw=raw_h, holdout_cal=cal_h,
                                                  bag0_holdout_raw=solo_raw_h,
                                                  bag0_holdout_cal=solo_cal_h)
        print(f"[pool={len(pool)}] BAGGED n={args.n_bags} val_rmsle={sc:.6f} "
              f"(solo mean {np.mean(solo):.6f}) holdout raw={raw_h:.6f} cal={cal_h:.6f} "
              f"| bag0 raw={solo_raw_h:.6f} cal={solo_cal_h:.6f} [{time.time()-tb:.0f}s]", flush=True)
        if best is None or sc < best[0]:
            best = (sc, len(pool), mean_log, iters, pool)

    if args.direct_baseline:
        anchors = pool_full[-args.baseline_anchors:]
        tb = time.time()
        Xs, ys = stack_anchors(anchors, cols, n_users)
        print(f"[direct] {len(anchors)} anchors {anchors[0]}..{anchors[-1]} "
              f"X {Xs.shape} in {time.time()-tb:.0f}s", flush=True)
        ys_log = np.log1p(ys)
        dpreds, diters, dsolo = [], [], []
        for s in range(args.baseline_seeds):
            p = dict(params)
            m, it = fit_lgb(Xs, ys_log, None, Xv, yv_log, p, "log_mse", args.seed + s)
            pl_ = m.predict(Xv)
            dpreds.append(pl_)
            diters.append(int(it))
            dsolo.append(rmsle(yv_raw, np.expm1(np.clip(pl_, 0, None))))
            print(f"  direct seed {args.seed+s}: it={it} rmsle={dsolo[-1]:.6f} "
                  f"[{time.time()-tb:.0f}s]", flush=True)
            del m
        del Xs, ys, ys_log
        dmean_log = np.mean(dpreds, axis=0)
        dsc = rmsle(yv_raw, np.expm1(np.clip(dmean_log, 0, None)))
        draw_h, dcal_h = cal_holdout(dmean_log, yv_log, yv_raw, half)
        dsolo_raw_h, dsolo_cal_h = cal_holdout(dpreds[0], yv_log, yv_raw, half)
        results["direct"] = dict(score=dsc, iters=diters, solo=dsolo, n_anchors=len(anchors),
                                 holdout_raw=draw_h, holdout_cal=dcal_h,
                                 seed0_holdout_raw=dsolo_raw_h, seed0_holdout_cal=dsolo_cal_h)
        print(f"[direct] STANDARD {len(anchors)} anchors x{args.baseline_seeds} "
              f"val_rmsle={dsc:.6f} holdout raw={draw_h:.6f} cal={dcal_h:.6f} "
              f"| seed0 raw={dsolo_raw_h:.6f} cal={dsolo_cal_h:.6f}", flush=True)
        save_preds(args.name + "_direct", "val", uid_val, np.expm1(np.clip(dmean_log, 0, None)))
    if args.direct_baseline and best is not None:
        # decorrelation check: even a losing family can pay in the blend (in-sample on val)
        eb = np.clip(best[2], 0, None) - yv_log
        ed = np.clip(dmean_log, 0, None) - yv_log
        ws = np.arange(0, 1.001, 0.05)
        bl = [(rmsle(yv_raw, np.expm1(np.clip(w * best[2] + (1 - w) * dmean_log, 0, None))), w)
              for w in ws]
        bl.sort()
        results["blend"] = dict(err_corr=float(np.corrcoef(eb, ed)[0, 1]),
                                best_w_bagged=float(bl[0][1]), best_score=float(bl[0][0]))
        print(f"[blend] err_corr={results['blend']['err_corr']:.4f} "
              f"best w_bagged={bl[0][1]:.2f} -> {bl[0][0]:.6f} (in-sample on val)", flush=True)

    if best is None:  # direct-only control run (--pool-anchors "")
        sc = results["direct"]["score"]
        log_score(args.name, sc, args.notes or
                  f"DIRECT-ONLY control {results['direct']['n_anchors']} anchors "
                  f"x{args.baseline_seeds} seeds it={results['direct']['iters']}")
        out = Path(__file__).resolve().parents[1] / "reports" / f"{args.name}.json"
        out.write_text(json.dumps({"results": results, "params": params}, indent=1, default=str))
        print(f"[DONE] {args.name} {time.time()-t0:.0f}s", flush=True)
        return

    sc, ps, mean_log, iters, pool = best
    note = args.notes or (f"bagged n={args.n_bags} pool={ps} gap{args.gap_days} it={iters}")
    if "direct" in results:
        d = results["direct"]
        bag = results[f"bagged_pool{ps}"]
        note += (f"; direct{d['n_anchors']}={d['score']:.6f} d_raw={sc - d['score']:+.6f}"
                 f"; holdout cal bagged={bag['holdout_cal']:.6f} direct={d['holdout_cal']:.6f} "
                 f"d_cal={bag['holdout_cal'] - d['holdout_cal']:+.6f}")
    save_preds(args.name, "val", uid_val, np.expm1(np.clip(mean_log, 0, None)))
    log_score(args.name, sc, note)
    out = Path(__file__).resolve().parents[1] / "reports" / f"{args.name}.json"
    out.write_text(json.dumps({"results": results, "n_bags": args.n_bags,
                               "params": params, "gap_days": args.gap_days,
                               "sample_seed": args.sample_seed}, indent=1, default=str))
    print(f"[JSON] {out}", flush=True)

    if args.no_test:
        print(f"[DONE] {args.name} {time.time()-t0:.0f}s", flush=True)
        return

    # ---- retrain for test: pool shifts forward (gap anchors + VAL become usable),
    # each bag = 250k sampled rows + the full VAL block (recency matters for test).
    rpool = [a for a in eligible_anchors(0, include_val=True) if a >= pool[0]]
    print(f"[retrain] pool {len(rpool)}: {rpool[0]}..{rpool[-1]}", flush=True)
    rassigns = draw_assignments(args.n_bags, len(rpool), n_users, args.sample_seed + 50000)
    Xb, yb = build_bags(rpool, rassigns, cols, n_users)
    test = load_anchor(TEST_ANCHOR)
    Xt = test.select(cols).to_numpy().astype(np.float32)
    uid_t = test["user_id"].to_numpy()
    del test
    iter_mult = 1.0 + 0.7 * 1.0  # rows double (sampled block + full val block)
    tpreds = []
    for b in range(args.n_bags):
        p = dict(params)
        p["n_estimators"] = max(50, int(iters[b] * iter_mult))
        X = np.vstack([Xb[b], Xv])
        y = np.concatenate([np.log1p(yb[b]), yv_log])
        Xb[b] = None; yb[b] = None
        m, _ = fit_lgb(X, y, None, None, None, p, "log_mse", args.seed + b)
        del X, y
        tpreds.append(m.predict(Xt))
        print(f"  retrain bag {b}: n_estimators={p['n_estimators']} [{time.time()-t0:.0f}s]", flush=True)
        del m
    pt = np.expm1(np.clip(np.mean(tpreds, axis=0), 0, None))
    save_preds(args.name, "test", uid_t, pt)
    print(f"[DONE] {args.name} val_rmsle={sc:.6f} total {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

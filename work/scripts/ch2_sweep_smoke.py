"""Smoke sweep for per-channel LGBM configs (channel V2, TEAM_PLAN #1).

Trains each channel config ONCE on the 2-anchor smoke protocol (gap-30 cutoff,
n_estimators cap 300) and scores every (search_cfg x cat_cfg) pair as
expm1(ps)+expm1(pc) on VAL. The channels are independent models, so 2+4 fits
cover all 8 pairs. For the best pair also runs the per-channel calibration +
Jensen-k path (honest half-split) to sanity-check the V2 calibration code.

Usage:
  USE_V2=1 USE_V3=1 USE_V4=1 POLARS_MAX_THREADS=3 OMP_NUM_THREADS=2 \
    ch2_sweep_smoke.py [--n-anchors 2] [--trees 300] [--threads 2]
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time
from datetime import timedelta
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from common import VAL_ANCHOR, feature_cols, load_anchor, rmsle
from exp_lib import log_score
from train_channel import (CHAMPION_PARAMS, CHANNELS, chtgt_anchors,
                           fit_channel_calibration, load_with_channels,
                           to_arrays)
from train_gbdt import fit_lgb

SEARCH_GRID = {
    "champ_nl255_mdl300": {},
    "nl511_mdl150": {"num_leaves": 511, "min_data_in_leaf": 150},
}
CAT_GRID = {
    "champ_nl255_mdl300": {},
    "nl127_mdl100": {"num_leaves": 127, "min_data_in_leaf": 100},
    "nl255_mdl50": {"num_leaves": 255, "min_data_in_leaf": 50},
    "nl63_mdl300": {"num_leaves": 63, "min_data_in_leaf": 300},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-anchors", type=int, default=2)
    ap.add_argument("--gap-days", type=int, default=30)
    ap.add_argument("--trees", type=int, default=300)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    os.environ["OMP_NUM_THREADS"] = str(args.threads)

    t0 = time.time()
    avail = chtgt_anchors()
    cutoff = VAL_ANCHOR - timedelta(days=args.gap_days)
    tr_anchors = [a for a in avail if a <= cutoff][-args.n_anchors:]
    print(f"smoke anchors: {[a.isoformat() for a in tr_anchors]}", flush=True)

    cols = feature_cols(load_anchor(VAL_ANCHOR))
    cols = [c for c in cols if c not in ("tgt_search", "tgt_cat")]
    val = load_with_channels([VAL_ANCHOR], cols)
    Xv, yv, yv_tot = to_arrays(val, cols)
    del val
    tr = load_with_channels(tr_anchors, cols)
    X, y, _ = to_arrays(tr, cols)
    del tr
    print(f"X {X.shape}, Xv {Xv.shape}, {len(cols)} feats, "
          f"load {time.time()-t0:.0f}s", flush=True)

    grids = {"search": SEARCH_GRID, "cat": CAT_GRID}
    preds: dict[str, dict[str, np.ndarray]] = {c: {} for c in CHANNELS}
    its: dict[str, dict[str, int]] = {c: {} for c in CHANNELS}
    for i, c in enumerate(CHANNELS):
        for tag, over in grids[c].items():
            p = dict(CHAMPION_PARAMS)
            p.update(over)
            p["n_estimators"] = args.trees
            tt = time.time()
            m, it = fit_lgb(X, np.log1p(y[c]), None, Xv, np.log1p(yv[c]),
                            p, "log_mse", args.seed + i)
            preds[c][tag] = m.predict(Xv)
            its[c][tag] = it
            ch = rmsle(yv[c], np.expm1(np.clip(preds[c][tag], 0, None)))
            print(f"[{c}/{tag}] ch_rmsle={ch:.6f} it={it} "
                  f"{time.time()-tt:.0f}s", flush=True)
            del m

    combos = []
    for s_tag, c_tag in itertools.product(SEARCH_GRID, CAT_GRID):
        tot = (np.expm1(np.clip(preds["search"][s_tag], 0, None))
               + np.expm1(np.clip(preds["cat"][c_tag], 0, None)))
        combos.append((rmsle(yv_tot, tot), s_tag, c_tag))
    combos.sort()
    print("--- combo ranking (raw sum) ---", flush=True)
    for sc, s_tag, c_tag in combos:
        print(f"  {sc:.6f}  S={s_tag} C={c_tag}", flush=True)

    best_sc, best_s, best_c = combos[0]
    cal = fit_channel_calibration(
        {"search": preds["search"][best_s], "cat": preds["cat"][best_c]},
        yv, yv_tot)
    print(f"[CAL best pair] k={cal['k']:.2f} holdout={cal['holdout']:.6f} "
          f"(no-k {cal['holdout_nok']:.6f}) full={cal['full']:.6f}", flush=True)

    log_score("ch3_sweep_smoke", best_sc,
              f"SMOKE sweep {args.n_anchors}a {args.trees}t: best S={best_s} "
              f"C={best_c} it={its['search'][best_s]}/{its['cat'][best_c]}; "
              f"cal holdout {cal['holdout']:.6f} k={cal['k']:.2f}")
    print("RESULT " + json.dumps({
        "best": {"search": best_s, "cat": best_c, "total": round(best_sc, 6)},
        "combos": [{"total": round(sc, 6), "search": s, "cat": c}
                   for sc, s, c in combos],
        "best_it": {"search": its["search"][best_s], "cat": its["cat"][best_c]},
        "cal": {"k": cal["k"], "holdout": round(cal["holdout"], 6),
                "holdout_nok": round(cal["holdout_nok"], 6),
                "full": round(cal["full"], 6)},
        "seconds": round(time.time() - t0),
    }), flush=True)


if __name__ == "__main__":
    main()

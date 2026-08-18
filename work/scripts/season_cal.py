"""Per-segment seasonal calibration from the 2025 analog.

Fits, on 2025 data, the per-segment log-shift between W1=[Jan15..Feb13] and
mdl_onyx=[Feb14..Mar15] (the val->test window analog), where segments are deciles of
user gmv in W1. Applies damped shifts to a test-pred file, segmenting users by

Usage: season_cal.py --pred NAME [--damp 0.5] [--max-shift 0.15]
Writes NAME_scal_test.parquet + a val no-op copy for bookkeeping.
"""
from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))
from common import PREDS_DIR, TRAIN_PARQUET, user_universe
from exp_lib import save_preds

W1_25 = (date(2025, 1, 15), date(2025, 2, 13))
W2_25 = (date(2025, 2, 14), date(2025, 3, 15))
W1_26 = (date(2026, 1, 15), date(2026, 2, 13))
N_SEG = 12


def user_window_gmv(lf, lo, hi):
    return (
        lf.filter(pl.col("event_date").is_between(lo, hi))
        .group_by("user_id").agg(pl.col("gmv").sum().alias("g"))
        .collect(engine="streaming")
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--damp", type=float, default=0.5)
    ap.add_argument("--max-shift", type=float, default=0.15)
    args = ap.parse_args()

    uni = user_universe()
    lf = pl.scan_parquet(TRAIN_PARQUET)
    g1 = uni.join(user_window_gmv(lf, *W1_25), on="user_id", how="left").fill_null(0.0)
    g2 = uni.join(user_window_gmv(lf, *W2_25), on="user_id", how="left").fill_null(0.0)
    l1 = np.log1p(g1["g"].to_numpy())
    l2 = np.log1p(g2["g"].to_numpy())

    #0 = zeros, rest = quantile bins of positives
    seg25 = np.zeros(len(l1), dtype=np.int32)
    pos = l1 > 0
    qs = np.quantile(l1[pos], np.linspace(0, 1, N_SEG))
    seg25[pos] = 1 + np.clip(np.searchsorted(qs, l1[pos], side="right") - 1, 0, N_SEG - 2)

    shifts = np.zeros(N_SEG)
    for s in range(N_SEG):
        m = seg25 == s
        if m.sum() >= 1000:
            shifts[s] = l2[m].mean() - l1[m].mean()
    shifts = np.clip(shifts * args.damp, -args.max_shift, args.max_shift)
    print("2025 segment shifts (damped):", np.round(shifts, 4).tolist())

    
    c1 = uni.join(user_window_gmv(lf, *W1_26), on="user_id", how="left").fill_null(0.0)
    lc = np.log1p(c1["g"].to_numpy())
    seg26 = np.zeros(len(lc), dtype=np.int32)
    posc = lc > 0
    seg26[posc] = 1 + np.clip(np.searchsorted(qs, lc[posc], side="right") - 1, 0, N_SEG - 2)

    dt = pl.read_parquet(PREDS_DIR / f"{args.pred}_test.parquet").sort("user_id")
    uni_sorted = uni.sort("user_id")
    assert (dt["user_id"].to_numpy() == uni_sorted["user_id"].to_numpy()).all()
    order = uni["user_id"].to_numpy().argsort()
    seg_sorted = seg26[order]

    lp = np.log1p(np.clip(dt["pred"].to_numpy(), 0, None))
    lp_adj = np.clip(lp + shifts[seg_sorted], 0, None)
    save_preds(f"{args.pred}_scal", "test", dt["user_id"].to_numpy(), np.expm1(lp_adj))
    # val copy unchanged (calibration targets the test window only)
    dv = pl.read_parquet(PREDS_DIR / f"{args.pred}_val.parquet")
    dv.write_parquet(PREDS_DIR / f"{args.pred}_scal_val.parquet")
    seg_counts = np.bincount(seg_sorted, minlength=N_SEG).tolist()
    print(f"applied to {args.pred}: seg counts {seg_counts}")
    print(f"mean lp before {lp.mean():.4f} after {lp_adj.mean():.4f}")


if __name__ == "__main__":
    main()

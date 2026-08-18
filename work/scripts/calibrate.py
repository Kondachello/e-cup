"""Per-quantile-bin calibration in log1p space, fitted on VAL predictions.

For a prediction file NAME: bins val preds into quantile bins (in log space),
computes per-bin shift = mean(log1p(y)) - mean(log1p(pred)), validates honesty
via 2-fold user split, then applies interpolated shifts to val+test preds.
Outputs NAME_cal_{val,test}.parquet.

Usage: calibrate.py --pred NAME [--bins 24]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))
from common import PREDS_DIR, VAL_ANCHOR, load_anchor, rmsle
from exp_lib import save_preds, log_score


def fit_shifts(lp: np.ndarray, ly: np.ndarray, bins: int):
    qs = np.quantile(lp, np.linspace(0, 1, bins + 1))
    qs[0] -= 1e-9; qs[-1] += 1e-9
    centers, shifts = [], []
    for i in range(bins):
        m = (lp > qs[i]) & (lp <= qs[i + 1])
        if m.sum() < 500:
            continue
        centers.append(lp[m].mean())
        shifts.append(ly[m].mean() - lp[m].mean())
    return np.array(centers), np.array(shifts)


def apply_shifts(lp: np.ndarray, centers: np.ndarray, shifts: np.ndarray):
    adj = np.interp(lp, centers, shifts)
    return np.clip(lp + adj, 0, None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--bins", type=int, default=24)
    args = ap.parse_args()

    val = load_anchor(VAL_ANCHOR, columns=["user_id", "target"]).sort("user_id")
    y = val["target"].to_numpy().astype(np.float64)
    ly = np.log1p(y)
    uid = val["user_id"].to_numpy()

    dv = pl.read_parquet(PREDS_DIR / f"{args.pred}_val.parquet").sort("user_id")
    assert (dv["user_id"].to_numpy() == uid).all()
    lp = np.log1p(np.clip(dv["pred"].to_numpy(), 0, None))
    base = rmsle(y, np.expm1(lp))

    # honesty check: fit on half users, eval on other half
    rng = np.random.default_rng(0)
    half = rng.permutation(len(uid)) < len(uid) // 2
    c1, s1 = fit_shifts(lp[half], ly[half], args.bins)
    holdout = rmsle(y[~half], np.expm1(apply_shifts(lp[~half], c1, s1)))
    base_holdout = rmsle(y[~half], np.expm1(lp[~half]))
    print(f"holdout: base {base_holdout:.6f} -> calibrated {holdout:.6f} "
          f"({'OK' if holdout < base_holdout else 'NO GAIN'})")

    centers, shifts = fit_shifts(lp, ly, args.bins)
    lv = apply_shifts(lp, centers, shifts)
    cal_val = rmsle(y, np.expm1(lv))
    print(f"full val: base {base:.6f} -> calibrated {cal_val:.6f} (in-sample, optimistic)")
    print("bin shifts:", np.round(shifts, 3).tolist())

    dt = pl.read_parquet(PREDS_DIR / f"{args.pred}_test.parquet").sort("user_id")
    lt = np.log1p(np.clip(dt["pred"].to_numpy(), 0, None))
    ltc = apply_shifts(lt, centers, shifts)

    save_preds(f"{args.pred}_cal", "val", uid, np.expm1(lv))
    save_preds(f"{args.pred}_cal", "test", dt["user_id"].to_numpy(), np.expm1(ltc))
    log_score(f"{args.pred}_cal", cal_val,
              f"binned log-shift calibration of {args.pred}; holdout {base_holdout:.6f}->{holdout:.6f}")


if __name__ == "__main__":
    main()

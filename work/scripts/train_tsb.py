"""TSB (Teunter-Syntetos-Babai) intermittent-demand predictor. Closes item Ш4.

Per-user exponential smoothing of two things: the probability that a day has demand
(updated EVERY day) and the demand size (updated only on demand days). The classic
against Croston: TSB decays the probability during silence, so lapsing users fade.
Forecast for the window = 30 * p_hat * z_hat, raw GMV scale.

Expectation is honest: kalman - the same family of "user's own history through a
smoother" - just scored weight 0.000 in the set. This either closes Ш4 with a number
or surprises us; both outcomes are results.

Discipline: alpha grid is selected on the DECEMBER window (anchor 2025-12-31, target
01.01-30.01), never on the validation window that margin.py scores. Selecting where
you measure is the trap the team already paid for (mdl_realgr, H2).

Usage:
  python work/scripts/train_tsb.py --name tsb
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))
from common import DATA_START, HORIZON, TEST_ANCHOR, TRAIN_PARQUET, VAL_ANCHOR, rmsle, user_universe
from exp_lib import log_score, save_preds

SELECT_ANCHOR = date(2025, 12, 31)
GRID_P = (0.02, 0.05, 0.10, 0.20)
GRID_Z = (0.05, 0.15, 0.30)


def day_idx(d: date) -> int:
    return (d - DATA_START).days


def load_matrix_gmv():
    """250k x 409 raw daily GMV; absent rows are honest zeros (no visit, no demand)."""
    uni = user_universe()
    uid = uni["user_id"].to_numpy()
    pos = {u: i for i, u in enumerate(uid)}
    df = (
        pl.scan_parquet(TRAIN_PARQUET)
        .filter(pl.col("gmv") > 0)
        .select("user_id", "event_date", "gmv")
        .collect()
    )
    rows = np.fromiter((pos[u] for u in df["user_id"].to_numpy()), dtype=np.int64, count=df.height)
    cols = (df["event_date"].to_numpy() - np.datetime64(DATA_START)).astype("timedelta64[D]").astype(np.int64)
    M = np.zeros((len(uid), day_idx(TEST_ANCHOR) + 1), dtype=np.float32)
    np.add.at(M, (rows, cols), df["gmv"].to_numpy().astype(np.float32))
    return uid, M


def window_target(anchor: date) -> np.ndarray:
    tgt = (
        pl.scan_parquet(TRAIN_PARQUET)
        .filter((pl.col("event_date") > anchor)
                & (pl.col("event_date") <= anchor + timedelta(days=HORIZON)))
        .group_by("user_id").agg(pl.col("gmv").sum().alias("y")).collect()
    )
    return (user_universe().join(tgt, on="user_id", how="left")
            .with_columns(pl.col("y").fill_null(0.0)).sort("user_id"))["y"].to_numpy()


def tsb_snapshots(M: np.ndarray, ap: float, az: float, snap_at: dict[int, str]):
    """One pass over days; p/z snapshots taken at each requested anchor index."""
    U, T = M.shape
    d_all = M > 0
    p = np.full(U, d_all[:, :60].mean(), dtype=np.float64)      # cold-start prior
    z = np.full(U, float(M[d_all].mean()) if d_all.any() else 1.0, dtype=np.float64)
    out = {}
    for t in range(T):
        d = d_all[:, t]
        p += ap * (d.astype(np.float64) - p)
        if d.any():
            z[d] += az * (M[d, t].astype(np.float64) - z[d])
        if t in snap_at:
            out[snap_at[t]] = HORIZON * p * np.clip(z, 0, None)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="tsb")
    args = ap.parse_args()
    t0 = time.time()

    uid, M = load_matrix_gmv()
    print(f"матрица {M.shape}, load {time.time()-t0:.0f}s", flush=True)
    snap = {day_idx(SELECT_ANCHOR): "dec", day_idx(VAL_ANCHOR): "val", day_idx(TEST_ANCHOR): "test"}

    y_dec = window_target(SELECT_ANCHOR)
    best = (9e9, None, None)
    for a_p in GRID_P:
        for a_z in GRID_Z:
            f = tsb_snapshots(M, a_p, a_z, snap)
            s_dec = rmsle(y_dec, f["dec"])
            print(f"  ap={a_p:.2f} az={a_z:.2f}  DEC={s_dec:.6f}", flush=True)
            if s_dec < best[0]:
                best = (s_dec, (a_p, a_z), f)
    (a_p, a_z), f = best[1], best[2]
    print(f"выбрано на декабре: ap={a_p} az={a_z} (DEC {best[0]:.6f})", flush=True)

    y_val = window_target(VAL_ANCHOR)
    s_val = rmsle(y_val, f["val"])
    save_preds(args.name, "val", uid, f["val"])
    save_preds(args.name, "test", uid, f["test"])
    log_score(args.name, s_val,
              f"TSB intermittent demand, ap={a_p} az={a_z} выбраны на DEC31; 30*p*z, без признаков")
    print(f"[DONE] {args.name} val={s_val:.6f} {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

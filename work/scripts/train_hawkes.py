"""Event-stream predictor: Hawkes-style self-exciting process on order events.

Task 4 item 3: look at history as a stream of events with intervals, not a dense
day x channel matrix. Models see 112-196 days of mostly-zero grid; this uses every
order in all 409 days, weighted by exponential decay - an interval representation.

Model. Order days of user u are events t_i with intensity
    lam_u(t) = mu_u + alpha * sum_{t_i < t} beta * exp(-beta (t - t_i))
Branching ratio alpha < 1, kernel decay beta. Closed forms used here:
  - excitation of the future window (T, T+30] by PAST events integrates to
        R_u = S_u * (1 - exp(-30 beta)),  S_u = sum exp(-beta (T - t_i))
  - stationary base rate from the observed count: mu_u = (1 - alpha) * n_u / T_obs
  - expected orders in the window, offspring included:
        E[N] = 30 * mu_u / (1 - alpha) ... for the base process the offspring factor
        is already inside the stationarity correction, so the used form is
        E[N] = 30 * mu_u + (alpha / (1 - alpha)) * R_u
GMV = E[N] * personal order size (EB-shrunk to global). Raw scale; margin.py
calibrates level anyway.

(alpha, beta) selected on the DECEMBER window, never on validation - same discipline
as train_tsb.py. Expectation is honest: this is a closure experiment for the last
untouched item of task 4; kalman and hazard from neighbouring families measured ~0.

Usage:
  python work/scripts/train_hawkes.py --name hawkes
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
GRID_BETA = (1 / 3.5, 1 / 7.0, 1 / 14.0, 1 / 28.0)      # kernel memory ~3.5-28 days
GRID_ALPHA = (0.15, 0.40, 0.65)


def day_idx(d: date) -> int:
    return (d - DATA_START).days


def order_events():
    return (
        pl.scan_parquet(TRAIN_PARQUET)
        .filter(pl.col("to_ord") > 0)
        .select("user_id", "event_date", "gmv")
        .with_columns(((pl.col("event_date") - pl.lit(DATA_START)).dt.total_days())
                      .alias("t").cast(pl.Int64))
        .collect()
    )


def window_target(anchor: date) -> np.ndarray:
    tgt = (
        pl.scan_parquet(TRAIN_PARQUET)
        .filter((pl.col("event_date") > anchor)
                & (pl.col("event_date") <= anchor + timedelta(days=HORIZON)))
        .group_by("user_id").agg(pl.col("gmv").sum().alias("y")).collect()
    )
    return (user_universe().join(tgt, on="user_id", how="left")
            .with_columns(pl.col("y").fill_null(0.0)).sort("user_id"))["y"].to_numpy()


def predict(ev: pl.DataFrame, uni: pl.DataFrame, anchor: date, alpha: float, beta: float):
    T = day_idx(anchor)
    h = ev.filter(pl.col("t") <= T)
    agg = (
        h.with_columns((-beta * (T - pl.col("t"))).exp().alias("w"))
        .group_by("user_id")
        .agg(pl.len().alias("n"), pl.col("w").sum().alias("S"),
             pl.col("gmv").sum().alias("g"))
    )
    d = (uni.join(agg, on="user_id", how="left")
         .with_columns(pl.col("n").fill_null(0), pl.col("S").fill_null(0.0),
                       pl.col("g").fill_null(0.0)).sort("user_id"))
    n = d["n"].to_numpy().astype(np.float64)
    S = d["S"].to_numpy().astype(np.float64)
    g = d["g"].to_numpy().astype(np.float64)
    g_bar = float(g.sum() / max(n.sum(), 1.0))                    # global mean order-day size
    size = (g + 5.0 * g_bar) / (n + 5.0)                          # EB shrink
    mu = (1.0 - alpha) * n / float(T + 1)
    R = S * (1.0 - np.exp(-HORIZON * beta))
    e_n = HORIZON * mu + (alpha / (1.0 - alpha)) * R
    return np.clip(e_n * size, 0, None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="hawkes")
    args = ap.parse_args()
    t0 = time.time()
    ev = order_events()
    uni = user_universe()
    print(f"событий-заказов: {ev.height}, load {time.time()-t0:.0f}s", flush=True)

    y_dec = window_target(SELECT_ANCHOR)
    best = (9e9, None)
    for beta in GRID_BETA:
        for alpha in GRID_ALPHA:
            s = rmsle(y_dec, predict(ev, uni, SELECT_ANCHOR, alpha, beta))
            print(f"  beta=1/{1/beta:.1f}д alpha={alpha:.2f}  DEC={s:.6f}", flush=True)
            if s < best[0]:
                best = (s, (alpha, beta))
    alpha, beta = best[1]
    print(f"выбрано на декабре: alpha={alpha} beta=1/{1/beta:.1f}д (DEC {best[0]:.6f})", flush=True)

    y_val = window_target(VAL_ANCHOR)
    pv = predict(ev, uni, VAL_ANCHOR, alpha, beta)
    s_val = rmsle(y_val, pv)
    save_preds(args.name, "val", uni.sort("user_id")["user_id"].to_numpy(), pv)
    pt = predict(ev, uni, TEST_ANCHOR, alpha, beta)
    save_preds(args.name, "test", uni.sort("user_id")["user_id"].to_numpy(), pt)
    log_score(args.name, s_val,
              f"hawkes event-stream: alpha={alpha} beta=1/{1/beta:.1f}d выбраны на DEC31; "
              f"все 409 дней, интервалы вместо матрицы")
    print(f"[DONE] {args.name} val={s_val:.6f} {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

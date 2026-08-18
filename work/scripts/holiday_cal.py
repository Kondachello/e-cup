"""Data-driven holiday/event calendar from global daily GMV (no external sources).

Recipe validated in promo_elast.py: peak days = top-15% by ratio of daily GMV to a
centered 28d rolling median. Consecutive/near-consecutive peak days are merged into
EVENTS (a gap of <= MERGE_GAP normal days still counts as one event).

Importable: events() -> list[(start_date, end_date)] and peak_mask arrays.
Run directly to print the calendar.
"""
from __future__ import annotations

import os

os.environ.setdefault("POLARS_MAX_THREADS", "3")

import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))
from common import DATA_END, DATA_START, TRAIN_PARQUET  # noqa: E402

TOP_Q = 0.85       # top-15% of days by ratio
MERGE_GAP = 2      # peaks separated by <= 2 non-peak days = one event
MIN_LEN = 1


def daily_gmv() -> tuple[list[date], np.ndarray]:
    d = (
        pl.scan_parquet(TRAIN_PARQUET)
        .group_by("event_date")
        .agg(pl.col("gmv").sum().alias("gmv_day"))
        .sort("event_date")
        .collect(engine="streaming")
    )
    cal = pl.DataFrame({"event_date": pl.date_range(DATA_START, DATA_END, "1d", eager=True)})
    d = cal.join(d, on="event_date", how="left").fill_null(0.0).sort("event_date")
    return d["event_date"].to_list(), d["gmv_day"].to_numpy().astype(np.float64)


def peak_mask(g: np.ndarray) -> np.ndarray:
    n = len(g)
    med = np.empty(n)
    for i in range(n):
        lo, hi = max(0, i - 14), min(n, i + 14)
        med[i] = np.median(g[lo:hi])
    ratio = g / np.maximum(med, 1.0)
    thr = float(np.quantile(ratio, TOP_Q))
    return ratio >= thr, ratio, thr


def group_events(dates: list[date], mask: np.ndarray) -> list[tuple[date, date]]:
    idx = np.nonzero(mask)[0]
    if len(idx) == 0:
        return []
    runs, start, prev = [], idx[0], idx[0]
    for i in idx[1:]:
        if i - prev <= MERGE_GAP + 1:
            prev = i
        else:
            runs.append((start, prev))
            start, prev = i, i
    runs.append((start, prev))
    return [(dates[a], dates[b]) for a, b in runs if (b - a + 1) >= MIN_LEN]


def events() -> tuple[list[tuple[date, date]], list[date], np.ndarray]:
    dates, g = daily_gmv()
    mask, ratio, thr = peak_mask(g)
    return group_events(dates, mask), dates, mask


if __name__ == "__main__":
    dates, g = daily_gmv()
    mask, ratio, thr = peak_mask(g)
    ev = group_events(dates, mask)
    print(f"days={len(dates)} peak_days={int(mask.sum())} thr={thr:.4f} events={len(ev)}")
    for a, b in ev:
        i0, i1 = dates.index(a), dates.index(b)
        print(f"  {a} .. {b}  len={(b-a).days+1:2d}  max_ratio={ratio[i0:i1+1].max():.3f} "
              f"gmv_peak={g[i0:i1+1].max()/1e6:.3f}M")

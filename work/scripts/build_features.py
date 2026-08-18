"""Build per-anchor user features + 30d GMV targets.

Usage:
  python build_features.py --anchors 2026-02-13,2026-01-14
  python build_features.py --preset all   # test + val + 14 train anchors
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
from common import (
    DATA_START, DATA_END, FEATURES_DIR, HORIZON, TEST_ANCHOR, TRAIN_PARQUET,
    VAL_ANCHOR, train_anchors, user_universe,
)

# [A-(w-1), A] trailing windows
MAIN_WINDOWS = [1, 3, 7, 14, 30, 60, 90, 180, 365]
# disjoint past bands (start_back, end_back): [A-start, A-end]
BANDS = [(59, 30), (89, 60), (179, 90), (364, 180)]
# year-ago windows (aligned to target window a year back, and wider context)
YEARAGO = [(364, 335, "ya_tgt"), (379, 320, "ya_wide")]

MAX_BACK = 379


def win_mask(anchor: date, start_back: int, end_back: int) -> pl.Expr:
    return pl.col("event_date").is_between(
        anchor - timedelta(days=start_back), anchor - timedelta(days=end_back)
    )


def agg_exprs(anchor: date) -> list[pl.Expr]:
    e: list[pl.Expr] = []
    A = pl.lit(anchor)

    for w in MAIN_WINDOWS:
        m = win_mask(anchor, w - 1, 0)
        e.append(pl.col("gmv").filter(m).sum().alias(f"gmv_sum_{w}"))
        e.append(pl.col("to_ord").filter(m).sum().alias(f"ord_cnt_{w}"))
        e.append((pl.col("to_ord") > 0).filter(m).sum().alias(f"ord_days_{w}"))
        e.append(m.sum().alias(f"active_days_{w}"))
        if w >= 7:
            e.append(pl.col("to_cart").filter(m).sum().alias(f"cart_cnt_{w}"))
            e.append(pl.col("searches").filter(m).sum().alias(f"searches_{w}"))
        if w in (30, 90, 365):
            e.append((pl.col("to_cart") > 0).filter(m).sum().alias(f"cart_days_{w}"))
            e.append(pl.col("search").filter(m).sum().alias(f"search_days_{w}"))
            e.append(pl.col("cat").filter(m).sum().alias(f"cat_days_{w}"))
            e.append(pl.col("gmv_search").filter(m).sum().alias(f"gmv_search_{w}"))
            e.append(pl.col("gmv_cat").filter(m).sum().alias(f"gmv_cat_{w}"))
            e.append(pl.col("gmv").filter(m & (pl.col("gmv") > 0)).mean().alias(f"gmv_daymean_{w}"))
            e.append(pl.col("gmv").filter(m & (pl.col("gmv") > 0)).max().alias(f"gmv_daymax_{w}"))
            e.append(pl.col("gmv").filter(m & (pl.col("gmv") > 0)).std().alias(f"gmv_daystd_{w}"))
        if w in (90, 365):
            e.append(pl.col("has_search_to_ord").filter(m).sum().alias(f"s2o_days_{w}"))
            e.append(pl.col("has_cat_to_ord").filter(m).sum().alias(f"c2o_days_{w}"))
            e.append(pl.col("has_search_to_cart").filter(m).sum().alias(f"s2c_days_{w}"))
            e.append(pl.col("has_cat_to_cart").filter(m).sum().alias(f"c2c_days_{w}"))

    for sb, eb in BANDS:
        m = win_mask(anchor, sb, eb)
        tag = f"b{eb}_{sb}"
        e.append(pl.col("gmv").filter(m).sum().alias(f"gmv_sum_{tag}"))
        e.append((pl.col("to_ord") > 0).filter(m).sum().alias(f"ord_days_{tag}"))
        e.append(m.sum().alias(f"active_days_{tag}"))
        e.append(pl.col("searches").filter(m).sum().alias(f"searches_{tag}"))

    for sb, eb, tag in YEARAGO:
        m = win_mask(anchor, sb, eb)
        e.append(pl.col("gmv").filter(m).sum().alias(f"gmv_sum_{tag}"))
        e.append((pl.col("to_ord") > 0).filter(m).sum().alias(f"ord_days_{tag}"))
        e.append(m.sum().alias(f"active_days_{tag}"))
        e.append(pl.col("to_ord").filter(m).sum().alias(f"ord_cnt_{tag}"))

    # recency (days since last X); tenure
    e.append((A - pl.col("event_date").max()).dt.total_days().alias("rec_active"))
    e.append((A - pl.col("event_date").min()).dt.total_days().alias("tenure"))
    e.append((A - pl.col("event_date").filter(pl.col("to_ord") > 0).max()).dt.total_days().alias("rec_order"))
    e.append((A - pl.col("event_date").filter(pl.col("to_cart") > 0).max()).dt.total_days().alias("rec_cart"))
    e.append((A - pl.col("event_date").filter(pl.col("search") > 0).max()).dt.total_days().alias("rec_search"))
    e.append((A - pl.col("event_date").filter(pl.col("cat") > 0).max()).dt.total_days().alias("rec_cat"))
    e.append((A - pl.col("event_date").filter(pl.col("gmv") > 0).max()).dt.total_days().alias("rec_gmv"))

    # order-day gap stats (full history in scan range)
    od = pl.col("event_date").filter(pl.col("to_ord") > 0).sort().diff().dt.total_days()
    e.append(od.mean().alias("ord_gap_mean"))
    e.append(od.std().alias("ord_gap_std"))
    e.append(od.min().alias("ord_gap_min"))
    e.append(od.max().alias("ord_gap_max"))
    ad = pl.col("event_date").sort().diff().dt.total_days()
    e.append(ad.mean().alias("act_gap_mean"))
    e.append(ad.std().alias("act_gap_std"))

    # totals over full scan range (≈ last 379d)
    e.append(pl.col("gmv").sum().alias("gmv_sum_full"))
    e.append((pl.col("to_ord") > 0).sum().alias("ord_days_full"))
    e.append(pl.len().alias("active_days_full"))

    # last active day snapshot
    e.append(pl.col("gmv").sort_by("event_date").last().alias("last_day_gmv"))
    e.append(pl.col("to_ord").sort_by("event_date").last().alias("last_day_ord"))
    e.append(pl.col("to_cart").sort_by("event_date").last().alias("last_day_cart"))
    e.append(pl.col("searches").sort_by("event_date").last().alias("last_day_searches"))
    return e


def derived_exprs() -> list[pl.Expr]:
    eps = 1e-6
    e = []
    # trends: recent vs previous bands
    e.append(((pl.col("gmv_sum_30") + 1).log() - (pl.col("gmv_sum_b30_59") + 1).log()).alias("gmv_trend_30"))
    e.append(((pl.col("active_days_30") + 1) / (pl.col("active_days_b30_59") + 1)).alias("act_trend_30"))
    e.append(((pl.col("gmv_sum_90") + 1).log() - (pl.col("gmv_sum_b90_179") + 1).log()).alias("gmv_trend_90"))
    e.append(((pl.col("searches_30") + 1) / (pl.col("searches_b30_59") + 1)).alias("search_trend_30"))
    # conversion / mix
    e.append((pl.col("ord_days_90") / (pl.col("active_days_90") + eps)).alias("ord_rate_90"))
    e.append((pl.col("ord_days_365") / (pl.col("active_days_365") + eps)).alias("ord_rate_365"))
    e.append((pl.col("ord_cnt_90") / (pl.col("cart_cnt_90") + eps)).alias("cart2ord_90"))
    e.append((pl.col("gmv_search_90") / (pl.col("gmv_sum_90") + eps)).alias("gmv_search_share_90"))
    e.append((pl.col("gmv_search_365") / (pl.col("gmv_sum_365") + eps)).alias("gmv_search_share_365"))
    e.append((pl.col("gmv_sum_90") / (pl.col("ord_days_90") + eps)).alias("gmv_per_ordday_90"))
    e.append((pl.col("gmv_sum_365") / (pl.col("ord_days_365") + eps)).alias("gmv_per_ordday_365"))
    e.append((pl.col("gmv_sum_365") / (pl.col("active_days_365") + eps)).alias("gmv_per_actday_365"))
    # regularity
    e.append((pl.col("active_days_365") / (pl.col("tenure") + 1.0)).alias("act_density"))
    e.append((pl.col("rec_order") / (pl.col("ord_gap_mean") + 1.0)).alias("rec_over_gap"))
    # log transforms of heavy-tailed sums
    for c in ["gmv_sum_7", "gmv_sum_30", "gmv_sum_90", "gmv_sum_365", "gmv_sum_ya_tgt", "gmv_sum_ya_wide", "gmv_sum_full"]:
        e.append((pl.col(c) + 1).log().alias(f"log_{c}"))
    return e


def seasonal_index(df_daily: pl.DataFrame, anchor: date) -> dict[str, float]:
    """Global expected GMV level of [A+1, A+30] using 2025 calendar mapping, smoothed."""
    s = df_daily.sort("event_date").with_columns(
        pl.col("gmv_sum").rolling_mean(7, center=True, min_samples=1).alias("gmv_smooth")
    )
    cal = {r["event_date"]: r["gmv_smooth"] for r in s.iter_rows(named=True)}
    days = [anchor + timedelta(days=i) for i in range(1, HORIZON + 1)]
    vals = []
    for d in days:
        try:
            d25 = date(2025, d.month, d.day)
        except ValueError:
            d25 = date(2025, 2, 28)
        if d25 in cal:
            vals.append(cal[d25])
    base = float(np.mean([v for v in cal.values()]))
    idx = float(np.mean(vals)) / base if vals else 1.0
    return {"seasonal_index": idx}


def build_anchor(lf: pl.LazyFrame, universe: pl.DataFrame, daily: pl.DataFrame, anchor: date) -> pl.DataFrame:
    t0 = time.time()
    hist = lf.filter(
        (pl.col("event_date") <= anchor)
        & (pl.col("event_date") >= anchor - timedelta(days=MAX_BACK))
    )
    feats = hist.group_by("user_id").agg(agg_exprs(anchor)).collect(engine="streaming")
    feats = feats.with_columns(derived_exprs())

    out = universe.join(feats, on="user_id", how="left")
    # fill count-like nulls with 0; keep stat/recency nulls as NaN-friendly nulls
    zero_fill = [c for c in out.columns if c.startswith((
        "gmv_sum", "ord_cnt", "ord_days", "active_days", "cart_cnt", "cart_days",
        "searches", "search_days", "cat_days", "gmv_search_", "gmv_cat_",
        "s2o_", "c2o_", "s2c_", "c2c_", "log_gmv", "last_day",
    ))]
    out = out.with_columns([pl.col(c).fill_null(0) for c in zero_fill])

    # year-ago windows: null-out when not fully covered by data (avoid fake zeros)
    for sb, eb, tag in YEARAGO:
        cov_start = anchor - timedelta(days=sb)
        cov = 1.0 if cov_start >= DATA_START else max(0.0, (anchor - timedelta(days=eb) - DATA_START).days + 1) / (sb - eb + 1)
        cov = min(cov, 1.0)
        out = out.with_columns(pl.lit(cov, dtype=pl.Float32).alias(f"ya_cov_{tag}"))
        if cov < 0.999:
            for c in [f"gmv_sum_{tag}", f"ord_days_{tag}", f"active_days_{tag}", f"ord_cnt_{tag}"]:
                out = out.with_columns(pl.lit(None, dtype=pl.Float32).alias(c))
            out = out.with_columns(pl.lit(None, dtype=pl.Float32).alias(f"log_gmv_sum_{tag}"))

    si = seasonal_index(daily, anchor)
    hist_days = (anchor - DATA_START).days + 1
    out = out.with_columns(
        pl.lit(si["seasonal_index"], dtype=pl.Float64).alias("seasonal_index"),
        pl.lit(hist_days, dtype=pl.Int32).alias("history_days"),
        pl.lit(anchor).alias("anchor_date"),
    )

    # target
    if anchor + timedelta(days=HORIZON) <= DATA_END:
        tgt = (
            lf.filter(pl.col("event_date").is_between(
                anchor + timedelta(days=1), anchor + timedelta(days=HORIZON)))
            .group_by("user_id").agg(pl.col("gmv").sum().alias("target"))
            .collect(engine="streaming")
        )
        out = out.join(tgt, on="user_id", how="left").with_columns(pl.col("target").fill_null(0.0))
    else:
        out = out.with_columns(pl.lit(None, dtype=pl.Float64).alias("target"))

    # downcast floats to f32 to halve size (keep target f64)
    casts = [pl.col(c).cast(pl.Float32) for c, dt in zip(out.columns, out.dtypes)
             if dt == pl.Float64 and c != "target"]
    out = out.with_columns(casts)
    print(f"  anchor {anchor}: {out.shape} in {time.time()-t0:.1f}s", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--anchors", type=str, default=None)
    ap.add_argument("--preset", type=str, default=None, choices=["all", "core"])
    args = ap.parse_args()

    if args.preset == "all":
        anchors = [TEST_ANCHOR, VAL_ANCHOR] + train_anchors(14)
    elif args.preset == "core":
        anchors = [TEST_ANCHOR, VAL_ANCHOR] + train_anchors(6)
    else:
        anchors = [date.fromisoformat(s) for s in args.anchors.split(",")]

    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    universe = user_universe()
    lf = pl.scan_parquet(TRAIN_PARQUET)
    daily = (
        lf.group_by("event_date").agg(pl.col("gmv").sum().alias("gmv_sum"))
        .collect(engine="streaming")
    )
    print(f"universe={len(universe)}, anchors={len(anchors)}", flush=True)

    for a in anchors:
        p = FEATURES_DIR / f"anchor={a.isoformat()}.parquet"
        if p.exists():
            print(f"  anchor {a}: exists, skip", flush=True)
            continue
        df = build_anchor(lf, universe, daily, a)
        df.write_parquet(p)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()

"""v6 features: renewal/hazard + TSB intermittent-demand features (research Q1).

Per anchor A (history = train rows with event_date <= A; order-days = to_ord>0),
intervals = day gaps between consecutive order-days of a user:
  adi                mean interval; null if <2 order-days
  overdue_ratio      days_since_last_order / adi, clipped at 10; null if adi null
  overdue_ratio_med  days_since_last_order / median interval, clipped at 10
  cv_intervals       std/mean of intervals (sample std, null if <3 order-days)
  p_renewal_30       empirical P(next interval <= d+30 | interval > d), d = current
                     pause, from the user's own intervals when n_intervals>=3 and
                     #(intervals > d)>=1; else fallback to segment survival table
                     (segment = ord-days-365 bucket 1-2/3-5/6-10/11+, users with 0
                     order-days in 365d go to the lowest bucket; grid d=0..365,
                     d clamped; 0.0 where the segment pool has no interval > d)
  tsb_p              TSB demand probability, daily update p+=0.05*(x-p), p0=0
                     => closed form 0.05 * sum over order-days t of 0.95^(A-t);
                     0.0 for users with activity but no orders
  tsb_z              TSB demand size: z=log1p(day gmv) on order-days, alpha=0.2,
                     init z_hat=z_first => weights az*(1-az)^(m-i), first (1-az)^(m-1)
  tsb_forecast30     log1p(tsb_p * expm1(tsb_z) * 30); 0.0 for no-order active users

Written as anchor=DATE.v6.parquet (user_id + 8 Float32 cols, full 250k universe,
nulls where undefined), joined by common.load_anchor when USE_V6=1.
"""
from __future__ import annotations

import sys
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))
from common import FEATURES_DIR, TRAIN_PARQUET, TEST_ANCHOR, VAL_ANCHOR, user_universe
from exp_lib import available_train_anchors

ALPHA_P = 0.05
ALPHA_Z = 0.2
GRID_MAX = 365          # segment survival grid d = 0..365
RENEW_WIN = 30
FEATS = ["adi", "overdue_ratio", "overdue_ratio_med", "cv_intervals",
         "p_renewal_30", "tsb_p", "tsb_z", "tsb_forecast30"]


def seg_expr(ord_365: pl.Expr) -> pl.Expr:
    """Quantile-ish buckets by #order-days in last 365d: 1-2 / 3-5 / 6-10 / 11+.
    0 order-days in 365d (but orders earlier) -> lowest bucket."""
    return (
        pl.when(ord_365 <= 2).then(0)
        .when(ord_365 <= 5).then(1)
        .when(ord_365 <= 10).then(2)
        .otherwise(3)
        .alias("seg")
    )


def segment_tables(iv_seg: pl.DataFrame) -> np.ndarray:
    """4 x (GRID_MAX+1) table: p_seg[s, d] = P(interval <= d+30 | interval > d)
    from the pooled intervals of segment s."""
    tab = np.zeros((4, GRID_MAX + 1), dtype=np.float64)
    need = GRID_MAX + RENEW_WIN + 2
    for s in range(4):
        iv = iv_seg.filter(pl.col("seg") == s)["iv"].to_numpy().astype(np.int64)
        if len(iv) == 0:
            continue
        cnt = np.bincount(iv, minlength=need)
        if len(cnt) < need:
            cnt = np.pad(cnt, (0, need - len(cnt)))
        cum = np.cumsum(cnt)
        tot = len(iv)
        d = np.arange(GRID_MAX + 1)
        S_d = tot - cum[d]                       # #(iv > d)
        S_d30 = tot - np.minimum(cum[d + RENEW_WIN], tot)   # #(iv > d+30)
        with np.errstate(divide="ignore", invalid="ignore"):
            p = np.where(S_d > 0, (S_d - S_d30) / np.maximum(S_d, 1), 0.0)
        tab[s] = p
    return tab


def build(anchor: date, uni: pl.DataFrame, ord_full: pl.DataFrame,
          first_act: pl.DataFrame) -> None:
    t0 = time.time()
    out_p = FEATURES_DIR / f"anchor={anchor.isoformat()}.v6.parquet"
    lo365 = anchor - timedelta(days=365)

    od = ord_full.filter(pl.col("event_date") <= anchor)  # sorted (user_id, date)
    od = od.with_columns(
        pl.col("event_date").diff().over("user_id").dt.total_days().alias("iv"),
        (pl.lit(anchor) - pl.col("event_date")).dt.total_days()
        .cast(pl.Float64).alias("k"),
        pl.col("gmv").clip(lower_bound=0.0).log1p().alias("z"),
        pl.int_range(pl.len()).over("user_id").alias("i"),
        pl.len().over("user_id").alias("m"),
    )
    od = od.with_columns(
        pl.when(pl.col("i") == 0)
        .then(pl.lit(1.0 - ALPHA_Z).pow((pl.col("m") - 1).cast(pl.Float64)))
        .otherwise(pl.lit(ALPHA_Z)
                   * pl.lit(1.0 - ALPHA_Z).pow((pl.col("m") - 1 - pl.col("i"))
                                               .cast(pl.Float64)))
        .alias("w")
    )

    d_expr = (pl.lit(anchor) - pl.col("event_date").max()).dt.total_days()
    g = od.group_by("user_id").agg(
        pl.len().alias("n_ord"),
        d_expr.alias("d_since"),
        pl.col("iv").mean().alias("adi"),
        pl.col("iv").median().alias("med_iv"),
        pl.col("iv").std().alias("std_iv"),
        pl.col("iv").count().alias("n_iv"),
        (pl.col("event_date") > lo365).sum().alias("ord_365"),
        (pl.col("iv") > d_expr).sum().alias("cnt_gt"),
        ((pl.col("iv") > d_expr) & (pl.col("iv") <= d_expr + RENEW_WIN))
        .sum().alias("cnt_in"),
        (pl.lit(1.0 - ALPHA_P).pow(pl.col("k")).sum() * ALPHA_P).alias("tsb_p"),
        (pl.col("w") * pl.col("z")).sum().alias("tsb_z"),
    ).with_columns(seg_expr(pl.col("ord_365")))

    # segment survival tables from pooled intervals
    iv_seg = (
        od.filter(pl.col("iv").is_not_null())
        .select("user_id", "iv")
        .join(g.select("user_id", "seg"), on="user_id", how="left")
    )
    tab = segment_tables(iv_seg)

    d_np = g["d_since"].to_numpy().astype(np.int64)
    seg_np = g["seg"].to_numpy().astype(np.int64)
    p_fb = tab[seg_np, np.clip(d_np, 0, GRID_MAX)]
    n_iv = g["n_iv"].to_numpy().astype(np.int64)
    cnt_gt = g["cnt_gt"].to_numpy().astype(np.float64)
    cnt_in = g["cnt_in"].to_numpy().astype(np.float64)
    personal_ok = (n_iv >= 3) & (cnt_gt >= 1)
    p_renew = np.where(personal_ok, cnt_in / np.maximum(cnt_gt, 1.0), p_fb)

    scored = g.with_columns(
        pl.Series("p_renewal_30", p_renew),
    ).with_columns(
        (pl.col("d_since") / pl.col("adi")).clip(upper_bound=10.0)
        .alias("overdue_ratio"),
        (pl.col("d_since") / pl.col("med_iv")).clip(upper_bound=10.0)
        .alias("overdue_ratio_med"),
        (pl.col("std_iv") / pl.col("adi")).alias("cv_intervals"),
        (pl.col("tsb_p") * pl.col("tsb_z").exp().sub(1.0) * 30.0).log1p()
        .alias("tsb_forecast30"),
    ).select(["user_id"] + FEATS)

    active = first_act.filter(pl.col("first_event") <= anchor).select(
        "user_id", pl.lit(True).alias("has_act"))
    out = uni.join(scored, on="user_id", how="left").join(
        active, on="user_id", how="left")
    # activity but no order-days: TSB p stays at its init 0 -> forecast 0
    out = out.with_columns(
        pl.when(pl.col("tsb_p").is_null() & pl.col("has_act"))
        .then(0.0).otherwise(pl.col("tsb_p")).alias("tsb_p"),
        pl.when(pl.col("tsb_forecast30").is_null() & pl.col("has_act"))
        .then(0.0).otherwise(pl.col("tsb_forecast30")).alias("tsb_forecast30"),
    ).drop("has_act")
    out = out.with_columns([pl.col(c).fill_nan(None).cast(pl.Float32) for c in FEATS])
    assert out.height == uni.height, f"universe join changed height at {anchor}"
    out.write_parquet(out_p)
    print(f"  v6 {anchor}: {out.shape} ord_users={g.height} "
          f"in {time.time()-t0:.1f}s", flush=True)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--anchors", type=str, default="")
    ap.add_argument("--only", type=str, default="")
    ns = ap.parse_args()
    if ns.anchors:
        anchors = [date.fromisoformat(x) for x in ns.anchors.split(",")]
    else:
        anchors = [TEST_ANCHOR, VAL_ANCHOR] + available_train_anchors()[-14:]
    if ns.only:
        anchors = [a for a in anchors if a == date.fromisoformat(ns.only)]
    uni = user_universe()
    lf = pl.scan_parquet(TRAIN_PARQUET)
    t0 = time.time()
    ord_full = (
        lf.filter(pl.col("to_ord") > 0)
        .select("user_id", "event_date", "gmv")
        .collect(engine="streaming")
        .sort(["user_id", "event_date"])
    )
    first_act = (
        lf.group_by("user_id").agg(pl.col("event_date").min().alias("first_event"))
        .collect(engine="streaming")
    )
    print(f"loaded order-days {ord_full.shape}, first_act {first_act.shape} "
          f"in {time.time()-t0:.1f}s", flush=True)
    for a in sorted(set(anchors)):
        if (FEATURES_DIR / f"anchor={a.isoformat()}.v6.parquet").exists():
            print(f"  v6 {a}: exists, skip", flush=True)
            continue
        build(a, uni, ord_full, first_act)
    print("V6 DONE", flush=True)


if __name__ == "__main__":
    main()

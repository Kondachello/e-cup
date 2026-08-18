"""Seasonality EDA - step 1: heavy aggregations from train.parquet.

Outputs (small files):
  work/data/seas_user_windows_2025.parquet - per-user gmv sums for the two 2025 analog windows
"""
import os

os.environ.setdefault("POLARS_MAX_THREADS", "3")
os.environ.setdefault("OMP_NUM_THREADS", "3")

from pathlib import Path

import polars as pl

ROOT = Path("/Users/alexanderkondakov/ozon-cup")
OUT = ROOT / "work" / "data"
OUT.mkdir(parents=True, exist_ok=True)

lf = pl.scan_parquet(ROOT / "train.parquet")
d = pl.col("event_date")

# ---------------------------------------------------------------- daily series
daily = (
    lf.group_by("event_date")
    .agg(
        pl.len().alias("rows"),
        pl.col("user_id").n_unique().alias("active_users"),
        pl.col("gmv").sum().alias("gmv"),
        pl.col("to_ord").sum().alias("orders"),
        pl.col("to_cart").sum().alias("carts"),
        pl.col("searches").sum().alias("searches"),
        pl.col("gmv_search").sum().alias("gmv_search"),
        pl.col("gmv_cat").sum().alias("gmv_cat"),
        (pl.col("gmv") > 0).sum().alias("buyers"),
    )
    .sort("event_date")
    .collect(engine="streaming")
)
print("daily:", daily.shape)
print(daily.head(3))
print(daily.tail(3))
print("rows==active_users everywhere:", (daily["rows"] == daily["active_users"]).all())

# ------------------------------------------- per-user 2025 analog window sums
# Analog of the competition setup shifted one year back:
#   anchor-like activity window : 2025-01-01..2025-01-14  (cohort definition)
#   W1 (val-like target window) : 2025-01-15..2025-02-13
#   mdl_onyx (test-like target window): 2025-02-14..2025-03-15  (incl. Feb 23 + Mar 8)
w0s, w0e = pl.date(2025, 1, 1), pl.date(2025, 1, 14)
w1s, w1e = pl.date(2025, 1, 15), pl.date(2025, 2, 13)
w2s, w2e = pl.date(2025, 2, 14), pl.date(2025, 3, 15)

user = (
    lf.filter((d >= w0s) & (d <= w2e))
    .group_by("user_id")
    .agg(
        ((d >= w0s) & (d <= w0e)).any().alias("act_jan14"),
        (d <= pl.date(2025, 1, 31)).any().alias("act_jan"),
        ((d >= w1s) & (d <= w1e)).any().alias("act_w1"),
        ((d >= w2s) & (d <= w2e)).any().alias("act_w2"),
        pl.col("gmv").filter((d >= w1s) & (d <= w1e)).sum().alias("gmv_w1"),
        pl.col("gmv").filter((d >= w2s) & (d <= w2e)).sum().alias("gmv_w2"),
    )
    .collect(engine="streaming")
)
user.write_parquet(OUT / "seas_user_windows_2025.parquet")
print("user windows:", user.shape)
print(user.select(pl.col("act_jan14").sum(), pl.col("act_jan").sum(), pl.col("act_w1").sum(), pl.col("act_w2").sum()))

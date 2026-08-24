#!/usr/bin/env python
"""eda3 gift pass 5: val-anchor (2026-01-15) pre-anchor windows, no leakage into val target."""
import polars as pl
from datetime import date

TRAIN = "/Users/alexanderkondakov/ozon-cup/train.parquet"
OUT = "/Users/alexanderkondakov/ozon-cup/work/reports/eda3_gift_valanchor.parquet"

WINS = {
    "vAh": (date(2025, 12, 2), date(2026, 1, 14)),    # 44d history before val anchor
    "vAl": (date(2026, 1, 8),  date(2026, 1, 14)),    # last 7d before val anchor
    # true test-anchor run-up week (observable; target unobservable) - for descriptive stats only
    "tAl": (date(2026, 2, 7),  date(2026, 2, 13)),
    "tAh": (date(2026, 1, 1),  date(2026, 2, 13)),
}
FIELDS = ["gmv", "to_ord", "to_cart", "search", "cat", "gmv_cat", "gmv_search"]
aggs = []
for nm, (s, e) in WINS.items():
    c = (pl.col("event_date") >= s) & (pl.col("event_date") <= e)
    aggs += [pl.col(f).filter(c).sum().alias(f"{nm}_{f}") for f in FIELDS]
    aggs.append(((pl.col("to_ord") > 0) & c).sum().alias(f"{nm}_dord"))
    aggs.append(c.sum().alias(f"{nm}_dact"))

pl.scan_parquet(TRAIN).group_by("user_id").agg(aggs).sort("user_id") \
  .collect(engine="streaming").write_parquet(OUT)
print("written", OUT)

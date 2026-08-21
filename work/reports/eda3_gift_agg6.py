#!/usr/bin/env python
"""eda3 gift pass 6: matched same-calendar-week YoY pairs (gift week vs neutral week)."""
import polars as pl
from datetime import date

TRAIN = "/Users/alexanderkondakov/ozon-cup/train.parquet"
OUT = "/Users/alexanderkondakov/ozon-cup/work/reports/eda3_gift_yoyweeks.parquet"

WINS = {
    "j25": (date(2025, 1, 8),  date(2025, 1, 14)),    # neutral week, year 1  (pair with vAl)
    "mid": (date(2025, 3, 16), date(2025, 12, 31)),   # level control, disjoint from all 4 weeks
    "f25b": (date(2025, 1, 24), date(2025, 2, 6)),    # 14d immediately before gift week 2025
    "f26b": (date(2026, 1, 24), date(2026, 2, 6)),    # 14d immediately before gift week 2026
}
FIELDS = ["gmv", "to_ord", "to_cart", "search", "cat", "gmv_cat"]
aggs = []
for nm, (s, e) in WINS.items():
    c = (pl.col("event_date") >= s) & (pl.col("event_date") <= e)
    aggs += [pl.col(f).filter(c).sum().alias(f"{nm}_{f}") for f in FIELDS]
    aggs.append(((pl.col("to_ord") > 0) & c).sum().alias(f"{nm}_dord"))
    aggs.append(c.sum().alias(f"{nm}_dact"))

pl.scan_parquet(TRAIN).group_by("user_id").agg(aggs).sort("user_id") \
  .collect(engine="streaming").write_parquet(OUT)
print("written", OUT)

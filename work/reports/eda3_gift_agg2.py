#!/usr/bin/env python
"""eda3 gift carriers pass 2: mirror windows + march target + clean post-base."""
import polars as pl
from datetime import date

TRAIN = "/Users/alexanderkondakov/ozon-cup/train.parquet"
OUT = "/Users/alexanderkondakov/ozon-cup/work/reports/eda3_gift_user_windows2.parquet"

WINS = {
    "Xp":   (date(2025, 1, 1),  date(2025, 1, 22)),   # mirror: year-ago base half
    "Yp":   (date(2025, 1, 23), date(2025, 2, 13)),   # mirror: year-ago target half
    "X":    (date(2026, 1, 1),  date(2026, 1, 22)),   # mirror: this-year base half
    "Y":    (date(2026, 1, 23), date(2026, 2, 13)),   # mirror: this-year target half (observable)
    "m8w":  (date(2025, 3, 1),  date(2025, 3, 8)),    # march-8 shopping window incl the day
    "postb": (date(2025, 3, 16), date(2025, 4, 12)),  # 28d post-base, disjoint from all gift runs
    "febt": (date(2025, 2, 14), date(2025, 2, 28)),   # feb tail between vrun and mrun
    "y2X":  (date(2025, 1, 1),  date(2025, 12, 31)),  # history before X (level control for mirror)
}
FIELDS = ["gmv", "to_ord", "to_cart", "search", "cat", "gmv_cat", "gmv_search"]

aggs = []
for nm, (s, e) in WINS.items():
    c = (pl.col("event_date") >= s) & (pl.col("event_date") <= e)
    aggs += [pl.col(f).filter(c).sum().alias(f"{nm}_{f}") for f in FIELDS]
    aggs.append(((pl.col("to_ord") > 0) & c).sum().alias(f"{nm}_dord"))
    aggs.append(c.sum().alias(f"{nm}_dact"))

lf = pl.scan_parquet(TRAIN)
lf.group_by("user_id").agg(aggs).sort("user_id").collect(engine="streaming").write_parquet(OUT)
print("written", OUT)

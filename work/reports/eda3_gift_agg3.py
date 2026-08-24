#!/usr/bin/env python
"""eda3 gift pass 3: geometric replicas of the (vrun, mrun) pair in neutral seasons."""
import polars as pl
from datetime import date

TRAIN = "/Users/alexanderkondakov/ozon-cup/train.parquet"
OUT = "/Users/alexanderkondakov/ozon-cup/work/reports/eda3_gift_user_windows3.parquet"

# replica of: vrun 07-13.02 | vrunb 10.01-06.02 | mrun 01-07.03 | postb 16.03-12.04
WINS = {
    "jr1":  (date(2025, 6, 6),  date(2025, 6, 12)),
    "jr1b": (date(2025, 5, 9),  date(2025, 6, 5)),
    "jr2":  (date(2025, 6, 28), date(2025, 7, 4)),
    "jrpb": (date(2025, 7, 13), date(2025, 8, 9)),
    "sr1":  (date(2025, 9, 5),  date(2025, 9, 11)),
    "sr1b": (date(2025, 8, 8),  date(2025, 9, 4)),
    "sr2":  (date(2025, 9, 27), date(2025, 10, 3)),
    "srpb": (date(2025, 10, 12), date(2025, 11, 8)),
}
FIELDS = ["gmv", "to_ord", "cat", "gmv_cat"]
aggs = []
for nm, (s, e) in WINS.items():
    c = (pl.col("event_date") >= s) & (pl.col("event_date") <= e)
    aggs += [pl.col(f).filter(c).sum().alias(f"{nm}_{f}") for f in FIELDS]
    aggs.append(c.sum().alias(f"{nm}_dact"))

lf = pl.scan_parquet(TRAIN)
lf.group_by("user_id").agg(aggs).sort("user_id").collect(engine="streaming").write_parquet(OUT)
print("written", OUT)

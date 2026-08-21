#!/usr/bin/env python
"""eda3 gift pass 4: five anchors x (44d history, last-7d run-up, 30d target) inside 2025."""
import polars as pl
from datetime import date

TRAIN = "/Users/alexanderkondakov/ozon-cup/train.parquet"
OUT = "/Users/alexanderkondakov/ozon-cup/work/reports/eda3_gift_anchors.parquet"

# anchor: (hist44 start, hist44 end, last7 start, last7 end, tgt start, tgt end)
ANCHORS = {
    "gA": (date(2025, 1, 1), date(2025, 2, 13), date(2025, 2, 7), date(2025, 2, 13),
           date(2025, 2, 14), date(2025, 3, 15)),                       # GIFT: feb14-mar15
    "gB": (date(2025, 11, 1), date(2025, 12, 14), date(2025, 12, 8), date(2025, 12, 14),
           date(2025, 12, 15), date(2026, 1, 13)),                      # GIFT: new-year
    "c1": (date(2025, 4, 23), date(2025, 6, 5), date(2025, 5, 30), date(2025, 6, 5),
           date(2025, 6, 6), date(2025, 7, 5)),                         # neutral
    "c2": (date(2025, 7, 23), date(2025, 9, 4), date(2025, 8, 29), date(2025, 9, 4),
           date(2025, 9, 5), date(2025, 10, 4)),                        # neutral
    "c3": (date(2025, 9, 22), date(2025, 11, 4), date(2025, 10, 29), date(2025, 11, 4),
           date(2025, 11, 5), date(2025, 12, 4)),                       # promo (11.11)
}
FIELDS = ["gmv", "to_ord", "to_cart", "search", "cat", "gmv_cat", "gmv_search"]

aggs = []
for a, (h0, h1, l0, l1, t0, t1) in ANCHORS.items():
    for tag, (s, e) in {"h": (h0, h1), "l": (l0, l1), "t": (t0, t1)}.items():
        c = (pl.col("event_date") >= s) & (pl.col("event_date") <= e)
        aggs += [pl.col(f).filter(c).sum().alias(f"{a}{tag}_{f}") for f in FIELDS]
        aggs.append(((pl.col("to_ord") > 0) & c).sum().alias(f"{a}{tag}_dord"))
        aggs.append(c.sum().alias(f"{a}{tag}_dact"))

lf = pl.scan_parquet(TRAIN)
lf.group_by("user_id").agg(aggs).sort("user_id").collect(engine="streaming").write_parquet(OUT)
print("written", OUT)

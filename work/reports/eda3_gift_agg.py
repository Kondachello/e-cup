#!/usr/bin/env python
"""eda3 gift carriers: per-user window aggregates from train.parquet (streaming)."""
import polars as pl
from datetime import date

TRAIN = "/Users/alexanderkondakov/ozon-cup/train.parquet"
OUT = "/Users/alexanderkondakov/ozon-cup/work/reports/eda3_gift_user_windows.parquet"

# (name, start, end) inclusive
RUNS = {
    "vrun":  (date(2025, 2, 7),  date(2025, 2, 13)),   # Valentine run-up
    "f23":   (date(2025, 2, 16), date(2025, 2, 22)),   # Feb-23 run-up
    "mrun":  (date(2025, 3, 1),  date(2025, 3, 7)),    # March-8 run-up
    "jun":   (date(2025, 6, 2),  date(2025, 6, 8)),    # control
    "oct":   (date(2025, 10, 6), date(2025, 10, 12)),  # control
    "nov11": (date(2025, 11, 5), date(2025, 11, 11)),  # sale run
    "dec7":  (date(2025, 12, 22), date(2025, 12, 28)), # NY gift run
}
BIG = {
    "gift":   (date(2025, 2, 14), date(2025, 3, 8)),   # lens window
    "p2win":  (date(2025, 2, 14), date(2025, 3, 15)),  # mdl_gabbro window
    "bjan":   (date(2025, 1, 8),  date(2025, 2, 6)),   # 30d pre base
    "bspr":   (date(2025, 3, 16), date(2025, 4, 14)),  # 30d post base
    "year":   (date(2025, 1, 1),  date(2026, 1, 14)),  # full pre-val history
    "vrun26": (date(2026, 2, 7),  date(2026, 2, 13)),  # Valentine run-up THIS year (val tail)
    "valw":   (date(2026, 1, 15), date(2026, 2, 13)),  # val window
}
FIELDS = ["gmv", "to_ord", "to_cart", "search", "cat", "gmv_cat", "gmv_search"]

def win_aggs(name, s, e, fields=FIELDS, days_ord=True):
    c = (pl.col("event_date") >= s) & (pl.col("event_date") <= e)
    aggs = [pl.col(f).filter(c).sum().alias(f"{name}_{f}") for f in fields]
    if days_ord:
        aggs.append(((pl.col("to_ord") > 0) & c).sum().alias(f"{name}_dord"))
        aggs.append(c.sum().alias(f"{name}_dact"))
    return aggs

aggs = []
for nm, (s, e) in {**RUNS, **BIG}.items():
    aggs += win_aggs(nm, s, e)
# local 28d bases for each run
from datetime import timedelta
for nm, (s, e) in RUNS.items():
    bs, be = s - timedelta(days=28), s - timedelta(days=1)
    aggs += win_aggs(nm + "b", bs, be)

lf = pl.scan_parquet(TRAIN)
df = lf.group_by("user_id").agg(aggs).sort("user_id")
df.collect(engine="streaming").write_parquet(OUT)
print("written", OUT)

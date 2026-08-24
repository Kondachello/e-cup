import os
os.environ.setdefault("POLARS_MAX_THREADS", "2")
import polars as pl
from datetime import date

ANCHOR = date(2026, 1, 14)
OUT = "/private/tmp/claude-501/-Users-alexanderkondakov-ozon-cup/b994bee8-2354-4524-9a2b-530595cd5feb/scratchpad/user_axes.parquet"

lf = pl.scan_parquet("/Users/alexanderkondakov/ozon-cup/train.parquet").filter(
    pl.col("event_date") <= ANCHOR
)

d = (pl.lit(ANCHOR) - pl.col("event_date")).dt.total_days()  # 0 = anchor day
w90 = d < 90
w28 = d < 28
w365 = d < 365
ordday = pl.col("to_ord") > 0

agg = lf.group_by("user_id").agg(
    # activity
    act90=w90.sum(),
    act28=w28.sum(),
    act365=w365.sum(),
    n_rows=pl.len(),
    # orders
    ord_days90=(w90 & ordday).sum(),
    ord_days365=(w365 & ordday).sum(),
    ord_cnt90=pl.col("to_ord").filter(w90).sum(),
    ord_cnt365=pl.col("to_ord").filter(w365).sum(),
    # gmv
    gmv90=pl.col("gmv").filter(w90).sum(),
    gmv365=pl.col("gmv").filter(w365).sum(),
    gmv_search365=pl.col("gmv_search").filter(w365).sum(),
    gmv_cat365=pl.col("gmv_cat").filter(w365).sum(),
    # recency
    rec_any=d.min(),
    rec_ord=d.filter(ordday).min(),
    first_seen=d.max(),
    # carts / search
    cart90=pl.col("to_cart").filter(w90).sum(),
    cart365=pl.col("to_cart").filter(w365).sum(),
    search90=pl.col("search").filter(w90).sum(),
    search_days90=(w90 & (pl.col("search") > 0)).sum(),
    cat90=pl.col("cat").filter(w90).sum(),
    # burstiness: weekly active-day counts over last 26 weeks
    # week index 0..25
    # collect activity per week via sum of indicator grouped later -- do via list
    # cheap proxy: std of day-index within 90d window
    d_mean90=d.filter(w90).mean(),
    d_std90=d.filter(w90).std(),
    # dow concentration of order days over 365
    # weekday of event_date
    # top share computed post-hoc from 7 counts
    **{
        f"dow{k}": (w365 & ordday & (pl.col("event_date").dt.weekday() == k + 1)).sum()
        for k in range(7)
    },
)

df = agg.collect(engine="streaming")
print(df.shape)
df.write_parquet(OUT)
print("saved", OUT)

"""eda3: линза «фаза покупательского цикла».
Шаг 1: пер-юзерные статистики заказных дней до якоря 2026-01-14 (день 378).
Выход: scratchpad/eda3_user_cycle.parquet
"""
import datetime as dt
import polars as pl

ANCHOR = dt.date(2026, 1, 14)
SCRATCH = "/private/tmp/claude-501/-Users-alexanderkondakov-ozon-cup/b994bee8-2354-4524-9a2b-530595cd5feb/scratchpad"

lf = (
    pl.scan_parquet("/Users/alexanderkondakov/ozon-cup/train.parquet")
    .filter((pl.col("event_date") <= ANCHOR) & (pl.col("to_ord") > 0))
    .select("user_id", "event_date", "to_ord", "gmv")
)
df = lf.collect(engine="streaming")
print("order-day rows:", df.height)
df = df.sort(["user_id", "event_date"])
df = df.with_columns(gap=pl.col("event_date").diff().over("user_id").dt.total_days())
stats = df.group_by("user_id").agg(
    n_ord_days=pl.len(),
    last_ord=pl.col("event_date").max(),
    first_ord=pl.col("event_date").min(),
    gap_mean=pl.col("gap").mean(),
    gap_std=pl.col("gap").std(),
    gap_median=pl.col("gap").median(),
    gap_last=pl.col("gap").last(),
    ord_sum=pl.col("to_ord").sum(),
    gmv_hist=pl.col("gmv").sum(),
)
stats = stats.with_columns(
    rec=(pl.lit(ANCHOR) - pl.col("last_ord")).dt.total_days(),
    cv=pl.col("gap_std") / pl.col("gap_mean"),
)
stats.write_parquet(f"{SCRATCH}/eda3_user_cycle.parquet")
print(stats.height, "users with >=1 order day")
print(stats.select(pl.col("n_ord_days").ge(5).sum().alias("n>=5"),
                   ((pl.col("n_ord_days") >= 5) & (pl.col("cv") <= 0.5)).sum().alias("n>=5 cv<=.5"),
                   ((pl.col("n_ord_days") >= 5) & (pl.col("cv") <= 0.33)).sum().alias("n>=5 cv<=.33"),
                   ((pl.col("n_ord_days") >= 8) & (pl.col("cv") <= 0.5)).sum().alias("n>=8 cv<=.5")))

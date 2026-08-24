# eda3 lens: activity ladder — monthly (28d blocks) per-user aggregates relative to anchors
import polars as pl
import datetime as dt

TRAIN = "train.parquet"
OUT = "work/reports/eda3_ladder_monthly.parquet"
ANCHORS = {"val": dt.date(2026, 1, 14), "test": dt.date(2026, 2, 13)}
K = 13  # months 0..12, 28-day blocks

frames = []
for name, anc in ANCHORS.items():
    lf = pl.scan_parquet(TRAIN)
    lf = lf.with_columns(((pl.lit(anc) - pl.col("event_date")).dt.total_days()).alias("d"))
    lf = lf.filter((pl.col("d") >= 0) & (pl.col("d") < 28 * K))
    lf = lf.with_columns((pl.col("d") // 28).alias("k").cast(pl.Int8))
    agg = lf.group_by("user_id", "k").agg(
        pl.len().alias("n_days"),
        pl.col("searches").sum().alias("srch"),
        pl.col("to_cart").sum().alias("cart"),
        pl.col("to_ord").sum().alias("ord"),
        pl.col("gmv").sum().alias("gmv"),
    ).with_columns(pl.lit(name).alias("anchor"))
    frames.append(agg)

df = pl.concat(frames).collect(engine="streaming")
df.write_parquet(OUT)
print(df.group_by("anchor").agg(pl.len(), pl.col("user_id").n_unique()))
print("saved", OUT, df.shape)

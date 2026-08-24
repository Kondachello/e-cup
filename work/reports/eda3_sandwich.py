"""eda3 step 3: sandwich-hole rate per calendar day (logging-outage detector)."""
import polars as pl

lf = pl.scan_parquet("/Users/alexanderkondakov/ozon-cup/train.parquet").select(["user_id", "event_date"])
df = (lf.sort(["user_id", "event_date"])
        .with_columns([
            (pl.col("event_date") - pl.col("event_date").shift(1).over("user_id")).dt.total_days().alias("dprev"),
            (pl.col("event_date").shift(-1).over("user_id") - pl.col("event_date")).dt.total_days().alias("dnext"),
        ]))

holes = (df.filter(pl.col("dnext") == 2)
           .select((pl.col("event_date") + pl.duration(days=1)).alias("day"))
           .group_by("day").agg(pl.len().alias("holes")))
bridges = (df.filter((pl.col("dprev") == 1) & (pl.col("dnext") == 1))
             .select(pl.col("event_date").alias("day"))
             .group_by("day").agg(pl.len().alias("bridges")))

out = (holes.join(bridges, on="day", how="full", coalesce=True)
            .fill_null(0)
            .with_columns((pl.col("holes") / (pl.col("holes") + pl.col("bridges"))).alias("hole_rate"))
            .sort("day")
            .collect(engine="streaming"))

import numpy as np
x = out["hole_rate"].to_numpy()
n = len(x)
y = np.log(x + 1e-6)
base = np.array([np.median(y[[t + k * 7 for k in (-4, -3, -2, -1, 1, 2, 3, 4) if 0 <= t + k * 7 < n]]) for t in range(n)])
r = y - base
mad = np.median(np.abs(r - np.median(r)))
z = r / (1.4826 * mad)
out = out.with_columns(pl.Series("z", z))
pl.Config.set_tbl_rows(40)
print("hole_rate mean/min/max:", float(np.mean(x)), float(np.min(x)), float(np.max(x)))
print(out.filter(pl.col("z").abs() >= 4).sort("z", descending=True))
# also gap-size distribution sanity
gaps = (df.filter(pl.col("dnext").is_not_null()).group_by("dnext").agg(pl.len()).sort("dnext").collect(engine="streaming"))
print(gaps.head(15))

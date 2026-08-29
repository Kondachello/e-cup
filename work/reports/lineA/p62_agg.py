"""P62: один polars-проход по train.parquet — агрегаты спящих/браузеров.

Якорь ANCHOR = 2026-02-13 (последний день train), di = (ANCHOR - event_date).days.
"""
import numpy as np, polars as pl
from datetime import date
from pathlib import Path

ROOT = Path("/Users/alexanderkondakov/ozon-cup")
SP = Path("/private/tmp/claude-501/-Users-alexanderkondakov-ozon-cup/0b55ab9f-3777-4ebc-bd91-937895c0e355/scratchpad")
ANCHOR = date(2026, 2, 13)

base = pl.read_csv(ROOT / "submissions/F8_priv.csv", schema_overrides={"user_id": pl.Int64}).sort("user_id")
uid_ref = base["user_id"].to_numpy()
assert len(uid_ref) == 250_000

g = pl.col("gmv") > 0
di = pl.col("di")
act = (pl.col("search") > 0) | (pl.col("cat") > 0)
acts = act | (pl.col("searches") > 0)

lf = (pl.scan_parquet(ROOT / "train.parquet")
      .with_columns(di=(pl.lit(ANCHOR) - pl.col("event_date")).dt.total_days().cast(pl.Int32))
      .group_by("user_id")
      .agg(
          nbuyd=di.filter(g).n_unique(),
          last_di=di.filter(g).min(),
          act30=di.filter(act & (di <= 29)).n_unique(),
          act30s=di.filter(acts & (di <= 29)).n_unique(),
          act90=di.filter(act & (di <= 89)).n_unique(),
          browse30=(pl.col("search") + pl.col("cat")).filter(di <= 29).sum(),
          searches30=pl.col("searches").filter(di <= 29).sum(),
          browse90=(pl.col("search") + pl.col("cat")).filter(di <= 89).sum(),
          browse7=(pl.col("search") + pl.col("cat")).filter(di <= 6).sum(),
          cart30=pl.col("to_cart").filter(di <= 29).sum(),
          gmv_sum=pl.col("gmv").sum(),
          gmv_hist=pl.col("gmv").filter(g).sum(),
          nrows=pl.len(),
      ))
df = lf.collect(engine="streaming")
print("train users:", df.height)

t = (pl.DataFrame({"user_id": uid_ref}).join(df, on="user_id", how="left").sort("user_id")
     .with_columns(pl.col("nbuyd","act30","act30s","act90","browse30","searches30",
                          "browse90","browse7","cart30","nrows").fill_null(0),
                   pl.col("gmv_sum","gmv_hist").fill_null(0.0)))
t.write_parquet(SP / "p62_agg.parquet")
print("saved", SP / "p62_agg.parquet", t.shape)
rec = t["last_di"].to_numpy().astype(np.float64)
rec = np.where(np.isnan(rec), 1e9, rec)
print("never-buyers:", float((t['nbuyd'].to_numpy()==0).mean()))
for lo, hi, nm in [(0,14,'0-14'),(15,90,'15-90'),(91,365,'91-365'),(91,1e8,'91+'),(366,1e8,'366+')]:
    print(f"  rec {nm:8s} share={float(((rec>=lo)&(rec<=hi)).mean()):.5f}")
print("max finite rec:", rec[rec<1e8].max())

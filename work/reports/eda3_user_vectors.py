"""eda3 step 4: per-user exposure vectors for logging-anomaly candidates, val + test anchors."""
import datetime as dt
import polars as pl

STEP = dt.date(2025, 4, 1)          # catalog logging regime change
VAL_A = dt.date(2026, 1, 14)
TEST_A = dt.date(2026, 2, 13)

lf = pl.scan_parquet("/Users/alexanderkondakov/ozon-cup/train.parquet")

zero_all = (pl.col("search") + pl.col("cat") + pl.col("to_cart") + pl.col("to_ord")
            + pl.col("has_search_to_cart") + pl.col("has_search_to_ord")
            + pl.col("has_cat_to_cart") + pl.col("has_cat_to_ord")
            + pl.col("search_to_cart") + pl.col("search_to_ord")
            + pl.col("cat_to_cart") + pl.col("cat_to_ord") + pl.col("searches") == 0) & (pl.col("gmv") == 0)

def vectors(anchor: dt.date, tag: str):
    w0 = anchor - dt.timedelta(days=364)
    in_w = pl.col("event_date").is_between(w0, anchor)
    pre = pl.col("event_date") < STEP
    return (lf.filter(in_w)
              .group_by("user_id")
              .agg([
                  (pl.col("cat") > 0).filter(pre).sum().alias(f"cat_pre_{tag}"),
                  (pl.col("cat") > 0).filter(~pre).sum().alias(f"cat_post_{tag}"),
                  (pl.col("cat_to_ord") > 0).filter(pre).sum().alias(f"c2o_pre_{tag}"),
                  (pl.col("cat_to_ord") > 0).filter(~pre).sum().alias(f"c2o_post_{tag}"),
                  pl.col("gmv_cat").filter(pre).sum().alias(f"gmvcat_pre_{tag}"),
                  pl.col("gmv_cat").filter(~pre).sum().alias(f"gmvcat_post_{tag}"),
                  zero_all.sum().alias(f"ghost_{tag}"),
                  pl.len().alias(f"act_{tag}"),
                  pre.sum().alias(f"act_pre_{tag}"),
              ]))

v = vectors(VAL_A, "v").collect(engine="streaming")
t = vectors(TEST_A, "t").collect(engine="streaming")
out = v.join(t, on="user_id", how="full", coalesce=True).fill_null(0)
out.write_parquet("/Users/alexanderkondakov/ozon-cup/work/reports/eda3_user_vectors.parquet")
print(out.shape)
print(out.select([pl.col(c).mean() for c in out.columns if c != "user_id"]).transpose(include_header=True))

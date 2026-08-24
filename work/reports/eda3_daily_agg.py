"""eda3 lens=logging anomalies. Step 1: full daily calendar audit of every counter."""
import polars as pl

lf = pl.scan_parquet("/Users/alexanderkondakov/ozon-cup/train.parquet")

cnt_cols = ["search", "cat", "search_to_cart", "search_to_ord", "cat_to_cart",
            "cat_to_ord", "to_cart", "to_ord", "searches"]
flag_cols = ["has_search_to_cart", "has_search_to_ord", "has_cat_to_cart", "has_cat_to_ord"]
gmv_cols = ["gmv", "gmv_search", "gmv_cat"]

aggs = [pl.len().alias("dau")]
for c in cnt_cols + gmv_cols:
    aggs.append(pl.col(c).sum().alias(f"{c}_sum"))
    aggs.append((pl.col(c) > 0).sum().alias(f"{c}_nnz"))
for c in flag_cols:
    aggs.append(pl.col(c).sum().alias(f"{c}_sum"))
    aggs.append(pl.col(c).max().alias(f"{c}_max"))
    aggs.append(pl.col(c).min().alias(f"{c}_min"))

# consistency violations per day
aggs += [
    ((pl.col("gmv") > 0) & (pl.col("to_ord") == 0)).sum().alias("viol_gmv_noord"),
    ((pl.col("to_ord") > 0) & (pl.col("gmv") == 0)).sum().alias("viol_ord_nogmv"),
    (pl.col("has_search_to_cart") != (pl.col("search_to_cart") > 0).cast(pl.Int64)).sum().alias("viol_flag_s2c"),
    (pl.col("has_search_to_ord") != (pl.col("search_to_ord") > 0).cast(pl.Int64)).sum().alias("viol_flag_s2o"),
    (pl.col("has_cat_to_cart") != (pl.col("cat_to_cart") > 0).cast(pl.Int64)).sum().alias("viol_flag_c2c"),
    (pl.col("has_cat_to_ord") != (pl.col("cat_to_ord") > 0).cast(pl.Int64)).sum().alias("viol_flag_c2o"),
    ((pl.col("gmv_search") > 0) & (pl.col("search_to_ord") == 0)).sum().alias("viol_gmvs_nos2o"),
    ((pl.col("gmv_cat") > 0) & (pl.col("cat_to_ord") == 0)).sum().alias("viol_gmvc_noc2o"),
    (pl.col("searches") < pl.col("search")).sum().alias("viol_searches_lt_search"),
    ((pl.col("to_ord") > 0) & (pl.col("search_to_ord") + pl.col("cat_to_ord") == 0)).sum().alias("ord_unattrib"),
    ((pl.col("to_cart") > 0) & (pl.col("search_to_cart") + pl.col("cat_to_cart") == 0)).sum().alias("cart_unattrib"),
    (pl.col("to_ord") < pl.col("search_to_ord") + pl.col("cat_to_ord")).sum().alias("viol_ord_lt_attrib"),
    (pl.col("to_cart") < pl.col("search_to_cart") + pl.col("cat_to_cart")).sum().alias("viol_cart_lt_attrib"),
    # a row with literally zero everything (ghost row)
    (pl.sum_horizontal([pl.col(c) for c in cnt_cols + flag_cols]) + pl.col("gmv") == 0).sum().alias("ghost_rows"),
]

daily = lf.group_by("event_date").agg(aggs).sort("event_date").collect(engine="streaming")
print("days:", daily.height, "range:", daily["event_date"].min(), daily["event_date"].max())

# missing calendar days?
import datetime as dt
d0, d1 = daily["event_date"].min(), daily["event_date"].max()
full = set((d0 + dt.timedelta(days=i)) for i in range((d1 - d0).days + 1))
have = set(daily["event_date"].to_list())
print("missing calendar days:", sorted(full - have))

# total violations
viol = [c for c in daily.columns if c.startswith("viol_")] + ["ghost_rows", "ord_unattrib", "cart_unattrib"]
print(daily.select([pl.col(c).sum() for c in viol]).transpose(include_header=True))

"""prep_data: train.parquet -> act/buys/users_order/user_global (+ sorted train)."""
import polars as pl
from paths import TRAIN_PARQUET, wp

def main():
    lf = pl.scan_parquet(str(TRAIN_PARQUET))
    df = lf.with_columns(
        pl.col("user_id").cast(pl.UInt32),
        *[pl.col(c).cast(pl.UInt8) for c in ["search","cat","has_search_to_cart","has_search_to_ord","has_cat_to_cart","has_cat_to_ord"]],
        *[pl.col(c).cast(pl.UInt32) for c in ["search_to_cart","search_to_ord","cat_to_cart","cat_to_ord","to_cart","to_ord","searches"]],
        pl.col("gmv_search").cast(pl.Float64), pl.col("gmv_cat").cast(pl.Float64),
    ).sort(["user_id","event_date"]).collect(engine="streaming")
    act = df.select("user_id","event_date","search","cat","searches","to_cart","to_ord","gmv","search_to_cart","cat_to_cart")
    act.write_parquet(wp("act.parquet"))
    buys = df.filter(pl.col("to_ord") > 0).select("user_id","event_date","gmv","to_ord","gmv_search","gmv_cat","search_to_ord","cat_to_ord","searches","to_cart")
    buys.write_parquet(wp("buys.parquet"))
    users = df["user_id"].unique().sort()
    users.to_frame().write_parquet(wp("users_order.parquet"))
    g = act.group_by("user_id").agg(
        pl.len().alias("n_days_active"),
        pl.col("event_date").min().alias("first_seen"),
        pl.col("event_date").max().alias("last_seen"),
        (pl.col("to_ord")>0).sum().alias("n_buy_days"),
        pl.col("gmv").sum().alias("gmv_total"),
    ).sort("user_id")
    g.write_parquet(wp("user_global.parquet"))
    print("prep done:", act.height, "act rows,", buys.height, "buy rows,", len(users), "users")

if __name__ == "__main__":
    main()

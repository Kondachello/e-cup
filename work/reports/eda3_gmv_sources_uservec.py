"""eda3 lens 'soglasovannost istochnikov GMV': build per-user beyond-shell channel vectors
on history <= 2026-01-14 (pre-val), plus H1/H2 mix stability numbers."""
import numpy as np
import polars as pl

CUT = pl.date(2026, 1, 14)
H_SPLIT = pl.date(2025, 7, 15)
lf = pl.scan_parquet("/Users/alexanderkondakov/ozon-cup/train.parquet").filter(pl.col("event_date") <= CUT)

gd = pl.col("gmv") > 0
x = (pl.col("gmv_cat") / pl.col("gmv"))  # daily cat share, on gmv days
t = (pl.col("event_date") - pl.date(2025, 1, 1)).dt.total_days().cast(pl.Float64)

agg = lf.group_by("user_id").agg([
    gd.sum().alias("n_gmv_days"),
    (pl.col("gmv_search") > 0).sum().alias("n_gs_days"),
    (pl.col("gmv_cat") > 0).sum().alias("n_gc_days"),
    ((pl.col("gmv_search") > 0) & (pl.col("gmv_cat") > 0)).sum().alias("n_mixed"),
    ((pl.col("gmv_cat") > 0) & (pl.col("cat") == 0)).sum().alias("n_gc_nocat"),
    ((pl.col("gmv_search") > 0) & (pl.col("search") == 0)).sum().alias("n_gs_nosearch"),
    pl.col("gmv").sum().alias("sum_gmv"),
    pl.col("gmv_cat").sum().alias("sum_gc"),
    x.filter(gd).sum().alias("sx"),
    (x ** 2).filter(gd).sum().alias("sx2"),
    t.filter(gd).sum().alias("st"),
    (t ** 2).filter(gd).sum().alias("st2"),
    (t * x).filter(gd).sum().alias("stx"),
    # H1/H2 rub mix
    pl.col("gmv").filter(pl.col("event_date") < H_SPLIT).sum().alias("gmv_h1"),
    pl.col("gmv_cat").filter(pl.col("event_date") < H_SPLIT).sum().alias("gc_h1"),
    pl.col("gmv").filter(pl.col("event_date") >= H_SPLIT).sum().alias("gmv_h2"),
    pl.col("gmv_cat").filter(pl.col("event_date") >= H_SPLIT).sum().alias("gc_h2"),
    # last 60d mix (fine recency beyond 30/90 shell windows? control-ish)
    pl.col("gmv").filter(pl.col("event_date") > pl.date(2025, 11, 15)).sum().alias("gmv_60"),
    pl.col("gmv_cat").filter(pl.col("event_date") > pl.date(2025, 11, 15)).sum().alias("gc_60"),
]).collect(engine="streaming")
print("users:", agg.height)

# switching of dominant channel across ordered gmv-days
ev = (pl.scan_parquet("/Users/alexanderkondakov/ozon-cup/train.parquet")
      .filter((pl.col("event_date") <= CUT) & gd)
      .select(["user_id", "event_date",
               (pl.col("gmv_cat") > pl.col("gmv_search")).cast(pl.Int8).alias("dom_cat")])
      .collect(engine="streaming")
      .sort(["user_id", "event_date"]))
sw = (ev.group_by("user_id", maintain_order=True)
      .agg([(pl.col("dom_cat").diff().abs().sum()).alias("n_switch"),
            pl.len().alias("n_ev")]))
agg = agg.join(sw.select(["user_id", "n_switch"]), on="user_id", how="left")

n = pl.col("n_gmv_days").cast(pl.Float64)
agg = agg.with_columns([
    (pl.col("sx") / n.clip(1)).alias("mean_daily_catshare"),
    ((pl.col("sx2") / n.clip(1)) - (pl.col("sx") / n.clip(1)) ** 2).clip(0).sqrt().alias("std_daily_catshare"),
    ((pl.col("stx") / n.clip(1) - (pl.col("st") / n.clip(1)) * (pl.col("sx") / n.clip(1)))
     / ((pl.col("st2") / n.clip(1) - (pl.col("st") / n.clip(1)) ** 2).clip(1e-9))).alias("catshare_slope"),
    (pl.col("sum_gc") / pl.col("sum_gmv").clip(1e-9)).alias("cat_rub_share"),
    (pl.col("n_gc_nocat") / pl.col("n_gc_days").clip(1)).alias("gc_nocat_share"),
    (pl.col("n_mixed") / n.clip(1)).alias("mixed_share"),
    (pl.col("n_switch").fill_null(0) / (n - 1).clip(1)).alias("switch_rate"),
    (pl.col("gc_h1") / pl.col("gmv_h1").clip(1e-9)).alias("catshare_h1"),
    (pl.col("gc_h2") / pl.col("gmv_h2").clip(1e-9)).alias("catshare_h2"),
    (pl.col("gc_60") / pl.col("gmv_60").clip(1e-9)).alias("catshare_60"),
])
agg.write_parquet("/Users/alexanderkondakov/ozon-cup/work/reports/eda3_gmv_sources_uservec.parquet")

# mix stability H1 vs H2 (users with gmv in both halves)
b = agg.filter((pl.col("gmv_h1") > 0) & (pl.col("gmv_h2") > 0))
h1 = b["catshare_h1"].to_numpy(); h2 = b["catshare_h2"].to_numpy()
w = np.log1p(b["sum_gmv"].to_numpy())
print(f"both-halves buyers: {b.height}")
print(f"corr(catshare_h1, catshare_h2) = {np.corrcoef(h1, h2)[0,1]:.4f}")
cw = np.cov(h1, h2, aweights=w); print(f"weighted corr = {cw[0,1]/np.sqrt(cw[0,0]*cw[1,1]):.4f}")
print(f"mean |dshare| = {np.abs(h2-h1).mean():.4f}; share |d|>0.5: {(np.abs(h2-h1)>0.5).mean():.4f}")
print(f"P(cat-dominant h2 | cat-dominant h1) = {(h2[h1>0.5]>0.5).mean():.4f}  (base P h2 dom = {(h2>0.5).mean():.4f})")
print(f"P(cat-dom h2 | search-dom h1) = {(h2[h1<=0.5]>0.5).mean():.4f}")

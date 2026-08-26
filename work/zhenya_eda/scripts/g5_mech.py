"""G5. МЕХАНИЗМ пустого визита. Лёгкий прогон на выборке юзеров (1 поток)."""
import polars as pl, numpy as np
df = pl.scan_parquet("train.parquet")
ids = pl.read_parquet("train.parquet", columns=["user_id"])["user_id"].unique().sort()
samp = ids.gather(np.linspace(0, len(ids)-1, 20000).astype(int))
d = df.filter(pl.col("user_id").is_in(samp.implode())).collect()
A = ((pl.col("searches")==0)&(pl.col("cat")==0)&(pl.col("to_cart")==0)&(pl.col("to_ord")==0))
d = d.with_columns([A.alias("emp"), (pl.col("to_ord")>0).alias("ord")]).sort(["user_id","event_date"])

# дни с последнего заказа для КАЖДОГО дня
d = d.with_columns(
    pl.when(pl.col("ord")).then(pl.col("event_date")).otherwise(None)
      .forward_fill().over("user_id").alias("last_ord"))
d = d.with_columns((pl.col("event_date")-pl.col("last_ord")).dt.total_days().alias("since_ord"))

print("=== доля ПУСТЫХ дней в зависимости от давности последнего заказа ===")
print(" дней с заказа | всего дней | доля пустых | доля дней с поиском")
for lo,hi,lab in [(0,0,"0 (день заказа)"),(1,1,"1"),(2,3,"2-3"),(4,7,"4-7"),(8,14,"8-14"),
                  (15,30,"15-30"),(31,90,"31-90"),(91,10**9,"90+")]:
    s = d.filter((pl.col("since_ord")>=lo)&(pl.col("since_ord")<=hi))
    if s.height:
        print(f"  {lab:>14} | {s.height:>10,} |    {100*s['emp'].mean():5.2f}%   |     {100*(s['searches']>0).mean():5.2f}%")
never = d.filter(pl.col("since_ord").is_null())
print(f"  {'заказов не было':>14} | {never.height:>10,} |    {100*never['emp'].mean():5.2f}%   |     {100*(never['searches']>0).mean():5.2f}%")

print("\n=== пустой день ПЕРЕД заказом? (за сколько дней до ближайшего будущего заказа) ===")
d2 = d.sort(["user_id","event_date"], descending=[False,True]).with_columns(
    pl.when(pl.col("ord")).then(pl.col("event_date")).otherwise(None)
      .forward_fill().over("user_id").alias("next_ord")).sort(["user_id","event_date"])
d2 = d2.with_columns((pl.col("next_ord")-pl.col("event_date")).dt.total_days().alias("to_ord_d"))
print(" дней ДО заказа | всего дней | доля пустых")
for lo,hi,lab in [(1,1,"1"),(2,3,"2-3"),(4,7,"4-7"),(8,14,"8-14"),(15,30,"15-30"),(31,10**9,"30+")]:
    s = d2.filter((pl.col("to_ord_d")>=lo)&(pl.col("to_ord_d")<=hi))
    if s.height: print(f"  {lab:>14} | {s.height:>10,} |    {100*s['emp'].mean():5.2f}%")

print("\n=== серии подряд идущих пустых дней ===")
e = d.filter(pl.col("emp")).select(["user_id","event_date"]).sort(["user_id","event_date"])
e = e.with_columns(pl.col("event_date").diff().dt.total_days().over("user_id").alias("g"))
e = e.with_columns(((pl.col("g")>1)|pl.col("g").is_null()).cum_sum().over("user_id").alias("run"))
rl = e.group_by(["user_id","run"]).len()["len"]
print(f"  длина серии: mean={rl.mean():.2f} p50={rl.median():.0f} p95={rl.quantile(.95):.0f} max={rl.max()}")
print(f"  доля одиночных пустых дней: {100*(rl==1).mean():.1f}%")

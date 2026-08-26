"""T1. Проверка трёх эмпирических утверждений из списка Сани."""
import polars as pl, numpy as np
from datetime import date, timedelta
df = pl.read_parquet("train.parquet")

print("=== 1. СЛОМ ЛОГИРОВАНИЯ КАТАЛОГА 01.04.2025 ===")
m = df.with_columns(pl.col("event_date").dt.strftime("%Y-%m").alias("mo")).group_by("mo").agg([
    (pl.col("cat")>0).mean().alias("cat_nnz"),
    (pl.col("cat_to_ord")>0).mean().alias("c2o_nnz"),
    (pl.col("search")>0).mean().alias("srch_nnz"),
    pl.len().alias("n")]).sort("mo")
print(m.head(8)); print("...")
pre = df.filter(pl.col("event_date").is_between(date(2025,1,1),date(2025,3,31)))
post= df.filter(pl.col("event_date").is_between(date(2025,4,1),date(2025,6,30)))
for c in ("cat","cat_to_ord","cat_to_cart","search","to_ord"):
    a,b=(pre[c]>0).mean(),(post[c]>0).mean()
    print(f"  {c:12s} до 01.04: {a:.4f}  после: {b:.4f}  изменение {100*(b/a-1):+.1f}%")

print("\n=== 2. СЕГМЕНТ gmv_cat>0 ПРИ cat=0 ===")
A=date(2026,1,14)
h=df.filter(pl.col("event_date").is_between(A-timedelta(days=364),A))
g=h.group_by("user_id").agg([
    ((pl.col("gmv_cat")>0)&(pl.col("cat")==0)).sum().alias("gcn"),
    (pl.col("cat")>0).sum().alias("cd")])
uid=np.sort(df["user_id"].unique().to_numpy())
g=pl.DataFrame({"user_id":uid}).join(g,on="user_id",how="left").fill_null(0)
gcn=g["gcn"].to_numpy()
v=pl.read_parquet("work/preds_pack/val_preds.parquet").sort("user_id")
ly=np.log1p(np.clip(v["target"].to_numpy().astype(float),0,None))
e=v["blend"].to_numpy().astype(float)-ly
mask=gcn>0
print(f"  юзеров с таким поведением за 365д: {int(mask.sum()):,} ({100*mask.mean():.1f}%)")
print(f"  средний остаток бленда у них: {e[mask].mean():+.5f}  (у прочих {e[~mask].mean():+.5f})")
se=e[mask].std()/np.sqrt(mask.sum())
print(f"  разница {e[mask].mean()-e[~mask].mean():+.5f} при SE {se:.5f} -> {(e[mask].mean()-e[~mask].mean())/se:.1f} сигм")
print(f"  ЗНАК: остаток = прогноз - факт. Отрицательный = НЕДОоценка.")

print("\n=== 3. СЕГМЕНТ «только поиск» (search_days_90/active_days_90 >= 0.95) ===")
h90=df.filter(pl.col("event_date").is_between(A-timedelta(days=89),A))
g2=h90.group_by("user_id").agg([(pl.col("search")>0).sum().alias("sd"), pl.len().alias("ad"),
                                 (pl.col("to_cart")>0).sum().alias("cd2"),(pl.col("to_ord")>0).sum().alias("od")])
g2=pl.DataFrame({"user_id":uid}).join(g2,on="user_id",how="left").fill_null(0)
sd,ad=g2["sd"].to_numpy().astype(float),g2["ad"].to_numpy().astype(float)
r=np.divide(sd,np.maximum(ad,1))
m2=(r>=0.95)&(ad>0)
print(f"  юзеров: {int(m2.sum()):,} ({100*m2.mean():.1f}%)")
print(f"  средний остаток: {e[m2].mean():+.5f}  (у прочих {e[~m2].mean():+.5f})")
se2=e[m2].std()/np.sqrt(m2.sum())
print(f"  разница {e[m2].mean()-e[~m2].mean():+.5f} при SE {se2:.5f} -> {(e[m2].mean()-e[~m2].mean())/se2:.1f} сигм")

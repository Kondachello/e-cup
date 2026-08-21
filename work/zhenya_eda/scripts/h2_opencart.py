"""H2. ОТКРЫТАЯ КОРЗИНА — состояние, обнуляемое покупкой.
У команды все признаки это окна фиксированной длины. Величина «сколько накоплено
с ПОСЛЕДНЕГО ЗАКАЗА» так не выражается: её граница плавает по юзеру.
Скрининг на остатке действующего бленда, контроль — равная ёмкость из старых величин.
"""
import os
import polars as pl, numpy as np
from datetime import date, timedelta
from sklearn.linear_model import Ridge
A = date(2026,1,14)
df = pl.read_parquet("train.parquet").filter(pl.col("event_date")<=A).sort(["user_id","event_date"])

# отметка последнего заказа и накопление ПОСЛЕ него
d = df.with_columns(pl.when(pl.col("to_ord")>0).then(pl.col("event_date")).otherwise(None)
                      .forward_fill().over("user_id").alias("lo"))
d = d.with_columns(((pl.col("lo").is_null()) | (pl.col("event_date")>pl.col("lo"))).alias("after"))
# строки строго ПОСЛЕ последнего заказа (для юзеров без заказов — вся история)
op = d.filter(pl.col("after")).group_by("user_id").agg([
    pl.col("to_cart").sum().alias("oc_carts"),
    (pl.col("to_cart")>0).sum().alias("oc_cartdays"),
    pl.col("searches").sum().alias("oc_srch"),
    pl.len().alias("oc_days"),
    (pl.col("cat")>0).sum().alias("oc_catdays"),
    ((pl.col("searches")==0)&(pl.col("cat")==0)&(pl.col("to_cart")==0)).sum().alias("oc_empty"),
    (pl.col("event_date").max()-pl.col("event_date").min()).dt.total_days().alias("oc_span"),
])
op = op.with_columns([
    (pl.col("oc_carts")/pl.max_horizontal(pl.col("oc_days"),pl.lit(1))).alias("oc_carts_pd"),
    (pl.col("oc_carts")/pl.max_horizontal(pl.col("oc_srch"),pl.lit(1))).alias("oc_conv"),
    (pl.col("oc_empty")/pl.max_horizontal(pl.col("oc_days"),pl.lit(1))).alias("oc_empsh"),
])
# то же, но нормировано на ТИПИЧНЫЙ межзаказный цикл юзера -> «перезрела ли корзина»
cyc = df.filter(pl.col("to_ord")>0).group_by("user_id").agg(
    pl.col("event_date").diff().dt.total_days().mean().alias("cycle"))
op = op.join(cyc, on="user_id", how="left").with_columns(
    (pl.col("oc_days")/pl.max_horizontal(pl.col("cycle"),pl.lit(1.0))).alias("oc_overdue"))

v = pl.read_parquet("work/preds_pack/val_preds.parquet").sort("user_id")
uid = v["user_id"].to_numpy()
ly = np.log1p(np.clip(v["target"].to_numpy().astype(np.float64),0,None))
resid = ly - v["blend"].to_numpy().astype(np.float64)
sb = float(np.sqrt(np.mean(resid**2)))
M = pl.DataFrame({"user_id":uid}).join(op, on="user_id", how="left").drop("user_id")
cols = M.columns
M = np.nan_to_num(M.to_numpy().astype(np.float64), nan=0., posinf=0., neginf=0.)
M = np.sign(M)*np.log1p(np.abs(M))
print(f"признаков открытой корзины: {M.shape[1]}  {cols}")

X = pl.read_parquet(os.environ.get("ZH_CACHE", "work/zhenya_eda/cache") + "/a2026-01-14.parquet")
CT = X.select([c for c in X.columns if c.startswith("ct_")]).to_numpy().astype(np.float64)
CT = np.sign(np.nan_to_num(CT))*np.log1p(np.abs(np.nan_to_num(CT)))
B = X.select([c for c in X.columns if c.startswith("b_")]).to_numpy().astype(np.float64)
B = np.sign(np.nan_to_num(B))*np.log1p(np.abs(np.nan_to_num(B)))

rng = np.random.default_rng(0); i = rng.permutation(len(ly)); h=len(ly)//2; TR,TE=i[:h],i[h:]
def r2(Z):
    mu,sd = Z[TR].mean(0), Z[TR].std(0)+1e-9
    m = Ridge(alpha=10.).fit((Z[TR]-mu)/sd, resid[TR]); p = m.predict((Z[TE]-mu)/sd)
    return 1-np.sum((resid[TE]-p)**2)/np.sum((resid[TE]-resid[TR].mean())**2)
k = M.shape[1]
a = r2(M)
ctl = np.mean([r2(B[:, rng.choice(B.shape[1], size=k, replace=False)]) for _ in range(5)])
g = r2(rng.normal(size=M.shape))
print(f"\n{'ОТКРЫТАЯ КОРЗИНА':34s} k={k:>3}  mdl_flint={a:+.6f}")
print(f"{'контроль: столько же СТАРЫХ':34s} k={k:>3}  mdl_flint={ctl:+.6f}   <- честный пол")
print(f"{'плацебо случайное':34s} k={k:>3}  mdl_flint={g:+.6f}")
print(f"\nпревышение над честным полом: {a-ctl:+.6f}")
print(f"в переводе на метрику: {sb - sb*np.sqrt(max(1-(a-ctl),0)):+.6f} RMSLE" if a>ctl else "ниже пола")
print("\nодиночные mdl_flint:")
for j,c in enumerate(cols):
    Z=M[:,[j]]; mu,sd=Z[TR].mean(0),Z[TR].std(0)+1e-9
    m=Ridge(alpha=10.).fit((Z[TR]-mu)/sd,resid[TR]); p=m.predict((Z[TE]-mu)/sd)
    print(f"   {c:14s} {1-np.sum((resid[TE]-p)**2)/np.sum((resid[TE]-resid[TR].mean())**2):+.6f}")

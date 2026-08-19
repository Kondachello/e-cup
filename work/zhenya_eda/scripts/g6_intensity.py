"""G6. Быстрая разведка: несёт ли ФОРМА распределения дневной интенсивности сигнал
сверх сумм. Только валидационный якорь, чтобы решить, стоит ли полная сборка."""
import polars as pl, numpy as np
from datetime import date, timedelta
from sklearn.linear_model import Ridge
A = date(2026,1,14)
df = pl.read_parquet("train.parquet").filter(pl.col("event_date")<=A)
v = pl.read_parquet("work/preds_pack/val_preds.parquet").sort("user_id")
uid = v["user_id"].to_numpy()
ly = np.log1p(np.clip(v["target"].to_numpy().astype(np.float64),0,None))
resid = ly - v["blend"].to_numpy().astype(np.float64)
sb = float(np.sqrt(np.mean(resid**2)))

a=[]
for w in (30,90,365):
    m = pl.col("event_date")>=A-timedelta(days=w-1)
    for col in ("searches","to_cart","to_ord"):
        a += [pl.col(col).filter(m).std().alias(f"in_{col}_sd_{w}"),
              pl.col(col).filter(m).max().alias(f"in_{col}_mx_{w}"),
              pl.col(col).filter(m).quantile(.9).alias(f"in_{col}_p90_{w}")]
    # энтропия распределения активности по дням
    s = pl.col("searches").filter(m)
    a += [(-( (s/pl.max_horizontal(s.sum(),pl.lit(1))) *
             ((s/pl.max_horizontal(s.sum(),pl.lit(1)))+1e-12).log() ).sum()).alias(f"in_ent_{w}")]
G = df.group_by("user_id").agg(a)
M = pl.DataFrame({"user_id":uid}).join(G,on="user_id",how="left").drop("user_id").to_numpy().astype(float)
M = np.nan_to_num(M); M = np.sign(M)*np.log1p(np.abs(M))
rng = np.random.default_rng(0); i = rng.permutation(len(ly)); h=len(ly)//2; TR,TE=i[:h],i[h:]
def r2(X):
    mu,sd = X[TR].mean(0), X[TR].std(0)+1e-9
    m = Ridge(alpha=10.).fit((X[TR]-mu)/sd, resid[TR]); p = m.predict((X[TE]-mu)/sd)
    return 1-np.sum((resid[TE]-p)**2)/np.sum((resid[TE]-resid[TR].mean())**2)
a_, b_ = r2(M), r2(rng.normal(size=M.shape))
print(f"INTENSITY (форма дневной интенсивности), k={M.shape[1]}")
print(f"  mdl_flint вне выборки {a_:+.6f}   плацебо {b_:+.6f}   выигрыш {sb-sb*np.sqrt(max(1-a_,0)) if a_>0 else 0:.6f}")
print(f"  вердикт: {'стоит полной сборки' if a_-b_>3e-4 else 'ниже порога, не строить'}")

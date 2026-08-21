"""J3. РЕШАЮЩИЙ ТЕСТ представления фазы: EWMA с полупериодом ~50 дней.

Гипотеза из J1/J2: фаза юзера — экспоненциальное скользящее среднее дневных рядов
с полупериодом 45-60 дней. У команды затухания 7/30/120 (тир v2) — оптимум обходят.
Проверка: остаток ДЕЙСТВУЮЩЕГО бленда, контроль равной ёмкости из старых признаков.
"""
import os
import polars as pl, numpy as np
from datetime import date, timedelta
from sklearn.linear_model import Ridge

A = date(2026,1,14); L=364
df = pl.read_parquet("train.parquet", columns=["user_id","event_date","gmv","to_ord","to_cart","searches"])
df = df.filter(pl.col("event_date").is_between(A-timedelta(days=L-1), A))
uids = np.sort(df["user_id"].unique().to_numpy()); uidx={u:i for i,u in enumerate(uids)}
d0=A-timedelta(days=L-1)
row=np.array([uidx[u] for u in df["user_id"].to_numpy()],dtype=np.int32)
col=(df["event_date"].to_numpy()-np.datetime64(d0)).astype("timedelta64[D]").astype(int)
age=np.arange(L-1,-1,-1,dtype=np.float32)

def ew(vals, hls):
    M=np.zeros((len(uids),L),dtype=np.float32); M[row,col]=vals
    return np.column_stack([(M*np.exp(-np.log(2)*age/h)).sum(1)/np.exp(-np.log(2)*age/h).sum() for h in hls])
SER = [("gmv", np.log1p(df["gmv"].to_numpy()).astype(np.float32)),
       ("ord", (df["to_ord"].to_numpy()>0).astype(np.float32)),
       ("cart",(df["to_cart"].to_numpy()>0).astype(np.float32)),
       ("srch",np.log1p(df["searches"].to_numpy()).astype(np.float32))]

v = pl.read_parquet("work/preds_pack/val_preds.parquet").sort("user_id")
ly = np.log1p(np.clip(v["target"].to_numpy().astype(np.float64),0,None))
resid = ly - v["blend"].to_numpy().astype(np.float64)
sb = float(np.sqrt(np.mean(resid**2)))
X = pl.read_parquet(os.environ.get("ZH_CACHE", "work/zhenya_eda/cache") + "/a2026-01-14.parquet")
B = X.select([c for c in X.columns if c.startswith("b_")]).to_numpy().astype(np.float64)
B = np.sign(np.nan_to_num(B))*np.log1p(np.abs(np.nan_to_num(B)))
rng=np.random.default_rng(0); i=rng.permutation(len(ly)); h2=len(ly)//2; TR,TE=i[:h2],i[h2:]
def r2(Z):
    mu,sd=Z[TR].mean(0),Z[TR].std(0)+1e-9
    m=Ridge(alpha=10.).fit((Z[TR]-mu)/sd,resid[TR]); p=m.predict((Z[TE]-mu)/sd)
    return 1-np.sum((resid[TE]-p)**2)/np.sum((resid[TE]-resid[TR].mean())**2)
def ctl(k, n=5):
    return float(np.mean([r2(B[:, rng.choice(B.shape[1], size=min(k,B.shape[1]), replace=False)]) for _ in range(n)]))

print(f"бленд val RMSLE={sb:.6f}\n")
VARIANTS = {
    "ФАЗА: полупериод 50 (4 ряда)":            [50],
    "ФАЗА: полупериоды 45+60 (4 ряда)":        [45,60],
    "команда: полупериоды 7/30/120 (4 ряда)":  [7,30,120],
    "ФАЗА + команда (7/30/50/120)":            [7,30,50,120],
}
print(f"{'вариант':40s} {'k':>4} {'mdl_flint':>11} {'контроль':>11} {'превышение':>12}")
for nm, hls in VARIANTS.items():
    Z = np.log1p(np.abs(np.hstack([ew(vals,hls) for _,vals in SER])))
    a = r2(Z); c = ctl(Z.shape[1])
    print(f"{nm:40s} {Z.shape[1]:>4} {a:>11.6f} {c:>11.6f} {a-c:>+12.6f}")

print(f"\n=== ПРЯМОЙ ВОПРОС: даёт ли полупериод 50 что-то СВЕРХ 7/30/120 ===")
Zt = np.log1p(np.abs(np.hstack([ew(vals,[7,30,120]) for _,vals in SER])))     # 12
Z5 = np.log1p(np.abs(np.hstack([ew(vals,[50]) for _,vals in SER])))           # 4
mu,sd = Zt[TR].mean(0), Zt[TR].std(0)+1e-9; Ztn=(Zt-mu)/sd
m = Ridge(alpha=1.0).fit(Ztn[TR], Z5[TR]); mdl_gneis2 = Z5 - m.predict(Ztn)
print(f"  доля дисперсии EWMA-50, НЕ объяснённой набором 7/30/120: {float(np.mean(mdl_gneis2.var(0)/(Z5.var(0)+1e-12))):.4f}")
print(f"  mdl_flint остатка бленда по этому остатку: {r2(mdl_gneis2):+.6f}  (контроль k=4: {ctl(4):+.6f})")

"""J2. Какие постоянные времени реально нужны для 30-дневного таргета,
и лежат ли они в оболочке существующих признаков команды.

Строим EWMA дневных рядов на сетке полупериодов, меряем:
  (1) сколько каждый масштаб даёт для таргета;
  (2) СКОЛЬКО ОСТАЁТСЯ после проекции на базовые признаки команды (оболочка).
Второе и есть ответ на «предложите представление, кодирующее фазу».
"""
import os
import polars as pl, numpy as np
from datetime import date, timedelta
from sklearn.linear_model import Ridge

A = date(2026,1,14); L = 364
HL = [1,2,3,5,7,14,30,45,60,90,120,180]        # полупериоды, дни
df = pl.read_parquet("train.parquet", columns=["user_id","event_date","gmv","to_ord","to_cart","searches"])
df = df.filter(pl.col("event_date").is_between(A-timedelta(days=L-1), A))
uids = np.sort(df["user_id"].unique().to_numpy()); uidx={u:i for i,u in enumerate(uids)}
d0 = A - timedelta(days=L-1)
row = np.array([uidx[u] for u in df["user_id"].to_numpy()], dtype=np.int32)
col = (df["event_date"].to_numpy()-np.datetime64(d0)).astype("timedelta64[D]").astype(int)

def ewma_feats(vals, name):
    M = np.zeros((len(uids), L), dtype=np.float32); M[row, col] = vals
    out = {}
    age = np.arange(L-1, -1, -1, dtype=np.float32)      # 0 = день якоря
    for h in HL:
        w = np.exp(-np.log(2)*age/h).astype(np.float32)
        out[f"{name}_h{h}"] = (M*w).sum(1)/w.sum()
    return out

F = {}
F.update(ewma_feats(np.log1p(df["gmv"].to_numpy()).astype(np.float32), "gmv"))
F.update(ewma_feats((df["to_ord"].to_numpy()>0).astype(np.float32), "ord"))
F.update(ewma_feats((df["to_cart"].to_numpy()>0).astype(np.float32), "cart"))
F.update(ewma_feats(np.log1p(df["searches"].to_numpy()).astype(np.float32), "srch"))
names = list(F)
E = np.log1p(np.abs(np.column_stack([F[n] for n in names])))
print(f"EWMA-признаков {E.shape[1]} ({len(HL)} полупериодов x 4 ряда)")

v = pl.read_parquet("work/preds_pack/val_preds.parquet").sort("user_id")
assert np.array_equal(v["user_id"].to_numpy(), uids)
ly = np.log1p(np.clip(v["target"].to_numpy().astype(np.float64),0,None))
X = pl.read_parquet(os.environ.get("ZH_CACHE", "work/zhenya_eda/cache") + "/a2026-01-14.parquet")
B = X.select([c for c in X.columns if c.startswith("b_")]).to_numpy().astype(np.float64)
B = np.sign(np.nan_to_num(B))*np.log1p(np.abs(np.nan_to_num(B)))

rng = np.random.default_rng(0); i = rng.permutation(len(ly)); h2=len(ly)//2; TR,TE=i[:h2],i[h2:]
def r2(Z, y):
    mu,sd = Z[TR].mean(0), Z[TR].std(0)+1e-9
    m = Ridge(alpha=10.).fit((Z[TR]-mu)/sd, y[TR]); p = m.predict((Z[TE]-mu)/sd)
    return 1-np.sum((y[TE]-p)**2)/np.sum((y[TE]-y[TR].mean())**2)

print(f"\n=== (1) mdl_flint таргета по ОДНОМУ полупериоду (4 ряда сразу) ===")
print(f"{'полупериод, дн':>15} {'mdl_flint таргета':>12}")
for hl in HL:
    idx = [k for k,n in enumerate(names) if n.endswith(f"_h{hl}")]
    print(f"{hl:>15} {r2(E[:,idx], ly):>12.4f}")
print(f"{'ВСЕ вместе':>15} {r2(E, ly):>12.4f}")
print(f"{'база команды':>15} {r2(B, ly):>12.4f}")

print(f"\n=== (2) ГЛАВНОЕ: что остаётся от EWMA после проекции на базу команды ===")
print("для каждого полупериода: доля дисперсии EWMA, НЕ объяснённая базой (вне оболочки)")
print(f"{'полупериод, дн':>15} {'ост. дисперсия':>15} {'mdl_flint остатка по таргету':>24}")
resid_lit = []
for hl in HL:
    idx = [k for k,n in enumerate(names) if n.endswith(f"_h{hl}")]
    Z = E[:, idx]
    mu,sd = B[TR].mean(0), B[TR].std(0)+1e-9
    Bn = (B-mu)/sd
    m = Ridge(alpha=10.).fit(Bn[TR], Z[TR])
    R = Z - m.predict(Bn)
    frac = float(np.mean(R.var(0)/(Z.var(0)+1e-12)))
    resid_lit.append((hl, frac, r2(R, ly)))
    print(f"{hl:>15} {frac:>15.4f} {resid_lit[-1][2]:>24.5f}")

print(f"\n=== (3) весь EWMA-блок вне оболочки: даёт ли он таргет ===")
mu,sd = B[TR].mean(0), B[TR].std(0)+1e-9; Bn=(B-mu)/sd
m = Ridge(alpha=10.).fit(Bn[TR], E[TR]); Rall = E - m.predict(Bn)
print(f"  mdl_flint таргета по остатку всего блока: {r2(Rall, ly):+.5f}")
print(f"  контроль: столько же СТАРЫХ признаков, тоже спроецированных на базу")
sel = rng.choice(B.shape[1], size=min(E.shape[1], B.shape[1]), replace=False)
Bs = B[:, sel]; m2 = Ridge(alpha=10.).fit(Bn[TR], Bs[TR]); Rc = Bs - m2.predict(Bn)
print(f"  mdl_flint таргета по остатку контроля:    {r2(Rc, ly):+.5f}")

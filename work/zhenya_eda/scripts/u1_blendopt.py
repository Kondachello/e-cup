"""mdl_malach. Есть ли запас в весах бленда? Честный OOF по юзерам, 5 фолдов, NNLS.
Это класс «пересборка бленда», κ замерена дважды: 0.601 и 0.529."""
import numpy as np, polars as pl
from scipy.optimize import nnls
v=pl.read_parquet("work/preds_pack/val_preds.parquet").sort("user_id")
ly=np.log1p(np.clip(v["target"].to_numpy().astype(float),0,None))
cur=v["blend"].to_numpy().astype(float)
cols=[c for c in v.columns if c not in ("user_id","target","blend")]
P=np.column_stack([v[c].to_numpy().astype(float) for c in cols])
rm=lambda p: float(np.sqrt(np.mean((p-ly)**2)))
print(f"моделей в пуле: {len(cols)}")
print(f"текущий бленд (колонка blend): val RMSLE {rm(cur):.6f}")
rng=np.random.default_rng(0); idx=rng.permutation(len(ly)); folds=np.array_split(idx,5)
oof=np.zeros(len(ly))
for i in range(5):
    te=folds[i]; tr=np.concatenate([folds[j] for j in range(5) if j!=i])
    w,_=nnls(P[tr],ly[tr]); oof[te]=P[te]@w
print(f"честный OOF-оптимум весов (NNLS, 5 фолдов): {rm(oof):.6f}")
print(f"  запас на валидации: {rm(cur)-rm(oof):+.6f}")
w_all,_=nnls(P,ly)
print(f"\nвеса на всей выборке (только >0.005):")
for c,x in sorted(zip(cols,w_all),key=lambda t:-t[1]):
    if x>0.005: print(f"   {c:24s} {x:.4f}")
print(f"   сумма весов {w_all.sum():.4f}")
d=(P@w_all)-cur
print(f"\nнаправление пересборки: sd={d.std():.5f}, q={float(np.mean(d*d)):.6f}")
g=rm(cur)-rm(oof)
for k in (0.529,0.565,0.601):
    print(f"  при κ={k}: ожидаемый тестовый выигрыш {k*k*g:.6f}")

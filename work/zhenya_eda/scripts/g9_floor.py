"""G9. ПОЛ МЕТОДА СКРИНИНГА. screen_repr.py сравнивает представление с ПЛАЦЕБО
из случайных чисел. Но случайные числа не коррелированы ни с чем, поэтому дают
отрицательный R^2. Честный контроль — представление ТОЙ ЖЕ РАЗМЕРНОСТИ из СТАРЫХ
признаков: по теории оно лежит в оболочке и должно давать ~0.
Меряем оба контроля как функцию числа колонок."""
import os
import numpy as np, polars as pl
from datetime import date
from pathlib import Path
from sklearn.linear_model import Ridge

X = pl.read_parquet(Path(os.environ.get("ZH_CACHE", "work/zhenya_eda/cache"))/f"a{date(2026,1,14)}.parquet")
v = pl.read_parquet("work/preds_pack/val_preds.parquet").sort("user_id")
ly = np.log1p(np.clip(v["target"].to_numpy().astype(np.float64),0,None))
resid = ly - v["blend"].to_numpy().astype(np.float64)
sb = float(np.sqrt(np.mean(resid**2)))
bcols = [c for c in X.columns if c.startswith("b_")]
B = X.select(bcols).to_numpy().astype(np.float64)
B = np.nan_to_num(B); B = np.sign(B)*np.log1p(np.abs(B))
rng = np.random.default_rng(0); i = rng.permutation(len(ly)); h=len(ly)//2; TR,TE=i[:h],i[h:]

def r2(M, alpha=10.0):
    mu,sd = M[TR].mean(0), M[TR].std(0)+1e-9
    m = Ridge(alpha=alpha).fit((M[TR]-mu)/sd, resid[TR]); p = m.predict((M[TE]-mu)/sd)
    return 1-np.sum((resid[TE]-p)**2)/np.sum((resid[TE]-resid[TR].mean())**2)

print(f"остаток бленда, sb={sb:.6f}\n")
print(f"{'k':>4} {'плацебо СЛУЧАЙНОЕ':>20} {'контроль ИЗ СТАРЫХ':>22} {'разница = скрытый пол':>24}")
for k in (8, 16, 32, 58, 80):
    g = np.mean([r2(rng.normal(size=(len(ly),k))) for _ in range(3)])
    ins=[]
    for _ in range(5):
        sel = rng.choice(B.shape[1], size=min(k,B.shape[1]), replace=False)
        ins.append(r2(B[:, sel]))
    ins = np.mean(ins)
    print(f"{k:>4} {g:>20.6f} {ins:>22.6f} {ins-g:>24.6f}")

print(f"\nвывод: контроль из СТАРЫХ признаков даёт положительный R^2 —")
print(f"то есть линейная модель на признаках, уже входящих в оболочку, «объясняет» остаток.")
print(f"Порог приёмки команды 0.0003 по ВЫИГРЫШУ; переведём пол в выигрыш:")
for k in (32, 58, 80):
    ins = np.mean([r2(B[:, rng.choice(B.shape[1], size=min(k,B.shape[1]), replace=False)]) for _ in range(5)])
    print(f"  k={k:>3}: пол = {sb - sb*np.sqrt(max(1-ins,0)):.6f} RMSLE")

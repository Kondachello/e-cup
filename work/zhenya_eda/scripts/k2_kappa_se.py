"""K2. Ошибка ЗАМЕРА κ. κ = c_test/c_val — отношение двух оценок. c_test меряется
на 50k публики, поэтому у него есть сэмплинговая ошибка. Считаем её честно и
проверяем, различимы ли вообще замеренных точек между собой."""
import os, numpy as np, polars as pl
from pathlib import Path
from sklearn.linear_model import Ridge
CACHE = Path(os.environ.get("ZH_CACHE", "work/zhenya_eda/cache"))

v = pl.read_parquet("work/preds_pack/val_preds.parquet").sort("user_id")
ly = np.log1p(np.clip(v["target"].to_numpy().astype(np.float64),0,None))
e = v["blend"].to_numpy().astype(np.float64) - ly
N_PUB = 50_000
X = pl.read_parquet(CACHE/"a2026-01-14.parquet")
B = X.select([c for c in X.columns if c.startswith("b_")]).to_numpy().astype(np.float64)
B = np.sign(np.nan_to_num(B))*np.log1p(np.abs(np.nan_to_num(B)))
rng = np.random.default_rng(0)

def se_c(h, n=N_PUB):
    """SE оценки c = -<e,h>/||h||^2 на выборке n юзеров"""
    hh = float(np.mean(h*h))
    return float(np.std(e*h)/(np.sqrt(n)*hh))

print("Оси разных классов, SE коэффициента при замере на 50k публики:\n")
print(f"{'ось':34s} {'|c_val|':>10} {'SE(c)':>10} {'SE(κ) при κ~0.5':>16}")
AX = {}
AX["уровень (константа)"] = np.ones(len(e))
q = np.quantile(v["blend"].to_numpy(), np.linspace(0,1,11)); q[0],q[-1]=-np.inf,np.inf
b = np.digitize(v["blend"].to_numpy(), q[1:-1])
sh = np.array([-e[b==k].mean() for k in range(10)]); AX["сегментная ступенька"] = sh[b]
mu,sd = B.mean(0), B.std(0)+1e-9
AX["стек по признакам"] = Ridge(alpha=10.).fit((B-mu)/sd, -e).predict((B-mu)/sd)
AX["случайная ось (крошка)"] = rng.normal(size=len(e))*0.01
cols = [c for c in v.columns if c not in ("user_id","target","blend")]
AX["дельта модели (mlpbin-бленд)"] = v["mlpbin_cal"].to_numpy().astype(np.float64) - v["blend"].to_numpy().astype(np.float64)

res={}
for nm,h in AX.items():
    c = -float(np.dot(e,h)/np.dot(h,h))
    s = se_c(h)
    res[nm]=(c,s)
    print(f"{nm:34s} {abs(c):>10.4f} {s:>10.4f} {s/max(abs(c),1e-9):>16.3f}")

print("\n=== ГЛАВНЫЙ ВОПРОС: различимы ли замеренных κ между собой? ===")
K = np.array([0.601,0.529,0.307,0.00,-0.20,0.05,0.20,0.09,0.63,1.12])
names = ["бленд1","бленд2","стек1","стек2","своп","сегмент","уровень","e_new","дешринк","крошки"]
print(f"  наблюдаемые: среднее {K.mean():.3f}, sd {K.std(ddof=1):.3f}, размах {K.min():.2f}..{K.max():.2f}")
for se in (0.10, 0.20, 0.30, 0.40):
    chi2 = float(np.sum((K-K.mean())**2)/se**2)
    print(f"  если SE(κ)={se:.2f}: хи-квадрат {chi2:.1f} при 9 ст.св. -> "
          f"{'ОДНОРОДНЫ (κ непредсказуема поштучно)' if chi2 < 16.9 else 'различимы'}")
print(f"\n  порог хи-квадрат 16.9 = 5% при 9 степенях свободы")
se_crit = float(np.sqrt(np.sum((K-K.mean())**2)/16.9))
print(f"  КРИТИЧЕСКАЯ SE: при SE(κ) >= {se_crit:.3f} десять точек статистически ОДНОРОДНЫ")

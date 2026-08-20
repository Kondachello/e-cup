"""I1. НЕУСТРАНИМЫЙ ПОЛ МЕТРИКИ через близнецов.

Если у двух юзеров признаковые истории совпадают, их таргеты — два независимых
розыгрыша одного условного распределения. Тогда
    E[(lp_i - lp_j)^2] = 2 * Var(lp | x)
а идеальная модель оставляет RMSLE = sqrt(E[Var(lp|x)]).
Близнецы неидеальны, поэтому оценка завышена; лечится экстраполяцией по
расстоянию между близнецами к нулю.
"""
import polars as pl, numpy as np
from datetime import date
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

X = pl.read_parquet("../zhenya/cache/a2026-01-14.parquet")
v = pl.read_parquet("work/preds_pack/val_preds.parquet").sort("user_id")
assert np.array_equal(X["user_id"].to_numpy(), v["user_id"].to_numpy())
ly = np.log1p(np.clip(v["target"].to_numpy().astype(np.float64),0,None))
blend = v["blend"].to_numpy().astype(np.float64)
sb = float(np.sqrt(np.mean((blend-ly)**2)))
print(f"бленд val RMSLE = {sb:.6f};  sd таргета = {ly.std():.4f}")

cols = [c for c in X.columns if c.startswith("b_")]
M = X.select(cols).to_numpy().astype(np.float64)
M = np.nan_to_num(M, nan=-1., posinf=1e9, neginf=-1e9)
M = np.sign(M)*np.log1p(np.abs(M))
M = (M - M.mean(0))/(M.std(0)+1e-9)
print(f"признаков {M.shape[1]}, юзеров {M.shape[0]:,}")

P = PCA(n_components=24, random_state=0).fit(M)
Z = P.transform(M)
print(f"PCA 24 компоненты объясняют {100*P.explained_variance_ratio_.sum():.1f}% дисперсии признаков")

nn = NearestNeighbors(n_neighbors=2, algorithm="auto", n_jobs=4).fit(Z)
dist, idx = nn.kneighbors(Z)
d1 = dist[:,1]; j = idx[:,1]
diff2 = (ly - ly[j])**2
print(f"\nрасстояние до ближайшего близнеца: p10={np.percentile(d1,10):.3f} "
      f"p50={np.percentile(d1,50):.3f} p90={np.percentile(d1,90):.3f}")

print(f"\n{'дециль расстояния':>18} {'ср.расст':>10} {'пар':>9} {'оценка пола':>13}")
q = np.quantile(d1, np.linspace(0,1,11)); b = np.digitize(d1, q[1:-1])
xs, ys = [], []
for k in range(10):
    m = b==k
    est = np.sqrt(np.mean(diff2[m])/2)
    xs.append(d1[m].mean()); ys.append(est)
    print(f"{k:>18} {d1[m].mean():>10.4f} {int(m.sum()):>9,} {est:>13.6f}")

xs, ys = np.array(xs), np.array(ys)
for deg, lab in ((1,"линейная"), (2,"квадратичная")):
    c = np.polyfit(xs[:6], ys[:6], deg)
    print(f"экстраполяция к нулевому расстоянию ({lab}, по 6 ближним децилям): {np.polyval(c,0):.6f}")

print(f"\n=== ЧТО ЭТО ЗНАЧИТ ===")
lo = np.polyval(np.polyfit(xs[:6], ys[:6], 1), 0)
print(f"  оценка неустранимого пола RMSLE:      ~{lo:.4f}")
print(f"  текущий бленд на валидации:            {sb:.4f}")
print(f"  остаток до пола:                       {sb-lo:.4f}")
print(f"  разрыв команды до топ-1 на паблике:    0.0031")
print(f"  то есть топ-1 использует примерно {100*0.0031/max(sb-lo,1e-9):.1f}% оставшегося запаса")

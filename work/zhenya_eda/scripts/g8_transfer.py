"""G8. СТРОГАЯ проверка переноса: контроль не гауссов шум, а РЕАЛЬНЫЕ старые признаки
той же размерности (правило равной ёмкости).

Гауссово плацебо — слишком слабый контроль: настоящие старые признаки коррелированы
со входами модели и тоже способны объяснять часть остатка. Поэтому сравниваем DT
не с шумом, а с CTL (58 старых величин на тех же окнах, столько же колонок).

Якоря подобраны так, чтобы ЦЕЛЕВЫЕ ОКНА НЕ ПЕРЕСЕКАЛИСЬ:
  A1 = 2025-10-20 -> окно 10-21..11-19
  A2 = 2025-12-01 -> окно 12-02..12-31   (разнос 42 дня)
Обучение модели строго на срезах <= 2025-09-20 (зазор 30 к A1).
"""
import os
import numpy as np, polars as pl, lightgbm as lgb, json
from datetime import date
from pathlib import Path
from sklearn.linear_model import Ridge

CACHE = Path(os.environ.get("ZH_CACHE", "work/zhenya_eda/cache"))
A1, A2 = date(2025, 10, 20), date(2025, 12, 1)
TRAIN = [date(2025, 8, 11), date(2025, 8, 25), date(2025, 9, 8)]

GROUPS = {"DT дневная таксономия": ["dt_"],
          "CTL равная ёмкость (старые)": ["ct_"],
          "TRANS переходы": ["tr_"],
          "BASE агрегаты": ["b_"]}


def load(A):
    return pl.read_parquet(CACHE / f"a{A}.parquet")


def mat(X, prefs):
    cols = [c for c in X.columns if any(c.startswith(p) for p in prefs)]
    M = X.select(cols).to_numpy().astype(np.float64)
    M = np.nan_to_num(M, nan=0.0, posinf=0.0, neginf=0.0)
    return np.sign(M) * np.log1p(np.abs(M)), cols


Xs, ys = [], []
for a in TRAIN:
    X = load(a)
    Xs.append(mat(X, ["b_"])[0])
    ys.append(np.log1p(X["target"].to_numpy().astype(np.float64)))
print(f"обучение на {len(TRAIN)} срезах {TRAIN[0]}..{TRAIN[-1]}, строк {sum(len(y) for y in ys):,}")
mdl = lgb.LGBMRegressor(objective="tweedie", tweedie_variance_power=1.45, learning_rate=.05,
                        num_leaves=63, min_child_samples=100, subsample=.8, colsample_bytree=.8,
                        n_estimators=700, verbose=-1, n_jobs=4, random_state=42)
mdl.fit(np.vstack(Xs), np.concatenate(ys))

D = {}
for tag, A in (("A1", A1), ("A2", A2)):
    X = load(A)
    pred = np.clip(mdl.predict(mat(X, ["b_"])[0]), 0, None)
    y = np.log1p(X["target"].to_numpy().astype(np.float64))
    D[tag] = (X, y - pred)
    print(f"  {tag} {A}: n={X.height:,} RMSLE={np.sqrt(np.mean((y-pred)**2)):.6f} "
          f"средний остаток {np.mean(y-pred):+.4f}")

X1, r1 = D["A1"]; X2, r2v = D["A2"]
u1, u2 = X1["user_id"].to_numpy(), X2["user_id"].to_numpy()
common = np.intersect1d(u1, u2)
i1, i2 = np.searchsorted(u1, common), np.searchsorted(u2, common)
r1, r2v = r1[i1], r2v[i2]
c1, c2 = r1 - r1.mean(), r2v - r2v.mean()
print(f"  общих юзеров {len(common):,}; разница уровня окон {r2v.mean()-r1.mean():+.4f} (снята центрированием)")


def r2(fitX, fity, evX, evy, alpha=10.0):
    mu, sd = fitX.mean(0), fitX.std(0) + 1e-9
    m = Ridge(alpha=alpha).fit((fitX - mu) / sd, fity)
    p = m.predict((evX - mu) / sd)
    return 1 - np.sum((evy - p) ** 2) / np.sum((evy - evy.mean()) ** 2)


rng = np.random.default_rng(0)
n = len(common); idx = rng.permutation(n); h = n // 2
print(f"\n{'представление':30s} {'k':>4} {'то же окно':>12} {'A1->A2':>10} {'A2->A1':>10}")
R = {}
for name, prefs in GROUPS.items():
    M1 = mat(X1, prefs)[0][i1]
    M2, cols = mat(X2, prefs)
    M2 = M2[i2]
    same = r2(M1[idx[:h]], c1[idx[:h]], M1[idx[h:]], c1[idx[h:]])
    f = r2(M1, c1, M2, c2)
    b = r2(M2, c2, M1, c1)
    R[name] = {"k": len(cols), "same": same, "fwd": f, "bwd": b}
    print(f"{name:30s} {len(cols):>4} {same:>12.6f} {f:>10.6f} {b:>10.6f}")

mdl_amber, mdl_gabbro = rng.normal(size=(n, 58)), rng.normal(size=(n, 58))
pg = r2(mdl_amber, c1, mdl_gabbro, c2)
print(f"{'гауссово плацебо (58)':30s} {58:>4} {'—':>12} {pg:>10.6f} {'—':>10}")

dt, ct = R["DT дневная таксономия"], R["CTL равная ёмкость (старые)"]
print(f"\nПРЯМОЕ СРАВНЕНИЕ при равной ёмкости (58 против 58):")
print(f"  перенос A1->A2:  DT {dt['fwd']:+.6f}  против CTL {ct['fwd']:+.6f}   разница {dt['fwd']-ct['fwd']:+.6f}")
print(f"  перенос A2->A1:  DT {dt['bwd']:+.6f}  против CTL {ct['bwd']:+.6f}   разница {dt['bwd']-ct['bwd']:+.6f}")
sb = 1.6664
mn = min(dt["fwd"], dt["bwd"])
if mn > 0:
    print(f"  верхняя оценка выигрыша DT: {sb - sb*np.sqrt(1-mn):.6f} RMSLE (порог 0.0003, шум 0.000022)")
Path(os.environ.get("ZH_OUT", "work/zhenya_eda/out")).mkdir(exist_ok=True, parents=True)
Path(os.environ.get("ZH_OUT", "work/zhenya_eda/out") + "/g8_transfer.json").write_text(json.dumps(R, indent=1))

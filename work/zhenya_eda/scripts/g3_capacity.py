"""G3. РЕШАЮЩИЙ ТЕСТ: контроль равной ёмкостью на настоящем валидационном якоре.

Правило команды (им отклонены тиры v6, v8, v10): новый набор признаков сравнивать
не с базой, а с базой ПЛЮС СТОЛЬКО ЖЕ СТАРЫХ признаков. Иначе меряется ёмкость,
а не информация.
Правило команды №1: сравнивать только ПОСЛЕ калибровки — сырой порядок обманывал 8 раз.
Правило команды: разброс между сидами может превышать эффект, поэтому несколько сидов.
"""
import os
import numpy as np, polars as pl, lightgbm as lgb, json, sys, gc
from datetime import date, timedelta
from pathlib import Path

CACHE = Path(os.environ.get("ZH_CACHE", "work/zhenya_eda/cache"))
VAL = date(2026, 1, 14)
TRAIN = [VAL - timedelta(days=30 + 14 * i) for i in range(0, 10)]
SEEDS = [42, 555]


def mat(X, prefs):
    cols = [c for c in X.columns if any(c.startswith(p) for p in prefs)]
    M = X.select(cols).to_numpy()
    M = np.nan_to_num(M.astype(np.float32), nan=-1.0, posinf=1e9, neginf=-1e9)
    return M, cols


def calibrated(lp, ly, nb=24):
    rng = np.random.default_rng(0)
    i = rng.permutation(len(ly)); h = len(ly) // 2
    out = lp.copy()
    for tr, te in ((i[:h], i[h:]), (i[h:], i[:h])):
        q = np.quantile(lp[tr], np.linspace(0, 1, nb + 1)); q[0], q[-1] = -np.inf, np.inf
        b1, b2 = np.digitize(lp[tr], q[1:-1]), np.digitize(lp[te], q[1:-1])
        sh = np.array([np.mean(ly[tr][b1 == k] - lp[tr][b1 == k]) if (b1 == k).sum() > 50 else 0.
                       for k in range(nb)])
        out[te] = lp[te] + sh[b2]
    return float(np.sqrt(np.mean((out - ly) ** 2)))


ARMS = {"база": ["b_"],
        "база + DT (таксономия дня)": ["b_", "dt_"],
        "база + CTL (столько же старых)": ["b_", "ct_"]}

missing = [a for a in TRAIN + [VAL] if not (CACHE / f"a{a}.parquet").exists()]
if missing:
    print("нет кэша:", missing); sys.exit(1)

XV = pl.read_parquet(CACHE / f"a{VAL}.parquet")
yv = np.log1p(XV["target"].to_numpy().astype(np.float64))
print(f"обучающих срезов {len(TRAIN)} ({TRAIN[-1]}..{TRAIN[0]}), зазор 30; валидация {VAL} n={XV.height:,}",
      flush=True)

res = {}
for name, prefs in ARMS.items():
    parts, ys = [], []
    for a in TRAIN:
        Xa = pl.read_parquet(CACHE / f"a{a}.parquet")
        M, _ = mat(Xa, prefs)
        parts.append(M); ys.append(np.log1p(Xa["target"].to_numpy().astype(np.float64)))
        del Xa; gc.collect()
    Xtr = np.vstack(parts); del parts; gc.collect()
    ytr = np.concatenate(ys)
    Xv, cols = mat(XV, prefs)
    raws, cals = [], []
    for s in SEEDS:
        m = lgb.LGBMRegressor(objective="tweedie", tweedie_variance_power=1.45, learning_rate=.05,
                              num_leaves=63, min_child_samples=100, subsample=.8,
                              colsample_bytree=.8, n_estimators=700, verbose=-1, n_jobs=4,
                              random_state=s).fit(Xtr, ytr)
        lp = np.clip(m.predict(Xv), 0, None)
        raws.append(float(np.sqrt(np.mean((lp - yv) ** 2))))
        cals.append(calibrated(lp, yv))
        del m; gc.collect()
    res[name] = (float(np.mean(raws)), float(np.mean(cals)), float(np.std(cals)), len(cols))
    print(f"{name:34s} k={len(cols):>3}  сырой {np.mean(raws):.6f}  "
          f"калиброванный {np.mean(cals):.6f}  разброс сидов {np.std(cals):.6f}", flush=True)
    del Xtr, ytr, Xv; gc.collect()

b = res["база"][1]; dt = res["база + DT (таксономия дня)"][1]; ct = res["база + CTL (столько же старых)"][1]
print(f"\n{'DT против базы':44s} {dt - b:+.6f}")
print(f"{'CTL против базы (это чистая ёмкость)':44s} {ct - b:+.6f}")
print(f"\n{'>>> DT против КОНТРОЛЯ РАВНОЙ ЁМКОСТИ':44s} {dt - ct:+.6f}   <- РЕШАЮЩЕЕ ЧИСЛО")
print(f"разброс между сидами ~{max(v[2] for v in res.values()):.6f}; порог приёмки 0.0003; шум 0.000022")
Path(os.environ.get("ZH_OUT", "work/zhenya_eda/out") + "/g3_capacity.json").write_text(json.dumps(
    {k: {"raw": v[0], "cal": v[1], "sd": v[2], "k": v[3]} for k, v in res.items()}, indent=1))

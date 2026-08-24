"""V2. Направление «поправка на слом каталога» на ТЕСТОВОМ якоре.

Запускать только если v1_erafix показал, что поправка помогает на валидации.
Строит те же две руки на якоре 2026-02-13 и берёт их разность — это направление,
которое чинит дрейф каталожных признаков между обучением и тестом.

Свойства: структурное (из данных, не из скоров), вне оболочки замеренных файлов
(слом никто не правил), применяется зондом по правилу 4x-шага.
"""
import os
import numpy as np, polars as pl, lightgbm as lgb
from datetime import date, timedelta
from pathlib import Path
import importlib.util

spec = importlib.util.spec_from_file_location("v1", str(Path(__file__).parent / "v1_erafix.py"))
src = open(Path(__file__).parent / "v1_erafix.py", encoding="utf-8").read().split("res = {}")[0]
ns = {}
exec(compile(src, "v1", "exec"), ns)
build, TRAIN, SEEDS, VAL = ns["build"], ns["TRAIN"], ns["SEEDS"], ns["VAL"]
OUT = Path(os.environ.get("ZH_OUT", "work/zhenya_eda/out"))
TEST = date(2026, 2, 13)
Q_TARGET = 0.0026          # правило 4x-шага (часть D)

preds = {}
for fix in (False, True):
    Xs, ys = [], []
    for a in TRAIN + [VAL]:                      # для теста доучиваем и на валидации
        M, y, _ = build(a, fix); Xs.append(M); ys.append(y)
    Xtr = np.vstack(Xs); ytr = np.concatenate(ys)
    Xte, _, uid = build(TEST, fix)
    p = np.mean([np.clip(lgb.LGBMRegressor(
        objective="tweedie", tweedie_variance_power=1.45, learning_rate=.05, num_leaves=63,
        min_child_samples=100, subsample=.8, colsample_bytree=.8, n_estimators=700,
        verbose=-1, n_jobs=4, random_state=s).fit(Xtr, ytr).predict(Xte), 0, None)
        for s in SEEDS], axis=0)
    preds[fix] = p
    print(f"{'С ПОПРАВКОЙ' if fix else 'без поправки':14s} тест: mean {p.mean():.5f} sd {p.std():.5f}",
          flush=True)

d = preds[True] - preds[False]
print(f"\nразность (сырая): mean {d.mean():+.6f}  sd {d.std():.6f}")

d = d - d.mean()
d = d * np.sqrt(Q_TARGET / float(np.mean(d * d)))
pl.DataFrame({"user_id": uid, "step": d}).write_parquet(OUT / "dir_erafix.parquet")
print(f"направление записано: q={float(np.mean(d*d)):.6f}, sd={d.std():.5f}")

# новизна к замеренным файлам
try:
    z = np.load(str(OUT / "lb_full.npz")); import json
    NAMES = json.load(open(OUT / "lb_meta.json"))["names"]
    assert np.array_equal(z["uid"], uid)
    B = np.column_stack([np.ones(len(uid))] + [z[f"lp_{x}"] for x in NAMES])
    Q, _ = np.linalg.qr(B); r = d - Q @ (Q.T @ d)
    print(f"новизна к замеренным файлам: "
          f"{float(np.sqrt(np.mean(r**2))/np.sqrt(np.mean(d**2))):.4f}")
except Exception as e:
    print(f"новизну посчитать не вышло: {e}")

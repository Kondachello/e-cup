"""Драйвер v2_eradir.py под наш репозиторий + переиспользование кэша v1b.

Отличия от оригинала Жени:
- пути наши (cwd=ROOT, ZH_OUT=work/zhenya_eda/out);
- Xtr для двух рук берётся из erafix_feat.npz (v1b), доучивание на валидации —
  vstack кэшированных Xtr и Xv, как в оригинале TRAIN + [VAL];
- тестовые признаки строятся свежими (в кэше их нет), по двум рукам.
Сиды, параметры, нормировка на q=0.0026 — дословно из v2_eradir.py.
"""
import os
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl
import lightgbm as lgb

ROOT = Path("/Users/alexanderkondakov/ozon-cup")
os.chdir(ROOT)
OUT = ROOT / "work" / "zhenya_eda" / "out"
CA = OUT / "erafix_feat.npz"

src = open(ROOT / "work/zhenya_eda/scripts/v1_erafix.py", encoding="utf-8").read().split("res = {}")[0]
ns = {}
exec(compile(src, "v1", "exec"), ns)
build, SEEDS = ns["build"], ns["SEEDS"]

TEST = date(2026, 2, 13)
Q_TARGET = 0.0026

z = np.load(CA)
preds = {}
uid = None
for fix in (False, True):
    Xtr = np.vstack([z[f"Xtr{int(fix)}"], z[f"Xv{int(fix)}"]])
    ytr = np.concatenate([z[f"ytr{int(fix)}"], z["yv"]])
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
print(f"направление записано: {OUT/'dir_erafix.parquet'} q={float(np.mean(d*d)):.6f}, sd={d.std():.5f}")

"""ЧЕСТНАЯ цена обрезки: модель, которая НИКОГДА не видела валидационный якорь.

Зачем. Сохранённые в work/models бустеры — артефакты ФАЗЫ ДООБУЧЕНИЯ, а она
включает валидационный якорь (train_gbdt.py: y_all = concat([y, gap, log1p(yv_raw)])).
Поэтому замер «обрезал признаки валидации — стало хуже» на них ВНУТРИВЫБОРОЧНЫЙ и
завышает цену: часть ухудшения это просто уход с запомненной поверхности.

Здесь обучается один диагностический бустер по протоколу зазора 30 дней на
обучающих якорях БЕЗ валидационного, и уже он смотрит на два плеча валидации.
Это не член бленда и никуда не идёт — только измерение.

Рецепт чемпиона проекта: tweedie на log1p-таргете (Р6 в KNOWLEDGE).
"""
from __future__ import annotations

import json
import sys
import time
from datetime import timedelta
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))
from calibrate import apply_shifts, fit_shifts
from common import (FEATURES_DIR, HORIZON, REPORTS_DIR, VAL_ANCHOR, WORK,
                    rmsle, train_anchors)

MB = 349
BINS = 24
N_TREES = 500
GAP_DAYS = 30


def cols_of() -> list[str]:
    meta = json.loads((WORK / "models" / "weak_an_d_meta.json").read_text())
    return meta["feature_cols"]


def anchor_frame(a: str, cut: bool = False) -> pl.DataFrame:
    sfx = f".mb{MB}" if cut else ""
    df = pl.read_parquet(FEATURES_DIR / f"anchor={a}{sfx}.parquet")
    for tier in ("extra", "v3"):
        p = FEATURES_DIR / f"anchor={a}{sfx}.{tier}.parquet"
        df = df.join(pl.read_parquet(p), on="user_id", how="left")
    df = df.join(pl.read_parquet(FEATURES_DIR / f"anchor={a}.v4.parquet"),
                 on="user_id", how="left")
    return df.sort("user_id")


def mat(df: pl.DataFrame, cols: list[str]) -> np.ndarray:
    return np.ascontiguousarray(
        df.select([pl.col(c).cast(pl.Float32) for c in cols]).to_numpy())


def main() -> None:
    import lightgbm as lgb

    cols = cols_of()
    # зазор 30: целевое окно обучающего якоря должно кончиться за 30 дней до
    # начала валидационного окна (2026-01-15)
    last_ok = VAL_ANCHOR + timedelta(days=1) - timedelta(days=GAP_DAYS + HORIZON)
    cand = [a for a in train_anchors(14) if a <= last_ok]
    use = cand[-6:]
    print(f"зазор 30 -> последний допустимый якорь {last_ok}; обучаюсь на {len(use)}: "
          f"{[str(a) for a in use]}", flush=True)

    Xs, ys = [], []
    for a in use:
        d = anchor_frame(a.isoformat())
        Xs.append(mat(d, cols))
        ys.append(d["target"].to_numpy().astype(np.float64))
        del d
    X = np.vstack(Xs); del Xs
    y = np.concatenate(ys); del ys
    print(f"обучающая матрица {X.shape}", flush=True)

    t0 = time.time()
    booster = lgb.LGBMRegressor(
        objective="tweedie", tweedie_variance_power=1.45, n_estimators=N_TREES,
        learning_rate=0.05, num_leaves=255, min_child_samples=300,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
        n_jobs=3, verbose=-1, random_state=7,
    ).fit(X, np.log1p(y))
    print(f"обучено за {time.time()-t0:.0f}s", flush=True)
    del X, y

    va = VAL_ANCHOR.isoformat()
    full, cut = anchor_frame(va), anchor_frame(va, cut=True)
    yv = full["target"].to_numpy().astype(np.float64)
    ly = np.log1p(yv)
    lf_ = np.clip(booster.predict(mat(full, cols)), 0, None)
    lc_ = np.clip(booster.predict(mat(cut, cols)), 0, None)

    n = len(yv)
    rng = np.random.default_rng(0)
    half = rng.permutation(n) < n // 2
    ev = ~half
    raw_f, raw_c = rmsle(yv[ev], np.expm1(lf_[ev])), rmsle(yv[ev], np.expm1(lc_[ev]))
    cf, sf = fit_shifts(lf_[half], ly[half], BINS)
    cc, sc = fit_shifts(lc_[half], ly[half], BINS)
    cal_f = rmsle(yv[ev], np.expm1(apply_shifts(lf_[ev], cf, sf)))
    cal_c = rmsle(yv[ev], np.expm1(apply_shifts(lc_[ev], cc, sc)))
    frz_c = rmsle(yv[ev], np.expm1(apply_shifts(lc_[ev], cf, sf)))

    print(f"\nВНЕ ВЫБОРКИ (модель не видела валидацию):")
    print(f"  сырой   FULL {raw_f:.6f}  CUT {raw_c:.6f}  цена {raw_c-raw_f:+.6f}")
    print(f"  калибр  FULL {cal_f:.6f}  CUT {cal_c:.6f}  цена {cal_c-cal_f:+.6f}")
    print(f"  заморож калибратор FULL-плеча: CUT {frz_c:.6f}  цена {frz_c-cal_f:+.6f}")
    d = lc_ - lf_
    print(f"\nсдвиг прогноза в логарифме: sd {d.std():.5f} среднее {d.mean():+.5f} "
          f"корр {np.corrcoef(lf_, lc_)[0,1]:.5f}")
    np.savez(REPORTS_DIR / "mb_fix_clean_probe.npz", full=lf_, cut=lc_, target=yv, half=half)
    print(f"\nпорог проекта 0.0003, шум замера LB 0.000022")


if __name__ == "__main__":
    main()

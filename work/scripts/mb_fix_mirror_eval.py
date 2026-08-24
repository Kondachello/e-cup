"""ЦЕНА ОБРЕЗКИ на валидации — !!! ЧИСЛА ЭТОГО СКРИПТА НЕДЕЙСТВИТЕЛЬНЫ !!!

ВНИМАНИЕ. Сохранённые в work/models бустеры — артефакты фазы ДООБУЧЕНИЯ, а она
включает валидационный якорь (train_gbdt.py: y_all = concat([y, gap, log1p(yv_raw)])).
Поэтому здесь модель смотрит на СВОИ ЖЕ обучающие строки, замер внутривыборочный и
завышает цену обрезки в тринадцать раз: +0.0047 здесь против честных +0.00038 в
work/scripts/mb_fix_clean_probe*.py. Скрипт оставлен как документация ловушки.
Для настоящего замера пользоваться mb_fix_clean_probe3.py.

Исходное описание:

Два плеча одних и тех же сохранённых моделей:
  FULL  — валидационные признаки как есть (обрезка не срабатывает),
  CUT   — валидационные признаки с MAX_BACK=349, отсечка 2025-01-30, то есть
          выброшены ровно те же 29 календарных дней, что теряет тестовый якорь.

Сравнение ТОЛЬКО после калибровки (правило проекта). Две формы:
  honest — таблица сдвигов подбирается на половине юзеров, замер на другой,
           отдельно для каждого плеча (какой набор признаков даёт модель лучше),
  frozen — общая таблица, подобранная на плече FULL, применяется к обоим
           (в точности то, что происходит на тесте: калибратор заморожен).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))
from calibrate import apply_shifts, fit_shifts
from common import FEATURES_DIR, REPORTS_DIR, VAL_ANCHOR, WORK, rmsle
from mb_fix_infer import LGB_MODELS, MULTIHEAD, gbdt_predict, lgb_booster, mat

MODELS = WORK / "models"
A = VAL_ANCHOR.isoformat()
MB = 349
BINS = 24


def frame(cut: bool) -> pl.DataFrame:
    sfx = f".mb{MB}" if cut else ""
    df = pl.read_parquet(FEATURES_DIR / f"anchor={A}{sfx}.parquet")
    df = df.join(pl.read_parquet(FEATURES_DIR / f"anchor={A}{sfx}.extra.parquet"),
                 on="user_id", how="left")
    df = df.join(pl.read_parquet(FEATURES_DIR / f"anchor={A}{sfx}.v3.parquet"),
                 on="user_id", how="left")
    df = df.join(pl.read_parquet(FEATURES_DIR / f"anchor={A}.v4.parquet"),
                 on="user_id", how="left")
    return df.sort("user_id")


def main() -> None:
    full, cut = frame(False), frame(True)
    y = full["target"].to_numpy().astype(np.float64)
    ly = np.log1p(y)
    n = len(y)
    rng = np.random.default_rng(0)
    half = rng.permutation(n) < n // 2          # на этой половине подбираем
    ev = ~half                                   # на этой замеряем

    print(f"val {A}: {n} юзеров, нулей {float((y == 0).mean()):.4f}\n")
    hdr = (f"{'модель':20s} {'сырой FULL':>11s} {'сырой CUT':>11s} "
           f"{'калибр FULL':>12s} {'калибр CUT':>12s} {'ЦЕНА':>10s} {'заморож ЦЕНА':>13s}")
    print(hdr)
    print("-" * len(hdr))

    rows, store = [], {"target": y}
    todo = [(m, None) for m in LGB_MODELS] + [(m, v) for m, v in MULTIHEAD.items()]
    for name, multi in todo:
        mp = MODELS / f"{name}_meta.json"
        if not mp.exists():
            continue
        meta = json.loads(mp.read_text())
        cols = meta["feature_cols"]
        try:
            Xf, Xc = mat(full, cols), mat(cut, cols)
        except AssertionError:
            print(f"{name:20s} пропуск (нет колонок)")
            continue
        if multi is None:
            pf, pc = gbdt_predict(meta, name, Xf), gbdt_predict(meta, name, Xc)
        elif multi[0] == "countaov":
            from train_countaov import COMBINE
            f = COMBINE[meta["mode"]]
            pf = f(lgb_booster(f"{name}__count").predict(Xf),
                   lgb_booster(f"{name}__aov").predict(Xf), meta["aov_damp"])
            pc = f(lgb_booster(f"{name}__count").predict(Xc),
                   lgb_booster(f"{name}__aov").predict(Xc), meta["aov_damp"])
        else:
            from train_channel import combine
            pf = combine({c: lgb_booster(f"{name}__{c}").predict(Xf) for c in meta["channels"]})
            pc = combine({c: lgb_booster(f"{name}__{c}").predict(Xc) for c in meta["channels"]})

        lf_ = np.log1p(np.clip(np.asarray(pf, dtype=np.float64), 0, None))
        lc_ = np.log1p(np.clip(np.asarray(pc, dtype=np.float64), 0, None))
        raw_f = rmsle(y[ev], np.expm1(lf_[ev]))
        raw_c = rmsle(y[ev], np.expm1(lc_[ev]))
        # honest: своя таблица у каждого плеча, подбор на half, замер на ev
        cf, sf = fit_shifts(lf_[half], ly[half], BINS)
        cc, sc = fit_shifts(lc_[half], ly[half], BINS)
        cal_f = rmsle(y[ev], np.expm1(apply_shifts(lf_[ev], cf, sf)))
        cal_c = rmsle(y[ev], np.expm1(apply_shifts(lc_[ev], cc, sc)))
        # frozen: таблица плеча FULL применяется к обоим
        frz_c = rmsle(y[ev], np.expm1(apply_shifts(lc_[ev], cf, sf)))
        print(f"{name:20s} {raw_f:11.6f} {raw_c:11.6f} {cal_f:12.6f} {cal_c:12.6f} "
              f"{cal_c - cal_f:+10.6f} {frz_c - cal_f:+13.6f}")
        rows.append((name, cal_f, cal_c, frz_c))
        store[f"{name}__full"], store[f"{name}__cut"] = lf_, lc_

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(REPORTS_DIR / "mb_fix_mirror_val.npz", half=half, **store)

    if rows:
        d = np.array([r[2] - r[1] for r in rows])
        dz = np.array([r[3] - r[1] for r in rows])
        print(f"\nЦЕНА ОБРЕЗКИ по {len(rows)} моделям (калиброванно, честно): "
              f"среднее {d.mean():+.6f}, медиана {np.median(d):+.6f}, "
              f"диапазон [{d.min():+.6f}, {d.max():+.6f}]")
        print(f"то же с ЗАМОРОЖЕННЫМ калибратором: среднее {dz.mean():+.6f}, "
              f"медиана {np.median(dz):+.6f}")
        print(f"плюс = обрезка ВРЕДИТ (исправление даст выигрыш такого размера)")
        print(f"\nпорог полезности проекта 0.0003; шум одного замера LB 0.000022")


if __name__ == "__main__":
    main()

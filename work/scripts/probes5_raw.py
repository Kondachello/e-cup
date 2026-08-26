"""Сырьё для пяти замерочных направлений (mdl_amber..mdl_realgr).

Считает один раз тяжёлые агрегаты и кладёт их в work/data/probes5_raw.npz:

  gmv_ya      GMV пользователя за 2025-02-14..2025-03-15 — тот же календарный
              отрезок, что тестовое окно (14 февраля, 23 февраля, 8 марта).
              Это ровно окно признака gmv_sum_ya_tgt (build_features.YEARAGO
              (364, 335) от якоря 2026-02-13), который ТОЖДЕСТВЕННО ПУСТ на всех
              обучающих якорях: данные начинаются 2025-01-01, поэтому «год назад»
              для июльских-декабрьских срезов лежит вне данных. Модели этой
              величины не видели ни разу — то самое условие «невидима на
              обучающих якорях», без которого поправка стоит ноль (KNOWLEDGE).
  gmv_rest    GMV за остальной 2025 год (2025-01-01..2025-12-31 минус окно);
              335 дней.
  first_day   номер первого дня с событием (от 2025-01-01), для отсечения тех,
              кто в феврале-марте 2025 ещё не наблюдался.
  days_ya     активных дней в том окне.
  blk1..blk3  активные дни (= строки, они совпадают: (user_id, event_date)
              уникальна на всех 30631006 строках) в трёх блоках отбора
              2026-01-15..02-13, 2025-12-16..2026-01-14, 2025-11-16..12-15.

Запуск:  POLARS_MAX_THREADS=2 .venv/bin/python work/scripts/probes5_raw.py
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))
from common import ROOT, TRAIN_PARQUET, user_universe          # noqa: E402
from silence_key import blocks_of                              # noqa: E402
from silence_target import build_cumsum                        # noqa: E402

OUT = ROOT / "work" / "data" / "probes5_raw.npz"
W_LO, W_HI = date(2025, 2, 14), date(2025, 3, 15)      # зеркало тестового окна
Y_LO, Y_HI = date(2025, 1, 1), date(2025, 12, 31)
DAY0 = date(2025, 1, 1)
TEST_ANCHOR = date(2026, 2, 13)


def main() -> None:
    assert not OUT.exists(), f"{OUT} уже есть — не перезаписываю"
    uni = user_universe()["user_id"].to_numpy()

    inw = pl.col("event_date").is_between(pl.lit(W_LO), pl.lit(W_HI))
    iny = pl.col("event_date").is_between(pl.lit(Y_LO), pl.lit(Y_HI))
    g = (
        pl.scan_parquet(TRAIN_PARQUET)
        .select("user_id", "event_date", "gmv")
        .group_by("user_id")
        .agg(
            pl.col("gmv").filter(inw).sum().alias("gmv_ya"),
            pl.col("gmv").filter(iny & ~inw).sum().alias("gmv_rest"),
            pl.col("event_date").min().alias("first_date"),
            pl.col("event_date").filter(inw).len().alias("days_ya"),
        )
        .collect(engine="streaming")
    )
    d = (pl.DataFrame({"user_id": uni}).join(g, on="user_id", how="left")
         .sort("user_id"))
    assert d.height == len(uni) and np.array_equal(d["user_id"].to_numpy(), uni)

    fd = d["first_date"].to_numpy()
    first_day = ((fd - np.datetime64(DAY0.isoformat())).astype("timedelta64[D]")
                 .astype(np.float64))

    C = build_cumsum()
    blk = blocks_of(C, TEST_ANCHOR)          # [посл.30, пред.30, ещё пред.30]
    for i, b in enumerate(blk):
        assert (b > 0).all(), f"блок {i}: есть пользователь с нулём строк"

    np.savez(
        OUT,
        user_id=uni.astype(np.int64),
        gmv_ya=d["gmv_ya"].to_numpy().astype(np.float64),
        gmv_rest=d["gmv_rest"].to_numpy().astype(np.float64),
        first_day=first_day,
        days_ya=d["days_ya"].to_numpy().astype(np.float64),
        blk1=blk[0].astype(np.int32), blk2=blk[1].astype(np.int32),
        blk3=blk[2].astype(np.int32),
    )
    mn = np.minimum(np.minimum(blk[0], blk[1]), blk[2])
    print(f"записан {OUT}")
    print(f"наблюдались до окна (first_day <= 2025-02-13): "
          f"{float((first_day <= 43).mean()):.4f}")
    print(f"GMV в окне 2025-02-14..03-15 > 0: {float((d['gmv_ya'].to_numpy() > 0).mean()):.4f}")
    print(f"минимум строк по трём блокам: доля ==1 {float((mn == 1).mean()):.4f}, "
          f"<=2 {float((mn <= 2).mean()):.4f}, <=3 {float((mn <= 3).mean()):.4f}, "
          f"медиана {np.median(mn):.0f}, среднее {mn.mean():.2f}")


if __name__ == "__main__":
    main()

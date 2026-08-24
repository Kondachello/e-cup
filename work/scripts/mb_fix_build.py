"""Пересборка признаков ТОЛЬКО тестового якоря с увеличенной глубиной истории.

Дефект: build_features.MAX_BACK=379 срабатывает единственный раз — на тестовом
якоре 2026-02-13 (отсечка 2025-01-30), обрезая 2025-01-01..2025-01-29. На
валидационном якоре 2026-01-14 отсечка 2024-12-31 < DATA_START, то есть не
срабатывает; на всех обучающих якорях тем более.

Скрипт НИЧЕГО не перезаписывает: пишет в anchor=2026-02-13.mb409.parquet,
имя вне контракта common.load_anchor (тот знает только .parquet/.extra./.v3.
/.v4./.v6./.v7./.v8./.v10./.v5./.v5s./.seqoof.).
"""
from __future__ import annotations

import sys
import time
from datetime import date
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).parent))
import build_features as bf
from common import FEATURES_DIR, TEST_ANCHOR, TRAIN_PARQUET, user_universe

MB = 409  # 2026-02-13 - 408 = 2025-01-01 ровно; 409 даёт день запаса
OUT = FEATURES_DIR / f"anchor={TEST_ANCHOR.isoformat()}.mb{MB}.parquet"


def main() -> None:
    assert TEST_ANCHOR == date(2026, 2, 13)
    if OUT.exists():
        print(f"{OUT.name} уже есть, выхожу")
        return
    bf.MAX_BACK = MB
    t0 = time.time()
    universe = user_universe()
    lf = pl.scan_parquet(TRAIN_PARQUET)
    daily = (lf.group_by("event_date").agg(pl.col("gmv").sum().alias("gmv_sum"))
             .collect(engine="streaming"))
    df = bf.build_anchor(lf, universe, daily, TEST_ANCHOR)
    df.write_parquet(OUT)
    print(f"записан {OUT} за {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()

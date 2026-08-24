"""Зеркало дефекта на ВАЛИДАЦИОННОМ якоре — единственный способ честно измерить цену.

На тесте (2026-02-13) MAX_BACK=379 даёт отсечку 2025-01-30. На валидации
(2026-01-14) MAX_BACK=349 даёт ТУ ЖЕ САМУЮ отсечку 2025-01-30, то есть выбрасывает
ровно те же 29 календарных дней 2025-01-01..2025-01-29. Модели те же, таргет
валидации наблюдаем — значит цена обрезки измеряется напрямую, а не гадается.

Пишет anchor=2026-01-14.mb349{,.extra,.v3}.parquet. Существующие файлы не трогает.
"""
from __future__ import annotations

import sys
import time
from datetime import date, timedelta
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).parent))
import build_features as bf
import build_features_v2 as b2
import build_features_v3 as b3
from common import DATA_START, FEATURES_DIR, TEST_ANCHOR, TRAIN_PARQUET, VAL_ANCHOR, user_universe

MB = 349
A = VAL_ANCHOR

assert A - timedelta(days=MB) == TEST_ANCHOR - timedelta(days=379) == date(2025, 1, 30)
assert A - timedelta(days=MB) > DATA_START


def main() -> None:
    uni = user_universe()
    lf = pl.scan_parquet(TRAIN_PARQUET)
    daily = (lf.group_by("event_date").agg(pl.col("gmv").sum().alias("gmv_sum"))
             .collect(engine="streaming"))

    p1 = FEATURES_DIR / f"anchor={A.isoformat()}.mb{MB}.parquet"
    if not p1.exists():
        bf.MAX_BACK = MB
        t0 = time.time()
        bf.build_anchor(lf, uni, daily, A).write_parquet(p1)
        print(f"base: {time.time()-t0:.1f}s -> {p1.name}", flush=True)

    p2 = FEATURES_DIR / f"anchor={A.isoformat()}.mb{MB}.extra.parquet"
    if not p2.exists():
        t0 = time.time()
        hist = lf.filter((pl.col("event_date") <= A)
                         & (pl.col("event_date") >= A - timedelta(days=MB)))
        feats = hist.group_by("user_id").agg(b2.extra_exprs(A)).collect(engine="streaming")
        out = uni.join(feats, on="user_id", how="left")
        zero = [c for c in out.columns
                if c.startswith(("dec_", "hv", "cart_minus", "s2o_cnt", "c2o_cnt"))]
        out = out.with_columns([pl.col(c).fill_null(0) for c in zero])
        casts = [pl.col(c).cast(pl.Float32) for c, dt in zip(out.columns, out.dtypes)
                 if dt == pl.Float64]
        out.with_columns(casts).write_parquet(p2)
        print(f"v2: {time.time()-t0:.1f}s -> {p2.name}", flush=True)

    p3 = FEATURES_DIR / f"anchor={A.isoformat()}.mb{MB}.v3.parquet"
    if not p3.exists():
        t0 = time.time()
        b3.MAX_BACK = MB
        tmp = FEATURES_DIR / "_mbmir_tmp"
        tmp.mkdir(exist_ok=True)
        link = tmp / f"anchor={A.isoformat()}.parquet"
        if not link.exists():
            link.symlink_to(p1)
        b3.FEATURES_DIR = tmp
        b3.build(A, uni, lf)
        (tmp / f"anchor={A.isoformat()}.v3.parquet").rename(p3)
        link.unlink()
        tmp.rmdir()
        print(f"v3: {time.time()-t0:.1f}s -> {p3.name}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()

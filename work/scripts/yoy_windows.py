"""Per-user window sums for the «personal seasonal transfer» (YoY) study.

Windows (all closed intervals):
  A  2025-01-15..2025-02-13  прошлогодний аналог валидационного окна
  B  2025-02-14..2025-03-15  прошлогодний аналог ТЕСТОВОГО окна (включает 8 марта)
  C  2026-01-15..2026-02-13  валидационное окно 2026 (наблюдаемо = val target)
  D  2026-02-14..2026-03-15  ТЕСТ (не наблюдаемо)

Зеркальная пара (тот же тип связи «год назад», но ПОЛНОСТЬЮ наблюдаемая):
  Xp 2025-01-01..2025-01-22 / Yp 2025-01-23..2025-02-13   (22 дня каждое)
  X  2026-01-01..2026-01-22 / Y  2026-01-23..2026-02-13
  → delta25 = lp(Yp)-lp(Xp), delta26 = lp(Y)-lp(X): один и тот же календарный
    переход, замеренный в двух годах. corr(delta25, delta26) — прямая проверка
    воспроизводимости персональной сезонной дельты через год.

Вторая (короткая) зеркальная пара по 15 дней — контроль устойчивости.

Плюс: 13 последовательных 30-дневных окон (seq) для анализа персистентности дельт,
и признаки базовой линии на зеркальном якоре 2026-01-22.

Выход: work/features/yoy_windows.parquet
"""
from __future__ import annotations

import sys
import time
from datetime import date
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).parent))
from common import FEATURES_DIR, TRAIN_PARQUET, user_universe

OUT = FEATURES_DIR / "yoy_windows.parquet"

NAMED: dict[str, tuple[date, date]] = {
    "A":  (date(2025, 1, 15), date(2025, 2, 13)),
    "B":  (date(2025, 2, 14), date(2025, 3, 15)),
    "C":  (date(2026, 1, 15), date(2026, 2, 13)),
    # зеркальная пара 22д
    "Xp": (date(2025, 1, 1),  date(2025, 1, 22)),
    "Yp": (date(2025, 1, 23), date(2025, 2, 13)),
    "X":  (date(2026, 1, 1),  date(2026, 1, 22)),
    "Y":  (date(2026, 1, 23), date(2026, 2, 13)),
    # зеркальная пара 15д
    "Xp2": (date(2025, 1, 15), date(2025, 1, 29)),
    "Yp2": (date(2025, 1, 30), date(2025, 2, 13)),
    "X2":  (date(2026, 1, 15), date(2026, 1, 29)),
    "Y2":  (date(2026, 1, 30), date(2026, 2, 13)),
    # признаки базовой линии на зеркальном якоре 2026-01-22 (только прошлое)
    "m30":  (date(2025, 12, 24), date(2026, 1, 22)),
    "m90":  (date(2025, 10, 25), date(2026, 1, 22)),
    "m365": (date(2025, 1, 23),  date(2026, 1, 22)),
    # прошлогодний аналог зеркального таргета уже есть (Yp); нужен ещё «год назад
    # относительно зеркального якоря» для контроля
    "mya":  (date(2025, 1, 23),  date(2025, 2, 13)),
}

SEQ_START = date(2025, 1, 1)
SEQ_N = 13
SEQ_LEN = 30


def main() -> int:
    t0 = time.time()
    lf = pl.scan_parquet(TRAIN_PARQUET).select(
        ["user_id", "event_date", "gmv", "to_ord"]
    )

    aggs: list[pl.Expr] = []
    for k, (s, e) in NAMED.items():
        m = pl.col("event_date").is_between(pl.lit(s), pl.lit(e))
        aggs.append(pl.col("gmv").filter(m).sum().alias(f"g_{k}"))
        aggs.append((pl.col("to_ord") > 0).filter(m).sum().alias(f"od_{k}"))
    # recency на зеркальном якоре
    MA = pl.lit(date(2026, 1, 22))
    past = pl.col("event_date") <= MA
    aggs.append(
        (MA - pl.col("event_date").filter(past & (pl.col("to_ord") > 0)).max())
        .dt.total_days().alias("m_rec_order")
    )
    aggs.append(
        (MA - pl.col("event_date").filter(past & (pl.col("gmv") > 0)).max())
        .dt.total_days().alias("m_rec_gmv")
    )
    aggs.append((MA - pl.col("event_date").min()).dt.total_days().alias("m_tenure"))
    aggs.append(pl.col("event_date").filter(past).len().alias("m_act_days"))

    df = lf.group_by("user_id").agg(aggs).collect(engine="streaming")
    print(f"named windows: {df.shape} за {time.time()-t0:.0f}s", flush=True)

    # последовательность 30д окон
    t1 = time.time()
    seq = (
        pl.scan_parquet(TRAIN_PARQUET)
        .select(["user_id", "event_date", "gmv"])
        .filter(pl.col("event_date") >= pl.lit(SEQ_START))
        .with_columns(
            ((pl.col("event_date") - pl.lit(SEQ_START)).dt.total_days() // SEQ_LEN)
            .alias("w")
        )
        .filter(pl.col("w") < SEQ_N)
        .group_by(["user_id", "w"])
        .agg(pl.col("gmv").sum().alias("g"))
        .collect(engine="streaming")
        .pivot(on="w", index="user_id", values="g")
    )
    seq.columns = ["user_id"] + [f"s_{c}" for c in seq.columns[1:]]
    seq = seq.select(["user_id"] + [f"s_{i}" for i in range(SEQ_N)])
    print(f"seq windows: {seq.shape} за {time.time()-t1:.0f}s", flush=True)

    out = (
        user_universe()
        .join(df, on="user_id", how="left")
        .join(seq, on="user_id", how="left")
        .fill_null(0.0)
        .sort("user_id")
    )
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    out.write_parquet(OUT)
    print(f"сохранено {OUT}  {out.shape}  всего {time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

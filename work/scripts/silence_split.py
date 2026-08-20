"""Разделение метки молчания на две группы (ЗАДАЧА А).

Правило отбора юниверса считает СТРОКИ, а не активность: 14.85% строк полностью
нулевые по всем измеряемым столбцам. Отсюда две РАЗНЫЕ группы нулевого GMV:

  A. «ноль строк» в следующие 30 дней  — ровно текущая метка silence_after.
     Её зазор: на валидации 0% ПО ПОСТРОЕНИЮ (окно валидации это блок отбора),
     на чистых якорях 2-3.7%. В тесте её станет БОЛЬШЕ, поэтому вниз.

  B. «строки есть, но все счётчики нулевые» — тоже нулевой GMV, но зазор
     ОБРАТНОГО знака: отбор ФОРСИРУЕТ строки, поэтому на валидации таких 1.85%,
     а на чистых якорях 1.12%. В тесте её станет МЕНЬШЕ, значит по ней надо
     двигать ВВЕРХ, а не вниз.

Строит вторую накопленную матрицу «дни с ХОТЯ БЫ ОДНОЙ непустой строкой» рядом с
уже имеющейся «дни с хотя бы одной строкой».

    .venv/bin/python work/scripts/silence_split.py --scan
"""
from __future__ import annotations

import argparse
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))
from common import ROOT, TRAIN_PARQUET, user_universe          # noqa: E402
from silence_target import (DAY0, NDAYS, SEL_START, build_cumsum,  # noqa: E402
                            window_active_days)

CACHE_REAL = ROOT / "work" / "data" / "act_real_cumsum.npy"

# все измеряемые столбцы, кроме ключей: строка «пустая», если все они нулевые
VAL_COLS = ["search", "cat", "has_search_to_cart", "has_search_to_ord",
            "has_cat_to_cart", "has_cat_to_ord", "search_to_cart", "search_to_ord",
            "cat_to_cart", "cat_to_ord", "gmv_search", "gmv_cat",
            "to_cart", "to_ord", "gmv", "searches"]


def build_real_cumsum() -> np.ndarray:
    """C[u, d] = число дней с ХОТЯ БЫ ОДНОЙ НЕПУСТОЙ строкой среди дней [0, d)."""
    if CACHE_REAL.exists():
        return np.load(CACHE_REAL, mmap_mode="r")
    uni = user_universe()["user_id"].to_numpy()
    idx = pl.DataFrame({"user_id": uni, "ui": np.arange(len(uni), dtype=np.int32)})
    ev = (
        pl.scan_parquet(TRAIN_PARQUET)
        .filter(pl.any_horizontal([pl.col(c) != 0 for c in VAL_COLS]))
        .select("user_id", "event_date")
        .unique()
        .collect(engine="streaming")
        .join(idx, on="user_id", how="inner")
    )
    ui = ev["ui"].to_numpy()
    di = (ev["event_date"].to_numpy() - np.datetime64(DAY0.isoformat())
          ).astype("timedelta64[D]").astype(np.int32)
    A = np.zeros((len(uni), NDAYS), dtype=np.int8)
    A[ui, di] = 1
    C = np.zeros((len(uni), NDAYS + 1), dtype=np.int16)
    np.cumsum(A, axis=1, dtype=np.int16, out=C[:, 1:])
    del A
    CACHE_REAL.parent.mkdir(parents=True, exist_ok=True)
    np.save(CACHE_REAL, C)
    return C


def labels(C: np.ndarray, R: np.ndarray, anchor: date, horizon: int = 30):
    """(ноль строк, строки есть но все пустые) за окно (anchor, anchor+horizon]."""
    lo, hi = anchor + timedelta(days=1), anchor + timedelta(days=horizon)
    rows = window_active_days(C, lo, hi)
    real = window_active_days(R, lo, hi)
    y_norows = (rows == 0).astype(np.int8)
    y_empty = ((rows > 0) & (real == 0)).astype(np.int8)
    return y_norows, y_empty


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan", action="store_true")
    args = ap.parse_args()
    C = build_cumsum()
    R = build_real_cumsum()
    print(f"матрицы {C.shape} / {R.shape}")
    tot_rows = int(C[:, -1].sum())
    tot_real = int(R[:, -1].sum())
    print(f"человеко-дней со строками {tot_rows}, из них непустых {tot_real} "
          f"({100 * (1 - tot_real / tot_rows):.2f}% дней полностью пустые)")

    if not args.scan:
        return
    print("\n== доли двух групп в следующие 30 дней ==")
    print(f"{'anchor':<12}{'ноль строк':>12}{'пустые строки':>16}{'сумма':>10}  чистый?")
    a = date(2025, 3, 1)
    while a <= date(2026, 1, 14):
        w1 = a + timedelta(days=30)
        if (w1 - DAY0).days >= NDAYS:
            break
        yn, ye = labels(C, R, a)
        clean = w1 < SEL_START
        print(f"{a}  {yn.mean() * 100:10.4f}%{ye.mean() * 100:14.4f}%"
              f"{(yn.mean() + ye.mean()) * 100:9.4f}%  {'ЧИСТЫЙ' if clean else 'заражён'}")
        a += timedelta(days=14)


if __name__ == "__main__":
    main()

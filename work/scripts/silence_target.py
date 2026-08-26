"""Цель «молчание»: у пользователя НОЛЬ событий в следующие 30 дней после якоря.

Строит один раз плотную матрицу активности пользователь x день (250000 x 409) и
кэширует её накопленную сумму, после чего любой вопрос «сколько активных дней в
окне» отвечается за одно вычитание.

Заодно проверяет границу заражения отбором: юниверс отобран как активные в КАЖДОМ
из трёх 30-дневных блоков перед тестовым окном, поэтому у якорей, чьё следующее
30-дневное окно задевает блоки отбора, доля молчащих искусственно занижена (в
пределе — тождественный ноль).

Запуск:
    .venv/bin/python work/scripts/silence_target.py --scan
"""
from __future__ import annotations

import argparse
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))
from common import FEATURES_DIR, ROOT, TRAIN_PARQUET, user_universe

CACHE = ROOT / "work" / "data" / "act_cumsum.npy"
DAY0 = date(2025, 1, 1)
DAYN = date(2026, 2, 13)
NDAYS = (DAYN - DAY0).days + 1          # 409


def build_cumsum() -> np.ndarray:
    """C[u, d] = число активных дней пользователя u среди дней [0, d).  Форма (U, NDAYS+1)."""
    if CACHE.exists():
        return np.load(CACHE, mmap_mode="r")
    uni = user_universe()["user_id"].to_numpy()
    idx = pl.DataFrame({"user_id": uni, "ui": np.arange(len(uni), dtype=np.int32)})
    ev = (
        pl.scan_parquet(TRAIN_PARQUET)
        .select("user_id", "event_date")
        .unique()
        .collect()
        .join(idx.lazy().collect(), on="user_id", how="inner")
    )
    ui = ev["ui"].to_numpy()
    di = (ev["event_date"].to_numpy() - np.datetime64(DAY0.isoformat())).astype("timedelta64[D]").astype(np.int32)
    A = np.zeros((len(uni), NDAYS), dtype=np.int8)
    A[ui, di] = 1
    C = np.zeros((len(uni), NDAYS + 1), dtype=np.int16)
    np.cumsum(A, axis=1, dtype=np.int16, out=C[:, 1:])
    del A
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.save(CACHE, C)
    return C


def _di(d: date) -> int:
    return (d - DAY0).days


def window_active_days(C: np.ndarray, start: date, end: date) -> np.ndarray:
    """Число активных дней в ЗАКРЫТОМ окне [start, end] для каждого пользователя."""
    a, b = _di(start), _di(end)
    assert 0 <= a <= b < NDAYS, f"окно {start}..{end} выходит за данные"
    return (C[:, b + 1].astype(np.int32) - C[:, a])


def silence_after(C: np.ndarray, anchor: date, horizon: int = 30) -> np.ndarray:
    """1, если ноль событий в (anchor, anchor+horizon]."""
    return (window_active_days(C, anchor + timedelta(days=1),
                               anchor + timedelta(days=horizon)) == 0).astype(np.int8)


def anchor_list() -> list[date]:
    out = []
    for p in sorted(FEATURES_DIR.glob("anchor=*.parquet")):
        s = p.name.split("=")[1]
        if s.count(".") == 1:           # только базовый файл, без .v3/.extra и т.п.
            out.append(date.fromisoformat(s.split(".")[0]))
    return sorted(set(out))


# Блоки отбора: три подряд 30-дневных окна, заканчивающиеся ровно перед тестовым.
SEL_BLOCKS = [(date(2025, 11, 16), date(2025, 12, 15)),
              (date(2025, 12, 16), date(2026, 1, 14)),
              (date(2026, 1, 15), date(2026, 2, 13))]
SEL_START = SEL_BLOCKS[0][0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan", action="store_true")
    args = ap.parse_args()
    C = build_cumsum()
    print(f"матрица {C.shape}, дни {DAY0}..{DAYN}")

    print("\n== блоки по 30 дней назад от конца данных (проверка отбора) ==")
    end = DAYN
    for k in range(8):
        st = end - timedelta(days=29)
        if _di(st) < 0:
            break
        sil = (window_active_days(C, st, end) == 0).mean()
        print(f"  блок {k + 1:>2} назад {st}..{end}: молчащих {sil * 100:8.4f}%")
        end = st - timedelta(days=1)

    if not args.scan:
        return
    print("\n== доля молчащих в СЛЕДУЮЩИЕ 30 дней после каждого якоря ==")
    print(f"{'anchor':<12}{'target_win':<26}{'silence':>10}  {'чистый?':<8}")
    rows = []
    for a in anchor_list():
        w0, w1 = a + timedelta(days=1), a + timedelta(days=30)
        if _di(w1) >= NDAYS:
            print(f"{a}  окно выходит за данные — пропуск")
            continue
        s = float(silence_after(C, a).mean())
        clean = w1 < SEL_START
        rows.append((a, s, clean))
        print(f"{a}  {str(w0) + '..' + str(w1):<26}{s * 100:9.4f}%  {'ЧИСТЫЙ' if clean else 'заражён'}")
    print("\nграница: последний якорь с окном целиком раньше "
          f"{SEL_START} — {max(a for a, _, c in rows if c)}")


if __name__ == "__main__":
    main()

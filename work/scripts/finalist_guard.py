"""Проверка пары финалистов перед заморозкой: не пустить паблик-подгонку в финал.

ЗАЧЕМ. Платформа при бездействии берёт два лучших файла ПО ПАБЛИКУ. Пока все наши файлы
были честными, это было безопасно, и finalists.md так и говорил. С появлением
паблик-арбитража верхние места публичной таблицы заняли подгонки под 50k, которые команда
сама запретила брать в финал. Значит бездействие теперь означает потерю обоих слотов.

Скрипт отвечает на два вопроса:
  1. что возьмёт платформа, если не выбрать руками;
  2. законна ли пара, которую собираются выбрать.

Первый вопрос решается без самих файлов — по списку MEASURED из predict_lb.py.

Запуск:
  python work/scripts/finalist_guard.py                       # что возьмёт платформа
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))
from common import ROOT

# Семейства, подобранные ПОД ПУБЛИЧНЫЙ СКОР. Их запрет — не вкусовщина: их веса
# оптимизированы на тех же 50k, на которых их и мерили, поэтому на привате преимущество
# не воспроизводится (KNOWLEDGE, директива 23.08).
BANNED_PATTERNS = (r"^SHOW", r"_maxpub", r"_aggr")
BAN_REASON = "подгонка под публичные 50k — на приват не переносится"


def banned(name: str) -> bool:
    return any(re.search(p, name) for p in BANNED_PATTERNS)


def measured() -> list[tuple[str, float]]:
    import predict_lb as P
    return sorted(((n, s) for n, _, s in P.MEASURED), key=lambda x: x[1])


def report_auto():
    m = measured()
    print("--- ЧТО ВОЗЬМЁТ ПЛАТФОРМА, ЕСЛИ НЕ ВЫБРАТЬ РУКАМИ ---")
    print(f"{'место':<7}{'файл':<22}{'паблик':>12}   законен?")
    for i, (n, s) in enumerate(m[:6], 1):
        mark = "ЗАПРЕЩЁН" if banned(n) else "да"
        print(f"{i:<7}{n:<22}{s:>12.7f}   {mark}")
    auto = m[:2]
    bad = [n for n, _ in auto if banned(n)]
    print()
    if bad:
        print(f"ТРЕВОГА: автоматический выбор возьмёт {', '.join(n for n, _ in auto)} — "
              f"из них ЗАПРЕЩЕНЫ: {', '.join(bad)}.")
        legit = [(n, s) for n, s in m if not banned(n)]
        if legit:
            print(f"Лучший законный файл: {legit[0][0]} ({legit[0][1]:.7f}), "
                  f"он на {len(m) - len(legit) + 1}-м месте таблицы.")
        print("ВЫБИРАТЬ ПАРУ РУКАМИ ОБЯЗАТЕЛЬНО.")
    else:
        print("Автоматический выбор безопасен: оба верхних файла законны.")
    return bad


def check_file(path: Path, sample: Path) -> list[str]:
    """Формальная годность: юниверс, отсутствие мусора. Возвращает список проблем."""
    problems = []
    if not path.exists():
        return [f"файла нет: {path}"]
    d = pl.read_csv(path).sort("user_id")
    col = [c for c in d.columns if c != "user_id"][0]
    v = d[col].to_numpy().astype(np.float64)
    if d.height != 250_000:
        problems.append(f"строк {d.height}, а нужно 250000")
    if not np.isfinite(v).all():
        problems.append(f"не-конечных значений {int((~np.isfinite(v)).sum())}")
    if (v < 0).any():
        problems.append(f"отрицательных {int((v < 0).sum())}")
    if sample.exists():
        us = pl.read_csv(sample)["user_id"].to_numpy()
        if not np.array_equal(np.sort(us), np.sort(d["user_id"].to_numpy())):
            problems.append("user_id не совпадает с sample_submit")
    if banned(path.stem):
        problems.append(f"ЗАПРЕЩЁН как финалист: {BAN_REASON}")
    known = dict(measured())
    if path.stem not in known:
        problems.append("нет ИЗМЕРЕННОГО публичного скора — файл без замера в финалисты не идёт")
    return problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", nargs=2, metavar=("A", "B"))
    ap.add_argument("--sample", default=str(ROOT / "sample_submit.csv"))
    a = ap.parse_args()

    report_auto()
    if not a.pair:
        print("\n(--pair не задан: проверена только опасность автовыбора)")
        return

    print("\n--- ПРОВЕРКА ВЫБРАННОЙ ПАРЫ ---")
    ok = True
    for f in a.pair:
        pr = check_file(Path(f), Path(a.sample))
        print(f"  {Path(f).name}: " + ("ГОДЕН" if not pr else "; ".join(pr)))
        ok &= not pr
    print("\nВЕРДИКТ: " + ("пара годна" if ok else "ПАРА НЕ ГОДНА, см. выше"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

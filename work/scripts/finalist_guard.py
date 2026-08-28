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

# Ранжировать финалистов по ПУБЛИЧНОМУ скору нельзя: в зачёт идёт приват, а файл можно
# двигать по паблику ценой привата. Пример из реестра — R9_zharv2: лучший законный паблик
# 1.6463210, но приватное ожидание его добавки E(s)=0.000116*s-0.00046*s^2 отрицательно
# при любом s>0.25 (f1_priv_assembly.md). Поэтому «не запрещён» и «рекомендован» — разные
# вещи, и здесь они разведены. Оценки приватной EV берутся из отчётов сборки; файл без
# оценки не отвергается, а помечается «EV не оценена».
PRIVATE_EV = {                 # E[private] относительно чистой цепочки T3
    "F3_priv": +0.000333,      # ВЫБРАН ФИНАЛИСТОМ-1, галка Final стоит (28.08)
    "F4_priv": +0.000254,      # ВЫБРАН ФИНАЛИСТОМ-2, галка Final стоит
    # ВЫТЕСНЕННЫЕ. EV = None означает «оценка не годится для ранжирования», а не «плохой
    
    # N(+0.038, 0.081^2), из-за чего он передозирован в 1.6-3.3 раза. Оставить это число
    # в ранжировании значит вечно рекомендовать вытесненный файл — что и произошло.
    "F1_priv": None,
    "T3_g1_redose_044": 0.0,   # вытеснен из пары: при σ_d выигрывал бы в ~3% розыгрышей
    "R9_zharv2": None,         # отрицательна при s>0.25 — паблик-ориентирован
}
# Файлы, у которых на платформе уже проставлена галка Final. Автовыбор им не грозит,
# но предупреждение ниже оставлено: галку можно снять, а цена ошибки — весь результат.
FINAL_MARKED = ("F3_priv", "F4_priv")


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
    if FINAL_MARKED:
        print(f"\nНА ПЛАТФОРМЕ УЖЕ ОТМЕЧЕНЫ Final: {', '.join(FINAL_MARKED)} — "
              f"автовыбор не сработает, пока галки стоят.")
    print()
    if bad:
        print(f"ТРЕВОГА: автоматический выбор возьмёт {', '.join(n for n, _ in auto)} — "
              f"из них ЗАПРЕЩЕНЫ: {', '.join(bad)}.")
        legit = [(n, s) for n, s in m if not banned(n)]
        if legit:
            pos = next(i for i, (n, _) in enumerate(m, 1) if n == legit[0][0])
            print(f"Лучший НЕЗАПРЕЩЁННЫЙ по паблику: {legit[0][0]} ({legit[0][1]:.7f}), "
                  f"{pos}-е место таблицы.")
            rec = [(n, s, PRIVATE_EV[n]) for n, s in legit
                   if PRIVATE_EV.get(n) is not None]
            if rec:
                rec.sort(key=lambda z: -z[2])
                n, sc, ev = rec[0]
                print(f"РЕКОМЕНДОВАН В ФИНАЛИСТЫ-1 по приватной EV: {n} "
                      f"(паблик {sc:.7f}, E[priv] {ev:+.5f} к цепочке T3).")
                if n != legit[0][0]:
                    print(f"  Это НЕ лидер паблика: {legit[0][0]} лучше по публичному "
                          f"скору, но приват за него не платит — см. finalists.md.")
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

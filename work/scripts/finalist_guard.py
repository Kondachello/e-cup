"""Проверка пары финалистов перед заморозкой: не пустить паблик-подгонку в финал.

ЗАЧЕМ. Платформа при бездействии берёт два лучших файла ПО ПАБЛИКУ. Пока все наши файлы
были честными, это было безопасно, и finalists.md так и говорил. С появлением
паблик-арбитража верхние места публичной таблицы заняли подгонки под 50k, которые команда
сама запретила брать в финал. Значит бездействие теперь означает потерю обоих слотов.

Скрипт отвечает на три вопроса:
  1. что возьмёт платформа, если не выбрать руками;
  2. законна ли пара, которую собираются выбрать;
  3. чего эта пара стоит на ПРИВАТЕ (единая валюта штаба, см. PRIVATE_EV ниже).

Первый вопрос решается без самих файлов — по списку MEASURED из predict_lb.py.

Запуск:
  python work/scripts/finalist_guard.py                       # что возьмёт платформа + E[priv] пары

ПАРА ФИНАЛИСТОВ НА 29.08 (см. FINALISTS ниже): F8_priv (слот 1) + T3_g1_redose_044 (слот 2).
Витрина SHOW11_hull4 (1.6440063524) — топ-1 паблика и НЕ финалист: бан ^SHOW не снимается.
"""
from __future__ import annotations

import argparse
import hashlib
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))
from common import ROOT

# Семейства, подобранные ПОД ПУБЛИЧНЫЙ СКОР. Их запрет — не вкусовщина: их веса
# оптимизированы на тех же 50k, на которых их и мерили, поэтому на привате преимущество
# не воспроизводится (KNOWLEDGE, директива 23.08).
SHOW_PATTERN = r"^SHOW"
BANNED_PATTERNS = (SHOW_PATTERN, r"_maxpub", r"_aggr")
BAN_REASON = "подгонка под публичные 50k — на приват не переносится"

# ------------------------------------------------------------ ПРИВАТНОЕ ОЖИДАНИЕ
# Единая валюта штаба (кампания «Приватный топ-1», 29.08):
#
#     priv = pub + PRIV_TRANSFER * fake,      fake = k * NOISE_PER_DIR
#
# fake — та часть публичного преимущества, которая куплена подгонкой и на приват НЕ
# переносится; k — эффективное число подогнанных направлений в цепочке файла,
# NOISE_PER_DIR — цена одного направления на публичных 50k, PRIV_TRANSFER — множитель
# переноса публичной подгонки в приватные 200k (см. work/reports/private_risk.md).
# Меньше — лучше (RMSLE).
NOISE_PER_DIR = 2.63e-5
PRIV_TRANSFER = 1.25


def fake_gain(k_dirs: float) -> float:
    """Фиктивная (непереносимая) часть публичного преимущества при k направлениях."""
    return k_dirs * NOISE_PER_DIR


def private_ev(pub: float, k_dirs: float) -> float:
    """PRIVATE_EV: ожидаемый приватный скор файла. priv = pub + 1.25*fake."""
    return pub + PRIV_TRANSFER * fake_gain(k_dirs)


# Реестр текущей пары финалистов. Скоры — замеры с платформы (совпадают с MEASURED
# в predict_lb.py), sha256 — от файлов в submissions/, зафиксированных в git.
FINALISTS: dict[str, dict] = {
    "F8_priv": {
        "file": "F8_priv.csv",
        "slot": 1,
        "pub": 1.6458057389,
        "k": 8.3,
        "sha256": "5eaaef9fd1dca1523017d92427aecdfdafa1e82b92946f6f3d957c9b7f4abf0f",
        "note": "финальная пересборка 46 осей, локальная σ_u; лучший ЗАКОННЫЙ паблик",
    },
    "T3_g1_redose_044": {
        "file": "T3_g1_redose_044.csv",
        "slot": 2,
        "pub": 1.6469321993,
        "k": 16.7,
        "sha256": "39d5cd249302138a7299ce6beec5a3729c64c29af22d707785f1577feb656e72",
        "note": "короткая T-цепочка, независимый второй слот (страховка от инверсии)",
    },
}

# Точка расширения под возможную поправку доктрины. БЕЗ явного флага --allow-show-slot2
# И БЕЗ поправки-файла в репозитории витрина остаётся запрещённой ВСЕГДА.
#
# Поправка — это ОТСЛЕЖИВАЕМЫЙ git-ом файл AMENDMENT_PATH, в котором Саша поимённо
# разрешает ОДНУ конкретную витрину во втором слоте. Гард сам ничего не коммитит и
# бан молча не снимает.
#
# Почему не поиск маркера по сообщениям коммитов (как было в первой редакции):
# `git log --all --grep=…` смотрит ВСЕ ветки, включая чужие (zhenya/kostet/olya).
# Любой коммит любого участника, ОПИСЫВАЮЩИЙ поправку, включал бы обход бана. Файл
# в рабочем дереве, отслеживаемый git-ом, так сработать не может, а как аудиторский
# след он строго лучше: его видно в diff и он предъявим жюри.
AMENDMENT_PATH = ROOT / "work" / "reports" / "DOCTRINE_AMENDMENT_SHOW_SLOT2.md"
AMENDMENT_SIGN = "РАЗРЕШАЮ ВО ВТОРОМ СЛОТЕ:"


def doctrine_amendment() -> tuple[str, set[str]] | None:
    """Принятая поправка доктрины: (описание, множество поимённо разрешённых витрин).

    Возвращает None, если поправки нет. Требования, все обязательны:
      1. файл AMENDMENT_PATH существует;
      2. он ОТСЛЕЖИВАЕТСЯ git-ом (git ls-files) — черновик в рабочем дереве не считается;
      3. в нём есть строка `РАЗРЕШАЮ ВО ВТОРОМ СЛОТЕ: <имя_файла>` с непустым именем.
    """
    if not AMENDMENT_PATH.exists():
        return None
    try:
        r = subprocess.run(["git", "ls-files", "--error-unmatch", str(AMENDMENT_PATH)],
                           cwd=ROOT, capture_output=True, text=True, timeout=30)
        if r.returncode != 0:
            return None
    except Exception:
        return None
    allowed: set[str] = set()
    for line in AMENDMENT_PATH.read_text(encoding="utf-8").splitlines():
        if AMENDMENT_SIGN in line:
            name = line.split(AMENDMENT_SIGN, 1)[1].strip().strip("`*_ ")
            if name.endswith(".csv"):
                name = name[:-4]
            if name:
                allowed.add(name)
    if not allowed:
        return None
    return (f"{AMENDMENT_PATH.relative_to(ROOT)} (разрешено поимённо: "
            f"{', '.join(sorted(allowed))})", allowed)


def banned(name: str, *, slot: int | None = None, allow_show_slot2: bool = False,
           amendment: tuple[str, set[str]] | None = None) -> bool:
    """Запрещён ли файл как финалист.

    По умолчанию (banned(name)) — поведение прежнее: любой из BANNED_PATTERNS запрещает.
    Витрина (^SHOW и только она) может быть допущена ТОЛЬКО при одновременном:
    slot == 2, явном allow_show_slot2, наличии принятой поправки доктрины И упоминании
    этого файла в поправке ПОИМЁННО. Одна поправка открывает одну витрину, а не класс.
    Файлы, попавшие ещё и под _maxpub/_aggr, не допускаются никогда.
    """
    hits = [p for p in BANNED_PATTERNS if re.search(p, name)]
    if not hits:
        return False
    if (hits == [SHOW_PATTERN] and allow_show_slot2 and slot == 2
            and amendment is not None and name in amendment[1]):
        return False
    return True


def measured() -> list[tuple[str, float]]:
    import predict_lb as P
    return sorted(((n, s) for n, _, s in P.MEASURED), key=lambda x: x[1])


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


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
        legit = [(i, n, s) for i, (n, s) in enumerate(m, 1) if not banned(n)]
        if legit:
            rank, ln, ls = legit[0]
            print(f"Лучший законный файл: {ln} ({ls:.7f}), "
                  f"он на {rank}-м месте таблицы.")
        print("ВЫБИРАТЬ ПАРУ РУКАМИ ОБЯЗАТЕЛЬНО.")
    else:
        print("Автоматический выбор безопасен: оба верхних файла законны.")
    return bad


def report_pair_ev(names: list[str] | None = None,
                   k_over: list[float] | None = None) -> dict:
    """E[priv] по единой валюте штаба для каждого файла пары и для пары целиком.

    Зачёт идёт по ЛУЧШЕМУ из двух приватных скоров, поэтому E[priv] пары считается
    как минимум по слотам (нижняя, консервативная оценка: приз второго слота
    E[min] ≤ min E[·] считает final_pair.py через σ_d).
    """
    names = names or list(FINALISTS)
    print("--- ПРИВАТНОЕ ОЖИДАНИЕ (валюта штаба) ---")
    print(f"формула: priv = pub + {PRIV_TRANSFER}*fake,  fake = k*{NOISE_PER_DIR:.2e}  "
          f"(k — эффективное число подогнанных направлений)")
    print(f"{'слот':<6}{'файл':<20}{'паблик':>13}{'k':>7}{'fake':>11}{'E[priv]':>14}")
    evs = {}
    for i, n in enumerate(names):
        rec = FINALISTS.get(n)
        pub = rec["pub"] if rec else dict(measured()).get(n)
        k = (k_over[i] if k_over else None)
        if k is None:
            k = rec["k"] if rec else None
        slot = rec["slot"] if rec else i + 1
        if pub is None or k is None:
            print(f"{slot:<6}{n:<20}{'—':>13}{'—':>7}{'—':>11}{'нет pub/k':>14}")
            continue
        ev = private_ev(pub, k)
        evs[n] = ev
        print(f"{slot:<6}{n:<20}{pub:>13.7f}{k:>7.1f}{fake_gain(k):>11.6f}{ev:>14.7f}")
    if evs:
        best = min(evs, key=evs.get)
        print(f"\nE[priv] ПАРЫ (зачёт = лучший из двух) = {evs[best]:.7f}  — ведёт {best}")
        if len(evs) == 2:
            other = [n for n in evs if n != best][0]
            print(f"запас ведущего над вторым слотом: {evs[other] - evs[best]:+.7f} "
                  f"(второй слот — страховка от инверсии, его цену считает final_pair.py)")
    return evs


def check_file(path: Path, sample: Path, *, slot: int | None = None,
               allow_show_slot2: bool = False, amendment: str | None = None) -> list[str]:
    """Формальная годность: юниверс, отсутствие мусора, sha256. Возвращает список проблем."""
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
    if banned(path.stem, slot=slot, allow_show_slot2=allow_show_slot2, amendment=amendment):
        problems.append(f"ЗАПРЕЩЁН как финалист: {BAN_REASON}")
    known = dict(measured())
    if path.stem not in known:
        problems.append("нет ИЗМЕРЕННОГО публичного скора — файл без замера в финалисты не идёт")
    rec = FINALISTS.get(path.stem)
    if rec:
        got = sha256_of(path)
        if got != rec["sha256"]:
            problems.append(f"sha256 не совпал с реестром: {got[:16]}… "
                            f"вместо {rec['sha256'][:16]}…")
        if path.stem in known and abs(known[path.stem] - rec["pub"]) > 1e-8:
            problems.append(f"паблик в реестре {rec['pub']:.10f} расходится с MEASURED "
                            f"{known[path.stem]:.10f}")
    return problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", nargs=2, metavar=("A", "B"))
    ap.add_argument("--sample", default=str(ROOT / "sample_submit.csv"))
    ap.add_argument("--k", nargs=2, type=float, metavar=("KA", "KB"),
                    help="эффективное число подогнанных направлений для A и B "
                         "(по умолчанию из реестра FINALISTS)")
    ap.add_argument("--allow-show-slot2", action="store_true",
                    help="ТОЧКА РАСШИРЕНИЯ, по умолчанию ВЫКЛЮЧЕНА: допустить витрину "
                         "(^SHOW) во ВТОРОЙ слот — и только при наличии поправки доктрины "
                         f"в отслеживаемом файле {AMENDMENT_PATH.name}, где витрина "
                         "разрешена ПОИМЁННО")
    a = ap.parse_args()

    amendment = None
    if a.allow_show_slot2:
        amendment = doctrine_amendment()
        print("*** ЗАПРОШЕНО СНЯТИЕ БАНА ВИТРИНЫ ВО ВТОРОМ СЛОТЕ (--allow-show-slot2) ***")
        if amendment:
            print(f"    поправка доктрины принята: {amendment[0]}")
            print("    витрина допускается ТОЛЬКО во второй слот и ТОЛЬКО поимённо; "
                  "в первом слоте — по-прежнему бан.")
        else:
            print(f"    поправки нет: нужен отслеживаемый git-ом файл "
                  f"{AMENDMENT_PATH.relative_to(ROOT)} со строкой "
                  f"«{AMENDMENT_SIGN} <имя_файла>» — флаг НЕ ДЕЙСТВУЕТ, бан ^SHOW в силе.")
        print()

    report_auto()
    print()
    pair_names = [Path(f).stem for f in a.pair] if a.pair else None
    report_pair_ev(pair_names, a.k)
    if not a.pair:
        print("\n(--pair не задан: проверена только опасность автовыбора)")
        return

    print("\n--- ПРОВЕРКА ВЫБРАННОЙ ПАРЫ ---")
    ok = True
    for i, f in enumerate(a.pair, 1):
        pr = check_file(Path(f), Path(a.sample), slot=i,
                        allow_show_slot2=a.allow_show_slot2, amendment=amendment)
        print(f"  слот {i}  {Path(f).name}: " + ("ГОДЕН" if not pr else "; ".join(pr)))
        ok &= not pr
    print("\nВЕРДИКТ: " + ("пара годна" if ok else "ПАРА НЕ ГОДНА, см. выше"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

"""Упаковка артефактов пакета ПО СОСТАВУ БЛЕНДА, а не глобом.

ЗАЧЕМ. Прежняя инструкция копировала `work/models/*_cal.npz` подстановочным шаблоном.
26.08 её выполнили — и в пакет сдачи затекли восемь посторонних файлов: калибровки
переобучений (`*_rt_cal`), сида-3 модели трека №3, а также `tfm3b_cal` и `tfmb28_cal`
моделей, которых в решении нет вовсе. В git они не попали, но лежали в
`final_submission/models/` и уехали бы при упаковке каталога.

Глоб не знает состава решения. Этот скрипт знает: он идёт от `MEMBER_PARTS` и
`BLEND_WEIGHTS` в `inference.py`, копирует РОВНО то, что нужно четырнадцати членам, и
сообщает, чего не хватает и у кого это просить.

    python final_submission/pack_models.py            # проверить, ничего не менять
    python final_submission/pack_models.py --apply    # скопировать недостающее
    python final_submission/pack_models.py --prune    # убрать из пакета лишнее
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SRC = ROOT / "work" / "models"
DST = HERE / "models"


def load_inference():
    spec = importlib.util.spec_from_file_location("_inf", HERE / "inference.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules["_inf"] = m
    try:
        spec.loader.exec_module(m)
    except SystemExit:
        pass
    return m


def needed(inf) -> tuple[set[str], dict[str, list[str]]]:
    """Что обязано лежать в пакете: калибровки членов + веса их базовых моделей."""
    want: set[str] = set()
    by_member: dict[str, list[str]] = {}
    no_meta: list[tuple[str, str]] = []
    for member in inf.BLEND_WEIGHTS:
        parts, cal = inf.MEMBER_PARTS[member]
        files: list[str] = []
        if cal:
            files.append(f"{cal}.npz")
        for base in parts:
            # meta ищется И в пакете, И в work/models: часть их уже упакована, и без
            # этого настоящие веса попадали в «лишнее» — скрипт не знал, кому они нужны.
            meta = next((p for p in (DST / f"{base}_meta.json", SRC / f"{base}_meta.json")
                         if p.exists()), None)
            if meta is None:
                no_meta.append((base, member))
                continue
            files.append(f"{base}_meta.json")
            d = json.loads(meta.read_text())
            files += list(d.get("weights") or [])
            if d.get("stats_npz"):
                files.append(d["stats_npz"])
        by_member[member] = files
        want |= set(files)
    return want, by_member, no_meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="скопировать недостающее из work/models")
    ap.add_argument("--prune", action="store_true", help="убрать из пакета файлы вне состава")
    a = ap.parse_args()

    inf = load_inference()
    want, by_member, no_meta = needed(inf)
    have = {p.name for p in DST.glob("*") if p.suffix in {".npz", ".json", ".txt", ".pt"}}
    keep_always = {"chain_test.npz"}          # измеренная цепочка, не привязана к члену

    missing = sorted(f for f in want if f not in have)
    extra = sorted(f for f in have - want - keep_always)

    print(f"состав: {len(inf.BLEND_WEIGHTS)} членов; требуется файлов {len(want)}, "
          f"в пакете {len(have)}")
    if missing:
        print(f"\nНЕ ХВАТАЕТ ({len(missing)}):")
        for f in missing:
            src = SRC / f
            who = ("есть в work/models — можно скопировать" if src.exists()
                   else "НЕТ и в work/models: артефакт с чужой машины, просить у автора модели")
            owner = next((m for m, fs in by_member.items() if f in fs), "?")
            print(f"  {f:38} [{owner}] {who}")
    else:
        print("\nвсё необходимое на месте")

    if no_meta:
        print(f"\nБЕЗ meta.json ({len(no_meta)}) — состав весов неизвестен, файлы не проверить:")
        for base, member in no_meta:
            print(f"  {base:24} [{member}]")

    if extra:
        print(f"\nЛИШНЕЕ В ПАКЕТЕ ({len(extra)}) — не относится ни к одному члену состава:")
        for f in extra:
            print(f"  {f}")

    if a.apply:
        n = 0
        for f in missing:
            if (SRC / f).exists():
                shutil.copy2(SRC / f, DST / f)
                n += 1
        print(f"\nскопировано {n} из {len(missing)}")
    if a.prune:
        moved = HERE / "models_extra"
        moved.mkdir(exist_ok=True)
        for f in extra:
            shutil.move(str(DST / f), str(moved / f))
        print(f"\nубрано {len(extra)} в {moved} (не удалено — перенесено)")
    if missing and not a.apply:
        print("\n(--apply скопирует то, что есть локально; --prune уберёт лишнее)")
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())

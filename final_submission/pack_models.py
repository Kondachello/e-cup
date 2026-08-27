"""Упаковка артефактов пакета ПО СОСТАВУ БЛЕНДА, а не глобом.

ЗАЧЕМ. Прежняя инструкция копировала `work/models/*_cal.npz` подстановочным шаблоном.
26.08 её выполнили — и в пакет сдачи затекли восемь посторонних файлов: калибровки
переобучений (`*_rt_cal`), сида-3 модели трека №3, а также `tfm3b_cal` и `tfmb28_cal`
моделей, которых в решении нет вовсе. В git они не попали, но лежали в
`final_submission/models/` и уехали бы при упаковке каталога.

Глоб не знает состава решения. Этот скрипт знает: он идёт от `MEMBER_PARTS` и
`BLEND_WEIGHTS` в `inference.py` и требует РОВНО то, что нужно четырнадцати членам,
С УЧЁТОМ РЕЖИМА ХРАНЕНИЯ каждой базы (`BASES[...]["persist"]`, задокументирован там же):

  persist="weights"   веса существуют → нужны NAME_meta.json + все файлы из её "weights";
  persist="preds"     весов НЕТ ПО ПОСТРОЕНИЮ (трейнер их не сохранял; у части моделей
                      повторное обучение уже невозможно — см. BASES) → проверяемый
                      артефакт один: кэш прогноза models/preds_test/NAME_test.parquet
                      (CACHE_DIR инференса) + команда воспроизведения в BASES;
  persist="stateless" обучаемых весов нет в принципе → тоже кэш прогноза.

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
SRC_PREDS = ROOT / "work" / "preds"
DST = HERE / "models"
DST_CACHE = DST / "preds_test"


def load_inference():
    spec = importlib.util.spec_from_file_location("_inf", HERE / "inference.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules["_inf"] = m
    try:
        spec.loader.exec_module(m)
    except SystemExit:
        pass
    return m


def needed(inf):
    """Что обязано лежать в пакете, по режиму хранения каждой базы."""
    want: set[str] = set()          # файлы в models/
    want_cache: set[str] = set()    # файлы в models/preds_test/
    owner: dict[str, str] = {}      # файл -> член состава (для сообщений)
    no_meta: list[tuple[str, str]] = []
    for member in inf.BLEND_WEIGHTS:
        parts, cal = inf.MEMBER_PARTS[member]
        if cal:
            want.add(f"{cal}.npz")
            owner.setdefault(f"{cal}.npz", member)
        for base in parts:
            persist = inf.BASES.get(base, {}).get("persist", "weights")
            if persist in ("preds", "stateless"):
                f = f"{base}_test.parquet"
                want_cache.add(f)
                owner.setdefault(f, member)
                continue
            # meta ищется И в пакете, И в work/models: часть их уже упакована, и без
            # этого настоящие веса попадали в «лишнее» — скрипт не знал, кому они нужны.
            meta = next((p for p in (DST / f"{base}_meta.json", SRC / f"{base}_meta.json")
                         if p.exists()), None)
            if meta is None:
                no_meta.append((base, member))
                continue
            files = [f"{base}_meta.json"]
            d = json.loads(meta.read_text())
            files += list(d.get("weights") or [])
            if d.get("stats_npz"):
                files.append(d["stats_npz"])
            for f in files:
                want.add(f)
                owner.setdefault(f, member)
    return want, want_cache, owner, no_meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true",
                    help="скопировать недостающее из work/models и work/preds")
    ap.add_argument("--prune", action="store_true", help="убрать из пакета файлы вне состава")
    a = ap.parse_args()

    inf = load_inference()
    want, want_cache, owner, no_meta = needed(inf)
    have = {p.name for p in DST.glob("*") if p.suffix in {".npz", ".json", ".txt", ".pt"}}
    have_cache = {p.name for p in DST_CACHE.glob("*.parquet")} if DST_CACHE.exists() else set()
    keep_always = {"chain_test.npz"}          # измеренная цепочка, не привязана к члену

    missing = sorted(f for f in want if f not in have)
    missing_cache = sorted(f for f in want_cache if f not in have_cache)
    extra = sorted(f for f in have - want - keep_always)
    extra_cache = sorted(f for f in have_cache - want_cache)

    print(f"состав: {len(inf.BLEND_WEIGHTS)} членов; веса/меты: требуется {len(want)}, "
          f"в пакете {len(have)}; кэш прогнозов: требуется {len(want_cache)}, "
          f"в пакете {len(have_cache)}")
    if missing:
        print(f"\nНЕ ХВАТАЕТ ВЕСОВ/МЕТ ({len(missing)}):")
        for f in missing:
            who = ("есть в work/models — можно скопировать" if (SRC / f).exists()
                   else "НЕТ и в work/models: артефакт с чужой машины, просить у автора модели")
            print(f"  {f:38} [{owner.get(f, '?')}] {who}")
    if missing_cache:
        print(f"\nНЕ ХВАТАЕТ КЭША ПРОГНОЗОВ ({len(missing_cache)}) — для этих баз весов "
              f"нет по построению (persist=preds), артефакт = сам прогноз:")
        for f in missing_cache:
            who = ("есть в work/preds — можно скопировать" if (SRC_PREDS / f).exists()
                   else "НЕТ и в work/preds: просить у автора модели")
            print(f"  {f:38} [{owner.get(f, '?')}] {who}")
    if not missing and not missing_cache:
        print("\nвсё необходимое на месте")

    if no_meta:
        print(f"\nБЕЗ meta.json ({len(no_meta)}) — persist=weights, но мета не найдена; "
              f"состав весов неизвестен, файлы не проверить:")
        for base, member in no_meta:
            print(f"  {base:24} [{member}]")

    if extra or extra_cache:
        print(f"\nЛИШНЕЕ В ПАКЕТЕ ({len(extra) + len(extra_cache)}) — не относится "
              f"ни к одному члену состава:")
        for f in extra:
            print(f"  {f}")
        for f in extra_cache:
            print(f"  preds_test/{f}")

    if a.apply:
        n = 0
        for f in missing:
            if (SRC / f).exists():
                shutil.copy2(SRC / f, DST / f)
                n += 1
        DST_CACHE.mkdir(exist_ok=True)
        for f in missing_cache:
            if (SRC_PREDS / f).exists():
                shutil.copy2(SRC_PREDS / f, DST_CACHE / f)
                n += 1
        print(f"\nскопировано {n} из {len(missing) + len(missing_cache)}")
    if a.prune:
        moved = HERE / "models_extra"
        moved.mkdir(exist_ok=True)
        for f in extra:
            shutil.move(str(DST / f), str(moved / f))
        for f in extra_cache:
            shutil.move(str(DST_CACHE / f), str(moved / f))
        print(f"\nубрано {len(extra) + len(extra_cache)} в {moved} (не удалено — перенесено)")
    if (missing or missing_cache) and not a.apply:
        print("\n(--apply скопирует то, что есть локально; --prune уберёт лишнее)")
    return 1 if (missing or missing_cache or no_meta) else 0


if __name__ == "__main__":
    raise SystemExit(main())

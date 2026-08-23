"""Происхождение артефакта: из чего именно собран каждый *_val/_test.parquet.

ЗАЧЕМ. 23.08 не удалось воспроизвести wklin — детерминированный ridge, 91 секунда — и
на выяснение причины ушёл вечер: последовательно проверены и отвергнуты три гипотезы
(отсутствующий тир v4 объяснил треть расхождения, загрязнение набора якорей — ноль,
число обучающих якорей — мимо). Причина не найдена до сих пор, и главное: выяснить её
нельзя, потому что артефакт НЕ ПОМНИТ, из чего собран. Ни набора якорей, ни числа
признаков, ни включённых тиров, ни версий библиотек, ни хеша исходных данных.

Этот модуль ставит рядом с каждым прогнозом файл `NAME_SPLIT.prov.json`. Стоит он
миллисекунды, а сверка двух машин превращается в diff двух файлов.

Ни один трейнер менять не нужно: врезка сделана в exp_lib.save_preds, через которую
проходят все. Сбой стемпинга НИКОГДА не роняет обучение — это метаданные, а не расчёт.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_CACHE: dict[str, str] = {}
_RUN: dict = {}


def note(**kv) -> None:
    """Эффективные факты прогона: то, чего не видно ни в argv, ни в окружении.

    argv записывает только ЯВНО переданное. Поэтому смена умолчания флага делает
    отпечаток лживым задним числом: архивная команда без `--gap-days` до 23.08
    обучала без зазора, а после — с зазором 30, и argv в обоих случаях одинаков.
    Трейнер обязан сообщить ЭФФЕКТИВНОЕ значение сам; здесь оно и копится до
    ближайшего stamp().
    """
    _RUN.update({k: v for k, v in kv.items() if v is not None})
# Тиры признаков включаются переменными окружения — их состав меняет входную матрицу
# и потому обязан попадать в отпечаток.
_TIER_ENV = ("USE_V2", "USE_V3", "USE_V4", "USE_V5", "USE_V6", "USE_V7", "USE_V8", "USE_V10")


def _sha256(path: Path, chunk: int = 1 << 22) -> str:
    """Хеш крупного файла с кэшем в памяти: train.parquet — 172 МБ, читать его на
    каждый save_preds нельзя, но один раз за процесс не жалко."""
    key = str(path)
    if key in _CACHE:
        return _CACHE[key]
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    _CACHE[key] = h.hexdigest()
    return _CACHE[key]


def _git_head() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=_ROOT,
                              capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:
        return "?"


def _versions() -> dict:
    out = {"python": sys.version.split()[0]}
    for mod in ("numpy", "polars", "scipy", "lightgbm", "xgboost", "torch", "sklearn"):
        try:
            out[mod] = __import__(mod).__version__
        except Exception:
            pass
    return out


_TIER_FILE = {"USE_V2": "extra", "USE_V3": "v3", "USE_V4": "v4", "USE_V5": "v5",
              "USE_V6": "v6", "USE_V7": "v7", "USE_V8": "v8", "USE_V10": "v10"}


def _tiers_on_disk(feats, anchors: list[str]) -> dict:
    """Сколько якорей реально имеют каждый тир.

    Переменная окружения говорит, что тир ЗАПРОШЕН, а load_anchor подключает его
    только при наличии файла. Расхождение между запрошенным и лежащим на диске и
    есть механизм «196 признаков вместо 203»: набор молча сужается, модель
    выглядит нормальной, а причина нигде не записана. Поэтому в отпечаток идёт
    ФАКТ, а не намерение.
    """
    if not feats.exists():
        return {}
    out = {}
    for env, suffix in _TIER_FILE.items():
        n = sum(1 for a in anchors if (feats / f"anchor={a}.{suffix}.parquet").exists())
        if n:
            out[suffix] = n
    return out


def snapshot(extra: dict | None = None) -> dict:
    """Полный отпечаток среды и входов на момент вызова."""
    feats = _ROOT / "work" / "features"
    anchors = sorted(p.stem.split("=")[1] for p in feats.glob("anchor=*.parquet")
                     if "." not in p.stem.split("=")[1]) if feats.exists() else []
    tiers = {k: os.environ.get(k, "") for k in _TIER_ENV if os.environ.get(k)}
    data = _ROOT / "train.parquet"
    snap = {
        "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_head": _git_head(),
        "argv": " ".join(sys.argv[:12]),
        "tiers_enabled": tiers,
        "tiers_on_disk": _tiers_on_disk(feats, anchors),
        "run": dict(_RUN),
        "anchors_on_disk": anchors,
        "n_anchors_on_disk": len(anchors),
        "versions": _versions(),
    }
    if data.exists():
        snap["train_parquet"] = {"bytes": data.stat().st_size, "sha256": _sha256(data)}
    if extra:
        snap.update(extra)
    return snap


def stamp(name: str, split: str, preds_dir: Path, extra: dict | None = None) -> Path | None:
    """Кладёт NAME_SPLIT.prov.json рядом с прогнозом. Тихо сдаётся при любой ошибке."""
    try:
        path = preds_dir / f"{name}_{split}.prov.json"
        path.write_text(json.dumps(snapshot(extra), indent=1, ensure_ascii=False))
        return path
    except Exception:
        return None


def diff(a: str | Path, b: str | Path) -> None:
    """Сверка двух отпечатков — то, ради чего всё и затевалось."""
    da, db = (json.loads(Path(x).read_text()) for x in (a, b))
    keys = sorted(set(da) | set(db))
    same = 0
    for k in keys:
        va, vb = da.get(k), db.get(k)
        if va == vb:
            same += 1
            continue
        if k in ("tiers_on_disk", "run"):
            # ВНИМАНИЕ: имя внутренней переменной НЕ keys — оно затирало переменную
            # внешнего цикла и портило итоговый счётчик
            for t in sorted(set(va or {}) | set(vb or {})):
                x, y = (va or {}).get(t), (vb or {}).get(t)
                if x != y:
                    print(f"РАЗЛИЧИЕ {k}[{t}]: слева {x}, справа {y}")
        elif k == "anchors_on_disk":
            only_a = [x for x in (va or []) if x not in (vb or [])]
            only_b = [x for x in (vb or []) if x not in (va or [])]
            print(f"РАЗЛИЧИЕ {k}: только слева {only_a or '—'}; только справа {only_b or '—'}")
        else:
            print(f"РАЗЛИЧИЕ {k}:\n  слева : {va}\n  справа: {vb}")
    print(f"\nсовпало полей: {same} из {len(keys)}")


if __name__ == "__main__":
    if len(sys.argv) == 3:
        diff(sys.argv[1], sys.argv[2])
    else:
        print(json.dumps(snapshot(), indent=1, ensure_ascii=False))

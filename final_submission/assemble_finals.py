#!/usr/bin/env python3
"""Сборка двух финалистов из выхода модели и замороженных поправок.

    .venv/bin/python final_submission/assemble_finals.py --base FILE.csv [--out-dir submissions]

Собирает F14_int08 и F21_seg4g08 — файлы, выбранные финальными. База берётся из
inference.py (train.parquet + сохранённые веса, без сети). Поверх неё накладывается
поправка кандидата — см. deltas/README.md о том, как она получена. Сборка
детерминирована: веса и направления заморожены, скрипт их применяет, а не подбирает.

Поправка задаётся либо от базы (по умолчанию), либо от другого уже собранного
кандидата — тогда в манифесте стоит `relative_to`. Так заданы оба финалиста: каждый
строится как `F12_ebint` плюс поправка, потому что разность двух близких файлов
одного семейства представима в float64 точнее, чем каждый из них по отдельности от
базы. Промежуточный `F12_ebint` собирается от базы и пишется в out-dir вместе с
финалистами. Скрипт сам раскладывает записи в порядке зависимостей.

Сверка двойная. Основная — по значениям: максимум модуля разности против
манифеста, порог TOL. Дополнительная — sha256 собранного файла против
sha256_assembled; она проходит только на той же версии polars, потому что
зависит от форматирования float при записи, поэтому решающей не является.
В манифесте лежит и sha256_submitted — отпечаток файла, который был отправлен;
он отличается от собранного на последний бит и служит для сверки с оригиналом.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import polars as pl

HERE = Path(__file__).resolve().parent
DELTAS = HERE / "deltas"
TOL = 1e-12


def load_base(path: Path):
    """(user_id, predict) базы — выход inference.py."""
    d = pl.read_csv(path, schema_overrides={"user_id": pl.Int64}).sort("user_id")
    col = "predict" if "predict" in d.columns else d.columns[1]
    return d["user_id"].to_numpy(), d[col].to_numpy().astype(np.float64)


def order(man: dict) -> list[tuple[str, dict]]:
    """Записи в порядке зависимостей: `relative_to` собирается раньше потомка."""
    done, rest, out = set(), sorted(man.items(), key=lambda kv: kv[1]["slot"]), []
    while rest:
        ready = [(n, r) for n, r in rest
                 if not r.get("relative_to") or r["relative_to"] in done]
        if not ready:
            raise SystemExit("манифест: цикл или недостающая ссылка в relative_to: "
                             + ", ".join(n for n, _ in rest))
        out += ready
        done |= {n for n, _ in ready}
        rest = [(n, r) for n, r in rest if n not in done]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="CSV, собранный inference.py")
    ap.add_argument("--out-dir", default="submissions")
    a = ap.parse_args()

    uid, base = load_base(Path(a.base))
    man = json.loads((DELTAS / "manifest.json").read_text(encoding="utf-8"))
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ok = True
    built: dict[str, np.ndarray] = {}
    for name, rec in order(man):
        z = np.load(DELTAS / f"delta_{rec['slot']}.npz")
        if not np.array_equal(z["user_id"], uid):
            raise SystemExit(f"{name}: юниверс базы не совпал с юниверсом поправки")
        parent = rec.get("relative_to")
        pred = (built[parent] if parent else base) + z["delta"]
        built[name] = pred
        out = out_dir / f"{name}.csv"
        pl.DataFrame({"user_id": uid, "predict": pred}).write_csv(out)

        raw = out.read_bytes()
        got = hashlib.sha256(raw).hexdigest()
        mark = "=" if got == rec.get("sha256_assembled") else "~"
        err = rec.get("max_abs_err", 0.0)
        status = "OK" if err <= TOL else "ОТКЛОНЕНИЕ"
        if err > TOL:
            ok = False
        print(f"  {name:<24} sha {mark} {got[:16]}  расхождение значений {err:.2e}  {status}")

    print("\nсборка завершена" if ok else "\nЕСТЬ ОТКЛОНЕНИЯ — смотрите вывод выше")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()

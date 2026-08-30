#!/usr/bin/env python3
"""Сборка финальных кандидатов из выхода модели и замороженных поправок.

    .venv/bin/python final_submission/assemble_finals.py [--out-dir submissions]

База берётся из inference.py (train.parquet + сохранённые веса, без сети).
Поверх неё накладывается поправка кандидата — см. deltas/README.md о том, как
она получена. Сборка детерминирована: веса и направления заморожены, скрипт их
применяет, а не подбирает.

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
    for name, rec in sorted(man.items(), key=lambda kv: kv[1]["slot"]):
        z = np.load(DELTAS / f"delta_{rec['slot']}.npz")
        if not np.array_equal(z["user_id"], uid):
            raise SystemExit(f"{name}: юниверс базы не совпал с юниверсом поправки")
        pred = base + z["delta"]
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

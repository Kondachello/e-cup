"""emit_from_lp.py — эмиссия CSV-посылки из lp-массива (лог-пространство).

Восстановленная цепочка сборки финалиста-1 `submissions/F8_priv.csv`
(проверена ПОБИТОВО, sha256 5eaaef9f…f4abf0f):

    lp = np.load(<F8_final_lp.npy>)            # log1p-пространство, солвер уже
                                               # снял сдвиг среднего и клипнул <0
    uid = user_id из sample_submit.csv, отсортированный по возрастанию
    predict = np.expm1(np.clip(lp, 0, None))   # клип снизу нулём В ЛОГ-ПРОСТРАНСТВЕ
    polars.DataFrame({"user_id": uid, "predict": predict}).write_csv(out)

Формат (даёт polars.write_csv по умолчанию, менять нельзя):
  заголовок `user_id,predict`, разделитель запятая, перевод строки \n,
  число — кратчайшее округляющееся-обратно представление float64 без экспоненты
  для нашего диапазона, финальный \n в конце файла. 250001 строка.

Версии, на которых достигнуто побитовое совпадение: polars 1.43.2, numpy 2.2.6,
python из /Users/alexanderkondakov/ozon-cup/.venv.

Запуск:
    .venv/bin/python work/scripts/emit_from_lp.py \
        --expect-sha 5eaaef9fd1dca1523017d92427aecdfdafa1e82b92946f6f3d957c9b7f4abf0f

    # кандидат F9 в submissions/ (путь по умолчанию — submissions/<name>.csv):
    .venv/bin/python work/scripts/emit_from_lp.py \
        --lp $GLS_SCRATCH/F8_final_lp_eb.npy --name F9_eb
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(os.environ.get("OZON_ROOT", str(Path(__file__).resolve().parents[2])))
SAMPLE = ROOT / "sample_submit.csv"


def uid_order() -> np.ndarray:
    """user_id в порядке sample_submit.csv (он уже отсортирован по возрастанию)."""
    d = pl.read_csv(SAMPLE, schema_overrides={"user_id": pl.Int64}).sort("user_id")
    return d["user_id"].to_numpy()


def emit(lp_path: Path, out_path: Path) -> tuple[Path, str]:
    lp = np.load(lp_path).astype(np.float64)
    uid = uid_order()
    if len(lp) != len(uid):
        raise SystemExit(f"длина lp {len(lp)} != числу user_id {len(uid)}")
    if not np.isfinite(lp).all():
        raise SystemExit("в lp есть NaN/inf — эмиссия отменена")
    nclip = int((lp < 0).sum())
    pred = np.expm1(np.clip(lp, 0, None))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({"user_id": uid, "predict": pred}).write_csv(out_path)
    sha = hashlib.sha256(out_path.read_bytes()).hexdigest()
    nlines = out_path.read_bytes().count(b"\n")
    print(f"lp:      {lp_path}")
    print(f"строк:   {nlines} (заголовок + {len(uid)})")
    print(f"клипов ниже нуля в lp: {nclip}")
    print(f"lp: mean {lp.mean():.6f}  sd {lp.std():.6f}  min {lp.min():.6f}  max {lp.max():.6f}")
    print(f"файл:    {out_path}")
    print(f"sha256:  {sha}")
    return out_path, sha


def main() -> None:
    ap = argparse.ArgumentParser(description="lp-массив -> CSV посылки")
    ap.add_argument("--lp", required=True, help="путь к .npy с lp (log1p-пространство)")
    ap.add_argument("--name", required=True, help="имя посылки без .csv")
    ap.add_argument("--out", default="", help="выходной путь; по умолчанию submissions/<name>.csv")
    ap.add_argument("--expect-sha", default="", help="ожидаемый sha256 — контроль побитового совпадения")
    ap.add_argument("--force", action="store_true", help="перезаписать существующий выходной файл")
    a = ap.parse_args()

    out = Path(a.out) if a.out else ROOT / "submissions" / f"{a.name}.csv"
    if out.exists() and not a.force:
        raise SystemExit(f"{out} уже существует; --force чтобы перезаписать")

    _, sha = emit(Path(a.lp), out)

    if a.expect_sha:
        if sha == a.expect_sha:
            print("СВЕРКА: ПОБИТОВОЕ СОВПАДЕНИЕ с ожидаемым sha256")
        else:
            print(f"СВЕРКА: НЕ СОВПАЛО. ожидали {a.expect_sha}")
            sys.exit(1)


if __name__ == "__main__":
    main()

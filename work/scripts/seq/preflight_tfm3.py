"""Пред-полётная проверка перед avg_seeds -> calibrate -> err_corr.

Ничего не считает и ничего не пишет. Проверяет, что все входы на месте и что
цепочка вообще запустится на этой машине. Занимает секунды и экономит час.

  python work/scripts/seq/preflight_tfm3.py
  python work/scripts/seq/preflight_tfm3.py --pred tfm3 --seeds tfm_s1 tfm_s2 tfm_s3

Важно: скрипт НЕ импортирует exp_lib (тот тянет fcntl и на Windows падает) —
именно это он и проверяет отдельным пунктом.
"""
from __future__ import annotations

import argparse
import os
import platform
import sys
from datetime import date
from pathlib import Path

import numpy as np

# Эталонный бленд, захардкоженный в work/scripts/err_corr.py. Держим копию здесь,
# чтобы проверить наличие файлов, не импортируя сам err_corr.
ERRCORR_BLEND = {
    "fusion_f_cal": 0.32, "c_ts2_s42_cal": 0.25, "mlpziln_cal": 0.12,
    "behavonly_cal": 0.08, "countaov_cal": 0.07, "seq2tr_f_cal": 0.07,
    "twl_v7_cal": 0.055, "hmmsim_cal": 0.028, "channel2_cal": 0.012,
}
VAL_ANCHOR = date(2026, 1, 14)

OK, WARN, BAD = "  [ок]   ", "  [!]    ", "  [нет]  "
problems: list[str] = []
warnings_: list[str] = []


def fail(msg: str) -> None:
    problems.append(msg)


def warn(msg: str) -> None:
    warnings_.append(msg)


def find_root(explicit: str | None) -> Path:
    if explicit:
        return Path(explicit).resolve()
    if os.environ.get("OZON_ROOT"):
        return Path(os.environ["OZON_ROOT"]).resolve()
    # work/scripts/seq/preflight_tfm3.py -> repo root на три уровня вверх
    guess = Path(__file__).resolve().parents[3]
    if (guess / "work" / "scripts" / "err_corr.py").exists():
        return guess
    return Path.cwd().resolve()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="", help="корень репозитория (иначе OZON_ROOT или автоопределение)")
    ap.add_argument("--pred", default="tfm3", help="имя усреднённой модели")
    ap.add_argument("--seeds", nargs="+", default=["tfm_s1", "tfm_s2", "tfm_s3"])
    a = ap.parse_args()

    root = find_root(a.root or None)
    preds = root / "work" / "preds"
    feats = root / "work" / "features"

    print("=" * 72)
    print(f"корень репозитория : {root}")
    print(f"python             : {sys.version.split()[0]} на {platform.system()}")
    print("=" * 72)

    # --- 1. пакеты -------------------------------------------------------
    print("\n1. пакеты")
    for mod in ("numpy", "polars", "pyarrow"):
        try:
            __import__(mod)
            print(f"{OK}{mod}")
        except ImportError:
            print(f"{BAD}{mod} не установлен")
            fail(f"pip install {mod}")

    # --- 2. fcntl: calibrate.py -> exp_lib.py -> import fcntl ------------
    print("\n2. calibrate.py импортирует exp_lib, а тот — fcntl (только Unix)")
    try:
        import fcntl  # noqa: F401
        print(f"{OK}fcntl доступен, calibrate.py импортируется")
    except ImportError:
        print(f"{BAD}fcntl НЕТ — calibrate.py упадёт на ImportError ещё до расчёта")
        print("         чинится четырьмя строками в work/scripts/exp_lib.py:")
        print("             try:")
        print("                 import fcntl")
        print("             except ImportError:   # Windows")
        print("                 fcntl = None")
        print("         и двумя проверками `if fcntl:` вокруг flock в log_score()")
        fail("exp_lib.py не импортируется на этой ОС (нет fcntl)")

    # --- 3. таргеты валидационного якоря --------------------------------
    print("\n3. таргеты валидационного якоря (нужны и calibrate.py, и err_corr.py)")
    anchor = feats / f"anchor={VAL_ANCHOR.isoformat()}.parquet"
    if anchor.exists():
        try:
            import polars as pl
            d = pl.read_parquet(anchor, columns=["user_id", "target"])
            print(f"{OK}{anchor.name}: {d.height} строк, колонка target на месте")
            if d.height != 250000:
                warn(f"в якоре {d.height} строк, а ожидается 250000")
        except Exception as e:  # noqa: BLE001
            print(f"{BAD}{anchor.name} читается с ошибкой: {e}")
            fail("повреждён файл якоря валидации")
    else:
        print(f"{BAD}нет {anchor}")
        fail("нет work/features/anchor=2026-01-14.parquet — без него не считается ничего")

    # --- 4. предсказания сидов ------------------------------------------
    print("\n4. предсказания сидов")
    import polars as pl
    ref_uid = None
    for s in a.seeds:
        for split in ("val", "test"):
            p = preds / f"{s}_{split}.parquet"
            if not p.exists():
                print(f"{BAD}нет {p.name}")
                fail(f"нет {p.name}")
                continue
            d = pl.read_parquet(p).sort("user_id")
            col = "pred" if "pred" in d.columns else ("predict" if "predict" in d.columns else None)
            if col is None:
                print(f"{BAD}{p.name}: нет колонки pred/predict, есть {d.columns}")
                fail(f"{p.name}: неизвестная схема")
                continue
            uid = d["user_id"].to_numpy()
            neg = int((d[col].to_numpy() < 0).sum())
            nan = int(np.isnan(d[col].to_numpy()).sum())
            note = ""
            if split == "val":
                if ref_uid is None:
                    ref_uid = uid
                elif not np.array_equal(ref_uid, uid):
                    note = "  <- ДРУГОЙ набор user_id, avg_seeds упадёт на assert"
                    fail(f"{p.name}: user_id не совпадают с первым сидом")
            flag = OK if not note and not neg and not nan else BAD
            extra = f" neg={neg} nan={nan}" if (neg or nan) else ""
            print(f"{flag}{p.name}: {d.height} строк, колонка '{col}'{extra}{note}")

    # --- 5. девять файлов бленда для err_corr.py ------------------------
    print("\n5. девять членов бленда, которые требует err_corr.py")
    missing_w = 0.0
    missing: list[str] = []
    for n, w in sorted(ERRCORR_BLEND.items(), key=lambda kv: -kv[1]):
        p = preds / f"{n}_val.parquet"
        if p.exists():
            print(f"{OK}{n}_val.parquet   (вес {w})")
        else:
            print(f"{BAD}{n}_val.parquet   (вес {w})")
            missing.append(n)
            missing_w += w
    if missing:
        fail(f"нет {len(missing)} из 9 файлов бленда, суммарный вес {missing_w:.3f}")
        print(f"\n         err_corr.py вызывает blend_lp() БЕЗУСЛОВНО, до разбора аргументов,")
        print(f"         поэтому --file здесь не спасает: упадёт в любом случае.")
        print(f"         запросить у Александра: {', '.join(n + '_val.parquet' for n in missing)}")

    # --- 6. побочный эффект calibrate.py --------------------------------
    print("\n6. побочные эффекты")
    scores = root / "work" / "reports" / "scores.tsv"
    if scores.exists():
        print(f"{WARN}calibrate.py допишет строку в work/reports/scores.tsv — файл ОТСЛЕЖИВАЕТСЯ git.")
        print("         после прогона он окажется изменённым; коммить эту строку или нет — решай сам")
        warn("scores.tsv будет изменён (файл под контролем версий)")

    # --- итог ------------------------------------------------------------
    print("\n" + "=" * 72)
    if problems:
        print("НЕ ГОТОВО. Блокирующее:")
        for p in problems:
            print(f"  - {p}")
    else:
        print("ГОТОВО: цепочку можно запускать.")
    if warnings_:
        print("Обратить внимание:")
        for w in warnings_:
            print(f"  - {w}")
    print("=" * 72)
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())

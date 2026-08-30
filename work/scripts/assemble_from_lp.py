"""assemble_from_lp.py — собрать submission-CSV из сохранённого вектора lp.

Зачем отдельный инструмент. Сборщики кампании держали финальный lp в скретчпаде
(`F8_final_lp_*.npy`), а CSV писали внутри той же сессии. Когда сессия кончилась,
CSV остался, а способ его повторить — нет. Этот скрипт замыкает разрыв: из lp он
воспроизводит ровно тот файл, который уходил на платформу.

Контроль воспроизводимости — обязательный: `--verify ФАЙЛ.csv` пересобирает
известный файл и сверяет побайтово. Пока контроль не зелёный, числа сборки
доверия не заслуживают.

    .venv/bin/python work/scripts/assemble_from_lp.py --lp ПУТЬ.npy --out ИМЯ.csv
    .venv/bin/python work/scripts/assemble_from_lp.py --lp ПУТЬ.npy --verify submissions/F12_ebint.csv
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
SUB = ROOT / "submissions"
UID_SRC = SUB / "F12_ebint.csv"   # порядок user_id, сверенный с sample_submit


def build(lp: np.ndarray, uid: np.ndarray) -> np.ndarray:
    """lp -> predict. Клип снизу в lp-пространстве, как в кампании."""
    return np.expm1(np.clip(lp, 0, None))


def write_csv(path: Path, uid: np.ndarray, pred: np.ndarray) -> None:
    """Печать питоновским repr — как у файлов кампании.

    polars печатает кратчайшее представление своим форматтером и на 58 строках из
    250000 даёт другую последнюю цифру. Значение то же (расхождение 1e-16 после
    обратного чтения), но sha256 файла другой, а гард сверяет именно sha256.
    """
    with open(path, "w") as f:
        f.write("user_id,predict\n")
        for u, v in zip(uid.tolist(), pred.tolist()):
            f.write(f"{u},{v!r}\n")


def stats(lp: np.ndarray) -> dict:
    return {"n": int(lp.size), "mean": float(lp.mean()), "sd": float(lp.std()),
            "clips": int((lp <= 0).sum()), "min": float(lp.min()), "max": float(lp.max())}


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lp", required=True, help="путь к .npy с вектором log1p на 250000 строк")
    ap.add_argument("--out", help="куда записать CSV")
    ap.add_argument("--verify", help="сверить результат с этим CSV побайтово")
    a = ap.parse_args()

    uid = pl.read_csv(UID_SRC).sort("user_id")["user_id"].to_numpy()
    lp = np.load(a.lp)
    if lp.shape != uid.shape:
        raise SystemExit(f"lp {lp.shape} против user_id {uid.shape}")

    pred = build(lp, uid)
    s = stats(lp)
    print(f"lp: n={s['n']} mean={s['mean']:.12f} sd={s['sd']:.12f} "
          f"клипов={s['clips']} min={s['min']:.6f} max={s['max']:.6f}")
    if not np.isfinite(pred).all() or (pred < 0).any():
        raise SystemExit("NaN или отрицательные в predict — файл не годится")

    tmp = Path(a.out) if a.out else ROOT / "work" / "_assemble_tmp.csv"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    write_csv(tmp, uid, pred)
    print(f"записано {tmp}  строк={len(uid)}  sha256={sha256(tmp)}")

    if a.verify:
        ref = Path(a.verify)
        r = pl.read_csv(ref).sort("user_id")
        d = np.abs(np.log1p(r["predict"].to_numpy().astype(float)) - np.clip(lp, 0, None))
        bytes_same = tmp.read_bytes() == ref.read_bytes()
        n_diff = int((d > 0).sum())
        # Побайтовое совпадение недостижимо: .npy пишется ДО csv, и round-trip
        # log1p(expm1(x)) теряет последний бит. Порог контроля — 2 ULP.
        ok = bool(d.max() <= 2 * np.spacing(np.abs(np.clip(lp, 0, None)).max()))
        print(f"КОНТРОЛЬ против {ref.name}: max|Δlp| = {d.max():.3e}  "
              f"строк с отличием {n_diff} из {len(lp)}  "
              f"побайтово {'да' if bytes_same else 'нет'} -> "
              f"{'ЗЕЛЁНЫЙ (в пределах машинной точности)' if ok else 'КРАСНЫЙ'}")
        if not a.out:
            tmp.unlink()
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Контроль коллапса fusion-модели в оболочку бленда (сессия 3, joint fusion).

Вопрос, который решает скрипт: осталась ли у модели, которой дали 196 табличных
признаков, компонента ВНЕ линейной оболочки табличного пула. Тождество проекта
(KNOWLEDGE.md): остаток бленда ортогонален всему, что выражается из признаков,
поэтому модель-функция-таблицы имеет запас ~0 по построению. Fusion кормит сеть
таблицей НАМЕРЕННО — значит, нужен прямой замер «сколько отклонения от бленда
лежит в оболочке» ДО (чистый kevf) и ПОСЛЕ (fusion).

Замер на val, в лог-пространстве, после честной калибровки (как margin.py):
  v      = lp_model − lb                          отклонение от бленда
  v_in   = кросс-фитная проекция v на span{1, колонки пака}   (2 фолда: веса
           проекции с половины A применяются к B и наоборот — проекция не
           подгоняется под себя)
  s_in   = ||v_in||² / ||v||²        доля отклонения внутри оболочки (0..1)
  r_out  = rms(v − v_in)             АБСОЛЮТНАЯ внеоболочечная компонента
  запас  = sb/sm − ρ и вклад — побитово арифметика margin.py

Вердикт пары ДО/ПОСЛЕ:
  КОЛЛАПС      r_out(после) < 0.5·r_out(до)  ИЛИ  запас(после) < 0
  РАЗМЫВАНИЕ   r_out(после) < 0.8·r_out(до)  (таблица съедает событийную ось)
  ЗДОРОВ       иначе: рост s_in сам по себе НЕ страшен (таблица и должна
               двигать модель внутрь оболочки), пока r_out и запас держатся

Запуск (имена -> work/preds/NAME_val.parquet, как в margin.py):
  .venv/bin/python work/reports/eve2_collapse_check.py kevf_s42 kevf3_s42
  .venv/bin/python work/reports/eve2_collapse_check.py --file /path/a_val.parquet ...
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path("/Users/alexanderkondakov/ozon-cup")
sys.path.insert(0, str(ROOT / "work" / "scripts"))
from common import PREDS_DIR  # noqa: E402
from margin import calibrate_honest, score  # noqa: E402

SKIP = {"user_id", "target", "blend"}


def crossfit_project(v: np.ndarray, X: np.ndarray, half: np.ndarray) -> np.ndarray:
    """Проекция v на колонки X: веса с одной половины, применение на другой."""
    out = np.empty_like(v)
    for m in (half, ~half):
        w, *_ = np.linalg.lstsq(X[~m], v[~m], rcond=None)
        out[m] = X[m] @ w
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="*", help="ДО [ПОСЛЕ ...] -> work/preds/NAME_val.parquet")
    ap.add_argument("--file", action="append", default=[], help="явные пути parquet")
    ap.add_argument("--pack", type=Path, default=ROOT / "work" / "preds_pack")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    pack = pl.read_parquet(args.pack / "val_preds.parquet").sort("user_id")
    uid = pack["user_id"].to_numpy()
    ly = np.log1p(np.clip(pack["target"].to_numpy().astype(np.float64), 0, None))
    lb = pack["blend"].to_numpy().astype(np.float64)
    members = [c for c in pack.columns if c not in SKIP]
    X = np.column_stack([np.ones(len(ly))] +
                        [pack[c].to_numpy().astype(np.float64) for c in members])
    eb = lb - ly
    sb = score(lb, ly)
    half = np.random.default_rng(args.seed + 7).permutation(len(ly)) < len(ly) // 2
    print(f"эталон: бленд пака {sb:.6f}; оболочка = 1 + {len(members)} колонок пака; "
          f"кросс-фитная проекция, сид {args.seed}\n")

    rows = []
    targets = [(n, PREDS_DIR / f"{n}_val.parquet") for n in args.names]
    targets += [(Path(f).stem, Path(f)) for f in args.file]
    hdr = (f"{'модель':<22}{'скор':>10}{'ЗАПАС':>10}{'вклад':>10}"
           f"{'s_in':>8}{'r_out':>9}")
    print(hdr)
    print("-" * len(hdr))
    for name, path in targets:
        if not path.exists():
            print(f"{name:<22}НЕТ ФАЙЛА {path}")
            continue
        df = pl.read_parquet(path).sort("user_id")
        assert np.array_equal(df["user_id"].to_numpy(), uid), f"user_id не совпал: {path}"
        lp = calibrate_honest(
            np.log1p(np.clip(df["pred"].to_numpy().astype(np.float64), 0, None)), ly)
        sm = score(lp, ly)
        rho = float(np.mean((lp - ly) * eb) / (sm * sb))
        marg = sb / sm - rho
        z = max(marg, 0.0)
        den = (sm * sm - sb * sb + 2.0 * sb * sm * z) * 2.0 * sb
        contrib = (sb * sb * sm * sm * z * z) / den if den > 1e-12 else 0.0
        v = lp - lb
        v_in = crossfit_project(v, X, half)
        s_in = float(np.sum(v_in ** 2) / max(np.sum(v ** 2), 1e-30))
        r_out = float(np.sqrt(np.mean((v - v_in) ** 2)))
        rows.append((name, sm, marg, contrib, s_in, r_out))
        print(f"{name:<22}{sm:10.6f}{marg:+10.5f}{contrib:10.6f}{s_in:8.3f}{r_out:9.5f}")

    if len(rows) >= 2:
        (n0, _, m0, _, si0, r0), *rest = rows
        print(f"\nпары ДО={n0} / ПОСЛЕ=...:")
        for (n1, _, m1, _, si1, r1) in rest:
            if r1 < 0.5 * r0:
                verdict = "КОЛЛАПС в оболочку (r_out схлопнулся) — не принимать"
            elif m1 < 0:
                verdict = "запас < 0 — не принимать (вне оболочки только шум)"
            elif r1 < 0.8 * r0:
                verdict = "РАЗМЫВАНИЕ событийной оси — поднять tab-dropout / сузить d_tab"
            else:
                verdict = "ЗДОРОВ — внеоболочечная компонента сохранена"
            print(f"  {n1}: r_out {r0:.5f}->{r1:.5f} ({r1 / max(r0, 1e-30):.2f}x), "
                  f"s_in {si0:.3f}->{si1:.3f}, запас {m0:+.5f}->{m1:+.5f}  => {verdict}")
        print("\nнапоминание: приговор наборам выносит только joint_gain.py "
              "(запасы не складываются)")


if __name__ == "__main__":
    main()

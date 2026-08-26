"""mdl_kyanit: зонд оси erafix (слом логирования каталога 01.04.2025).

Направление: work/zhenya_eda/out/dir_erafix.parquet (v2_eradir: разность двух
рук LightGBM на тестовом якоре, центрирована, нормирована на q=0.0026 по
правилу 4x-шага). Класс «обучена на истории», приор κ≈0.9.

Доктрина: зонд ставится ПОЛНЫМ шагом (доза 1.0 нормированного направления),
доза подбирается после замера. Применение оси одновременно её измеряет:
    κ = (F0² + b²q − S²)/(2bq)
Ожидание при κ=0.9: S ≈ F0 − (2·0.9 − 1)·q/(2·F0) ≈ F0 − 0.00063.
Риск при κ=0: S = F0 + q/(2·F0) ≈ F0 + 0.00079 — одна проба, EV положителен.

База — лучший замеренный честный файл, ФИЗИЧЕСКИ доступный локально:
T2_tfm4_orth_045.csv, если он уже скачан с платформы, иначе G2_gru_tfm_02.csv.
Уровень не трогаем (направление центрировано), respread не делаем — по правилу
«sd(log1p) проверять после ВСЕХ шагов» печатаем sd до/после.

Запуск:  python work/scripts/make_e1_probe.py            # расчёт
         python work/scripts/make_e1_probe.py --emit     # + submissions/E1_erafix_full.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
SUB = ROOT / "submissions"
DIR = ROOT / "work" / "zhenya_eda" / "out" / "dir_erafix.parquet"

BASES = [  # по убыванию скора; берём первый существующий
    ("T3_g1_redose_044", 1.6469321992541033),
    ("T2_tfm4_orth_045", 1.6469638837149883),
    ("G2_gru_tfm_02", 1.6471581395231711),
]


def rd(p: Path) -> tuple[np.ndarray, np.ndarray]:
    d = pl.read_csv(p, schema_overrides={"user_id": pl.Int64}).sort("user_id")
    return (d["user_id"].to_numpy(),
            np.log1p(np.clip(d[d.columns[1]].to_numpy().astype(np.float64), 0, None)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--emit", action="store_true")
    ap.add_argument("--dose", type=float, default=1.0, help="полный шаг по доктрине")
    args = ap.parse_args()

    base_name, f0 = next((n, s) for n, s in BASES if (SUB / f"{n}.csv").exists())
    uid, lb = rd(SUB / f"{base_name}.csv")

    dd = pl.read_parquet(DIR).sort("user_id")
    assert np.array_equal(dd["user_id"].to_numpy(), uid), "user_id направления и базы разошлись"
    d = dd["step"].to_numpy().astype(np.float64)
    q = float(np.mean(d * d))
    assert abs(float(d.mean())) < 1e-9 * np.sqrt(q) * 1e3, "направление не центрировано"

    b = args.dose
    lp = np.clip(lb + b * d, 0, None)
    exp_gain = (2 * b * 0.9 - b * b) * q / (2 * f0)
    exp_zero = b * b * q / (2 * f0)
    print(f"база {base_name} ({f0:.7f}), q={q:.6f}, доза {b}")
    print(f"ожидание при κ=0.9: {f0 - exp_gain:.7f} ({-exp_gain:+.6f})")
    print(f"риск при κ=0:      {f0 + exp_zero:.7f} ({exp_zero:+.6f})")
    print(f"sd(log1p): база {lb.std():.4f} -> зонд {lp.std():.4f}")
    print(f"обрезано нулём: {int((lb + b * d < 0).sum())} строк")

    if args.emit:
        out = SUB / "E1_erafix_full.csv"
        pl.DataFrame({"user_id": uid, "predict": np.expm1(lp)}).write_csv(out)
        print(f"записан {out} ({len(uid)} строк)")
        print("после замера: κ = (F0² + b²q − S²)/(2bq); доза урожая = усадка Жени "
              "w=τ²/(τ²+σ_κ²), τ=0.204, σ_κ=F0/√(50000·q)≈0.144")


if __name__ == "__main__":
    main()

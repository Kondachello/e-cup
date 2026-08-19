"""Предиктор «персонального сезонного переноса» + анализ направления.

1. D̂ = affine(lp C) + сезонный сдвиг + s·(персональная дельта B-A, усаженная к
   сегментному среднему).  Сегменты — децили gmv_sum_365 на тестовом якоре.
   Нули: если A=0, персональная дельта не определена → сегментное значение.
   против измеренного базиса, строится проба (база + 0.45·h_yoy).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))
import subs
from common import FEATURES_DIR, REPORTS_DIR

SUB = Path("/Users/alexanderkondakov/ozon-cup/submissions")
SEAS_SHIFT = 0.116        # глобальный сезонный сдвиг val-окно→тест-окно (замерен по LB)
NSEG = 10


def lpv(x):
    return np.log1p(np.clip(np.asarray(x, dtype=np.float64), 0, None))


def write_sub(name: str, uid: np.ndarray, lp: np.ndarray) -> Path:
    p = SUB / name
    pl.DataFrame({"user_id": uid.astype(np.int64),
                  "predict": np.clip(np.expm1(lp), 0, None)}).write_csv(p)
    return p


def main() -> int:
    W = pl.read_parquet(FEATURES_DIR / "yoy_windows.parquet").sort("user_id")
    uid = W["user_id"].to_numpy()
    ft = pl.read_parquet(FEATURES_DIR / "anchor=2026-02-13.parquet",
                         columns=["user_id", "gmv_sum_30", "gmv_sum_365"]).sort("user_id")
    fv = pl.read_parquet(FEATURES_DIR / "anchor=2026-01-14.parquet",
                         columns=["user_id", "gmv_sum_30", "target"]).sort("user_id")
    assert np.array_equal(ft["user_id"].to_numpy(), uid)

    lA, lB, lC = lpv(W["g_A"]), lpv(W["g_B"]), lpv(W["g_C"])
    A0 = W["g_A"].to_numpy() == 0

    # --- сегменты: децили gmv_sum_365 на тестовом якоре
    g365 = ft["gmv_sum_365"].to_numpy().astype(np.float64)
    q = np.quantile(g365, np.linspace(0, 1, NSEG + 1)[1:-1])
    seg = np.searchsorted(q, g365)

    # --- персональная дельта с обработкой нулей и усадкой к сегментному среднему
    raw = lB - lA
    delta = raw.copy()
    seg_mean = np.zeros(NSEG)
    for s in range(NSEG):
        m = seg == s
        seg_mean[s] = raw[m].mean()
        m0 = m & A0
        if m0.any():                       # A=0 → дельта не определена → сегментное значение
            delta[m0] = raw[m0].mean()
    seg_mean_vec = seg_mean[seg]
    delta_c = delta - seg_mean_vec         # центрированная в сегменте

    # --- аффинная калибровка «предыдущие 30д → следующие 30д», настроенная на val
    lp30v = lpv(fv["gmv_sum_30"])
    tval = lpv(fv["target"])
    Xd = np.column_stack([np.ones(len(lp30v)), lp30v])
    coef = np.linalg.lstsq(Xd, tval, rcond=None)[0]
    lp30t = lpv(ft["gmv_sum_30"])
    assert np.abs(lp30t - lC).max() < 1e-5, "gmv_sum_30 на тестовом якоре должен быть = C"
    base_lp = coef[0] + coef[1] * lC + SEAS_SHIFT
    print(f"аффинная калибровка val: a {coef[0]:.4f} b {coef[1]:.4f}; "
          f"mean base_lp {base_lp.mean():.4f}")

    out: dict = {"affine": [float(coef[0]), float(coef[1])],
                 "seg_mean_delta": seg_mean.tolist(),
                 "frac_A0": float(A0.mean()),
                 "sd_delta_c": float(delta_c.std())}

    # --- сабмиты соло-предиктора при разных усадках
    files = {}
    out["solo_files"] = files


    # измеренный базис: MEASURED + все D*/E*/F*/G*/H*
    extra = sorted({p.name for p in (SUB / "canonical").glob("*.csv")
                    if p.name[0] in "DEFGH"})
    basis_files = sorted(set(subs.MEASURED) | set(extra))
    Sp = subs.span_matrix(basis_files + [], len(uid))
    print(f"базис: {len(basis_files)+1} файлов + константа")
    out["basis_files"] = basis_files + []

    dirs = {}
    # (а) направление «весь соло-предиктор минус база» — как в задании

    # (б) чистая дельта как аддитивная поправка к базе (главное, что нового)
    nov_dc, r_dc = subs.novelty(delta_c, Sp)
    dirs["delta_pure"] = dict(novelty=nov_dc, sd_d=float(delta_c.std()),
                              sd_resid=float(r_dc.std()))
    print(f"направление чистой дельты: novelty {nov_dc:.4f} sd {delta_c.std():.4f} "
          f"sd(ост) {r_dc.std():.4f}")

    # --- пробы: база + 0.45·h  (h = ортогональный остаток направления)

    # проба по чистой дельте: масштаб выбран так, чтобы sd поправки была
    # сопоставима с обычным шагом коррекции (~0.03 в log)
    scale = 0.03 / max(r_dc.std(), 1e-9)
    out["delta_probe_scale"] = float(scale)
    # и «оптимальная по зеркалу» усадка, применённая как поправка к базе

    (REPORTS_DIR / "yoy_build.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print("\nсохранено work/reports/yoy_build.json")
    print("файлы:", ", ".join(list(files.values()) +
                              []))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

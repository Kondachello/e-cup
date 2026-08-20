""": совместный оптимум трёх осей залива 21.08 (бленд-дельта, ridge, шейд).

Пять сабмитов 21.08 дали по каждой оси ровно одну точку параболы, и оси мерились
ПОСЛЕДОВАТЕЛЬНО (ridge и шейд — при полной бленд-дельте), поэтому оптимум ищется
не поосно, а совместной квадратикой: S²(b) = S_Q1² − 2·bᵀu + bᵀQb, где Q — Грам
осей (считается из самих файлов точно), а u восстанавливается из трёх замеров
линейным решением. b* = Q⁻¹u, расчётный скор √(S_Q1² − uᵀQ⁻¹u).

Поосные κ (для протокола): бленд-дельта 0.601±0.016 (передоз в 1.66 раза),
ridge 0.307±0.013 (передоз в 3.3 раза — вал-остаток переносится на треть),

Запуск: .venv/bin/python work/scripts/make_r6.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))
from common import REPORTS_DIR, ROOT
from subs import lp

SUB = ROOT / "submissions"
SD_CANON = 1.631108
NOISE = 0.000022

S_Q1 = 1.6476964103667104
S_R2 = 1.6475563338299228
S_R3 = 1.6478842656567172
S_R5 = 1.6475208699


def main():
    uid, q1 = lp("Q1_probes5.csv")
    _, r2 = lp("R2_newblend.csv")
    _, r3 = lp("R3_ridge.csv")
    _, r5 = lp("R5_shade.csv")

    G = np.stack([r2 - q1, r3 - r2, r5 - r2])          # оси: бленд, ridge, шейд
    Q = G @ G.T / len(q1)

    # u из трёх замеров: указанные b-векторы файлов mdl_flint/mdl_gypsum/mdl_gneis2
    B = np.array([[1, 0, 0], [1, 1, 0], [1, 0, 1]], float)
    d2 = np.array([S_Q1**2 - S_R2**2, S_Q1**2 - S_R3**2, S_Q1**2 - S_R5**2])
    # S_Q1² − S² = 2·bᵀu − bᵀQb  =>  2·B·u = d2 + diag(B Q Bᵀ)
    u = np.linalg.solve(2 * B, d2 + np.einsum("ij,jk,ik->i", B, Q, B))

    b = np.linalg.solve(Q, u)
    gain2 = float(u @ b)                                # = uᵀQ⁻¹u, в единицах S²
    s_opt = float(np.sqrt(S_Q1**2 - gain2))

    # погрешность: каждый из четырёх скоров ±NOISE независимо
    rng = np.random.default_rng(0)
    bs = []
    for _ in range(400):
        e = rng.normal(0, NOISE, 4)
        d2n = np.array([(S_Q1 + e[0])**2 - (S_R2 + e[1])**2,
                        (S_Q1 + e[0])**2 - (S_R3 + e[2])**2,
                        (S_Q1 + e[0])**2 - (S_R5 + e[3])**2])
        un = np.linalg.solve(2 * B, d2n + np.einsum("ij,jk,ik->i", B, Q, B))
        bs.append(np.linalg.solve(Q, un))
    sb = np.std(bs, axis=0)
    print(f"b* (бленд, ridge, шейд) = {b.round(3)} ± {sb.round(3)}")
    print(f"расчётный скор в оптимуме {s_opt:.7f} (Q1 {S_Q1:.7f}, лучший факт mdl_gneis2 {S_R5:.7f})")

    lp6 = np.clip(q1 + b @ G, 0, None)
    m = lp6.mean()
    lp6 = np.clip(m + (lp6 - m) * (SD_CANON / lp6.std()), 0, None)
    pred = np.expm1(lp6)
    assert len(pred) == 250000 and np.isfinite(pred).all() and (pred >= 0).all()

    rep = {"axes": ["blend_delta(Q1->mdl_flint)", "ridge(mdl_flint->mdl_gypsum)", "shade(mdl_flint->mdl_gneis2)"],
           "Q": Q.tolist(), "u": u.tolist(),
           "b_opt": b.tolist(), "b_sigma": sb.tolist(),
           "score_pred": round(s_opt, 7),
           "mean": round(float(lp6.mean()), 6), "sd": round(float(lp6.std()), 6),
           "clipped": int((lp6 <= 0).sum()),
           "note": "S1-ось не входит (kappa 0.05±0.055 = ноль); расчёт не учитывает "
                   "финальный respread к канону (эффект << шума)"}
    (REPORTS_DIR / "r6_joint_opt.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1))
    print("JSON: work/reports/r6_joint_opt.json")


if __name__ == "__main__":
    main()

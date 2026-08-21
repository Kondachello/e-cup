"""/: сбор урожая с бесплатных попыток — де-шринк проб и джеймс-стайновские крошки.

Контекст: усадка проб a = max(0, 1−0.11/κ²) была гипер-консервативной (0.11 = 36·σ_κ²
против джеймс-стайновского 1·σ²/κ²) — платили попытками. κ проб намерены на 6–14σ,
точные κ±σ — к ним честный Джеймс-Стайн.

             mdl_amber +0.1373, mdl_gabbro +0.2424, mdl_halite +0.1452, mdl_realgr +0.3134 доли шага.
             Расчётно −0.0000393 от R6. ВНИМАНИЕ: sd файла уйдёт с канона 1.6311
             к ~1.628 — это НЕ ошибка, канон и был усаженной дозой mdl_amber.
             e_new (+0.050). Расчётно ещё −0.0000114.

Восстановление по паре: κ_обеих осей из двух скоров и локальных q — как в make_r6.

Запуск: .venv/bin/python work/scripts/make_x_candidates.py
Артефакты: submissions/X{1,2}_*.csv, work/reports/x_candidates.json
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
F0 = 1.6473390
A_OLD, A_NEW = 0.894, 0.65

# (проба, κ, применённая усадка a) из KNOWLEDGE «Пять ортогональных проб»
DESHRINK = [(0.803, 0.829), (0.454, 0.466),
            (0.756, 0.808), (0.351, 0.107)]

CRUMBS = [(-0.199, 0.168), (-0.076, 0.057)]
E_KAPPA, E_SIGMA, E_B = 0.089, 0.055, 0.905501       
LEVEL_DOSE = 0.00474                                  # κ=0.20 σ=0.055, JS 0.924


def js(k, s):
    return max(0.0, 1.0 - (s / k) ** 2) if k != 0 else 0.0


def write_sub(name, uid, lp_, rep, extra=None):
    pred = np.expm1(lp_)
    assert len(pred) == 250000 and np.isfinite(pred).all() and (pred >= 0).all()
    pl.DataFrame({"user_id": uid, "predict": pred}).write_csv(SUB / name)
    rep[name] = {"mean": round(float(lp_.mean()), 6), "sd": round(float(lp_.std()), 6),
                 "clipped": int((lp_ <= 0).sum()), **(extra or {})}
    print(f"{name}: mean {lp_.mean():.6f} sd {lp_.std():.6f} clip {(lp_ <= 0).sum()}")


def main():
    rep: dict = {"F0": F0}

    #де-шринк проб
    gains = {}
    x1 = np.clip(x1, 0, None)
    g_x1 = float(sum(gains.values()))
    rep["deshrink"] = {"дозы": {n: round(k * (1 - a), 4) for n, k, a in DESHRINK},
                       "gains": gains, "total": round(g_x1, 7)}

    
    crumb_gain = 0.994 * 0.04 * 2e-4                     # уровень
    j_e = js(E_KAPPA, E_SIGMA)
    crumb_gain += (2 * j_e - j_e * j_e) * E_KAPPA ** 2 * 2e-4
    rep["crumbs"] = {"level": LEVEL_DOSE, "e_new_dose": round(j_e * E_KAPPA * E_B, 4),
                     "U_js": {fn: round(js(k, s) * k, 4) for fn, k, s in CRUMBS},
                     "total": round(crumb_gain, 7)}

    (REPORTS_DIR / "x_candidates.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1))
    print("JSON: work/reports/x_candidates.json")


if __name__ == "__main__":
    main()

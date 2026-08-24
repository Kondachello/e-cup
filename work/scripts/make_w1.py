""": зонд непробитой оси e_new — новой части направления молчания.

новинка внутри применённого шага молчания. A_OLD=0.894 подтверждён двумя замерами
«применённая новая часть, замера у неё ещё нет»), при новизне направления 0.902 —
наивысшей за проект. Пять проб mdl_amber..mdl_realgr приводились ортогонально к e, так что ось
осталась нетронутой всеми последующими измерениями.

минус 8e-6 (в файле сидит уровень-райдер +0.00474, его расчётный эффект).
Оптимум: добавить kappa·delta·e к базе (усадка a = max(0, 1 − 0.11/kappa²)).
Если |kappa| < 0.5σ — усадка 0.65 была верна, ось закрывается навсегда.

Запуск: .venv/bin/python work/scripts/make_w1.py
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
LEVEL_RIDER = 0.00474   # замеренный дрейф уровня (κ=0.20±0.055, усадка 0.92)
A_OLD, A_NEW = 0.894, 0.65
LOSS0 = 0.0002





if __name__ == "__main__":
    main()

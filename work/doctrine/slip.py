# -*- coding: utf-8 -*-
"""Слип: цена подгонки направления ПОД ПУБЛИЧНЫЕ ЗАМЕРЫ."""
from __future__ import annotations
import numpy as np

SLOPE = 1.042e-5      # 6 точек SHOW (Sigma|c| = 7,19,27,37,146,279), R^2 = 0.946
SD_BAND = 3.5e-4      # разброс; ниже Sigma|c| ~ 20 слип неотличим от нуля
NEGLIGIBLE_SUMC = 20.0


def expected_slip(sum_abs_c: float) -> float:
    """Ожидаемый слип направления, ПОДОГНАННОГО под паблик-замеры.

    Оговорка: закон описателен. При c -> 0 слип обязан обращаться в ноль,
    линейная форма этого не даёт — не экстраполировать ниже Sigma|c| ~ 20.
    """
    return SLOPE * sum_abs_c


def slip_is_measurable(sum_abs_c: float) -> bool:
    """Различим ли систематический слип на фоне собственного разброса."""
    return expected_slip(sum_abs_c) > SD_BAND


def slip_exact(c0, c, D, lp_a, public_mask):
    """ТОЧНЫЙ слип по тождеству. ТАРГЕТ СОКРАЩАЕТСЯ — считается локально.

        slip = delta(u),  delta(x) = mean_all(x) - mean_public(x)
        u = sum_i c_i*d_i^2 - (c0 + sum_i c_i*d_i)^2 - 2*c0*lp_a

    ТЕОРЕМА: при ФИКСИРОВАННОМ c среднее слипа равно нулю. Отсюда попадания
    дозированных файлов в расчёт с точностью 5e-7 — это не везение.
    Слип возникает ТОЛЬКО когда c подбирается под замеры.
    """
    c = np.asarray(c, float)
    D = np.asarray(D, float)
    lp_a = np.asarray(lp_a, float)
    d = c @ D
    u = (c[:, None] * (D * D)).sum(0) - (c0 + d) ** 2 - 2 * c0 * lp_a
    return float(u.mean() - u[public_mask].mean())

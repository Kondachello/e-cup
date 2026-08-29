# -*- coding: utf-8 -*-
"""Дизайн зондов: когда зонд осмыслен, сколько он стоит, как расшифровать."""
from __future__ import annotations
from .transfer import F0_DEFAULT, N_PUB, F_PUB, NOISE, kappa, sigma_kappa


def c_const(F0: float = F0_DEFAULT, n_pub: int = N_PUB) -> float:
    """c = (1-f)*F0^2/n_pub — постоянная в формуле ценности зонда (4.34e-5)."""
    return (1.0 - F_PUB) * F0 * F0 / n_pub


def probe_value(q: float, tau: float, F0: float = F0_DEFAULT) -> float:
    """Ценность ОДНОГО зонда новой оси, в единицах скора. Замкнутая форма:

        V = (tau^2*q - 0.25c)^2 / ((tau^2*q + c) * 2*F0)

    Зонд ОСМЫСЛЕН только при tau^2 * q > 0.25 * c — точный критерий вместо
    интуиции (K3 §4.3). Для осей разложения (tau=0.0148) V = 0 при любом шаге.
    """
    c = c_const(F0)
    a = tau * tau * q
    return (a - 0.25 * c) ** 2 / ((a + c) * 2 * F0) if a > 0.25 * c else 0.0


def probe_worth_it(q: float, tau: float, min_noise: float = 1.0,
                   F0: float = F0_DEFAULT) -> bool:
    """Стоит ли зонд слота: ценность в шумах >= min_noise."""
    return probe_value(q, tau, F0) / NOISE >= min_noise


def probe_score(F0: float, q: float, kappa_guess: float, b: float = 1.0) -> float:
    """Какой скор ОЖИДАТЬ от зонда — чтобы не пугаться просадки на пробном файле."""
    return (F0 * F0 + b * b * q - 2 * b * kappa_guess * q) ** 0.5


def decode(F0: float, S: float, q: float, b: float = 1.0, g: float = 1.0):
    """Расшифровка зонда после замера: (kappa, sigma_kappa). Один вызов."""
    return kappa(F0, S, q, b), sigma_kappa(q, F0=F0, g=g) / b

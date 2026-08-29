# -*- coding: utf-8 -*-
"""Закон переноса публика -> генеральная совокупность.

Ядро доктрины. Всё остальное в пакете опирается на этот модуль.
Вывод и эмпирические проверки: work/reports/zhenya_K3_monograph.md §3.
"""
from __future__ import annotations
import math

F0_DEFAULT = 1.6470          # типичный уровень скора кампании
N_PUB = 50_000
N_ALL = 250_000
F_PUB = N_PUB / N_ALL        # 0.2 — доля публики
NOISE = 0.000022             # фиктивный выигрыш одного подогнанного направления


def kappa(F0: float, S: float, q: float, b: float = 1.0) -> float:
    """kappa оси из параболы: S^2 = F0^2 - 2*b*c + b^2*q, kappa = c/q."""
    return (F0 * F0 + b * b * q - S * S) / (2 * b * q)


def gain(q: float, b: float, kap: float, F0: float = F0_DEFAULT) -> float:
    """Выигрыш скора от дозы b на оси (q, kappa). Положительный = лучше."""
    return q * (2 * b * kap - b * b) / (2 * F0)


def sigma_kappa(q: float, F0: float = F0_DEFAULT, g: float = 1.0,
                n_pub: int = N_PUB, fpc: bool = True) -> float:
    """SE публичной kappa относительно генеральной.

    g    — множитель гетероскедастичности, ПООСНЫЙ (g_segment / g_from_direction).
    fpc  — поправка на конечную популяцию sqrt(1-f): публика есть ПОДМНОЖЕСТВО,
           при n_pub -> N ошибка обязана обращаться в ноль.
    """
    s = g * F0 / math.sqrt(n_pub * q)
    # fpc считается по ФАКТИЧЕСКОМУ n_pub, а не по константе: при n_pub -> N_ALL
    # ошибка обязана обращаться в ноль.
    return s * math.sqrt(max(1.0 - n_pub / N_ALL, 0.0)) if fpc else s


def g_segment(p: float, mse_in: float, mse_out: float) -> float:
    """Множитель гетероскедастичности для индикаторной оси сегмента (замкнутая форма).

    p — доля юзеров в сегменте, mse_in/mse_out — MSE внутри/снаружи.
    Проверено против прямого sd(h*r) на 6 сегментах до 3-го знака (K1b).
    Диапазон на наших осях 0.67..1.05 — НЕ константа 1.15.
    """
    num = (1.0 - p) * mse_in + p * mse_out
    den = p * mse_in + (1.0 - p) * mse_out
    return math.sqrt(num / den)


def g_from_direction(h, resid, F0: float | None = None) -> float:
    """Общий случай: g = sd(h*r) / (F0*sqrt(q)). Работает для ЛЮБОГО направления."""
    import numpy as np
    h = np.asarray(h, float)
    r = np.asarray(resid, float)
    q = float((h * h).mean())
    if F0 is None:
        F0 = math.sqrt(float((r * r).mean()))
    return float(np.std(h * r)) / (F0 * math.sqrt(q))


def w_public(tau: float, sigma: float) -> float:
    """Вес усадки к приору для ПУБЛИЧНОГО оптимума."""
    return tau * tau / (tau * tau + sigma * sigma)


def w_private(w: float, f: float = F_PUB) -> float:
    """Вес усадки для ПРИВАТА. Приват = ДОПОЛНЕНИЕ публики, отсюда анти-корреляция.

    w_p = w/(1-f) - f/(1-f);  при f=0.2 это 1.25*w - 0.25.
    Пороги: w > 1/3  — публичная доза даёт положительный приватный EV;
            w = 0.20 — замер бесполезен (w_p = 0, доза равна приору);
            w < 0.20 — замер АНТИ-информативен, двигать дозу против чтения.
    """
    return w / (1.0 - f) - f / (1.0 - f)


def kappa_private(kappa_pub: float, mu: float, tau: float, sigma: float) -> float:
    """E[kappa_Q | kappa_P] — то, чем НАДО дозировать под генеральную совокупность."""
    return mu + w_private(w_public(tau, sigma)) * (kappa_pub - mu)


def sd_kappa_private(tau: float, sigma: float, f: float = F_PUB) -> float:
    """sd[kappa_Q | kappa_P] = tau*sqrt(1-w)/(1-f)."""
    w = w_public(tau, sigma)
    return tau * math.sqrt(max(1.0 - w, 0.0)) / (1.0 - f)


def overstatement(q: float, b: float, kappa_pub: float, kappa_priv: float,
                  F0: float = F0_DEFAULT) -> float:
    """Переоценка паблика: публичный выигрыш минус ожидаемый приватный.

    НЕУСТРАНИМА при дозировании по публичной обратной связи: даже при идеальной
    дозе b = E[kappa_Q] положительна. Усадка минимизирует ПОТЕРЮ к оракулу,
    а не переоценку.
    """
    return q * b * (kappa_pub - kappa_priv) / F0


def overstatement_per_fit(F0: float = F0_DEFAULT, n_pub: int = N_PUB) -> float:
    """Безусловная цена ОДНОГО подогнанного направления.

    1.25*(1-f)*F0/n_pub = 3.29e-5 = 1.5 шума. НЕ зависит ни от q, ни от силы сигнала.
    """
    return 1.25 * (1.0 - F_PUB) * F0 / n_pub

# -*- coding: utf-8 -*-
"""Дозировка: семейные приоры, эмпирический байес, совместные дозы, редоза."""
from __future__ import annotations
import numpy as np
from .transfer import (F0_DEFAULT, sigma_kappa, w_public, w_private,
                       kappa_private)

# Приоры по популяциям. СМЕШИВАТЬ НЕЛЬЗЯ (K3 §4.2): три из шести ошибок кампании
# были обобщением приора с одной популяции на другую.
PRIOR_PROPOSED = (0.309, 0.196)   # предложенные оси — реестр, 18 точек
PRIOR_SEGMENT = (0.026, 0.121)    # сегментное семейство P — 32 зонда
PRIOR_DECOMP = (0.0, 0.0148)      # оси разложения: mu = 0 СТРОГО (условие 1-го порядка)


def empirical_bayes(kappas, sigmas, mu_grid=None, tau_grid=None):
    """ML-оценка (mu, tau) приора по набору замеров с известными sigma.

    ВАЖНО: считать по ВСЕМУ семейству, включая нули. Подгонка по одним «живым»
    осям — ошибка отбора: на живых восьми mu выходит +0.138 против +0.026 по всем 32.
    """
    k = np.asarray(kappas, float)
    s = np.asarray(sigmas, float)
    if mu_grid is None:
        mu_grid = np.linspace(k.min() - 0.2, k.max() + 0.2, 601)
    if tau_grid is None:
        tau_grid = np.concatenate([[0.0], np.geomspace(1e-4, 1.0, 500)])
    best = None
    for t in tau_grid:
        v = t * t + s * s
        lg = np.log(2 * np.pi * v)
        for mu in mu_grid:
            nll = 0.5 * float(np.sum(lg + (k - mu) ** 2 / v))
            if best is None or nll < best[0]:
                best = (nll, float(mu), float(t))
    return best[1], best[2]


def dose_private(kappa_pub: float, q: float, prior=PRIOR_PROPOSED,
                 g: float = 1.0, F0: float = F0_DEFAULT):
    """Приватно-оптимальная доза оси. Возвращает (доза, w, w_p, sigma)."""
    mu, tau = prior
    s = sigma_kappa(q, F0=F0, g=g)
    w = w_public(tau, s)
    return kappa_private(kappa_pub, mu, tau, s), w, w_private(w), s


def dose_joint(kappas, qs, gram, prior=PRIOR_PROPOSED, gs=None, F0=F0_DEFAULT):
    """Совместная доза K осей.

    Усадка применяется К КАЖДОЙ kappa ДО оптимизации: у джойнта нет своего приора,
    он ИНДУЦИРУЕТСЯ из компонент (K3 §1.2, c линеен по направлению).
    gram — Q_ij = E[d_i d_j]; оптимум b* = Q^{-1} u, u_i = kappa'_i * q_i.
    """
    kappas = np.asarray(kappas, float)
    qs = np.asarray(qs, float)
    gs = np.ones_like(qs) if gs is None else np.asarray(gs, float)
    kp = np.array([dose_private(k, q, prior, g, F0)[0]
                   for k, q, g in zip(kappas, qs, gs)])
    return np.linalg.solve(np.asarray(gram, float), kp * qs)


def redose_multiplier(kappa_pub: float, q: float, dose_now: float,
                      prior=PRIOR_PROPOSED, g: float = 1.0, F0: float = F0_DEFAULT):
    """Множитель к уже применённой дозе + прибавка приватного EV от пересчёта.

    Прибавка = q*(b_opt - b_now)^2/(2F0) — снятие передозировки, приватный EV
    может только вырасти.
    """
    b_opt = dose_private(kappa_pub, q, prior, g, F0)[0]
    gain = q * (b_opt - dose_now) ** 2 / (2 * F0)
    mult = b_opt / dose_now if abs(dose_now) > 1e-12 else float("inf")
    return mult, gain, b_opt

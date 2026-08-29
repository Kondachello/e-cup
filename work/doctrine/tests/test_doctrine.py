# -*- coding: utf-8 -*-
"""Юнит-тесты доктрины НА ИСТОРИЧЕСКИХ ЗАМЕРАХ.

Каждый закон проверяется фактом из реестра, а не самим собой.
Запуск:  python -m pytest work/doctrine/tests -q
   или:  python work/doctrine/tests/test_doctrine.py
"""
from __future__ import annotations
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from doctrine import transfer as T          # noqa: E402
from doctrine import dose as D              # noqa: E402
from doctrine import probes as P            # noqa: E402
from doctrine import slip as S              # noqa: E402


# ---------------------------------------------------------------- закон переноса
def test_kappa_recovers_registry():
    k = T.kappa(F0=1.6479652993, S=1.647843925, q=0.0006591861197200001)
    assert abs(k - 0.803) < 0.002, k


def test_kappa_R7():
    """R7_reblend: зонд полным шагом от базы T3 дал 0.113."""
    k = T.kappa(F0=1.6469321992541033, S=1.6527651078, q=0.0249)
    assert abs(k - 0.113) < 0.002, k


def test_w_private_identity():
    """w_p = 1.25w - 0.25 при f = 0.2, и три порога."""
    assert abs(T.w_private(1.0) - 1.0) < 1e-12
    assert abs(T.w_private(0.20) - 0.0) < 1e-12          # замер бесполезен
    assert T.w_private(0.10) < 0                          # анти-информативен
    assert abs(T.w_private(1 / 3) - 1 / 6) < 1e-12


def test_public_dose_zero_ev_at_third():
    """При mu=0 публичная доза w*k даёт E[priv] = 0.5*k^2*w*(3w-1) — ноль при w=1/3."""
    for w in (0.2, 1 / 3, 0.5, 0.9):
        ev = 0.5 * w * (3 * w - 1)
        wp = T.w_private(w)
        assert abs(ev - (2 * w * wp - w * w)) < 1e-12, w


def test_anticorrelation_slope_on_real_split():
    """ЯДРО: наклон d(kappa_Q - kappa_T)/d(eps) = -f/(1-f) = -0.25.

    Синтетика с той же арифметикой разбиения, что и в a1_verify_wp.py
    (там же — замер на настоящих lp: -0.2503/-0.2499/-0.2500).
    """
    rng = np.random.default_rng(0)
    N, npub = 50_000, 10_000
    f = npub / N
    h = rng.normal(size=N)
    h -= h.mean()
    r = rng.normal(size=N) * (1.0 + 0.3 * np.abs(h))     # гетероскедастичность
    u, d2 = h * r, h * h
    kT = u.sum() / d2.sum()
    eps, dq = [], []
    for _ in range(400):
        idx = np.argpartition(rng.random(N), npub)[:npub]
        up, dp = u[idx].sum(), d2[idx].sum()
        eps.append(up / dp - kT)
        dq.append((u.sum() - up) / (d2.sum() - dp) - kT)
    slope = float(np.polyfit(eps, dq, 1)[0])
    assert abs(slope + f / (1 - f)) < 0.01, slope


def test_sigma_kappa_fpc():
    """fpc обязателен: при n_pub -> N ошибка -> 0."""
    s_full = T.sigma_kappa(0.001, n_pub=T.N_ALL)
    assert s_full < 1e-9, s_full
    assert T.sigma_kappa(0.001, fpc=True) < T.sigma_kappa(0.001, fpc=False)


def test_overstatement_per_fit():
    """Цена подогнанного направления = 3.29e-5 = 1.5 шума, не зависит от q."""
    v = T.overstatement_per_fit()
    assert abs(v - 3.294e-5) < 1e-7, v
    assert abs(v / T.NOISE - 1.497) < 0.01


# ------------------------------------------------------- гетероскедастичность g
def test_g_segment_closed_form():
    """Замкнутая форма против прямого sd(h*r). Воспроизводит K1b до 3-го знака."""
    rng = np.random.default_rng(1)
    N = 200_000
    for p, ratio in ((0.369, 1.482), (0.136, 0.342), (0.05, 0.617)):
        m = rng.random(N) < p
        r = rng.normal(size=N) * np.where(m, math.sqrt(ratio), 1.0)
        h = m.astype(float) - p
        mse_in, mse_out = float((r[m] ** 2).mean()), float((r[~m] ** 2).mean())
        g_form = T.g_segment(p, mse_in, mse_out)
        g_dir = T.g_from_direction(h, r)
        assert abs(g_form - g_dir) < 0.02, (p, g_form, g_dir)


def test_g_below_one_for_quiet_segments():
    """Сегмент с МАЛОЙ ошибкой даёт g < 1: замер по нему ТОЧНЕЕ формулы.
    Это и опровергло blanket 1.15 (никогда-не-покупавшие: 0.688)."""
    assert T.g_segment(p=0.136, mse_in=0.375, mse_out=1.098) < 0.75


# ------------------------------------------------------------------- дозировка
def test_priors_are_distinct():
    """Три приора СМЕШИВАТЬ НЕЛЬЗЯ — источник трёх из шести ошибок кампании."""
    assert D.PRIOR_DECOMP[0] == 0.0
    assert D.PRIOR_DECOMP[1] < D.PRIOR_SEGMENT[1] < D.PRIOR_PROPOSED[1]


def test_empirical_bayes_recovers_tau_z():
    """ЭБ по 20 прямым парам Z-разложения даёт tau_z = 0.0148 (сверено с Сашей)."""
    Z = [(0.02470, 0.010), (0.21957, 0.006), (0.03475, -0.064), (0.13151, -0.012),
         (0.06326, 0.034), (0.06581, 0.032), (0.11696, -0.071), (0.03773, 0.037),
         (0.03854, -0.041), (0.02630, 0.013), (0.02910, 0.009), (0.04377, -0.037),
         (0.11421, 0.015), (0.06681, -0.018), (0.04703, -0.007), (0.25102, -0.005),
         (0.03788, -0.057), (0.03707, 0.014), (0.38802, -0.025), (0.34704, -0.015)]
    k = [x[1] for x in Z]
    s = [T.sigma_kappa(x[0]) for x in Z]
    mu, tau = D.empirical_bayes(k, s)
    assert abs(tau - 0.0148) < 0.004, tau
    assert abs(mu + 0.011) < 0.02, mu


def test_redose_R5_multiplier():
    """Редоза R5_shade: 1.002 -> ~0.13 (множитель 0.13), Часть G §2."""
    mult, gain, b_opt = D.redose_multiplier(kappa_pub=1.317, q=0.00007, dose_now=1.002)
    assert 0.10 < mult < 0.17, mult
    assert gain > 0


def test_dose_joint_matches_scalar_when_orthogonal():
    """Ортогональные оси: совместная доза = поосной."""
    ks, qs = [0.5, 0.3], [0.01, 0.02]
    gram = np.diag(qs)
    b = D.dose_joint(ks, qs, gram)
    for i, (k, q) in enumerate(zip(ks, qs)):
        assert abs(b[i] - D.dose_private(k, q)[0]) < 1e-9


# ----------------------------------------------------------------------- зонды
def test_probe_value_zero_for_decomposition():
    """Зонды осей разложения БЕСПОЛЕЗНЫ при любом шаге (tau = 0.0148)."""
    for q in (0.003, 0.01, 0.02, 0.05):          # реальный диапазон наших осей
        assert P.probe_value(q, D.PRIOR_DECOMP[1]) / T.NOISE < 0.05, q


def test_probe_value_threshold():
    """Критерий осмысленности: tau^2*q > 0.25*c."""
    c = P.c_const()
    tau = 0.196
    q_thr = 0.25 * c / (tau * tau)
    assert P.probe_value(q_thr * 0.99, tau) == 0.0
    assert P.probe_value(q_thr * 2.0, tau) > 0.0


def test_probe_value_grows_with_step():
    """Правило 4x-шага: ценность растёт по q."""
    v = [P.probe_value(q, 0.196) for q in (0.003, 0.01, 0.02, 0.05)]
    assert all(v[i] < v[i + 1] for i in range(len(v) - 1)), v


def test_decode_roundtrip():
    """probe_score -> decode возвращает исходную kappa."""
    F0, q, k = 1.647, 0.0026, 0.45
    s = P.probe_score(F0, q, k)
    k_back, _ = P.decode(F0, s, q)
    assert abs(k_back - k) < 1e-9


# ------------------------------------------------------------------------ слип
def test_slip_zero_mean_at_fixed_c():
    """ТЕОРЕМА: при фиксированном c среднее слипа = 0."""
    rng = np.random.default_rng(3)
    N, npub, K = 40_000, 8_000, 5
    lp_a = rng.normal(2.3, 1.5, N)
    Dm = rng.normal(0, 0.05, (K, N))
    c = rng.normal(0, 0.5, K)
    vals = []
    for _ in range(300):
        mask = np.zeros(N, bool)
        mask[np.argpartition(rng.random(N), npub)[:npub]] = True
        vals.append(S.slip_exact(0.01, c, Dm, lp_a, mask))
    m, sd = float(np.mean(vals)), float(np.std(vals))
    se = sd / math.sqrt(len(vals))
    assert abs(m) < 3 * se, (m, se)          # среднее нулевое в пределах 3 SE


def test_slip_law_six_points():
    """Закон 1.042e-5*Sigma|c| на шести точках SHOW: R^2 >= 0.94."""
    x = np.array([7, 19, 27, 37, 146, 279], float)
    y = np.array([-45, -146, 182, 711, 1180, 3080], float) * 1e-6
    pred = S.expected_slip(x)
    r2 = 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    assert r2 >= 0.94, r2


def test_slip_negligible_below_20():
    assert not S.slip_is_measurable(15)
    assert S.slip_is_measurable(60)


# ------------------------------------------------------------------------ main
if __name__ == "__main__":
    fns = [(n, f) for n, f in sorted(globals().items())
           if n.startswith("test_") and callable(f)]
    bad = 0
    for n, f in fns:
        try:
            f()
            print(f"  OK   {n}")
        except AssertionError as e:
            bad += 1
            print(f"  FAIL {n}: {e}")
    print(f"\n{len(fns) - bad}/{len(fns)} тестов прошло")
    sys.exit(1 if bad else 0)

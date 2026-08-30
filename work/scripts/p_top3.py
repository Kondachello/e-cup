# -*- coding: utf-8 -*-
"""P(топ-3) как считающая функция.  Кампания E-CUP 2026, задача 3 (RMSLE, меньше = лучше).

Цель заказчика (29.08): ТОП-3 ПРИВАТА.  Значит максимизируем НЕ E[priv],
а P(наш зачётный приват <= приват третьего места).

Модель поля — Женина (часть M4, work/reports/zhenya_M4_realboard.md):
    priv_i = pub_i + min( phi_i * (REF - pub_i),  n_i * PER )
    phi_i ~ Beta(1.5, 12)      — доля подгонки, откалибрована на нашем замеренном phi=0.086
    PER   = 3.294e-5           — безусловная цена одного подогнанного направления
    REF   = 1.6535955          — общая точка отсчёта (A1), от неё меряются все улучшения
    n_i   — число посылок команды (физический кап подгонки)

Наша сторона — ПАРА файлов, разыгранная из ОДНОГО общего вектора случайности:
    g1 = mu_our + sigma_our * z              (первый финалист, F8-класс)
    g2 = g1 + delta_mu + sd_d * w            (второй финалист, T3-класс)
    X  = min(g1, g2)                          — зачётный приват (RMSLE: меньше = лучше)
Разыгрывать g1 и g2 независимо НЕЛЬЗЯ — это выдумывает паре ценность,
которой у неё нет (часть J / M3.2 Жени).

Ключевое тождество, на котором стоит весь модуль:
    rank = 1 + #{i : priv_i < X}   =>   rank <= 3  <=>  X < C_(3),
где C_(3) — ТРЕТЬЯ ПОРЯДКОВАЯ СТАТИСТИКА (третий снизу) приватов соперников.
То есть P(топ-3) — обычное одномерное сравнение X с порогом C_(3).
Это делает карту чувствительности дешёвой: поле разыгрывается ОДИН раз,
дальше по всей сетке (mu, sigma) переигрывается только наша сторона
(общие случайные числа => разности между ячейками карты почти без MC-шума).

Запуск:
    python work/scripts/p_top3.py --repro    # воспроизведение Жениных 20.8 %
    python work/scripts/p_top3.py --map      # карта чувствительности + обменный курс
    python work/scripts/p_top3.py --all      # всё + разложение + гамма-путь
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np

# ---------------------------------------------------------------- константы

NOISE = 0.000022          # шум замера паблика (work/doctrine/transfer.py)
PER = 1.25 * (1.0 - 0.2) * 1.6470 / 50_000      # = 3.294e-5, overstatement_per_fit()
REF = 1.6535955005        # общая точка отсчёта A1
PHI_A, PHI_B = 1.5, 12.0  # Beta(1.5, 12)

MU_US = 1.646203          # E[приват F8], часть M2
SIGMA_US = 0.000284       # наша собственная сигма привата, часть M2
DELTA_T3 = 0.0014         # E[приват T3] - E[приват F8] = 1.647603 - 1.646203
SD_D_T3 = 0.0011 * 0.0595 # sd разности пары F8/T3 (правило M3: min(0.0011*rms, 0.0009))

# Реальная доска (скрин 304 команды, топ-11 с числом решений). Наша строка-витрина
# исключена: витрина не может быть финалистом (доктрина).
# ИСПРАВЛЕНО 30.08 по СКРИНУ доски (308 команд, видны места 1-18).
# Было три команды и устаревшее число HSE — поле недосчитывало ПЯТЬ соперников
# впереди нас, и E[C_(3)] выходило 1.6467753 вместо 1.6458054, а P(топ-3) — 93 %
# вместо 26.5 %. Место 1 «итмони» 1.6440063524 / 63 посылки — ЭТО МЫ (витрина
# SHOW11), в поле соперников НЕ входит: витрина финалистом быть не может.
BOARD = [
    ("Ежи",                    1.6443877459, 73),
    ("ICEQ",                   1.6445536520, 97),
    ("Daniil Stepanov",        1.6449845018, 96),
    ("DeepTech AI",            1.6452091364, 75),
    ("mdl_larvik",                      1.6457819828, 11),
    ("y76a7c338e6ca",          1.6458200196, 94),
    ("Zababa Zabuba",          1.6458920186, 10),
    ("HSE Бобры",              1.6460816563, 77),   # было 1.6462497938/72 — устарело
    ("speedrun",               1.6463752873,  9),
    ("tak nazivaemaya krutka", 1.6464562896, 42),
    ("Марк Кукурелья",         1.6464720190, 15),
    ("we08deb807b7d",          1.6465637107, 17),
    ("ML Clan",                1.6466129691, 105),
    ("Alexander Alekseev",     1.6466553631, 30),
    ("ferrari strategy",       1.6466589721, 69),
    ("0STG0T",                 1.6466724851, 81),
    ("ddb4fc2ba9832",          1.6466728474,  9),
]
# Хвост (места 19+) — реконструкция, а не данные. Места 1-18 теперь настоящие,
# поэтому хвост начинается ниже последнего замеренного и укорочен.
TAIL_N, TAIL_PUB0, TAIL_STEP, TAIL_SUB = 10, 1.6467, 0.0002, 30


# ---------------------------------------------------------------- поле

def sample_field(ns: int, seed: int = 2026, tail: bool = True) -> np.ndarray:
    """Приваты соперников: матрица (ns, n_команд). Меньше = лучше."""
    rng = np.random.default_rng(seed)
    cols = []
    for _, pub, n in BOARD:
        bias = np.minimum(rng.beta(PHI_A, PHI_B, ns) * (REF - pub), n * PER)
        cols.append(pub + bias)
    if tail:
        for j in range(TAIL_N):
            pub = TAIL_PUB0 + TAIL_STEP * j
            bias = np.minimum(rng.beta(PHI_A, PHI_B, ns) * (REF - pub), TAIL_SUB * PER)
            cols.append(pub + bias)
    return np.column_stack(cols)


def threshold_k(field: np.ndarray, k: int = 3) -> np.ndarray:
    """C_(k) — k-я снизу (лучшая) реализация привата среди соперников.

    rank <= k  <=>  X < C_(k).  Для k=3 это порог третьего места.
    """
    return np.partition(field, k - 1, axis=1)[:, k - 1]


# ---------------------------------------------------------------- наша сторона

def sample_us(ns: int, seed: int = 777, pair: bool = True):
    """Стандартизованные (mu=0, sigma=1) наши розыгрыши: z и добавка второго файла.

    Возвращает (z, w). Наш приват при (mu, sigma):
        g1 = mu + sigma*z;  g2 = g1 + delta + sd_d*w;  X = min(g1, g2).
    Общий вектор z — ровно та самая ОДНА общая случайность пары.
    """
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(ns)
    w = rng.standard_normal(ns) if pair else np.zeros(ns)
    return z, w


def our_private(mu: float, sigma: float, z: np.ndarray, w: np.ndarray,
                delta: float = DELTA_T3, sd_d: float = SD_D_T3) -> np.ndarray:
    g1 = mu + sigma * z
    if delta is None:
        return g1
    g2 = g1 + delta + sd_d * w
    return np.minimum(g1, g2)


# ---------------------------------------------------------------- главная функция

class Objective:
    """Предрассчитанное поле + наши стандартизованные розыгрыши.

    Создаётся один раз, дальше P_top3(mu, sigma) стоит ~1 мс.
    Общие случайные числа => разности по сетке считаются с точностью
    много выше, чем 1/sqrt(NS) на абсолютном уровне.
    """

    def __init__(self, ns: int = 200_000, seed_field: int = 2026, seed_us: int = 777,
                 pair: bool = True, tail: bool = True):
        self.ns = ns
        self.field = sample_field(ns, seed_field, tail=tail)
        self.c3 = threshold_k(self.field, 3)
        self.c1 = threshold_k(self.field, 1)
        self.c5 = threshold_k(self.field, 5)
        self.z, self.w = sample_us(ns, seed_us, pair=pair)
        self.pair = pair

    def X(self, mu: float, sigma: float) -> np.ndarray:
        return our_private(mu, sigma, self.z, self.w,
                           delta=(DELTA_T3 if self.pair else None))

    def P_top3(self, mu_our: float = MU_US, sigma_our: float = SIGMA_US) -> float:
        """ВЕРОЯТНОСТЬ ТОП-3: P(наш зачётный приват < приват третьего места)."""
        return float((self.X(mu_our, sigma_our) < self.c3).mean())

    def P_topk(self, k: int, mu_our: float = MU_US, sigma_our: float = SIGMA_US) -> float:
        ck = threshold_k(self.field, k)
        return float((self.X(mu_our, sigma_our) < ck).mean())

    def rank_dist(self, mu_our: float = MU_US, sigma_our: float = SIGMA_US) -> dict:
        x = self.X(mu_our, sigma_our)
        rank = 1 + (self.field < x[:, None]).sum(1)
        return dict(
            p1=float((rank == 1).mean()),
            p23=float(((rank >= 2) & (rank <= 3)).mean()),
            p45=float(((rank >= 4) & (rank <= 5)).mean()),
            p6_10=float(((rank >= 6) & (rank <= 10)).mean()),
            p11=float((rank >= 11).mean()),
            top3=float((rank <= 3).mean()),
            top5=float((rank <= 5).mean()),
            top10=float((rank <= 10).mean()),
            median=int(np.median(rank)),
        )


# --------- модуль-уровневая обёртка, как просил ТЗ: P_top3(mu_our, sigma_our)

_OBJ = None


def P_top3(mu_our: float = MU_US, sigma_our: float = SIGMA_US, ns: int = 200_000) -> float:
    """Вероятность, что наш ЗАЧЁТНЫЙ приват (лучший из пары) попадёт в тройку."""
    global _OBJ
    if _OBJ is None or _OBJ.ns != ns:
        _OBJ = Objective(ns=ns)
    return _OBJ.P_top3(mu_our, sigma_our)


# ---------------------------------------------------------------- 1. репродукция

def reproduce_zhenya() -> dict:
    """Побитовое повторение m4_realboard.py: seed 3008, NS=300k, sigma 0.00028, без пары."""
    rng = np.random.default_rng(3008)
    ns = 300_000
    ours = rng.normal(MU_US, 0.00028, ns)
    cols = []
    for _, pub, n in BOARD:
        cols.append(pub + np.minimum(rng.beta(PHI_A, PHI_B, ns) * (REF - pub), n * PER))
    for j in range(TAIL_N):
        pub = TAIL_PUB0 + TAIL_STEP * j
        cols.append(pub + np.minimum(rng.beta(PHI_A, PHI_B, ns) * (REF - pub), TAIL_SUB * PER))
    fl = np.column_stack(cols)
    rank = 1 + (fl < ours[:, None]).sum(1)
    e_priv = {nm: float(fl[:, i].mean()) for i, (nm, _, _) in enumerate(BOARD)}
    return dict(
        p1=float((rank == 1).mean()), p23=float(((rank >= 2) & (rank <= 3)).mean()),
        p3=float((rank <= 3).mean()), p5=float((rank <= 5).mean()),
        p10=float((rank <= 10).mean()), med=int(np.median(rank)), e_priv=e_priv,
    )


# ---------------------------------------------------------------- 3. карта

SHIFTS = np.round(np.arange(-0.0004, 0.0002 + 1e-12, 2e-5), 8)
SIGMA_MULTS = np.round(np.arange(0.5, 2.5 + 1e-12, 0.25), 4)


def sensitivity_map(obj: Objective, shifts=SHIFTS, mults=SIGMA_MULTS) -> np.ndarray:
    m = np.empty((len(shifts), len(mults)))
    for i, s in enumerate(shifts):
        for j, k in enumerate(mults):
            m[i, j] = obj.P_top3(MU_US + s, SIGMA_US * k)
    return m


# ---------------------------------------------------------------- 4. обменный курс

def exchange_rate(obj: Objective, mu: float = MU_US, sigma: float = SIGMA_US,
                  h_mu: float = 1e-5, h_sig_mult: float = 0.05) -> dict:
    """Наклон линии уровня P(топ-3) в плоскости (E, sigma).

    dP/dmu и dP/dsigma центральными разностями на ОБЩИХ случайных числах.
    Курс = -(dP/dsigma)/(dP/dmu) = сколько единиц mu можно ОТДАТЬ (в худшую
    сторону, т.е. +mu при RMSLE) за одну единицу прироста sigma.
    """
    dP_dmu = (obj.P_top3(mu + h_mu, sigma) - obj.P_top3(mu - h_mu, sigma)) / (2 * h_mu)
    hs = SIGMA_US * h_sig_mult
    dP_ds = (obj.P_top3(mu, sigma + hs) - obj.P_top3(mu, sigma - hs)) / (2 * hs)
    rate = -dP_ds / dP_dmu if dP_dmu != 0 else float("nan")
    return dict(
        dP_dmu=dP_dmu, dP_dsigma=dP_ds, rate=rate,
        rate_noises_per_sigma_unit=rate,                       # безразмерный наклон
        mu_per_1x_sigma=rate * SIGMA_US,                       # за +1x базовой сигмы
        mu_per_1x_sigma_noises=rate * SIGMA_US / NOISE,
        mu_per_1noise_sigma=rate * NOISE,                      # за +1 шум сигмы
        mu_per_1noise_sigma_noises=rate,
    )


# ---------------------------------------------------------------- 5. разложение

def decomposition(obj: Objective, mu: float = MU_US, sigma: float = SIGMA_US) -> dict:
    """Кто нас вносит в тройку: наша удача или просадка поля.

    Тождество для успеха (X < C3):
        C3 - X = (C3 - E[C3])  +  (E[C3] - mu)  +  (mu - X)
                  поле просело     базовый разрыв   мы улучшились
    Базовый разрыв фиксирован и отрицателен (мы позади). Условно на успехе
    его должны перекрыть два слагаемых — их условные средние и дают доли.
    """
    x = obj.X(mu, sigma)
    c3 = obj.c3
    ok = x < c3
    e_c3 = float(c3.mean())
    field_luck = c3 - e_c3            # >0 => поле оказалось хуже ожидания
    our_luck = mu - x                 # >0 => мы оказались лучше ожидания
    fl = float(field_luck[ok].mean())
    ol = float(our_luck[ok].mean())
    tot = fl + ol
    # контрфактические вероятности
    p_field_only = float((c3 > mu).mean())          # мы = ровно ожидание
    p_us_only = float((x < e_c3).mean())            # поле = ровно ожидание
    return dict(
        p=float(ok.mean()), E_C3=e_c3, sd_C3=float(c3.std()),
        gap=e_c3 - mu, field_luck=fl, our_luck=ol,
        share_field=fl / tot if tot else float("nan"),
        share_us=ol / tot if tot else float("nan"),
        p_field_only=p_field_only, p_us_only=p_us_only,
        corr_check=float(np.corrcoef(c3, x)[0, 1]),
    )


# ---------------------------------------------------------------- ценность пары

def pair_value(obj: Objective, mu: float = MU_US, sigma: float = SIGMA_US,
               deltas=(0.0, 0.0002, 0.0005, 0.0010, 0.0014, 0.0020, 0.0030),
               sd_ds=(0.00005, 0.0001, 0.0002, 0.0003, 0.0005, 0.0008, 0.0012)) -> dict:
    """Сколько П.П. P(топ-3) добавляет ВТОРОЙ файл при (delta_E, sd разности).

    Под целью E второй файл почти бесполезен (M3: максимум +0.000003).
    Под целью хвоста он ценен ровно развязкой: P(хотя бы один в тройке).
    Разыгрывается ОДИН общий вектор z, второй файл = первый + delta + sd_d*w.
    """
    g1 = mu + sigma * obj.z
    base = float((g1 < obj.c3).mean())
    grid = np.empty((len(deltas), len(sd_ds)))
    for i, d in enumerate(deltas):
        for j, sd in enumerate(sd_ds):
            g2 = g1 + d + sd * obj.w
            grid[i, j] = float((np.minimum(g1, g2) < obj.c3).mean())
    return dict(base=base, deltas=list(deltas), sd_ds=list(sd_ds), grid=grid)


# ---------------------------------------------------------------- точка перелома

def variance_breakeven(obj: Objective, lo: float = -0.0006, hi: float = 0.0002,
                       h_sig_mult: float = 0.05) -> dict:
    """Сдвиг E, при котором dP/dsigma меняет знак.

    Правее (хуже) этой точки дисперсия ПОЛЕЗНА, левее (лучше) — ВРЕДНА.
    """
    hs = SIGMA_US * h_sig_mult

    def dpds(shift):
        return (obj.P_top3(MU_US + shift, SIGMA_US + hs)
                - obj.P_top3(MU_US + shift, SIGMA_US - hs)) / (2 * hs)

    a, b = lo, hi
    if dpds(a) * dpds(b) > 0:
        return dict(shift=float("nan"), mu=float("nan"), p=float("nan"))
    for _ in range(60):
        m = 0.5 * (a + b)
        if dpds(a) * dpds(m) <= 0:
            b = m
        else:
            a = m
    s = 0.5 * (a + b)
    return dict(shift=s, mu=MU_US + s, p=obj.P_top3(MU_US + s, SIGMA_US),
                dist_from_now_noises=s / NOISE)


# ---------------------------------------------------------------- устойчивость

def robustness(ns: int = 200_000) -> dict:
    """Устойчивость главного вывода к сидам и к допущениям модели поля."""
    out = {}
    # a) сиды
    vals, gains = [], []
    for s in range(11, 16):
        o = Objective(ns=ns, seed_field=s * 101, seed_us=s * 37)
        p1 = o.P_top3(MU_US, SIGMA_US)
        p15 = o.P_top3(MU_US, SIGMA_US * 1.5)
        vals.append(p1)
        gains.append(p15 - p1)
    out["seeds_p_base"] = [float(v) for v in vals]
    out["seeds_gain_15x"] = [float(v) for v in gains]

    # b) без реконструированного хвоста
    o = Objective(ns=ns, tail=False)
    out["no_tail"] = dict(p1x=o.P_top3(MU_US, SIGMA_US),
                          p15x=o.P_top3(MU_US, SIGMA_US * 1.5))

    # c) другие приоры phi и снятый кап
    global PHI_A, PHI_B
    for tag, (pa, pb) in {"beta(1.5,12) базовый": (1.5, 12.0),
                          "beta(2,10) mean .167": (2.0, 10.0),
                          "beta(1,15) mean .0625": (1.0, 15.0),
                          "beta(3,6)  mean .333": (3.0, 6.0)}.items():
        PHI_A, PHI_B = pa, pb
        o = Objective(ns=ns)
        out.setdefault("phi_prior", {})[tag] = dict(
            p1x=o.P_top3(MU_US, SIGMA_US), p15x=o.P_top3(MU_US, SIGMA_US * 1.5),
            p2x=o.P_top3(MU_US, SIGMA_US * 2.0))
    PHI_A, PHI_B = 1.5, 12.0

    # d) без пары (один файл)
    o = Objective(ns=ns, pair=False)
    out["single_file"] = dict(p1x=o.P_top3(MU_US, SIGMA_US),
                              p15x=o.P_top3(MU_US, SIGMA_US * 1.5))
    return out


# ---------------------------------------------------------------- гамма-путь

GAMMA_JSON = "/Users/alexanderkondakov/ozon-cup/work/reports/lineA/a3_gamma_path.json"


def gamma_candidates() -> list:
    """(gamma, E[priv], sigma_total) для риджа.

    ДОПУЩЕНИЕ (явно): наша sigma 0.000284 = sqrt(sigma_общая^2 + sd(F8)^2),
    где sd(F8) — собственная sd переноса доз при gamma=0.1 (9.537e-5).
    Смена gamma меняет ТОЛЬКО эту компоненту. Без такого разделения
    компоненты дозы считались бы дважды.
    """
    with open(GAMMA_JSON, encoding="utf-8") as f:
        j = json.load(f)
    sd_f8 = [p["sd"] for p in j["path"] if abs(p["gamma"] - 0.1) < 1e-12][0]
    var_shared = SIGMA_US ** 2 - sd_f8 ** 2
    out = []
    for p in j["path"]:
        mu = MU_US - p["vs_F8_noises"] * NOISE
        sig = math.sqrt(var_shared + p["sd"] ** 2)
        out.append((p["gamma"], mu, sig, p["vs_F8_noises"], p["sd"], p["p_worse"]))
    return out


# ---------------------------------------------------------------- печать

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repro", action="store_true")
    ap.add_argument("--map", action="store_true")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--ns", type=int, default=200_000)
    ap.add_argument("--json-out", default="")
    a = ap.parse_args()
    if not (a.repro or a.map or a.all):
        a.all = True
    res = {}

    if a.repro or a.all:
        print("=" * 78)
        print("1. ВОСПРОИЗВЕДЕНИЕ M4 (seed 3008, NS=300k, sigma 0.00028, без пары)")
        print("=" * 78)
        r = reproduce_zhenya()
        print(f"  P(1)={r['p1']*100:.1f}%  P(2-3)={r['p23']*100:.1f}%  "
              f"P(топ-3)={r['p3']*100:.2f}%  P(топ-5)={r['p5']*100:.1f}%  медиана {r['med']}")
        print("  Женя (m4_realboard.json): P(1)=0.94%  P(топ-3)=20.81%  "
              "P(топ-5)=53.70%  медиана 5")
        print("  E[приват] по командам:")
        for nm, pub, n in BOARD:
            print(f"    {nm:24s}{pub:12.7f}{n:5d}{r['e_priv'][nm]:12.6f}")
        res["repro"] = {k: v for k, v in r.items() if k != "e_priv"}

    obj = Objective(ns=a.ns)

    if a.map or a.all:
        print()
        print("=" * 78)
        print(f"2. БАЗОВАЯ ТОЧКА (пара F8+T3, NS={a.ns:,})")
        print("=" * 78)
        rd = obj.rank_dist()
        print(f"  mu={MU_US:.6f}  sigma={SIGMA_US:.6f}")
        print(f"  P(1)={rd['p1']*100:.2f}%  P(2-3)={rd['p23']*100:.2f}%  "
              f"P(4-5)={rd['p45']*100:.2f}%  P(6-10)={rd['p6_10']*100:.2f}%")
        print(f"  P(топ-3)={rd['top3']*100:.2f}%  P(топ-5)={rd['top5']*100:.2f}%  "
              f"медиана {rd['median']}")
        se = math.sqrt(rd["top3"] * (1 - rd["top3"]) / a.ns)
        print(f"  MC-погрешность P(топ-3): +-{se*100:.2f} п.п. (1 sigma)")
        res["base"] = rd

        print()
        print("=" * 78)
        print("3. КАРТА ЧУВСТВИТЕЛЬНОСТИ P(топ-3), %   (строки — сдвиг E, столбцы — sigma)")
        print("=" * 78)
        m = sensitivity_map(obj)
        print("  сдвиг E |" + "".join(f"{k:>7.2f}x" for k in SIGMA_MULTS))
        print("   (шумов)|" + "".join(f"{SIGMA_US*k/NOISE:>7.1f}ш" for k in SIGMA_MULTS))
        print("  " + "-" * 76)
        for i, s in enumerate(SHIFTS):
            mark = " <<<" if abs(s) < 1e-12 else ""
            print(f"{s:+9.5f}|" + "".join(f"{m[i,j]*100:8.2f}" for j in range(len(SIGMA_MULTS)))
                  + mark)
        res["map"] = dict(shifts=SHIFTS.tolist(), mults=SIGMA_MULTS.tolist(),
                          p=m.tolist())

        j1 = int(np.argmin(np.abs(SIGMA_MULTS - 1.0)))
        i0 = int(np.argmin(np.abs(SHIFTS)))
        print()
        print("  ГЛАВНЫЙ ОТВЕТ — при текущем ожидании (сдвиг 0) по строке sigma:")
        for j, k in enumerate(SIGMA_MULTS):
            d = (m[i0, j] - m[i0, j1]) * 100
            print(f"    sigma={k:.2f}x ({SIGMA_US*k:.6f}):  P(топ-3)={m[i0,j]*100:6.2f}%   "
                  f"{d:+6.2f} п.п. к базе")
        res["sigma_row_at_zero"] = m[i0].tolist()

    if a.all:
        print()
        print("=" * 78)
        print("4. ОБМЕННЫЙ КУРС (наклон линии уровня P(топ-3))")
        print("=" * 78)
        er = exchange_rate(obj)
        print(f"  dP/dE     = {er['dP_dmu']:+.4g} на единицу скора "
              f"({er['dP_dmu']*NOISE*100:+.3f} п.п. на 1 шум ухудшения E)")
        print(f"  dP/dsigma = {er['dP_dsigma']:+.4g} на единицу sigma "
              f"({er['dP_dsigma']*NOISE*100:+.3f} п.п. на 1 шум прироста sigma)")
        print(f"  КУРС: за +1 шум sigma можно отдать {er['mu_per_1noise_sigma']/NOISE:.2f} "
              f"шума E ({er['mu_per_1noise_sigma']:+.7f})")
        print(f"        за +1x базовой sigma (+{SIGMA_US:.6f}) — "
              f"{er['mu_per_1x_sigma_noises']:.1f} шума E ({er['mu_per_1x_sigma']:+.7f})")
        res["exchange"] = er
        print(f"\n  курс в разных рабочих точках (шума E за 1 шум sigma):")
        print(f"    {'sigma':>8}{'P(топ-3)':>10}{'dP/dE,пп/ш':>12}"
              f"{'dP/dsig,пп/ш':>14}{'курс':>8}")
        tbl = []
        for k in (0.75, 1.0, 1.25, 1.5, 2.0):
            e2 = exchange_rate(obj, MU_US, SIGMA_US * k)
            p = obj.P_top3(MU_US, SIGMA_US * k)
            tbl.append(dict(mult=k, p=p, **e2))
            print(f"    {SIGMA_US*k:8.6f}{p*100:9.2f}%{e2['dP_dmu']*NOISE*100:12.3f}"
                  f"{e2['dP_dsigma']*NOISE*100:14.3f}{e2['rate']:8.2f}")
        res["exchange_table"] = tbl

        print()
        print("=" * 78)
        print("5. РАЗЛОЖЕНИЕ: наша удача против просадки поля")
        print("=" * 78)
        d = decomposition(obj)
        print(f"  порог третьего места C_(3): E={d['E_C3']:.6f}  sd={d['sd_C3']:.6f}")
        print(f"  базовый разрыв E[C3]-mu = {d['gap']:+.6f} "
              f"({d['gap']/NOISE:+.1f} шума)")
        print(f"  условно на успехе: поле просело на {d['field_luck']:+.6f}, "
              f"мы улучшились на {d['our_luck']:+.6f}")
        print(f"  ДОЛИ: поле {d['share_field']*100:.1f} %   мы {d['share_us']*100:.1f} %")
        print(f"  контрфакт: P(топ-3 | мы ровно в ожидании) = {d['p_field_only']*100:.2f} %")
        print(f"             P(топ-3 | поле ровно в ожидании) = {d['p_us_only']*100:.2f} %")
        res["decomp"] = d

        print()
        print("=" * 78)
        print("6. ГАММА-ПУТЬ ЧЕРЕЗ НОВУЮ ЦЕЛЕВУЮ ФУНКЦИЮ")
        print("=" * 78)
        print(f"  {'gamma':>7}{'vsF8,ш':>9}{'E[priv]':>12}{'sigma':>11}{'sig/база':>9}"
              f"{'P(топ-3)':>10}{'P(топ-5)':>10}{'d к F8':>9}")
        gp = []
        p_f8 = obj.P_top3(MU_US, SIGMA_US)
        for g, mu, sig, vsf8, sd, pw in gamma_candidates():
            p3 = obj.P_top3(mu, sig)
            p5 = obj.P_topk(5, mu, sig)
            gp.append(dict(gamma=g, mu=mu, sigma=sig, p3=p3, p5=p5, vs_f8_noises=vsf8))
            print(f"  {g:7.4f}{vsf8:+9.2f}{mu:12.6f}{sig:11.6f}{sig/SIGMA_US:9.3f}"
                  f"{p3*100:9.2f}%{p5*100:9.2f}%{(p3-p_f8)*100:+9.2f}")
        res["gamma_path"] = gp

        print()
        print("=" * 78)
        print("7. ТОЧКА ПЕРЕЛОМА ДИСПЕРСИИ И УСТОЙЧИВОСТЬ")
        print("=" * 78)
        be = variance_breakeven(obj)
        print(f"  dP/dsigma меняет знак при сдвиге E = {be['shift']:+.6f} "
              f"({be['dist_from_now_noises']:+.1f} шума от текущего),")
        print(f"  то есть при mu = {be['mu']:.6f}, где P(топ-3) = {be['p']*100:.1f} %.")
        print(f"  ПРАВЕЕ (хуже) — дисперсия помогает. Мы правее на "
              f"{-be['dist_from_now_noises']:.1f} шума.")
        res["breakeven"] = be

        rb = robustness(a.ns)
        print(f"\n  по 5 сидам: P(база) = "
              f"{min(rb['seeds_p_base'])*100:.2f}..{max(rb['seeds_p_base'])*100:.2f} %, "
              f"прирост при 1.5x sigma = "
              f"{min(rb['seeds_gain_15x'])*100:+.2f}..{max(rb['seeds_gain_15x'])*100:+.2f} п.п.")
        print(f"  без реконструированного хвоста: {rb['no_tail']['p1x']*100:.2f} % -> "
              f"{rb['no_tail']['p15x']*100:.2f} % при 1.5x")
        print(f"  одиночный файл вместо пары:     {rb['single_file']['p1x']*100:.2f} % -> "
              f"{rb['single_file']['p15x']*100:.2f} % при 1.5x")
        print("  приор phi:")
        for k, v in rb["phi_prior"].items():
            print(f"    {k:24s} 1.0x {v['p1x']*100:6.2f} %   1.5x {v['p15x']*100:6.2f} %"
                  f"   2.0x {v['p2x']*100:6.2f} %   "
                  f"({(v['p15x']-v['p1x'])*100:+.2f} п.п.)")
        res["robust"] = rb

        print()
        print("=" * 78)
        print("8. ЦЕННОСТЬ ВТОРОГО ФАЙЛА ПОД ЦЕЛЬЮ ТОП-3 (прирост P, п.п.)")
        print("=" * 78)
        pv = pair_value(obj)
        print(f"  одиночный F8: P(топ-3) = {pv['base']*100:.2f} %")
        print(f"  {'delta E, ш':>11}|" + "".join(f"{s/NOISE:>8.1f}ш" for s in pv["sd_ds"]))
        print(f"  {'(2й хуже)':>11}|" + "".join(f"{s:>9.5f}" for s in pv["sd_ds"])
              + "   <- sd разности")
        print("  " + "-" * 74)
        for i, d in enumerate(pv["deltas"]):
            print(f"  {d/NOISE:>11.1f}|"
                  + "".join(f"{(pv['grid'][i,j]-pv['base'])*100:+9.2f}"
                            for j in range(len(pv["sd_ds"]))))
        print("  Действующая пара F8+T3: delta = 63.6 шума, sd разности = "
              f"{SD_D_T3/NOISE:.1f} шума -> прирост "
              f"{(obj.P_top3()-pv['base'])*100:+.2f} п.п.")
        res["pair_value"] = dict(base=pv["base"], deltas=pv["deltas"],
                                 sd_ds=pv["sd_ds"], grid=pv["grid"].tolist())

    if a.json_out:
        os.makedirs(os.path.dirname(a.json_out), exist_ok=True)
        with open(a.json_out, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=1)
        print(f"\nJSON -> {a.json_out}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()

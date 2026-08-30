# -*- coding: utf-8 -*-
"""P(топ-5) как считающая функция. Пятая порядковая статистика поля C_(5).

Кампания E-CUP 2026, задача 3 (RMSLE, меньше = лучше). Цель штаба сменилась
с ТОП-3 на ТОП-5, а вся считающая машинка (`p_top3.py`) построена на ТРЕТЬЕЙ
порядковой статистике. Пятую не считал никто. Этот модуль её строит — той же
машинкой, ничего не переопределяя.

Тождество ровно то же, с заменой 3 -> 5:

    rank = 1 + #{i : priv_i < X}   =>   rank <= 5  <=>  X < C_(5),

где C_(5) — ПЯТАЯ порядковая статистика (пятый снизу) приватов соперников, а
X = min(g1, g2) — зачётный приват пары. Пара разыгрывается из ОДНОГО общего
вектора случайности z:

    g1 = mu_1 + sigma_us * z                (слот-1)
    g2 = g1 + delta + sd_d * w              (слот-2)
    X  = min(g1, g2)

Независимый розыгрыш g1 и g2 ЗАПРЕЩЁН доктриной (часть J / M3.2 Жени): он
выдумывает паре ценность, которой у неё нет. Все константы поля, модель поля
и BOARD импортируются из `p_top3.py` — штабной модуль не редактируется.

ГЛАВНОЕ, ЧТО ОТЛИЧАЕТ ПЯТУЮ СТАТИСТИКУ ОТ ТРЕТЬЕЙ (§2-§3 отчёта):
C_(3) реализуется в основном командами с БОЛЬШИМ числом посылок, у которых
приват = pub + phi*(REF-pub) и sd порядка 30 шумов. C_(5) — командами с МАЛЫМ
n, у которых упирается физический кап n*PER, приват почти равен паблику и sd
порядка 3 шумов. Отсюда: sd(C_(5)) вдвое МЕНЬШЕ sd(C_(3)), но чувствительность
к СОСТАВУ поля (одна команда добавилась/исчезла) в 4-14 раз ВЫШЕ.

Запуск:
    .venv/bin/python work/scripts/p_top5.py --control   # контроль 65.59 %
    .venv/bin/python work/scripts/p_top5.py --all       # всё
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from p_top3 import (BOARD, NOISE, PER, PHI_A, PHI_B, REF, SIGMA_US, TAIL_N,
                    TAIL_PUB0, TAIL_STEP, TAIL_SUB, sample_us)

ROOT = Path(__file__).resolve().parents[2]
LINEA = ROOT / "work" / "reports" / "lineA"

NS = 400_000                 # как в pair_top3.py — тот же розыгрыш, тот же сид
SEED_FIELD, SEED_US = 2026, 777

# --- валюта штаба (pair_top3.py) -------------------------------------------
MU_SLOT1 = 1.646073          # E[приват F12] в валюте части M2 (в ней все отчёты)
PUB_F12 = 1.6456761695614883 # замер слота-1
PUB_F13 = 1.6457023817       # замер слота-2
K_F12 = 9.30                 # реестр (пессимистический край); точный дифференциал 7.76
K_F12_ALT = 7.76
K_F13 = 11.27                # линия A3: 8.30 + 2.97
PRIV_TRANSFER, NOISE_PER_DIR = 1.25, 2.63e-5
F_SCALE = 1.646

# Прогнозы паблика кандидатов (линейный закон слипа, запечатанные прогнозы).
# k у всех НЕ выведен — поэтому по ним параметризуемся, а не подставляем число.
CANDIDATES = {
    "ext283":  dict(pub=1.6452272, axes=84, gamma=0.20, k_hint=None),
    "extebz":  dict(pub=1.6453939, axes=84, gamma=0.20, k_hint=None),
    "ebz362":  dict(pub=1.6455183, axes=46, gamma=0.10, k_hint=12.37),
    "int08":   dict(pub=1.6455437, axes=46, gamma=0.08, k_hint=10.78),
    "int10":   dict(pub=1.6455523, axes=46, gamma=0.10, k_hint=10.78),
    "f10g08":  dict(pub=1.6455796, axes=46, gamma=0.08, k_hint=12.76),
    "a4allnc": dict(pub=1.6455849, axes=84, gamma=0.10, k_hint=None),
    "f10g10":  dict(pub=1.6455894, axes=46, gamma=0.10, k_hint=12.76),
}
K_GRID = (6.0, 8.0, 8.3, 9.3, 10.0, 11.27, 12.0, 13.0, 14.0, 16.0, 18.0, 20.0)

# --- ИСПРАВЛЕНИЕ 30.08 ~21:30 (work/reports/kcalc_3008.md) --------------------
# k выведен. Реестровая тройка 8.30 / 9.30 / 11.27 — это ЗАМЕР, ДИФФЕРЕНЦИАЛ и
# ГРАНИЦА, а не три значения одной величины. Точный k(F12) = 7.755, а не 9.30:
# граница давала каждому претенденту БЕЗ сегментного расщепления 1.54 бесплатных
# направления = 2.3 шума форы. Наш слот-2 a3g0 — именно такой, поэтому его
# отставание от F12 равно 6.46 шума, а не 4.14, и P(топ-*) пары были завышены.
#
# EV_PRIV — готовая приватная валюта из kcalc §5, Z-калиброванная колонка,
# В ШУМАХ против замеренного F12 (минус = ЛУЧШЕ F12). Ею считаются пары в §9;
# параметризация по k (§5 отчёта) остаётся как выкладка «что было бы, если».
K_F12_EXACT = 7.755
EV_PRIV = {
    "seg4":   -0.89,   # k занижен на EB-гиперпараметры расщепления — см. оговорку
    "int08":  -0.71,
    "int10":  -0.32,
    "f12":     0.00,   # якорь: E[priv] = MU_SLOT1 = 1.646073
    "seg4eb": +0.21,
    "segsil": +0.87,
    "ebz209": +6.07,
    "f10g08": +6.19,
    "a3g0":   +6.46,   # F13_g0, действующий слот-2
    "f10g10": +6.63,
    "base":   +6.69,   # F8
    "ebz362": +8.22,
    "a4allnc": +16.74,
    "extebz": +17.68,
    "ext283": +17.87,
    "extctl": +19.92,
}
MEASURED = {"f12", "a3g0", "base"}          # замерены на платформе на 30.08 20:00
TONIGHT = {"int08", "int10", "ext283", "extebz"}   # F14/F17/F15/F18, замер до 23:30
# Развилка уровня (work/reports/rank/r2c_countersign_f8.md): k(F8) = 23.0,
# коридор 16..28; маршрут C даёт k(F12) = 22.33 против M2-переноса 12.07.
K_ROUTEC_F12 = 22.33
K_ROUTEC_F8 = 23.03
K_AUDIT_F8 = 8.30


# ---------------------------------------------------------------- поле
# Поле разыгрывается ОДНИМ пред-нарисованным массивом beta: столбец j всегда
# берёт U[:, j]. Тогда удаление команды не сдвигает случайность остальных, и
# производные по составу поля считаются на общих случайных числах.

def draw_u(ns: int = NS, ncols: int | None = None, seed: int = SEED_FIELD) -> np.ndarray:
    """Матрица beta(PHI_A, PHI_B) размера (ns, ncols).

    Порядок заполнения совпадает с последовательными вызовами
    rng.beta(a, b, ns) в p_top3.sample_field — контроль в control().
    """
    if ncols is None:
        ncols = len(BOARD) + TAIL_N
    rng = np.random.default_rng(seed)
    return rng.beta(PHI_A, PHI_B, (ncols, ns)).T


def build_field(u: np.ndarray, board=BOARD, shift: float = 0.0,
                drop: int | None = None, add: list | None = None,
                tail: bool = True) -> np.ndarray:
    """Приваты соперников (ns, n_команд). priv = pub + min(phi*(REF-pub), n*PER)."""
    cols, j = [], 0
    for i, (_, pub, n) in enumerate(board):
        if i != drop:
            p = pub + shift
            cols.append(p + np.minimum(u[:, j] * (REF - p), n * PER))
        j += 1
    for t in range(TAIL_N):
        if tail:
            p = TAIL_PUB0 + TAIL_STEP * t + shift
            cols.append(p + np.minimum(u[:, j] * (REF - p), TAIL_SUB * PER))
        j += 1
    if add:
        for m, (_, pub, n) in enumerate(add):
            p = pub + shift
            cols.append(p + np.minimum(u[:, j + m] * (REF - p), n * PER))
    return np.column_stack(cols)


def ck(field: np.ndarray, k: int) -> np.ndarray:
    """C_(k) — k-я снизу (лучшая) реализация привата среди соперников."""
    return np.partition(field, k - 1, axis=1)[:, k - 1]


# ---------------------------------------------------------------- наша сторона

class Pair:
    """Пара (слот-1, слот-2) из ОДНОГО вектора случайности z."""

    def __init__(self, ns: int = NS, seed_us: int = SEED_US):
        self.z, self.w = sample_us(ns, seed_us)

    def X(self, mu1: float, delta: float, sd_d: float,
          sigma: float = SIGMA_US) -> np.ndarray:
        g1 = mu1 + sigma * self.z
        if sd_d == 0.0 and delta == 0.0:
            return g1
        return np.minimum(g1, g1 + delta + sd_d * self.w)

    def P(self, thr: np.ndarray, mu1: float, delta: float, sd_d: float,
          sigma: float = SIGMA_US) -> float:
        return float((self.X(mu1, delta, sd_d, sigma) < thr).mean())


# ---------------------------------------------------------------- состояния GLS

_STATES: dict[str, dict] = {}


def state(tag: str) -> dict:
    if tag not in _STATES:
        z = np.load(LINEA / f"gls_state_{tag}.npz", allow_pickle=True)
        _STATES[tag] = dict(names=[str(x) for x in z["names"]],
                            d=z["d_fin"], V=z["mdl_vivian"])
    return _STATES[tag]


def sd_d(tag_a: str, tag_b: str, force_V: str | None = None) -> float:
    """sd разности приватов двух файлов: sqrt(1.25^2 * dd' V dd)/F_SCALE.

    Оси выравниваются по именам; 46 модельных имён — подмножество 84.
    V берётся из состояния с БОЛЬШИМ числом осей (иначе усечение выбрасывает
    ровно те Z-оси, на которых 84-осевые кандидаты и отличаются).
    force_V='f12' воспроизводит усечённый счёт каталога — печатается справочно.
    """
    a, b = state(tag_a), state(tag_b)
    host = a if len(a["names"]) >= len(b["names"]) else b
    nm = host["names"]
    V = state(force_V)["mdl_vivian"] if force_V else host["mdl_vivian"]
    idx = [nm.index(x) for x in state(force_V)["names"]] if force_V else None

    def emb(s):
        out = np.zeros(len(nm))
        for i, x in enumerate(s["names"]):
            out[nm.index(x)] = s["d"][i]
        return out

    dd = emb(a) - emb(b)
    if idx is not None:
        dd = dd[idx]
    return float(np.sqrt(PRIV_TRANSFER ** 2 * dd @ V @ dd) / F_SCALE)


def mu1_of(k_f12: float = K_F12) -> float:
    """E[приват F12] в валюте M2 при заданном k(F12).

    ВНИМАНИЕ, тут легко ошибиться. MU_SLOT1 = 1.646073 — НЕ pub + 1.25*k*2.63e-5
    при k = 9.30 (это дало бы 1.6459819). Это якорь валюты M2: постоянный перенос
    12.07 направления, одинаковый для наших файлов (у F8 в той же валюте 12.08).
    Поэтому исправление k(F12) наше ожидание НЕ двигает — оно двигает только
    РАЗНОСТИ (offset), то есть отставание слота-2 и кандидатов.
    Абсолютный уровень — отдельная развилка, level_fork().
    """
    return MU_SLOT1


def offset(pub: float, k: float, k_f12: float = K_F12) -> float:
    """Сдвиг E[приват] кандидата против F12 в валюте гарда.

        E_gard = pub + 1.25 * k * 2.63e-5 ;  offset = E_gard(c) - E_gard(F12)
    """
    return (pub - PUB_F12) + PRIV_TRANSFER * NOISE_PER_DIR * (k - k_f12)


# ---------------------------------------------------------------- 1. контроль

def control(u, pair, k_f12: float = K_F12) -> dict:
    f = build_field(u)
    c3, c5 = ck(f, 3), ck(f, 5)
    d13 = sd_d("f12", "a3g0")
    dl13 = offset(PUB_F13, K_F13, k_f12)
    mu1 = mu1_of(k_f12)
    out = dict(
        k_f12=k_f12, mu1=mu1,
        E_C3=float(c3.mean()), sd_C3=float(c3.std()),
        E_C5=float(c5.mean()), sd_C5=float(c5.std()),
        gap3=float(c3.mean() - mu1), gap5=float(c5.mean() - mu1),
        sd_d=d13, delta=dl13,
        p3_solo=pair.P(c3, mu1, 0.0, 0.0), p5_solo=pair.P(c5, mu1, 0.0, 0.0),
        p3_pair=pair.P(c3, mu1, dl13, d13), p5_pair=pair.P(c5, mu1, dl13, d13),
        sd_d_V12_truncated=sd_d("f12", "a3g0", force_V="f12"),
    )
    # сверка генератора поля с p_top3.sample_field (побитово)
    import p_top3 as mdl_halite
    out["field_bitwise_equal_to_p_top3"] = bool(
        np.array_equal(f, mdl_halite.sample_field(u.shape[0], SEED_FIELD)))
    return out


# ---------------------------------------------------------------- 2. структура поля

def field_structure(u) -> list:
    f = build_field(u)
    names = [b[0] for b in BOARD] + [f"хвост{j+1}" for j in range(TAIL_N)]
    pubs = [b[1] for b in BOARD] + [TAIL_PUB0 + TAIL_STEP * j for j in range(TAIL_N)]
    subs = [b[2] for b in BOARD] + [TAIL_SUB] * TAIL_N
    order = np.argsort(f, axis=1)
    i3, i5 = order[:, 2], order[:, 4]
    rows = []
    for i, (nm, pub, n) in enumerate(zip(names, pubs, subs)):
        col = f[:, i]
        # доля розыгрышей, где кап n*PER связывает (приват = pub + n*PER)
        cap_bind = float((u[:, i] * (REF - pub) >= n * PER).mean())
        rows.append(dict(name=nm, pub=pub, n=n, cap_noises=n * PER / NOISE,
                         e_priv=float(col.mean()), sd_noises=float(col.std()) / NOISE,
                         cap_bind=cap_bind,
                         is_c3=float((i3 == i).mean()), is_c5=float((i5 == i).mean())))
    return rows


# ---------------------------------------------------------------- 3. чувствительность

def sens_shift(u, pair, hs=(-4, -3, -2, -1, 0, 1, 2, 3, 4)) -> dict:
    """Порог и P при равномерном сдвиге ВСЕГО поля на s шумов.

    s < 0 = соперники улучшились (льют до 23:59), s > 0 = поле просело.
    """
    d13, dl13 = sd_d("f12", "a3g0"), offset(PUB_F13, K_F13)
    rows = []
    for s in hs:
        f = build_field(u, shift=s * NOISE)
        c3, c5 = ck(f, 3), ck(f, 5)
        rows.append(dict(s=s, E_C3=float(c3.mean()), E_C5=float(c5.mean()),
                         p3=pair.P(c3, MU_SLOT1, dl13, d13),
                         p5=pair.P(c5, MU_SLOT1, dl13, d13)))
    h = NOISE
    fp, fm = build_field(u, shift=+h), build_field(u, shift=-h)
    der = {}
    for k in (1, 3, 5, 10):
        der[k] = float((ck(fp, k).mean() - ck(fm, k).mean()) / (2 * h))
    dp = {}
    for k, c in ((3, 3), (5, 5)):
        dp[k] = float((pair.P(ck(fp, c), MU_SLOT1, dl13, d13)
                       - pair.P(ck(fm, c), MU_SLOT1, dl13, d13)) / (2 * h) * NOISE)
    return dict(rows=rows, dE=der, dP_per_noise=dp)


def sens_composition(u, pair, n_show: int = 10) -> dict:
    """Производная порога по УДАЛЕНИЮ и по ДОБАВЛЕНИЮ одной команды."""
    d13, dl13 = sd_d("f12", "a3g0"), offset(PUB_F13, K_F13)
    f0 = build_field(u)
    b3, b5 = float(ck(f0, 3).mean()), float(ck(f0, 5).mean())
    p3, p5 = pair.P(ck(f0, 3), MU_SLOT1, dl13, d13), pair.P(ck(f0, 5), MU_SLOT1, dl13, d13)

    rem = []
    for i, (nm, pub, n) in enumerate(BOARD[:n_show]):
        f = build_field(u, drop=i)
        c3, c5 = ck(f, 3), ck(f, 5)
        rem.append(dict(name=nm, pub=pub, n=n,
                        d3=(float(c3.mean()) - b3) / NOISE,
                        d5=(float(c5.mean()) - b5) / NOISE,
                        dp3=(pair.P(c3, MU_SLOT1, dl13, d13) - p3) * 100,
                        dp5=(pair.P(c5, MU_SLOT1, dl13, d13) - p5) * 100))

    # добавление: новая команда с пабликом pub и n посылок.
    u2 = np.concatenate([u, draw_u(u.shape[0], 1, seed=99991)], axis=1)
    grid = [(1.6450, 60), (1.6452, 60), (1.6453, 60), (1.6455, 60), (1.6456, 60),
            (1.6455, 12), (1.6456, 12), (1.6457, 12), (1.6458, 12), (1.6459, 12),
            (1.6459, 60), (1.6460, 12)]
    addr = []
    for pub, n in grid:
        f = build_field(u2, add=[("новая", pub, n)])
        c3, c5 = ck(f, 3), ck(f, 5)
        addr.append(dict(pub=pub, n=n,
                         d3=(float(c3.mean()) - b3) / NOISE,
                         d5=(float(c5.mean()) - b5) / NOISE,
                         dp3=(pair.P(c3, MU_SLOT1, dl13, d13) - p3) * 100,
                         dp5=(pair.P(c5, MU_SLOT1, dl13, d13) - p5) * 100))
    return dict(base=dict(E_C3=b3, E_C5=b5, p3=p3, p5=p5), remove=rem, add=addr)


# ---------------------------------------------------------------- 4. кандидаты

def cand_configs(tag: str, k: float, k_f12: float = K_F12):
    """Три конфигурации кандидата. Возвращает (mu1, delta, sd_d) для каждой."""
    pub = CANDIDATES[tag]["pub"]
    off_c = offset(pub, k, k_f12)
    off_13 = offset(PUB_F13, K_F13, k_f12)
    mu1 = mu1_of(k_f12)
    return {
        # кандидат в слот-2, слот-1 остаётся F12 (счёт pair_top3.py)
        "slot2_with_F12": (mu1, off_c, sd_d("f12", tag)),
        # кандидат в слот-1, слот-2 остаётся F13_g0 (F12 вытеснен)
        "slot1_with_F13": (mu1 + off_c, off_13 - off_c, sd_d(tag, "a3g0")),
        # кандидат в слот-1, слот-2 = F12 (вытеснен F13_g0)
        "slot1_with_F12": (mu1 + off_c, -off_c, sd_d(tag, "f12")),
    }


def cand_table(u, pair, k_f12: float = K_F12, ks=K_GRID) -> dict:
    f = build_field(u)
    c3, c5 = ck(f, 3), ck(f, 5)
    d13, dl13 = sd_d("f12", "a3g0"), offset(PUB_F13, K_F13, k_f12)
    mu1 = mu1_of(k_f12)
    base = dict(p3=pair.P(c3, mu1, dl13, d13), p5=pair.P(c5, mu1, dl13, d13),
                p3_solo=pair.P(c3, mu1, 0, 0), p5_solo=pair.P(c5, mu1, 0, 0),
                mu1=mu1)
    out = {}
    for tag in CANDIDATES:
        cfgs = {}
        for name in ("slot2_with_F12", "slot1_with_F13", "slot1_with_F12"):
            grid = []
            for k in ks:
                mu1, dl, sd = cand_configs(tag, k, k_f12)[name]
                grid.append(dict(k=k, p3=pair.P(c3, mu1, dl, sd),
                                 p5=pair.P(c5, mu1, dl, sd)))
            kstar5 = _kstar(pair, c5, tag, name, base["p5"], k_f12)
            kstar3 = _kstar(pair, c3, tag, name, base["p3"], k_f12)
            cfgs[name] = dict(grid=grid, kstar5=kstar5, kstar3=kstar3,
                              sd_d=cand_configs(tag, 10.0, k_f12)[name][2])
        out[tag] = cfgs
    return dict(base=base, cands=out, k_f12=k_f12)


def _kstar(pair, thr, tag, cfg, target, k_f12, lo=0.0, hi=40.0):
    """k, при котором конфигурация перестаёт бить текущую пару (P = target)."""
    def g(k):
        mu1, dl, sd = cand_configs(tag, k, k_f12)[cfg]
        return pair.P(thr, mu1, dl, sd) - target
    if g(lo) <= 0:
        return float("nan")          # не бьёт даже при k = 0
    if g(hi) > 0:
        return float("inf")          # бьёт при любом мыслимом k
    for _ in range(50):
        m = 0.5 * (lo + hi)
        if g(m) > 0:
            lo = m
        else:
            hi = m
    return 0.5 * (lo + hi)


# ---------------------------------------------------------------- 5. карта решения

MEASURED_SHIFTS = (-20, -10, -6, -4, -2, 0, 2, 4, 6, 10, 20)


def decision_map(u, pair, shifts=MEASURED_SHIFTS, k_f12: float = K_F12) -> dict:
    """Карта решения по ЗАМЕРЕННЫМ файлам: меняется ли оптимальная пара со сдвигом.

    Только F12, F13_g0 и F8 — это единственные три файла с фактическим замером
    паблика. Кандидаты сюда не входят: их место — таблица k*(сдвиг) ниже, потому
    что их порядок определяется не полем, а невыведенным k.
    """
    d13, dl13 = sd_d("f12", "a3g0"), offset(PUB_F13, K_F13, k_f12)
    d8 = sd_d("f12", "base")
    dl8 = offset(1.6458055, 8.30, k_f12)      # F8: паблик по линейному закону слипа
    mu1 = mu1_of(k_f12)
    opts = [("F12+F13_g0", (mu1, dl13, d13)),
            ("F12+F8", (mu1, dl8, d8)),
            ("F12 соло", (mu1, 0.0, 0.0)),
            ("F13_g0+F12*", (mu1 + dl13, -dl13, d13))]
    rows = []
    for s in shifts:
        f = build_field(u, shift=s * NOISE)
        c3, c5 = ck(f, 3), ck(f, 5)
        v5 = {nm: pair.P(c5, *cfg) for nm, cfg in opts}
        v3 = {nm: pair.P(c3, *cfg) for nm, cfg in opts}
        rows.append(dict(s=s, best5=max(v5, key=v5.get), best3=max(v3, key=v3.get),
                         p5=v5, p3=v3))
    return dict(options=[nm for nm, _ in opts], rows=rows)


def scenarios(u, shifts=(-4, -2, 0, 2, 4)) -> list:
    """Сценарии доски: равномерные сдвиги + шоки состава (одна команда)."""
    sc = [(f"{s:+d}ш", build_field(u, shift=s * NOISE)) for s in shifts]
    u2 = np.concatenate([u, draw_u(u.shape[0], 1, seed=99991)], axis=1)
    sc.append(("+нов.n12@1.6456", build_field(u2, add=[("новая", 1.6456, 12)])))
    sc.append(("+нов.n60@1.6453", build_field(u2, add=[("новая", 1.6453, 60)])))
    sc.append(("-Ежи", build_field(u, drop=0)))
    sc.append(("-T(n11)", build_field(u, drop=4)))
    return sc


def kstar_map(u, pair, shifts=(-4, -2, 0, 2, 4), k_f12: float = K_F12) -> dict:
    """k*, при котором кандидат перестаёт бить текущую пару, — по сценариям доски.

    Это и есть карта устойчивости решения: если k* почти не зависит от доски,
    решение держится на k, а не на скрине, и свежий скрин его не переворачивает.
    """
    d13 = sd_d("f12", "a3g0")
    dl13 = offset(PUB_F13, K_F13, k_f12)
    out = {}
    for lab, f in scenarios(u, shifts):
        c3, c5 = ck(f, 3), ck(f, 5)
        base5 = pair.P(c5, mu1_of(k_f12), dl13, d13)
        base3 = pair.P(c3, mu1_of(k_f12), dl13, d13)
        row = {}
        for tag in CANDIDATES:
            row[tag] = dict(
                slot2=_kstar(pair, c5, tag, "slot2_with_F12", base5, k_f12),
                slot1=_kstar(pair, c5, tag, "slot1_with_F13", base5, k_f12))
        out[lab] = dict(base5=base5, base3=base3, kstar=row)
    return out


# ---------------------------------------------------------------- 6. дисперсия

def variance_answer(u, pair, k_f12: float = K_F12) -> dict:
    f = build_field(u)
    c3, c5 = ck(f, 3), ck(f, 5)
    d13, dl13 = sd_d("f12", "a3g0"), offset(PUB_F13, K_F13, k_f12)
    h = 0.05 * SIGMA_US
    hd = 0.05 * d13
    hm = 1e-5

    MU = mu1_of(k_f12)

    def P(k, mu=MU, sig=SIGMA_US, dl=dl13, sd=d13):
        return pair.P(c3 if k == 3 else c5, mu, dl, sd, sig)

    out = {}
    for k in (3, 5):
        out[k] = dict(
            p=P(k),
            dP_dsigma=(P(k, sig=SIGMA_US + h) - P(k, sig=SIGMA_US - h)) / (2 * h),
            dP_dsd_d=(P(k, sd=d13 + hd) - P(k, sd=d13 - hd)) / (2 * hd),
            dP_dmu=(P(k, mu=MU + hm) - P(k, mu=MU - hm)) / (2 * hm),
        )
        out[k]["dP_dsigma_pp_per_noise"] = out[k]["dP_dsigma"] * NOISE * 100
        out[k]["dP_dsd_d_pp_per_noise"] = out[k]["dP_dsd_d"] * NOISE * 100
        out[k]["dP_dmu_pp_per_noise"] = out[k]["dP_dmu"] * NOISE * 100
        # обменный курс: сколько шумов mu можно отдать за 1 шум sigma
        out[k]["rate"] = (-out[k]["dP_dsigma"] / out[k]["dP_dmu"]
                          if out[k]["dP_dmu"] else float("nan"))
    # точка перелома знака dP/dsigma по сдвигу mu.
    # Ищем ПЕРВУЮ смену знака от текущей точки, сканируя сетку: на краях
    # (P -> 0 или P -> 1) производная тоже уходит в ноль, и слепой бисект
    # ловит там ложный корень.
    grid = np.arange(-30.0, 40.01, 0.5) * NOISE

    def dsig(k, s):
        return (P(k, mu=MU + s, sig=SIGMA_US + h)
                - P(k, mu=MU + s, sig=SIGMA_US - h)) / (2 * h)

    for k in (3, 5):
        vals = [dsig(k, s) for s in grid]
        i0 = int(np.argmin(np.abs(grid)))          # текущая точка
        br = None
        for j in range(i0, len(grid) - 1):         # вправо (mu хуже)
            if vals[j] * vals[j + 1] <= 0:
                br = (grid[j], grid[j + 1]); break
        if br is None:
            for j in range(i0, 0, -1):             # влево (mu лучше)
                if vals[j] * vals[j - 1] <= 0:
                    br = (grid[j - 1], grid[j]); break
        if br is None:
            out[k]["breakeven_shift"] = float("nan")
            out[k]["breakeven_p"] = float("nan")
            continue
        lo, hi = br
        for _ in range(40):
            m = 0.5 * (lo + hi)
            if dsig(k, lo) * dsig(k, m) <= 0:
                hi = m
            else:
                lo = m
        s = 0.5 * (lo + hi)
        out[k]["breakeven_shift"] = s
        out[k]["breakeven_noises"] = s / NOISE
        out[k]["breakeven_p"] = P(k, mu=MU + s)
    # сдвиг ПОЛЯ, при котором знак dP/dsigma меняется (эквивалент того же
    # вопроса, но в валюте доски: соперники льют до 23:59)
    for k in (3, 5):
        def dsig_field(sf):
            ff = build_field(u, shift=sf * NOISE)
            cc = ck(ff, k)
            return (pair.P(cc, MU, dl13, d13, SIGMA_US + h)
                    - pair.P(cc, MU, dl13, d13, SIGMA_US - h)) / (2 * h)
        g = np.arange(-30.0, 30.01, 1.0)
        v = [dsig_field(x) for x in g]
        br = None
        for j in range(len(g) - 1):
            if v[j] * v[j + 1] <= 0:
                br = (g[j], g[j + 1]); break
        if br is None:
            out[k]["field_breakeven_noises"] = float("nan")
            continue
        lo, hi = br
        for _ in range(30):
            m = 0.5 * (lo + hi)
            if dsig_field(lo) * dsig_field(m) <= 0:
                hi = m
            else:
                lo = m
        out[k]["field_breakeven_noises"] = 0.5 * (lo + hi)
    # профиль dP/dsigma по сдвигу mu
    out["dsigma_profile"] = [
        dict(s=float(s / NOISE), p3=P(3, mu=MU + s), p5=P(5, mu=MU + s),
             d3=dsig(3, s) * NOISE * 100, d5=dsig(5, s) * NOISE * 100)
        for s in np.array([-10, -5, -2, 0, 2, 4, 5.64, 8, 10, 15, 20]) * NOISE]
    # строка по sigma
    row = []
    for m in (0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5):
        row.append(dict(mult=m, sigma=SIGMA_US * m,
                        p3=P(3, sig=SIGMA_US * m), p5=P(5, sig=SIGMA_US * m)))
    out["sigma_row"] = row
    # строка по sd_d (развязка пары)
    rowd = []
    for m in (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0):
        rowd.append(dict(mult=m, sd_noises=d13 * m / NOISE,
                         p3=P(3, sd=d13 * m), p5=P(5, sd=d13 * m)))
    out["sd_d_row"] = rowd
    return out


# ------------------------------------------- 9. пары в ИСПРАВЛЕННОЙ приватной валюте

def pair_P(pair, thr3, thr5, a: str, b: str | None, mu_shift: float = 0.0,
           ev: dict | None = None):
    """P(топ-3), P(топ-5), sd_d, delta для пары (a = слот-1, b = слот-2).

    Ожидания берутся из EV_PRIV (шумы против F12), якорь mu(F12) = MU_SLOT1.
    b = None — одиночный файл.
    """
    ev = ev or EV_PRIV
    mu1 = MU_SLOT1 + (ev[a] + mu_shift) * NOISE
    if b is None:
        return pair.P(thr3, mu1, 0.0, 0.0), pair.P(thr5, mu1, 0.0, 0.0), 0.0, 0.0
    dl = (ev[b] - ev[a]) * NOISE
    s = sd_d(a, b)
    return pair.P(thr3, mu1, dl, s), pair.P(thr5, mu1, dl, s), s, dl


def pairs_corrected(u, pair, ev: dict | None = None, mu_shift: float = 0.0) -> dict:
    """Полный перебор пар в исправленной приватной валюте + порог переворота."""
    ev = ev or EV_PRIV
    f = build_field(u, shift=0.0)
    c3, c5 = ck(f, 3), ck(f, 5)
    rows = []
    for a in ev:
        for b in list(ev) + [None]:
            if b == a:
                continue
            p3, p5, s, dl = pair_P(pair, c3, c5, a, b, mu_shift, ev)
            rows.append(dict(a=a, b=b, p3=p3, p5=p5, sd_d=s / NOISE, delta=dl / NOISE,
                             ready=(a in MEASURED and (b is None or b in MEASURED)),
                             tonight=(a in MEASURED | TONIGHT
                                      and (b is None or b in MEASURED | TONIGHT))))
    rows.sort(key=lambda r: -r["p5"])
    cur = next(r for r in rows if r["a"] == "f12" and r["b"] == "a3g0")
    best_ready = next(r for r in rows if r["ready"])
    best_night = next(r for r in rows if r["tonight"])
    best_any = rows[0]
    return dict(rows=rows, current=cur, best_ready=best_ready,
                best_tonight=best_night, best_any=best_any)


def flip_threshold(u, pair, tag: str = "int08", ev: dict | None = None) -> float:
    """ΔE[priv] кандидата, при котором пара с ним сравнивается с текущей парой."""
    ev = dict(ev or EV_PRIV)
    f = build_field(u)
    c3, c5 = ck(f, 3), ck(f, 5)
    base = pair_P(pair, c3, c5, "f12", "a3g0", ev=ev)[1]

    def g(d):
        ev[tag] = d
        return max(pair_P(pair, c3, c5, "f12", tag, ev=ev)[1],
                   pair_P(pair, c3, c5, tag, "f12", ev=ev)[1]) - base

    lo, hi = -5.0, 12.0
    if g(lo) <= 0 or g(hi) > 0:
        return float("nan")
    for _ in range(50):
        m = 0.5 * (lo + hi)
        if g(m) > 0:
            lo = m
        else:
            hi = m
    return 0.5 * (lo + hi)


def level_fork(u, pair) -> dict:
    """Развилка уровня k(F8) = 8.30 (аудит) против 23.0 (маршрут C, R2c).

    Ветка «односторонне» — двигаем только НАШЕ ожидание. Ветка «согласованно» —
    приор поля phi масштабируется тем же множителем, потому что он откалиброван
    той же машинкой на нашем же замере.
    """
    import p_top3 as mdl_halite
    unit = PRIV_TRANSFER * NOISE_PER_DIR
    mu_routec = PUB_F12 + K_ROUTEC_F12 * unit
    shift = (mu_routec - MU_SLOT1) / NOISE
    out = dict(m2_transfer_dirs=(MU_SLOT1 - PUB_F12) / unit,
               mu_routec=mu_routec, shift_noises=shift)
    f = build_field(u)
    c3, c5 = ck(f, 3), ck(f, 5)
    out["staff"] = {nm: pair_P(pair, c3, c5, a, b)[1]
                    for nm, (a, b) in {"f12+a3g0": ("f12", "a3g0"),
                                       "f12+int08": ("f12", "int08"),
                                       "int08+f12": ("int08", "f12")}.items()}
    out["one_sided"] = {nm: pair_P(pair, c3, c5, a, b, mu_shift=shift)[1]
                        for nm, (a, b) in {"f12+a3g0": ("f12", "a3g0"),
                                           "f12+int08": ("f12", "int08"),
                                           "int08+f12": ("int08", "f12")}.items()}
    fac = K_ROUTEC_F8 / K_AUDIT_F8
    g = globals()
    a0, b0 = g["PHI_A"], g["PHI_B"]
    conc = a0 + b0
    m1 = min(a0 / conc * fac, 0.95)
    g["PHI_A"], g["PHI_B"] = conc * m1, conc * (1 - m1)
    u2 = draw_u(u.shape[0])
    f2 = build_field(u2)
    c3b, c5b = ck(f2, 3), ck(f2, 5)
    out["consistent"] = {nm: pair_P(pair, c3b, c5b, a, b, mu_shift=shift)[1]
                         for nm, (a, b) in {"f12+a3g0": ("f12", "a3g0"),
                                            "f12+int08": ("f12", "int08"),
                                            "int08+f12": ("int08", "f12")}.items()}
    out["consistent_field"] = dict(factor=fac, phi_mean=m1,
                                   E_C3=float(c3b.mean()), E_C5=float(c5b.mean()))
    g["PHI_A"], g["PHI_B"] = a0, b0
    return out


# ---------------------------------------------------------------- 8. устойчивость

def robustness(ns: int = NS) -> dict:
    """Устойчивость E[C_(5)] и P(топ-5) к сиду и к приору phi.

    Приор Beta(1.5,12) откалиброван на НАШЕМ замеренном phi = 0.086 — то есть
    на одной точке. Если у соперников подгонка агрессивнее, поле проседает,
    и оба порога едут. Меняем PHI_A/PHI_B в p_top3 и пересобираем поле.
    """
    import p_top3 as mdl_halite
    d13, dl13 = sd_d("f12", "a3g0"), offset(PUB_F13, K_F13)
    out = {"seeds": [], "phi": {}}
    for sd_seed in (2026, 1111, 2222, 3333, 4444):
        u = draw_u(ns, seed=sd_seed)
        f = build_field(u)
        pair = Pair(ns, seed_us=777 + sd_seed)
        c3, c5 = ck(f, 3), ck(f, 5)
        out["seeds"].append(dict(seed=sd_seed, E_C3=float(c3.mean()),
                                 E_C5=float(c5.mean()),
                                 p3=pair.P(c3, MU_SLOT1, dl13, d13),
                                 p5=pair.P(c5, MU_SLOT1, dl13, d13)))
    pair = Pair(ns)
    a0, b0 = mdl_halite.PHI_A, mdl_halite.PHI_B
    g = globals()
    for tag, (pa, pb) in {"Beta(1.5,12) базовый": (1.5, 12.0),
                          "Beta(2,10) mean .167": (2.0, 10.0),
                          "Beta(1,15) mean .0625": (1.0, 15.0),
                          "Beta(3,6)  mean .333": (3.0, 6.0)}.items():
        g["PHI_A"], g["PHI_B"] = pa, pb
        u = draw_u(ns)
        f = build_field(u)
        c3, c5 = ck(f, 3), ck(f, 5)
        out["phi"][tag] = dict(E_C3=float(c3.mean()), E_C5=float(c5.mean()),
                               sd_C5=float(c5.std()),
                               p3=pair.P(c3, MU_SLOT1, dl13, d13),
                               p5=pair.P(c5, MU_SLOT1, dl13, d13))
    g["PHI_A"], g["PHI_B"] = a0, b0
    return out


# ---------------------------------------------------------------- печать

def main() -> None:
    ap = argparse.ArgumentParser()
    for flag in ("control", "field", "sens", "cands", "map", "sigma", "robust",
                 "pairs", "all"):
        ap.add_argument(f"--{flag}", action="store_true")
    ap.add_argument("--ns", type=int, default=NS)
    ap.add_argument("--k-f12", type=float, default=K_F12)
    ap.add_argument("--json-out", default="")
    a = ap.parse_args()
    if not any(getattr(a, f) for f in ("control", "field", "sens", "cands",
                                       "map", "sigma", "robust", "pairs")):
        a.all = True
    u = draw_u(a.ns)
    pair = Pair(a.ns)
    res = {}
    W = 78

    if a.control or a.all:
        print("=" * W)
        print(f"1. КОНТРОЛЬ: пятая порядковая статистика C_(5), NS={a.ns:,}")
        print("=" * W)
        c = control(u, pair, a.k_f12)
        res["control"] = c
        print(f"  генератор поля побитово = p_top3.sample_field: "
              f"{c['field_bitwise_equal_to_p_top3']}")
        print(f"  E[C_(3)] = {c['E_C3']:.7f}   sd = {c['sd_C3']:.7f} "
              f"({c['sd_C3']/NOISE:.2f} шума)")
        print(f"  E[C_(5)] = {c['E_C5']:.7f}   sd = {c['sd_C5']:.7f} "
              f"({c['sd_C5']/NOISE:.2f} шума)")
        print(f"  наше E[приват] пары (слот-1 F12) mu = {c['mu1']:.7f} "
              f"при k(F12) = {c['k_f12']}")
        print(f"  разрыв до C_(3): {c['gap3']:+.7f} = {c['gap3']/NOISE:+.2f} шума "
              f"(мы ПОЗАДИ)")
        print(f"  разрыв до C_(5): {c['gap5']:+.7f} = {c['gap5']/NOISE:+.2f} шума "
              f"(мы ВПЕРЕДИ)")
        print(f"  пара: delta = {c['delta']:+.7f} ({c['delta']/NOISE:+.2f} ш), "
              f"sd_d = {c['sd_d']:.6e} ({c['sd_d']/NOISE:.2f} ш)")
        print(f"  F12 соло:        P(топ-3) = {c['p3_solo']*100:6.2f} %   "
              f"P(топ-5) = {c['p5_solo']*100:6.2f} %")
        print(f"  пара F12+F13_g0: P(топ-3) = {c['p3_pair']*100:6.2f} %   "
              f"P(топ-5) = {c['p5_pair']*100:6.2f} %")
        se = math.sqrt(c["p5_pair"] * (1 - c["p5_pair"]) / a.ns) * 100
        print(f"  КОНТРОЛЬ ШТАБА:  P(топ-3) = 27.08 %, P(топ-5) = 65.59 %  -> "
              f"{'СОШЛОСЬ' if abs(c['p5_pair']*100-65.59)<0.02 else 'РАСХОЖДЕНИЕ'}")
        print(f"  MC-погрешность абсолютного P: +-{se:.3f} п.п. (1 sigma)")
        unit = PRIV_TRANSFER * NOISE_PER_DIR
        print(f"\n  ЧТО ЗДЕСЬ ВОСПРОИЗВЕДЕНО, А ЧТО ИСПРАВЛЕНО НИЖЕ.")
        print(f"    mu(F12) = {MU_SLOT1:.7f} — это якорь валюты M2: постоянный перенос")
        print(f"    {(MU_SLOT1 - PUB_F12)/unit:.2f} направления, НЕ зависящий от k(F12) "
              f"(тот же перенос у F8: "
              f"{(1.646203 - 1.6458057389)/unit:.2f}).")
        print(f"    Поэтому исправление k(F12) 9.30 -> {K_F12_EXACT} наше mu не двигает — "
              f"оно двигает")
        print(f"    ОТСТАВАНИЕ слота-2: F13_g0 отстаёт на {EV_PRIV['a3g0']:.2f} шума, "
              f"а не на {c['delta']/NOISE:.2f}.")
        print(f"    Числа 27.08 / 65.59 воспроизведены как КОНТРОЛЬ старого счёта; "
              f"действующие — в §9.")

    if a.field or a.all:
        print()
        print("=" * W)
        print("2. КТО СТАВИТ ПОРОГ: структура поля")
        print("=" * W)
        rows = field_structure(u)
        res["field"] = rows
        print(f"  {'команда':24s}{'паблик':>11s}{'n':>5s}{'кап,ш':>8s}{'E[priv]':>11s}"
              f"{'sd,ш':>7s}{'кап,%':>7s}{'=C3,%':>7s}{'=C5,%':>7s}")
        for r in rows:
            if r["is_c3"] < 0.001 and r["is_c5"] < 0.001:
                continue
            print(f"  {r['name']:24s}{r['pub']:11.7f}{r['n']:5d}{r['cap_noises']:8.1f}"
                  f"{r['e_priv']:11.6f}{r['sd_noises']:7.2f}{r['cap_bind']*100:7.1f}"
                  f"{r['is_c3']*100:7.1f}{r['is_c5']*100:7.1f}")
        small = [r for r in rows if r["n"] <= 20]
        s3 = sum(r["is_c3"] for r in small) * 100
        s5 = sum(r["is_c5"] for r in small) * 100
        print(f"\n  ИТОГ: команды с n <= 20 (кап n*PER связывает, приват ~ паблик) "
              f"ставят C_(3) в {s3:.1f} % розыгрышей, C_(5) — в {s5:.1f} %.")

    if a.sens or a.all:
        print()
        print("=" * W)
        print("3. ЧУВСТВИТЕЛЬНОСТЬ ПОРОГА: сдвиг поля")
        print("=" * W)
        sh = sens_shift(u, pair)
        res["shift"] = sh
        print(f"  {'сдвиг,ш':>9s}{'E[C_(3)]':>12s}{'E[C_(5)]':>12s}"
              f"{'P(топ-3)':>11s}{'P(топ-5)':>11s}")
        for r in sh["rows"]:
            mark = "  <<< сейчас" if r["s"] == 0 else ""
            print(f"  {r['s']:+9d}{r['E_C3']:12.7f}{r['E_C5']:12.7f}"
                  f"{r['p3']*100:10.2f}%{r['p5']*100:10.2f}%{mark}")
        print(f"\n  dE[C_(k)]/d(сдвиг паблика поля):  "
              + "   ".join(f"k={k}: {v:.4f}" for k, v in sh["dE"].items()))
        print(f"  dP/d(сдвиг поля), п.п. на 1 шум: топ-3 {sh['dP_per_noise'][3]*100:+.2f}, "
              f"топ-5 {sh['dP_per_noise'][5]*100:+.2f}   "
              f"(топ-5 чувствительнее в "
              f"{sh['dP_per_noise'][5]/sh['dP_per_noise'][3]:.2f} раза)")

        print()
        print("=" * W)
        print("4. ЧУВСТВИТЕЛЬНОСТЬ ПОРОГА: состав поля (одна команда)")
        print("=" * W)
        sc = sens_composition(u, pair)
        res["composition"] = sc
        print("  УДАЛЕНИЕ команды (она снялась / оказалась нашей витриной):")
        print(f"  {'команда':24s}{'n':>5s}{'dE[C3],ш':>10s}{'dE[C5],ш':>10s}"
              f"{'dP3,пп':>9s}{'dP5,пп':>9s}{'C5/C3':>8s}")
        for r in sc["remove"]:
            rat = r["d5"] / r["d3"] if abs(r["d3"]) > 1e-9 else float("inf")
            print(f"  {r['name']:24s}{r['n']:5d}{r['d3']:10.2f}{r['d5']:10.2f}"
                  f"{r['dp3']:+9.2f}{r['dp5']:+9.2f}{rat:8.1f}")
        print("\n  ДОБАВЛЕНИЕ новой команды (паблик, число посылок):")
        print(f"  {'паблик':>10s}{'n':>5s}{'dE[C3],ш':>10s}{'dE[C5],ш':>10s}"
              f"{'dP3,пп':>9s}{'dP5,пп':>9s}{'C5/C3':>8s}")
        for r in sc["add"]:
            rat = r["d5"] / r["d3"] if abs(r["d3"]) > 1e-9 else float("inf")
            print(f"  {r['pub']:10.4f}{r['n']:5d}{r['d3']:10.2f}{r['d5']:10.2f}"
                  f"{r['dp3']:+9.2f}{r['dp5']:+9.2f}{rat:8.1f}")

    if a.cands or a.all:
        print()
        print("=" * W)
        print(f"5. КАНДИДАТЫ, ПАРАМЕТРИЗОВАННЫЕ ПО k   (k(F12) = {a.k_f12})")
        print("=" * W)
        ct = cand_table(u, pair, a.k_f12)
        res["cands"] = ct
        b = ct["base"]
        print(f"  база — текущая пара F12+F13_g0: P(топ-3) = {b['p3']*100:.2f} %, "
              f"P(топ-5) = {b['p5']*100:.2f} %;  F12 соло: {b['p3_solo']*100:.2f} % / "
              f"{b['p5_solo']*100:.2f} %")
        for cfg, title in (("slot2_with_F12", "КАНДИДАТ В СЛОТ-2 (слот-1 = F12)"),
                           ("slot1_with_F13", "КАНДИДАТ В СЛОТ-1 (слот-2 = F13_g0)"),
                           ("slot1_with_F12", "КАНДИДАТ В СЛОТ-1 (слот-2 = F12)")):
            print(f"\n  --- {title}: P(топ-5), % ---")
            print(f"  {'кандидат':9s}{'sd_d,ш':>8s}|"
                  + "".join(f"{k:>7.1f}" for k in K_GRID) + f"{'k*(mdl_realgr)':>9s}{'k*(mdl_halite)':>8s}")
            for tag in CANDIDATES:
                c = ct["cands"][tag][cfg]
                ks5 = c["kstar5"]
                ks3 = c["kstar3"]
                f5 = ("никогда" if math.isnan(ks5) else
                      "всегда" if math.isinf(ks5) else f"{ks5:.2f}")
                f3 = ("никогда" if math.isnan(ks3) else
                      "всегда" if math.isinf(ks3) else f"{ks3:.2f}")
                print(f"  {tag:9s}{c['sd_d']/NOISE:8.2f}|"
                      + "".join(f"{g['p5']*100:7.2f}" for g in c["grid"])
                      + f"{f5:>9s}{f3:>8s}")
            print(f"  k* — то k, ВЫШЕ которого кандидат перестаёт бить текущую пару "
                  f"({b['p5']*100:.2f} % / {b['p3']*100:.2f} %).")

    if a.map or a.all:
        print()
        print("=" * W)
        print("6. КАРТА РЕШЕНИЯ: при каком сдвиге поля меняется оптимальная пара")
        print("=" * W)
        dm = decision_map(u, pair, k_f12=a.k_f12)
        res["decision_map"] = dm
        print("  A. ТОЛЬКО ЗАМЕРЕННЫЕ ФАЙЛЫ (F12, F13_g0, F8) — P(топ-5), %:")
        print(f"  {'сдвиг,ш':>9s}" + "".join(f"{m:>13s}" for m in dm["options"])
              + "   лучшая по mdl_realgr / по mdl_halite")
        for r in dm["rows"]:
            print(f"  {r['s']:+9d}"
                  + "".join(f"{r['p5'][m]*100:13.2f}" for m in dm["options"])
                  + f"   {r['best5']} / {r['best3']}")
        real = ["F12+F13_g0", "F12+F8", "F12 соло"]   # физически разные наборы галок
        f5 = {max(real, key=lambda m: r["p5"][m]) for r in dm["rows"]}
        f3 = {max(real, key=lambda m: r["p3"][m]) for r in dm["rows"]}
        print(f"  ПЕРЕВОРОТ среди физически разных наборов галок на "
              f"{dm['rows'][0]['s']:+d}..{dm['rows'][-1]['s']:+d} шума: "
              f"{'НЕТ' if len(f5) == 1 and len(f3) == 1 else 'ЕСТЬ'} "
              f"(по mdl_realgr: {', '.join(sorted(f5))}; по mdl_halite: {', '.join(sorted(f3))})")
        r0 = next(r for r in dm["rows"] if r["s"] == 0)
        print(f"  * F13_g0+F12 — ТА ЖЕ пара, роли слотов переставлены. Разница "
              f"{(r0['p5']['F12+F13_g0']-r0['p5']['F13_g0+F12*'])*100:+.2f} п.п. по mdl_realgr — "
              f"это цена КОНВЕНЦИИ модели")
        print(f"    (кто из двух файлов несёт sd_d*w), а не свойство файлов. "
              f"Она того же порядка, что вся ценность второго слота "
              f"({(r0['p5']['F12+F13_g0']-r0['p5']['F12 соло'])*100:+.2f} п.п.).")

        km = kstar_map(u, pair, k_f12=a.k_f12)
        res["kstar_map"] = km
        labs = list(km)
        print("\n  B. k* по сценариям доски — порог k, выше которого кандидат перестаёт")
        print("     бить текущую пару по P(топ-5). База mdl_realgr по сценариям: "
              + ", ".join(f"{l} {km[l]['base5']*100:.1f}%" for l in labs))
        print(f"  {'кандидат':9s}{'роль':>7s}|" + "".join(f"{l:>16s}" for l in labs)
              + f"{'размах k*':>11s}")
        for tag in CANDIDATES:
            for role, rl in (("slot2", "слот-2"), ("slot1", "слот-1")):
                vals = [km[l]["kstar"][tag][role] for l in labs]
                print(f"  {tag:9s}{rl:>7s}|" + "".join(f"{v:16.2f}" for v in vals)
                      + f"{max(vals)-min(vals):11.2f}")
        print("  РАЗМАХ k* по ВСЕМ сценариям доски — сравнивать с разбросом самих оценок k")
        print("  (реестр против route D против eb-калибровки: 1.5-2.1 направления).")

    if a.sigma or a.all:
        print()
        print("=" * W)
        print("7. ДИСПЕРСИЯ ПОД ЦЕЛЬЮ ТОП-5: беречь или наращивать")
        print("=" * W)
        va = variance_answer(u, pair, a.k_f12)
        res["variance"] = va
        for k in (3, 5):
            v = va[k]
            print(f"  топ-{k}: P = {v['p']*100:.2f} %")
            print(f"    dP/dsigma_us = {v['dP_dsigma']:+.4g}  "
                  f"({v['dP_dsigma_pp_per_noise']:+.3f} п.п. на 1 шум прироста sigma) "
                  f"-> {'НАРАЩИВАТЬ' if v['dP_dsigma']>0 else 'БЕРЕЧЬ'}")
            print(f"    dP/dsd_d     = {v['dP_dsd_d']:+.4g}  "
                  f"({v['dP_dsd_d_pp_per_noise']:+.3f} п.п. на 1 шум прироста sd_d)")
            print(f"    dP/dmu       = {v['dP_dmu']:+.4g}  "
                  f"({v['dP_dmu_pp_per_noise']:+.3f} п.п. на 1 шум ухудшения mu)")
            print(f"    курс: за +1 шум sigma можно отдать {v['rate']:.2f} шума mu")
            if not math.isnan(v.get("breakeven_p", float("nan"))):
                print(f"    знак dP/dsigma меняется при сдвиге mu "
                      f"{v['breakeven_noises']:+.2f} шума, там P = "
                      f"{v['breakeven_p']*100:.2f} %")
            fb = v.get("field_breakeven_noises", float("nan"))
            if not math.isnan(fb):
                word = (f"поле просело на {fb:.2f} шума" if fb > 0
                        else f"соперники улучшились на {-fb:.2f} шума")
                print(f"    то же в валюте доски: знак меняется при сдвиге ПОЛЯ "
                      f"{fb:+.2f} шума ({word})")
        print(f"\n  профиль dP/dsigma по сдвигу нашего mu (+ = хуже):")
        print(f"  {'сдвиг,ш':>9s}{'mdl_halite,%':>8s}{'dP3/dsig,пп/ш':>15s}"
              f"{'mdl_realgr,%':>8s}{'dP5/dsig,пп/ш':>15s}")
        for r in va["dsigma_profile"]:
            print(f"  {r['s']:+9.2f}{r['p3']*100:8.2f}{r['d3']:15.3f}"
                  f"{r['p5']*100:8.2f}{r['d5']:15.3f}")
        print(f"\n  {'sigma':>10s}{'x база':>8s}{'P(топ-3)':>11s}{'P(топ-5)':>11s}")
        for r in va["sigma_row"]:
            print(f"  {r['sigma']:10.6f}{r['mult']:8.2f}{r['p3']*100:10.2f}%"
                  f"{r['p5']*100:10.2f}%")
        print(f"\n  развязка пары sd_d (при том же mu и sigma):")
        print(f"  {'sd_d,ш':>10s}{'x база':>8s}{'P(топ-3)':>11s}{'P(топ-5)':>11s}")
        for r in va["sd_d_row"]:
            print(f"  {r['sd_noises']:10.2f}{r['mult']:8.2f}{r['p3']*100:10.2f}%"
                  f"{r['p5']*100:10.2f}%")

    if a.pairs or a.all:
        print()
        print("=" * W)
        print("9. ПАРЫ В ИСПРАВЛЕННОЙ ПРИВАТНОЙ ВАЛЮТЕ (kcalc_3008.md §5, k(F12)=7.755)")
        print("=" * W)
        pc = pairs_corrected(u, pair)
        res["pairs_corrected"] = pc
        cur = pc["current"]
        solo = next(r for r in pc["rows"] if r["a"] == "f12" and r["b"] is None)
        print(f"  ТЕКУЩАЯ ПАРА F12 + F13_g0 при отставании {cur['delta']:+.2f} шума "
              f"(было +4.14):")
        print(f"    P(топ-5) = {cur['p5']*100:.2f} %   (штаб объявлял 65.59 % -> "
              f"ЗАВЫШЕНО на {(0.6559-cur['p5'])*100:.2f} п.п.)")
        print(f"    P(топ-3) = {cur['p3']*100:.2f} %   (штаб объявлял 27.08 % -> "
              f"ЗАВЫШЕНО на {(0.2708-cur['p3'])*100:.2f} п.п.)")
        print(f"    F12 соло = {solo['p5']*100:.2f} % / {solo['p3']*100:.2f} %  -> "
              f"второй слот покупает {(cur['p5']-solo['p5'])*100:+.2f} п.п. mdl_realgr "
              f"(было +0.69)")
        hdr = (f"  {'слот-1':9s}{'слот-2':9s}{'sd_d,ш':>8s}{'Δ,ш':>8s}"
               f"{'P(топ-5)':>10s}{'ΔP5':>8s}{'P(топ-3)':>10s}")

        def show(rows, n=None):
            for r in (rows if n is None else rows[:n]):
                print(f"  {r['a']:9s}{(r['b'] or '— соло'):9s}{r['sd_d']:8.2f}"
                      f"{r['delta']:+8.2f}{r['p5']*100:9.2f}%"
                      f"{(r['p5']-cur['p5'])*100:+8.2f}{r['p3']*100:9.2f}%")

        print("\n  A. ТОЛЬКО ЗАМЕРЕННЫЕ ФАЙЛЫ (F12, F13_g0, F8) — что можно поставить сейчас")
        print(hdr)
        show([r for r in pc["rows"] if r["ready"]])
        print("\n  B. + сегодняшние замеры (F14 int08, F17 int10, F15 ext283, F18 extebz),")
        print("     если они лягут в коридор §9 — топ-8")
        print(hdr)
        show([r for r in pc["rows"] if r["tonight"] and not r["ready"]], 8)
        print("\n  C. что остаётся на столе НЕотправленным (справочно, топ-4)")
        print(hdr)
        show([r for r in pc["rows"] if not r["tonight"]], 4)
        br, bn, ba = pc["best_ready"], pc["best_tonight"], pc["best_any"]
        print(f"\n  ЛУЧШАЯ из ЗАМЕРЕННЫХ:      {br['a']}+{br['b']}  "
              f"mdl_realgr = {br['p5']*100:.2f} %  ({(br['p5']-cur['p5'])*100:+.2f} п.п.)")
        print(f"  ЛУЧШАЯ с сегодняшним замером: {bn['a']}+{bn['b']}  "
              f"mdl_realgr = {bn['p5']*100:.2f} %  ({(bn['p5']-cur['p5'])*100:+.2f} п.п.)")
        print(f"  ЛУЧШАЯ вообще (файл не отправлен): {ba['a']}+{ba['b']}  "
              f"mdl_realgr = {ba['p5']*100:.2f} %  ({(ba['p5']-cur['p5'])*100:+.2f} п.п.)")
        ft = flip_threshold(u, pair, "int08")
        print(f"\n  УСТОЙЧИВОСТЬ решения по int08: пара с ним бьёт текущую, пока его")
        print(f"  ΔE[priv] лучше {ft:+.2f} шума. Замеренная оценка −0.71 "
              f"(интервал −0.71 … +0.56 по §5.2 kcalc). Запас {ft+0.71:.2f} шума.")

        print()
        print("=" * W)
        print("10. РАЗВИЛКА УРОВНЯ: k(F8) = 8.30 (аудит) против 23.0 (маршрут C)")
        print("=" * W)
        lf = level_fork(u, pair)
        res["level_fork"] = lf
        print(f"  M2-якорь mu(F12) = {MU_SLOT1:.7f} — это перенос "
              f"{lf['m2_transfer_dirs']:.2f} направлений,")
        print(f"  то есть уже между точным дифференциалом (7.755) и маршрутом C "
              f"({K_ROUTEC_F12}).")
        print(f"  Маршрут C: mu(F12) = {lf['mu_routec']:.7f}, "
              f"{lf['shift_noises']:+.2f} шума к якорю.")
        print(f"  Согласованная ветка: приор phi x{lf['consistent_field']['factor']:.2f} "
              f"-> mean {lf['consistent_field']['phi_mean']:.4f}, "
              f"E[C_(5)] = {lf['consistent_field']['E_C5']:.7f}")
        print(f"\n  {'пара':14s}{'валюта штаба':>16s}{'уровень C, односторонне':>26s}"
              f"{'уровень C, согласованно':>26s}")
        for nm in ("f12+a3g0", "f12+int08", "int08+f12"):
            print(f"  {nm:14s}{lf['staff'][nm]*100:15.2f}%"
                  f"{lf['one_sided'][nm]*100:25.2f}%{lf['consistent'][nm]*100:25.2f}%")
        for br_ in ("staff", "one_sided", "consistent"):
            b = max(lf[br_], key=lf[br_].get)
            print(f"  лучшая в ветке {br_:11s}: {b}")

    if a.robust or a.all:
        print()
        print("=" * W)
        print("8. УСТОЙЧИВОСТЬ: сид поля и приор phi")
        print("=" * W)
        rb = robustness(a.ns)
        res["robust"] = rb
        print(f"  {'сид':>8s}{'E[C_(3)]':>12s}{'E[C_(5)]':>12s}{'P(топ-3)':>11s}{'P(топ-5)':>11s}")
        for r in rb["seeds"]:
            print(f"  {r['seed']:8d}{r['E_C3']:12.7f}{r['E_C5']:12.7f}"
                  f"{r['p3']*100:10.2f}%{r['p5']*100:10.2f}%")
        p5s = [r["p5"] for r in rb["seeds"]]
        print(f"  разброс P(топ-5) по сидам: {min(p5s)*100:.2f}..{max(p5s)*100:.2f} %")
        print(f"\n  {'приор phi':24s}{'E[C_(3)]':>12s}{'E[C_(5)]':>12s}"
              f"{'sd C5,ш':>9s}{'P(топ-3)':>11s}{'P(топ-5)':>11s}")
        for tag, v in rb["phi"].items():
            print(f"  {tag:24s}{v['E_C3']:12.7f}{v['E_C5']:12.7f}"
                  f"{v['sd_C5']/NOISE:9.2f}{v['p3']*100:10.2f}%{v['p5']*100:10.2f}%")

    if a.json_out:
        os.makedirs(os.path.dirname(a.json_out) or ".", exist_ok=True)
        with open(a.json_out, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=1, default=float)
        print(f"\nJSON -> {a.json_out}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()

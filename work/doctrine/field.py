# -*- coding: utf-8 -*-
"""Модель поля соперников и ранговая задача."""
from __future__ import annotations
import math
import numpy as np
from .transfer import NOISE

# Три мира по подгонке поля. bias НЕНАБЛЮДАЕМ — отсюда три мира, а не одна модель.
WORLDS = ("A", "B", "C")


def bias_aggressive(n_subs: int, own_gap: float, own_n: int = 57,
                    n_free: int = 9) -> float:
    """Мир A: поле подгоняло как мы. Якоря — команда с n_free посылками (bias=0)
    и мы сами (own_gap = разрыв «наша витрина -> наш приват»)."""
    if n_subs <= n_free:
        return 0.0
    return own_gap * math.log(n_subs / n_free) / math.log(own_n / n_free)


def bias_moderate(n_subs: int, rate: float = 0.33) -> float:
    """Мир B: умеренная подгонка. rate откалиброван на НАШЕЙ честной цепочке."""
    return rate * NOISE * n_subs


def bias_none(n_subs: int) -> float:
    """Мир C: поле не подгоняло."""
    return 0.0


BIAS = {"A": bias_aggressive, "B": bias_moderate, "C": bias_none}


def rank_distribution(our_pop, teams, world, own_gap=None, n_tail=8,
                      tail_pub=1.64655, rng=None, sd_frac=0.5, sd_floor=2e-4):
    """Распределение нашего ранга. teams = [(имя, паблик, посылок), ...].

    our_pop — массив розыгрышей нашего приватного скора (меньше = лучше).
    """
    rng = np.random.default_rng(0) if rng is None else rng
    ns = len(our_pop)
    cols = []
    for _, pub, n in teams:
        b = (bias_aggressive(n, own_gap) if world == "A"
             else BIAS[world](n))
        cols.append(rng.normal(pub + b, max(sd_frac * b, sd_floor), ns))
    for _ in range(n_tail):
        b = (bias_aggressive(40, own_gap) if world == "A" else BIAS[world](40))
        cols.append(rng.normal(tail_pub + b, max(sd_frac * b, sd_floor), ns))
    fl = np.column_stack(cols)
    return 1 + (fl < np.asarray(our_pop)[:, None]).sum(1)


def pair_best(pop_a, pop_b):
    """Лучший из пары финалистов (меньше = лучше). Постериоры ДОЛЖНЫ быть
    посчитаны из ОДНИХ розыгрышей kappa_Q — иначе корреляция потеряна."""
    return np.minimum(np.asarray(pop_a), np.asarray(pop_b))


def rank_optimal_overdose(b_star: float, deficit: float, curvature: float) -> float:
    """Квадратичная лемма: при отставании D оптимум ПЕРЕДОЗА существует,

        b_rank = sqrt(b*^2 + D/a) > b*

    E деградирует квадратично, sd растёт линейно — рычаг СУЩЕСТВУЕТ, но короткий.
    Лотерею убивает КОРРЕЛЯЦИЯ кандидатов (0.93..1.00), а не квадратичность.
    """
    return math.sqrt(b_star * b_star + deficit / curvature)

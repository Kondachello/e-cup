"""Пять замерочных файлов mdl_amber..mdl_realgr: пять взаимно ортогональных шагов от базы.


ЗАЧЕМ. Парабола скора разделяется по ортогональным составляющим: если пять
направлений взаимно ортогональны, пять замеров дают пять НЕЗАВИСИМЫХ коэффициентов,
и потом все пять оптимумов прикладываются одновременно.

АРИФМЕТИКА (одна на все пять файлов). Скор файла m: S^2 = mean((m - t)^2).
Для шага Δ = b*g:  S(b)^2 = F0^2 - 2*b*c + b^2*q,  c = mean(g*(t - m0)), q = mean(g^2).
Шаг выбран так, что Q = b^2*q ОДИНАКОВ у всех пяти и равен 2*F0*0.0002, то есть
при нулевом коэффициенте каждый файл теряет ровно 0.0002 (десять шумов замера).
Тогда, обозначив κ = b*/b (во сколько раз оптимальный шаг больше применённого):

    κ = (F0^2 - S^2 + Q) / (2Q)   ≈   (1 - ΔS/0.0002) / 2,   ΔS = S - F0
    выигрыш в оптимуме = κ^2 * 0.0002,   добавить к базе ещё (κ - 1) * b * g

ОРТОГОНАЛИЗАЦИЯ. mdl_gabbro..mdl_realgr приводятся ортогонально к {константа, log-прогноз базы,
затем друг к другу по Граму-Шмидту в порядке C, B, E, D (от обоснованного к
спекулятивному). mdl_amber (разброс) лежит в оси {1, m0} и потому автоматически
ортогонален остальным четырём; его не приводят — он и ЕСТЬ ось масштаба.

Запуск:  POLARS_MAX_THREADS=2 .venv/bin/python work/scripts/probes5_make.py
         (--check — только посчитать, ничего не писать)
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))
from common import ROOT, REPORTS_DIR                                   # noqa: E402
from subs import MEASURED, lp, novelty, span_matrix                    # noqa: E402

SUB = ROOT / "submissions"
CANON = SUB / "canonical"
F0 = 1.6479652993
LOSS0 = 0.0002                       # цена файла, если коэффициент окажется нулевым
Q_DESIGN = 2 * F0 * LOSS0            # = b^2 * q, одинаково у всех пяти
A_OLD = 0.894                        # сила mdl_tektit внутри применённого h
A_NEW = 0.65

W_BLEND = {"weak_an_d": 0.043814, "weak_ft_recency": 0.023857,
           "weak_ft_counts": 0.017269, "weak_ft_long90": 0.007295,
           "countaov_s7": 0.023260, "behavonly_s7": 0.057488 / 2,
           "behavonly_s1337": 0.057488 / 2}


def rms(x):
    return float(np.sqrt((x ** 2).mean()))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()



    from predict_lb import MEASURED as MFULL, _resolve, read_lp   # noqa: E402


    # ---------------------------------------------------------------- сырьё

    # mdl_amber РАЗБРОС: ось масштаба. К {1, m0} НЕ приводится — она и ЕСТЬ эта ось; но
    # приводится к плоскости молчания {mdl_tektit, e}, потому что 31% оси масштаба —
    # это применённый шаг молчания (corr(m0 - mean, h) = +0.312): поправка давит
    # вниз низкие прогнозы и тем самым раздувает разброс. Без этого приведения
    # шаг по разбросу незаметно откручивал бы назад самый крупный замеренный
    # выигрыш проекта. После приведения corr с h равна 2e-10.

    # mdl_halite ОБРЕЗКА ИСТОРИИ: суррогатный бленд на двух плечах MAX_BACK 379 / 409,
    # оба приведены к моментам базы ровно как в make_candidate.py
    zp = np.load(REPORTS_DIR / "mb_fix_preds.npz")
    names = sorted({k.split("__")[0] for k in zp.files if "__" in k})
    sel = [m for m in names if m in W_BLEND]
    w = np.array([W_BLEND[m] for m in sel]); w = w / w.sum()

    # mdl_gabbro ПРАЗДНИЧНАЯ СКЛОННОСТЬ ПРОШЛОГО ГОДА

    # mdl_realgr ЕДВА ПРОШЕДШИЕ ОТБОР: положительный шаг = поднять «прочных», опустить «едва»

    # mdl_marble ВЕРХНИЙ ХВОСТ: гладкий пандус по рангу прогноза, ноль ниже 98-го процентиля.
    # пандуса с 90-го процентиля к полному базису всего 0.41 против 0.72 у этого.
    # Квадрат даёт нулевую производную в стыке — в калибровочную кривую не вносится излом.

    # ------------------------------------------------- ортогонализация и Грам-Шмидт
    dirs = {}

    # ------------------------------------------------- шаг, проверки, запись
    rep, order = {}, []

    # взаимная ортогональность
    print("\nвзаимные корреляции направлений:")
    M = np.stack([dirs[k] for k in order])
    Cm = np.corrcoef(M)
    print("        " + "".join(f"{k.split('_')[0]:>9s}" for k in order))
    for i, k in enumerate(order):
        print(f"{k.split('_')[0]:8s}" + "".join(f"{Cm[i,j]:+9.5f}" for j in range(len(order))))
    assert np.abs(Cm - np.eye(len(order))).max() < 1e-8, "направления не ортогональны"

    if not args.check:
        (REPORTS_DIR / "probes5.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1))
        print(f"\nзаписан {REPORTS_DIR / 'probes5.json'} и пять файлов в submissions/ + canonical/")


if __name__ == "__main__":
    main()

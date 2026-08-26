"""Сборка результата задач А и Б: арифметика направлений и новый файл.

ИДЕЯ. Поправка на молчание работает через ИЗБЫТОК нулевого GMV в тестовом окне по
сравнению с тем, что заложила модель. Избыток складывается из двух групп с
ПРОТИВОПОЛОЖНЫМИ знаками, потому что правило отбора считает СТРОКИ:

    Δ(x) = p_norows(x) * 1                      (в вале 0%, в тесте ~3.45% — вниз)
         + p_empty(x)  * (1 - ν_вал / R_тест)   (в вале 1.85%, в тесте ~1.12% — ВВЕРХ)

Применённое направление построено только по первой группе, а признаки, которые
предсказывают первую, предсказывают и вторую (корреляция направлений +0.46). Значит
часть приложенной силы давит вниз тех, кого надо поднимать.

АРИФМЕТИКА ПЕРЕНОСА. На чистом якоре 2025-10-15 (метки видны, окно не задевает
блоки отбора) для направления d считается выравнивание cA(d) = -cov(d, m*Y), где
Y = w_n*y_norows + w_e*y_empty — псевдометка ИЗБЫТКА нулей, приведённая к тестовым
лидерборда c_LB(d) = λ * [cA(d_якорь)/rms(d_якорь)] * rms(d_тест); λ прибит ЗАМЕРОМ:
у применённого табличного направления c_LB = 0.929*q_old (три замера).  c_LB линеен
по d, поэтому у СМЕСИ он складывается из компонент, а не пересчитывается через её
собственную норму (иначе в оценку протекает отношение норм и смесь завышается втрое).

ПРИМЕНЕНИЕ. Новинка ортогонализуется к применённому h И К КОНСТАНТЕ (уровень
прогноза прибит лидербордом, любая правка в среднем стоит ноль); замеренное не
трогается, добавляется только ортогональная часть с усадкой.

"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))
from common import ROOT                                               # noqa: E402
from silence_key import (B30, B60, FIT_OFFSETS, REC6, _bin,           # noqa: E402
                         blocks_of, fit_table, recency)
from silence_model import sig_level                                    # noqa: E402
from silence_split import build_real_cumsum                            # noqa: E402
from silence_target import build_cumsum                                # noqa: E402
from subs import MEASURED, lp, novelty, span_matrix                    # noqa: E402

OUT = ROOT / "work" / "reports"
EVAL_ANCHOR = date(2025, 10, 15)
TEST_ANCHOR = date(2026, 2, 13)

S_OPT_OLD = 0.929        # оптимальная сила применённого табличного направления (3 замера)

R_N_TEST = 0.0345        # уровень «ноль строк» в тесте (привязан к лидерборду)
P_LEVEL_OLD = 0.030843   # уровень, при котором СТРОИЛОСЬ применённое направление
NU_E_VAL = 0.018503      # доля «пустых» в окне валидации 2026-01-15..02-13
R_E_TEST = 0.011170      # доля «пустых» в тесте ~ ближайший чистый якорь


def rms(x):
    return float(np.sqrt((x ** 2).mean()))


def block_table_p(C, R, anchor, fits):
    """Ключ b8x8_rec6 (задача Б): блоки НЕ складываются, 8x8, плюс давность 6 корзин."""
    def f(C, a):
        bl = blocks_of(C, a)
        return ((_bin(bl[0], B30) * 8 + _bin(bl[1] + bl[2], B60)) * 6
                + _bin(recency(C, a), REC6)), 64 * 6
    return fit_table(f, C, R, fits, 64 * 6)[f(C, anchor)[0]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shrink", type=float, default=0.7)
    ap.add_argument("--r-e-test", type=float, default=R_E_TEST)
    ap.add_argument("--boot", type=int, default=300)
    args = ap.parse_args()





    # ---------------- направления
    fits_ev = [EVAL_ANCHOR - timedelta(days=k) for k in FIT_OFFSETS]
    fits_te = [EVAL_ANCHOR - timedelta(days=k) for k in (56, 42, 28, 14, 0)]

    def dir_of(p, w, lv):
        u = sig_level(p, lv) * w
        return -(u - float(u.mean()))


    def solve(w_e, sel=None):
        """Все коэффициенты лидерборда при данном w_e (sel = бутстрап-индексы якоря)."""

        return out

    # ортогонализация к {константа, h}: уровень прибит лидербордом, среднее стоит ноль



    P, res = {}, {}

    # ---------------- бутстрап по пользователям чистого якоря
    bs = {k: [] for k in ("nor", "emp", "comb")}
    print("\nбутстрап c⊥ (5..95%):")
    for k in bs:
        v = np.array(bs[k])
        res[k]["boot_ci"] = [float(np.percentile(v, 5)), float(np.percentile(v, 95))]
        res[k]["boot_sd"] = float(v.std())
        print(f"  {k:<5} [{res[k]['boot_ci'][0]:+.7f}, {res[k]['boot_ci'][1]:+.7f}]  "
              f"sd {res[k]['boot_sd']:.7f}")

    # ---------------- устойчивость к неизвестной доле «пустых» в тесте
    # направление ЗАФИКСИРОВАНО (строится при центральном w_e), меняется только правда
    print("\nустойчивость (направление фиксировано, меняется истинная доля «пустых»):")

    # ---------------- файл
    out = ROOT / "submissions" / f"{args.name}.csv"
    print(f"записан {out}")

    print(f"записан {OUT / 'silence_ab_apply.json'}")


if __name__ == "__main__":
    main()

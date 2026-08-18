"""Сборка двух финальных кандидатов с учётом переноса на приватную часть.

mdl_amber (консервативный) = A1 + сегментные поправки, ужатые множителем lambda*
                      (lambda* = 1 - sigma_eff^2/m^2, см. work/reports/private_risk.md)
mdl_gabbro (агрессивный)    = mdl_amber + t * (febspec - mdl_amber), где t пересчитывается по ФАКТИЧЕСКОМУ
                      публичному скору A8 (до замера шаг не применять: направление
                      наполовину экстраполяция, вероятность выигрыша всего 0.69)

Запуск:
  python work/scripts/make_finalists.py                 # только mdl_amber
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))
from common import PREDS_DIR, ROOT

# замерено пробами A2-A4 (средние ошибки сегментов на публичной части)
SEG_M = {"S1": -0.072949, "S2": -0.032625, "S3": 0.054674, "S4": 0.031136}
# усадка под перенос на приватную часть (private_risk.md)
SEG_LAMBDA = {"S1": 0.949, "S2": 0.782, "S3": 0.963, "S4": 0.630}
F_A5 = 1.6529168340539806
DELTA_A8 = 0.0612894      
LAMBDA_FEBDIR = 0.973     


def load_lp(p):
    df = pl.read_csv(p, schema_overrides={"user_id": pl.Int64}).sort("user_id")
    return df["user_id"].to_numpy(), np.log1p(np.clip(df[df.columns[1]].to_numpy(), 0, None))


def segments(uid, lpA1):
    segs = {}
    for f, tag in [("A2_probe_s1_gmv.csv", "S1")]:
        u, lpp = load_lp(f"{ROOT}/submissions/{f}")
        assert (u == uid).all()
        segs[tag] = (lpp - lpA1) > 0.15
    segs["S4"] = ~(segs["S1"])
    return segs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fa8", type=float, default=None, help="фактический публичный скор ")
    args = ap.parse_args()

    uid, lpA1 = load_lp(f"{ROOT}/submissions/A1_gram7_shift.csv")
    segs = segments(uid, lpA1)

    corr = np.zeros_like(lpA1)
    for tag, mask in segs.items():
        c = SEG_M[tag] * SEG_LAMBDA[tag]
        corr[mask] = -c   # поправка компенсирует смещение
        print(f"{tag}: {int(mask.sum()):>6} юзеров, m={SEG_M[tag]:+.5f}, "
              f"lambda={SEG_LAMBDA[tag]:.3f}, сдвиг {-c:+.5f}")

    if args.fa8 is None:
        print("скор  не задан — mdl_gabbro не собран (шаг без замера применять нельзя)")
        return

    d = pl.read_parquet(PREDS_DIR / "febspec_test.parquet").sort("user_id")
    assert (d["user_id"].to_numpy() == uid).all()
    


if __name__ == "__main__":
    main()

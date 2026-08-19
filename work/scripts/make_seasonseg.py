#!/usr/bin/env python
"""make_seasonseg.py — сегментная сезонная поправка поверх .

lp_new = lp_G5 + strength * eff_b(user),  eff — посегментное ОТКЛОНЕНИЕ сезонного
переноса (0.62 из замеров, 0.31 — половинная сила).

Usage: POLARS_MAX_THREADS=3 PYTHONPATH=work/scripts .venv/bin/python \
         work/scripts/make_seasonseg.py --seg act_days
"""
from __future__ import annotations

import argparse
import json
import os

os.environ.setdefault("POLARS_MAX_THREADS", "3")

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

from season_seg import features  # noqa: E402
from season_seg2 import assign  # noqa: E402

ROOT = "/Users/alexanderkondakov/ozon-cup"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seg", default="act_days")
    ap.add_argument("--field", default="eff", choices=["eff", "eff_res", "eff_trend"])
    ap.add_argument("--strengths", default="0.62,0.31")
    a = ap.parse_args()

    rows = {r["name"]: r for r in json.load(open(f"{ROOT}/work/reports/season_seg2.json"))}
    r = rows[a.seg]
    eff = np.array(r[a.field])
    edges = np.array(r["edges"])

    d = pl.read_parquet(f"{ROOT}/work/features/seasseg_test26.parquet").sort("user_id")
    F = features(d)
    act = d["f_days"].to_numpy() > 0
    lab = assign(F[a.seg], act, edges)
    ids = sorted(set(lab[lab >= 0].tolist()))
    assert len(ids) == len(eff), f"корзин {len(ids)} != длины eff {len(eff)}"
    shift_b = {v: eff[i] for i, v in enumerate(ids)}
    shift = np.zeros(len(lab))
    for v, e in shift_b.items():
        shift[lab == v] = e
    assert (lab >= 0).all(), "на тесте есть неактивные — поправка не определена"


    print(f"сегментация {a.seg} ({a.field}), K={len(ids)}")
    for i, v in enumerate(ids):
        n = int((lab == v).sum())
        print(f"  корзина {i}: n={n:7d} ({n/len(lab):6.2%})  eff={eff[i]:+.4f}")
    print(f"  взвешенное среднее eff = {(shift).mean():+.6f} (должно быть ~0)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

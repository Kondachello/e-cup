"""Empirical check of the pairwise-contribution formula on the whole library.

Sasha established from three honest points that the constant 7.1 in `gain = 7.1*margin^2`
is wrong, and proposed the exact algebra. Three points settle that 7.1 is wrong; they do
not settle the SHAPE. This measures both against the library.

For every model: predict the gain from (score, margin) with each formula, then MEASURE it
honestly - fit the single blending weight on one half of the users, score the other, swap,
average. Everything on calibrated predictions, shifts fitted on the fitting half only.

    exact:  gain = sb^2*sm^2*z^2 / ((sm^2 - sb^2 + 2*sb*sm*z) * 2*sb)
    old:    gain = 7.1 * z^2

Usage: python work/scripts/formula_check.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))
from common import PREDS_DIR, ROOT
from margin import calibrate_split, fit_shifts, score
from joint_gain import nnls_weights

SKIP = {"blend"}


def main():
    pack = pl.read_parquet(ROOT / "work" / "preds_pack" / "val_preds.parquet").sort("user_id")
    uid = pack["user_id"].to_numpy()
    ly = np.log1p(np.clip(pack["target"].to_numpy().astype(np.float64), 0, None))
    lb = pack["blend"].to_numpy().astype(np.float64)
    sb = score(lb, ly)
    eb = lb - ly
    print(f"эталон (колонка blend пакета): {sb:.6f}, n={len(ly)}\n")

    rng = np.random.default_rng(0)
    half = rng.permutation(len(ly)) < len(ly) // 2

    def exact(sm, z):
        den = (sm ** 2 - sb ** 2 + 2 * sb * sm * z) * 2 * sb
        return sb ** 2 * sm ** 2 * z ** 2 / den if den > 0 else float("nan")

    rows = []
    for f in sorted(PREDS_DIR.glob("*_val.parquet")):
        n = f.name[: -len("_val.parquet")]
        if n in SKIP:
            continue
        d = pl.read_parquet(f).sort("user_id")
        if d.height != len(uid) or not np.array_equal(d["user_id"].to_numpy(), uid):
            continue
        lp = calibrate_split(np.log1p(np.clip(d["pred"].to_numpy().astype(np.float64), 0, None)),
                             ly, half, 24)
        sm = score(lp, ly)
        e = lp - ly
        z = sb / sm - float(np.mean(e * eb) / (sm * sb))      # uncentered, per the identity
        # measured: honest two-way cross-fit
        A = np.column_stack([lb, lp])
        gs = []
        for m in (half, ~half):
            w = nnls_weights(A[m], ly[m])
            gs.append(sb - score(A[~m] @ w, ly[~m]))
        rows.append((n, sm, z, float(np.mean(gs)), exact(sm, z), 7.1 * z * z))

    rows.sort(key=lambda r: -r[3])
    print(f"{'модель':<24}{'скор':>9}{'ЗАПАС':>10}{'ИЗМЕРЕНО':>11}{'точная':>10}{'7.1z²':>10}")
    print("-" * 74)
    for n, sm, z, meas, ex, old in rows[:18]:
        print(f"{n:<24}{sm:>9.4f}{z:>+10.5f}{meas:>+11.6f}{ex:>10.6f}{old:>10.6f}")

    ok = [r for r in rows if r[3] > 2e-5 and np.isfinite(r[4])]
    if ok:
        m = np.array([r[3] for r in ok]); ex = np.array([r[4] for r in ok]); od = np.array([r[5] for r in ok])
        print(f"\nна {len(ok)} моделях с измеримым вкладом:")
        for tag, pred in (("точная формула", ex), ("старая 7.1z²", od)):
            r = pred / m
            print(f"  {tag:<16} медиана отношения предсказ./измер. = {np.median(r):.2f}  "
                  f"разброс {np.percentile(r,10):.2f}..{np.percentile(r,90):.2f}")


if __name__ == "__main__":
    main()

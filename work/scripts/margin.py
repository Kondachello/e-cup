"""Acceptance in the team's currency: ЗАПАС (margin) against the current blend.

Replaces the old "error correlation" check, which was empty: for a model inside the
blend's linear hull the correlation is identically sb/sm, i.e. it is determined by the
model's own score and says nothing (verified to five decimals on 101 models).

    ЗАПАС = sb / sm - rho          (sb = blend score, sm = model score, rho = err corr)
    вклад ≈ 7.1 · ЗАПАС²

Reference points: record margin over the whole project 0.00193; a margin of 0.0065 is
needed for a +0.0003 gain; leaderboard measurement noise is 0.000022.

Everything is measured on CALIBRATED predictions, because the raw score misled the team
eight times. Calibration is the honest cross-fit from the pack README: one half of the
users fits the per-bin log shifts, the other half is corrected by them, and vice versa,
so a shift is never measured on the rows it was fitted on.

The reference blend comes from the `blend` column of work/preds_pack/val_preds.parquet
(already calibrated, already log1p). That column is the whole point of the pack: it
tracks the live blend, unlike a hardcoded list of component names that goes stale.

Usage:
  python work/scripts/margin.py NAME [NAME2 ...]
  python work/scripts/margin.py --file /path/to/preds.parquet --pack work/preds_pack
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))
from common import PREDS_DIR, ROOT

RECORD = 0.00193      # best margin ever measured in the project
NOISE = 0.000022      # leaderboard measurement noise, one unit
THRESHOLD = 0.0003    # gain the team calls meaningful


def fit_shifts(lp: np.ndarray, ly: np.ndarray, bins: int):
    qs = np.quantile(lp, np.linspace(0, 1, bins + 1))
    qs[0] -= 1e-9
    qs[-1] += 1e-9
    centers, shifts = [], []
    for i in range(bins):
        m = (lp > qs[i]) & (lp <= qs[i + 1])
        if m.sum() < 500:                       # a shift from fewer rows is too noisy
            continue
        centers.append(lp[m].mean())
        shifts.append(ly[m].mean() - lp[m].mean())
    return np.array(centers), np.array(shifts)


def calibrate_honest(lp: np.ndarray, ly: np.ndarray, bins: int = 24, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    half = rng.permutation(len(ly)) < len(ly) // 2
    out = np.empty_like(lp)
    for m in (half, ~half):
        c, s = fit_shifts(lp[m], ly[m], bins)
        out[~m] = np.clip(lp[~m] + np.interp(lp[~m], c, s), 0, None)
    return out


def score(lp: np.ndarray, ly: np.ndarray) -> float:
    return float(np.sqrt(np.mean((lp - ly) ** 2)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="*", help="model names -> work/preds/NAME_val.parquet")
    ap.add_argument("--file", action="append", default=[], help="explicit parquet path(s)")
    ap.add_argument("--pack", type=Path, default=ROOT / "work" / "preds_pack")
    ap.add_argument("--bins", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--raw", action="store_true", help="also show the uncalibrated numbers")
    args = ap.parse_args()

    pack = pl.read_parquet(args.pack / "val_preds.parquet").sort("user_id")
    uid = pack["user_id"].to_numpy()
    ly = np.log1p(np.clip(pack["target"].to_numpy().astype(np.float64), 0, None))
    lb = pack["blend"].to_numpy().astype(np.float64)      # already log1p and calibrated
    eb = lb - ly
    sb = score(lb, ly)
    print(f"эталон: бленд из пакета, скор {sb:.6f}, n={len(ly)}")
    print(f"ориентиры: рекорд запаса {RECORD}, нужно {0.0065} ради прироста {THRESHOLD}, "
          f"шум {NOISE}\n")

    hdr = f"{'модель':<24}{'скор':>10}{'корр':>9}{'ЗАПАС':>10}{'вклад':>10}  вердикт"
    print(hdr)
    print("-" * len(hdr))

    targets = [(n, PREDS_DIR / f"{n}_val.parquet") for n in args.names]
    targets += [(Path(f).stem, Path(f)) for f in args.file]
    for name, path in targets:
        if not path.exists():
            print(f"{name:<24}НЕТ ФАЙЛА {path}")
            continue
        df = pl.read_parquet(path).sort("user_id")
        assert np.array_equal(df["user_id"].to_numpy(), uid), f"порядок user_id не совпал: {path}"
        lp_raw = np.log1p(np.clip(df["pred"].to_numpy().astype(np.float64), 0, None))
        for tag, lp in ((("сырой", lp_raw),) if args.raw else ()) + \
                       (("", calibrate_honest(lp_raw, ly, args.bins, args.seed)),):
            sm = score(lp, ly)
            e = lp - ly
            rho = float(np.corrcoef(e, eb)[0, 1])
            margin = sb / sm - rho
            # the identity is margin = (sb/sm)(1 - beta): margin <= 0 means beta >= 1, i.e. the
            # optimiser gives this model zero weight. Squaring a negative margin would turn a
            # useless model into a large "contribution", so the floor is not cosmetic.
            contrib = 7.1 * max(margin, 0.0) ** 2
            verdict = ("ГОДИТСЯ" if contrib >= THRESHOLD else
                       "слабо, но не шум" if contrib >= 2 * NOISE else "шум")
            label = f"{name} {tag}".strip()
            print(f"{label:<24}{sm:10.6f}{rho:9.5f}{margin:+10.5f}{contrib:10.6f}  {verdict}")


if __name__ == "__main__":
    main()

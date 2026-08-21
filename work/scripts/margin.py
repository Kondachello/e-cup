"""Acceptance in the team's currency: ЗАПАС (margin) against the current blend.

Replaces the old "error correlation" check, which was empty: for a model inside the
blend's linear hull the correlation is identically sb/sm, i.e. it is determined by the
model's own score and says nothing (verified to five decimals on 101 models).

    ЗАПАС = sb / sm - rho          (sb = blend score, sm = model score, rho = err corr)
    вклад = sb²·sm²·з² / [(sm² − sb² + 2·sb·sm·з) · 2·sb]     (точная алгебра пары)

The old shorthand `вклад ≈ 7.1·ЗАПАС²` is dead: it explained 9% of the variance and
under-priced strong models 3-20x (KNOWLEDGE «ПОПРАВКА К ЗАКОНУ ВКЛАДА»). With the exact
pair algebra the contribution of a strong model (sm/sb -> 1) is almost LINEAR in margin,
so the margin needed for +0.0003 depends on the score: 1.67 -> 0.00166, 1.70 -> 0.00412,
1.83 -> 0.00811. Sets are still measured by joint_gain.py - margins do not add up.

Reference points: record margin over the whole project 0.00193; leaderboard measurement
noise is 0.000022.

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


def apply_shifts(lp: np.ndarray, centers: np.ndarray, shifts: np.ndarray) -> np.ndarray:
    return np.clip(lp + np.interp(lp, centers, shifts), 0, None)


def calibrate_split(lp: np.ndarray, ly: np.ndarray, fit_mask: np.ndarray,
                    bins: int = 24) -> np.ndarray:
    """Shifts fitted ONLY on fit_mask, applied everywhere.

    calibrate_honest cross-fits over the whole population, which is right when the whole
    population is what gets scored. It is WRONG when a later step splits the users again:
    the rows used for model selection then carry shifts fitted with the held-out rows'
    targets, so the held-out set stops being held out. Sasha caught this in library_sweep
    and joint_gain, where the calibration seed and the DEV/EVAL seed were the same integer
    and therefore produced bit-identical halves. Here the shifts are a parameter learned on
    the fit side and applied to the other side, like any other parameter.
    """
    c, sh = fit_shifts(lp[fit_mask], ly[fit_mask], bins)
    out = np.empty_like(lp)
    out[~fit_mask] = apply_shifts(lp[~fit_mask], c, sh)
    # inside the fit side, cross-fit so those rows are not calibrated on themselves
    sub = fit_mask.copy()
    idx = np.flatnonzero(fit_mask)
    half = np.zeros(len(lp), bool)
    half[idx[: len(idx) // 2]] = True
    for m in (half & sub, (~half) & sub):
        o = sub & ~m
        c2, sh2 = fit_shifts(lp[o], ly[o], bins)
        out[m] = apply_shifts(lp[m], c2, sh2)
    return out


def calibrate_honest(lp: np.ndarray, ly: np.ndarray, bins: int = 24, seed: int = 0) -> np.ndarray:
    # Сид со смещением: перестановка калибровки не должна совпадать с fit/score-сплитами
    # замерителей (joint_gain, library_sweep) — при одинаковом default_rng(seed) сплит
    # калибровки и сплит отбора были одной перестановкой, узкая утечка EVAL в отбор.
    rng = np.random.default_rng(seed + 100_003)
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
    print(f"ориентиры: рекорд запаса {RECORD}, шум {NOISE}; вклад — точная алгебра пары, "
          f"наборы мерить joint_gain\n")

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
            # UNcentered correlation: the identity margin = sb/sm - rho holds for
            # E[e*eb]/(sm*sb), not for corrcoef. Zhenya measured the difference: corrcoef
            # distorted 6 of 30 models - every uncalibrated one (mean error != 0). After
            # calibration the two coincide, so old numbers on _cal models stand.
            rho = float(np.mean(e * eb) / (sm * sb))
            margin = sb / sm - rho
            # the identity is margin = (sb/sm)(1 - beta): margin <= 0 means beta >= 1, i.e. the
            # optimiser gives this model zero weight. A negative margin would flip the sign of
            # the exact formula's numerator, so the floor is not cosmetic.
            z = max(margin, 0.0)
            den = (sm * sm - sb * sb + 2.0 * sb * sm * z) * 2.0 * sb
            contrib = (sb * sb * sm * sm * z * z) / den if den > 1e-12 else 0.0
            verdict = ("ГОДИТСЯ" if contrib >= THRESHOLD else
                       "слабо, но не шум" if contrib >= 2 * NOISE else "шум")
            label = f"{name} {tag}".strip()
            print(f"{label:<24}{sm:10.6f}{rho:9.5f}{margin:+10.5f}{contrib:10.6f}  {verdict}")


if __name__ == "__main__":
    main()

"""MIRROR VALIDATION: score predictions on the only genuine analogue of the test window.

The test window 2026-02-14..2026-03-15 contains March-8. Our entire validation
machinery lives on the January window 2026-01-15..2026-02-13, which has no
holiday at all. Measured cost of that blind spot: a meta-model with an honest
+0.00096 val gain transferred to the test at 2% (ceiling +0.000016).

The data contain exactly one structural analogue: anchor 2025-02-13, target
2025-02-14..2025-03-15 -- same calendar slot, March-8 inside, one year earlier.
Only 44 days of history exist there (data start 2025-01-01), hence the companion
builder build_features_short.py (windows <= 42d).

WHY THE CALIBRATED NUMBER IS THE ONE THAT COUNTS
The platform roughly doubled during 2025, so the March-2025 window sits at a very
different LEVEL than January-2026 (mean log1p target 1.7154 vs 2.2421). A raw
RMSLE comparison across the two windows measures mostly that level gap, not model
quality -- and the project has been burned by raw comparisons before. So every
score is reported twice: raw, and after an HONEST binned log-shift calibration
(cross-fitted on two halves of the users: shifts fitted on one half are only ever
applied to the other half, so nothing is measured in-sample).

Usage:
  # score a prediction file on the mirror window
  mirror_val.py --file work/preds/foo_mirror.parquet
  mirror_val.py --pred febspec2          # -> work/preds/febspec2_mirror.parquet
  # score on the January window instead (same honest-calibration protocol)
  mirror_val.py --pred febspec2 --window jan
  # constant-prediction floors of both windows
  mirror_val.py --floors
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))
from common import PREDS_DIR, rmsle  # noqa: F401

MIRROR_ANCHOR = date(2025, 2, 13)   # target 2025-02-14 .. 2025-03-15  (March-8 inside)
JAN_ANCHOR = date(2026, 1, 14)      # target 2026-01-15 .. 2026-02-13  (the usual val)
WINDOWS = {"mirror": MIRROR_ANCHOR, "jan": JAN_ANCHOR}
CAL_BINS = 24
CAL_SEED = 0
MIN_BIN = 500


# ----------------------------------------------------------------- calibration
def fit_shifts(lp: np.ndarray, ly: np.ndarray, bins: int = CAL_BINS):
    """Per-quantile-bin additive shift in log1p space (same recipe as calibrate.py)."""
    qs = np.quantile(lp, np.linspace(0, 1, bins + 1))
    qs[0] -= 1e-9
    qs[-1] += 1e-9
    centers, shifts = [], []
    for i in range(bins):
        m = (lp > qs[i]) & (lp <= qs[i + 1])
        if m.sum() < MIN_BIN:
            continue
        centers.append(lp[m].mean())
        shifts.append(ly[m].mean() - lp[m].mean())
    if not centers:                                    # degenerate predictor
        return np.array([0.0]), np.array([float(ly.mean() - lp.mean())])
    return np.array(centers), np.array(shifts)


def apply_shifts(lp: np.ndarray, centers: np.ndarray, shifts: np.ndarray) -> np.ndarray:
    return np.clip(lp + np.interp(lp, centers, shifts), 0, None)


def honest_cal(lp: np.ndarray, ly: np.ndarray, bins: int = CAL_BINS,
               seed: int = CAL_SEED) -> np.ndarray:
    """Cross-fitted calibrated log-predictions: half the users fit, the other half
    is measured, then swapped. No user is ever scored with shifts fitted on itself."""
    n = len(lp)
    half = np.random.default_rng(seed).permutation(n) < n // 2
    out = np.empty(n)
    for m in (half, ~half):
        c, s = fit_shifts(lp[m], ly[m], bins)
        out[~m] = apply_shifts(lp[~m], c, s)
    return out


def honest_shift(lp: np.ndarray, ly: np.ndarray, seed: int = CAL_SEED) -> np.ndarray:
    """Same protocol but with a single global shift (level only) -- the diagnostic
    that separates 'wrong level' from 'wrong shape'."""
    n = len(lp)
    half = np.random.default_rng(seed).permutation(n) < n // 2
    out = np.empty(n)
    for m in (half, ~half):
        out[~m] = np.clip(lp[~m] + (ly[m].mean() - lp[m].mean()), 0, None)
    return out


# ----------------------------------------------------------------------- target
def window_target(window: str = "mirror") -> tuple[np.ndarray, np.ndarray]:
    """(user_id sorted, target) for the evaluation window."""
    anchor = WINDOWS[window]
    from build_features_short import load_short, path_for
    if path_for(anchor).exists():
        d = load_short(anchor, columns=["user_id", "target"]).sort("user_id")
    else:                                              # fall back to the main tier
        from common import load_anchor
        d = load_anchor(anchor, columns=["user_id", "target"]).sort("user_id")
    y = d["target"].to_numpy().astype(np.float64)
    assert np.isfinite(y).all(), f"{window} window target is not fully observed"
    return d["user_id"].to_numpy(), y


def score(uid: np.ndarray, pred: np.ndarray, window: str = "mirror",
          bins: int = CAL_BINS, seed: int = CAL_SEED) -> dict:
    """RMSLE raw / after honest global shift / after honest binned calibration."""
    uid_ref, y = window_target(window)
    order = np.argsort(uid)
    uid, pred = uid[order], np.asarray(pred, dtype=np.float64)[order]
    assert np.array_equal(uid, uid_ref), f"user_id mismatch on the {window} window"
    ly = np.log1p(np.clip(y, 0, None))
    lp = np.log1p(np.clip(pred, 0, None))
    r = lambda l: float(np.sqrt(np.mean((l - ly) ** 2)))  # noqa: E731
    return {
        "window": window,
        "anchor": WINDOWS[window].isoformat(),
        "n": int(len(y)),
        "rmsle": r(lp),
        "rmsle_shift": r(honest_shift(lp, ly, seed)),
        "rmsle_cal": r(honest_cal(lp, ly, bins, seed)),
        "mean_log_err": float((lp - ly).mean()),
        "sd_pred": float(lp.std()),
        "sd_target": float(ly.std()),
    }


def score_file(path: Path, window: str = "mirror", **kw) -> dict:
    d = pl.read_parquet(path).sort("user_id")
    out = score(d["user_id"].to_numpy(), d["pred"].to_numpy(), window, **kw)
    out["file"] = str(path)
    return out


def floors(window: str) -> dict:
    """Metric floors of a window: zero prediction and the best constant."""
    _, y = window_target(window)
    ly = np.log1p(np.clip(y, 0, None))
    c = float(np.expm1(ly.mean()))
    return {"window": window, "zero": float(np.sqrt(np.mean(ly ** 2))),
            "best_const": c,
            "const_rmsle": float(np.sqrt(np.mean((np.log1p(c) - ly) ** 2))),
            "mean_log1p": float(ly.mean()), "zero_share": float((y == 0).mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", action="append", default=[],
                    help="model name -> work/preds/NAME_{window}.parquet")
    ap.add_argument("--file", action="append", default=[])
    ap.add_argument("--window", default="mirror", choices=sorted(WINDOWS))
    ap.add_argument("--bins", type=int, default=CAL_BINS)
    ap.add_argument("--seed", type=int, default=CAL_SEED)
    ap.add_argument("--floors", action="store_true")
    ap.add_argument("--json", type=str, default="")
    args = ap.parse_args()

    res = []
    if args.floors:
        for w in sorted(WINDOWS):
            print(json.dumps(floors(w)), flush=True)
    suffix = args.window if args.window != "jan" else "val"
    targets = [(n, PREDS_DIR / f"{n}_{suffix}.parquet") for n in args.pred]
    targets += [(Path(f).stem, Path(f)) for f in args.file]
    for name, p in targets:
        if not p.exists():
            print(f"{name}: MISSING {p}", flush=True)
            continue
        s = score_file(p, args.window, bins=args.bins, seed=args.seed)
        s["name"] = name
        res.append(s)
        print(f"{name:28s} [{args.window}] raw {s['rmsle']:.6f}  "
              f"+shift {s['rmsle_shift']:.6f}  +cal {s['rmsle_cal']:.6f}  "
              f"mean_log_err {s['mean_log_err']:+.4f}", flush=True)
    if args.json and res:
        Path(args.json).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()

"""Verdict for the v10 per-channel-funnel tier: twl_v10 vs its strict control.

Protocol (lesson from v7: compare only against a control on the SAME anchors, never
against historical numbers): twl_v10 and twl_v10ctl share seed, config, anchors and
feature set except USE_V10. The control runs with --no-test, so it has no test preds
and calibrate.py cannot be used on it. Instead both models get the SAME honest
calibration here: cross-fitted binned log-shift on VAL (K folds by user, shifts fitted
on K-1 folds and applied to the held-out one), which is unbiased and comparable.

Output: work/reports/v10_verdict.json  (+ stdout summary)
Run: POLARS_MAX_THREADS=3 .venv/bin/python work/scripts/v10_verdict.py
"""
from __future__ import annotations

import os

os.environ.setdefault("POLARS_MAX_THREADS", "3")

import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from common import FEATURES_DIR, PREDS_DIR, REPORTS_DIR, VAL_ANCHOR, rmsle  # noqa: E402
from calibrate import apply_shifts, fit_shifts  # noqa: E402

MODELS = ["twl_v10", "twl_v10ctl"]
KFOLD = 5
BINS = 24
SEED = 0


def xfit_calibrated(lp: np.ndarray, ly: np.ndarray) -> np.ndarray:
    """Cross-fitted binned log-shift calibration (no in-sample optimism)."""
    rng = np.random.default_rng(SEED)
    fold = rng.integers(0, KFOLD, len(lp))
    out = np.empty_like(lp)
    for f in range(KFOLD):
        tr, te = fold != f, fold == f
        c, s = fit_shifts(lp[tr], ly[tr], BINS)
        out[te] = apply_shifts(lp[te], c, s)
    return out


def main():
    val = pl.read_parquet(FEATURES_DIR / f"anchor={VAL_ANCHOR.isoformat()}.parquet",
                          columns=["user_id", "target"]).sort("user_id")
    uid = val["user_id"].to_numpy()
    y = val["target"].to_numpy().astype(np.float64)
    ly = np.log1p(y)

    res = {}
    for name in MODELS:
        p = PREDS_DIR / f"{name}_val.parquet"
        if not p.exists():
            print(f"MISSING {p}", flush=True)
            return
        d = pl.read_parquet(p).sort("user_id")
        assert (d["user_id"].to_numpy() == uid).all(), name
        lp = np.log1p(np.clip(d["pred"].to_numpy().astype(np.float64), 0, None))
        lc = xfit_calibrated(lp, ly)
        res[name] = {
            "raw": round(rmsle(y, np.expm1(lp)), 6),
            "cal": round(rmsle(y, np.expm1(lc)), 6),
            "mean_logpred": round(float(lp.mean()), 4),
            "sd_logpred": round(float(lp.std()), 4),
        }
        print(f"{name:<12} raw {res[name]['raw']:.6f}  cal(xfit) {res[name]['cal']:.6f}", flush=True)

    d_raw = res["twl_v10"]["raw"] - res["twl_v10ctl"]["raw"]
    d_cal = res["twl_v10"]["cal"] - res["twl_v10ctl"]["cal"]
    out = {
        "models": res,
        "delta_raw": round(d_raw, 6),
        "delta_cal": round(d_cal, 6),
        "helps": bool(d_cal < 0),
        # reference scale: v7 tier moved -0.00065 / +0.00014 across two seeds and was
        # rejected as noise; a single-seed effect below ~0.001 is not evidence.
        "above_noise_floor": bool(abs(d_cal) > 0.001),
        "verdict": ("v10 HELPS" if d_cal < -0.001 else
                    "v10 HURTS" if d_cal > 0.001 else "NOISE (|delta| <= 0.001)"),
    }
    (REPORTS_DIR / "v10_verdict.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print("\n=== RAW JSON ===")
    print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()

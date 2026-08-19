"""Verdict for the seq3 tier (per-channel funnel channels in the fusion sequence input).

Protocol (lesson from v7/v10: compare only against a control on the SAME data, never
against historical numbers): fusion_v3 (--n-ch 12) and fusion_v3ctl (--n-ch 8) read the
SAME work/seq3 tensors with the same anchors, seed, L=112, quantisation and tabular
features; the only difference is that the control cannot see channels 8..11
(search_to_cart, search_to_ord, cat_to_cart, cat_to_ord).  So the delta isolates the new
channels and not L=112-vs-196 or the uint8 storage.

Both arms are compared AFTER calibration (raw comparisons mislead: models sit ~0.25 high
in log space and a level difference swamps the structural one).  Calibration here is the
cross-fitted binned log-shift (K folds by user, shifts fitted on K-1 and applied to the
held-out fold), which is unbiased and identical for both arms.  fusion_f (the seq2/L=196
model that holds weight 0.32 in the blend) is reported as context only, NOT as the
decision criterion.

Output: work/reports/seq3_verdict.json (+ stdout summary)
Run: POLARS_MAX_THREADS=3 .venv/bin/python work/scripts/seq3_verdict.py
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

NEW, CTL = "fusion_v3", "fusion_v3ctl"
REFS = ["fusion_f"]          # context only (seq2, L=196, 12 anchors)
KFOLD = 5
BINS = 24
SEED = 0
NOISE = 0.001                # v7 moved -0.00065/+0.00014 across seeds and was rejected


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

    res, lcal = {}, {}
    for name in [NEW, CTL] + REFS:
        p = PREDS_DIR / f"{name}_val.parquet"
        if not p.exists():
            if name in REFS:
                print(f"(ref {name} missing, skipped)", flush=True)
                continue
            print(f"MISSING {p} — run the queue jobs first", flush=True)
            return
        d = pl.read_parquet(p).sort("user_id")
        assert (d["user_id"].to_numpy() == uid).all(), name
        lp = np.log1p(np.clip(d["pred"].to_numpy().astype(np.float64), 0, None))
        lc = xfit_calibrated(lp, ly)
        lcal[name] = lc
        res[name] = {
            "raw": round(rmsle(y, np.expm1(lp)), 6),
            "cal": round(rmsle(y, np.expm1(lc)), 6),
            "mean_logpred": round(float(lp.mean()), 4),
            "sd_logpred": round(float(lp.std()), 4),
        }
        print(f"{name:<14} raw {res[name]['raw']:.6f}  cal(xfit) {res[name]['cal']:.6f}",
              flush=True)

    d_cal = res[NEW]["cal"] - res[CTL]["cal"]
    # paired bootstrap over users on the calibrated squared errors
    e_new = (lcal[NEW] - ly) ** 2
    e_ctl = (lcal[CTL] - ly) ** 2
    rng = np.random.default_rng(1)
    boot = []
    for _ in range(400):
        i = rng.integers(0, len(ly), len(ly))
        boot.append(np.sqrt(e_new[i].mean()) - np.sqrt(e_ctl[i].mean()))
    lo, hi = np.percentile(boot, [2.5, 97.5])

    out = {
        "models": res,
        "delta_raw": round(res[NEW]["raw"] - res[CTL]["raw"], 6),
        "delta_cal": round(d_cal, 6),
        "delta_cal_ci95": [round(float(lo), 6), round(float(hi), 6)],
        "corr_cal_with_ctl": round(float(np.corrcoef(lcal[NEW], lcal[CTL])[0, 1]), 5),
        "helps": bool(d_cal < 0),
        "above_noise_floor": bool(abs(d_cal) > NOISE),
        "verdict": ("seq3 funnel channels HELP" if d_cal < -NOISE else
                    "seq3 funnel channels HURT" if d_cal > NOISE else
                    f"NOISE (|delta| <= {NOISE})"),
    }
    if REFS[0] in res:
        out["ref_fusion_f_cal"] = res[REFS[0]]["cal"]
        out["note_ref"] = ("fusion_f is seq2 L=196 with 12 final anchors — context only, "
                           "not the control")
    (REPORTS_DIR / "seq3_verdict.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1))
    print("\n=== RAW JSON ===")
    print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()

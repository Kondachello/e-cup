"""Verdict for the weekly low-rank direction: the linear model and the booster arms.

Two questions, two different measurements.

A. Does the weekly representation help the CHAMPION BOOSTER as one more feature tier?
   Arms, all sharing seed/config/anchors and differing only in the tier:
     twl_v5      + 48-component joint weekly basis (32 used)
     twl_v5s     + 2 supervised weekly columns (concentrated form)
     twl_v5cap   + 32 columns of PURE CAPACITY (rotated ranks of existing features)
     twl_v5ctl   nothing added
   twl_v5cap is the control the v6/v8/v10 post-mortems demand: a tier must beat equal
   capacity, not merely fewer features. Comparison is by CROSS-FITTED calibrated RMSLE --
   raw ordering has misled this project seven times.

B. Does the linear model contribute to the BLEND? That is the real question, and it is not
   answered by the model's own score (febspec scored 1.83 solo and still contributed). The
   measurement is err_corr's MARGIN = sb/sm - rho, the share of the model outside the blend's
   linear hull, with contribution ~= 7.1 * margin^2. Acceptance: contribution > 0.0003,
   which needs margin > 0.0065; the project record is 0.00193 (febspec2_cal).

Output: work/reports/v5_verdict.json
Run: POLARS_MAX_THREADS=3 .venv/bin/python work/scripts/v5_verdict.py
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
from err_corr import BLEND  # noqa: E402

BOOST_ARMS = ["twl_v5", "twl_v5s", "twl_v5cap", "twl_v5ctl"]
LINEAR = ["wklin", "wklin_wk", "wklin_base"]
KFOLD, BINS, SEED = 5, 24, 0
RECORD_MARGIN = 0.00193          # febspec2_cal, the project record
CONTRIB_K = 7.1                  # contribution ~= CONTRIB_K * margin^2
THRESHOLD = 0.0003


def xfit(lp: np.ndarray, ly: np.ndarray) -> np.ndarray:
    """Cross-fitted binned log-shift calibration: comparable across arms, no in-sample gain."""
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

    def lp_of(name: str):
        p = PREDS_DIR / f"{name}_val.parquet"
        if not p.exists():
            return None
        d = pl.read_parquet(p).sort("user_id")
        assert np.array_equal(d["user_id"].to_numpy(), uid), name
        return np.log1p(np.clip(d["pred"].to_numpy().astype(np.float64), 0, None))

    # ---- A. booster arms, cross-fitted calibrated
    arms = {}
    for n in BOOST_ARMS:
        lp = lp_of(n)
        if lp is None:
            print(f"MISSING {n}", flush=True)
            continue
        lc = xfit(lp, ly)
        arms[n] = {"raw": round(rmsle(y, np.expm1(lp)), 6),
                   "cal": round(rmsle(y, np.expm1(lc)), 6),
                   "mean_logpred": round(float(lp.mean()), 4),
                   "sd_logpred": round(float(lp.std()), 4)}
        print(f"{n:<12} raw {arms[n]['raw']:.6f}  cal(xfit) {arms[n]['cal']:.6f}", flush=True)
    ctl = arms.get("twl_v5ctl", {}).get("cal")
    deltas = {n: round(arms[n]["cal"] - ctl, 6)
              for n in arms if ctl is not None and n != "twl_v5ctl"}

    # ---- B. margin against the honest blend
    lb = None
    missing = [n for n in BLEND if not (PREDS_DIR / f"{n}_val.parquet").exists()]
    if not missing:
        lb = sum(w * lp_of(n) for n, w in BLEND.items())
    models = {}
    if lb is not None:
        eb = lb - ly
        sb = float(np.sqrt(np.mean(eb ** 2)))
        print(f"\nblend val_rmsle={sb:.6f}", flush=True)
        # CALIBRATED ONLY. Uncalibrated models fail the margin test for a reason that has
        # nothing to do with their structure -- every raw model is biased ~0.25 high in log,
        # which alone drives the margin negative (see the audit note in TEAM_PLAN).
        cands = [f"{n}_cal" for n in LINEAR] + ["twl_v5_cal", "twl_v5s_cal"]
        for n in cands:
            lp = lp_of(n)
            if lp is None:
                continue
            e = lp - ly
            sm = float(np.sqrt(np.mean(e ** 2)))
            rho = float(np.corrcoef(e, eb)[0, 1])
            margin = sb / max(sm, 1e-12) - rho
            d = e - eb
            w = float(-np.dot(eb, d) / max(np.dot(d, d), 1e-12))
            best = float(np.sqrt(np.mean(((1 - w) * lb + w * lp - ly) ** 2)))
            models[n] = {"val_rmsle": round(rmsle(y, np.expm1(lp)), 6),
                         "err_corr": round(rho, 5),
                         "corr_expected": round(sb / max(sm, 1e-12), 5),
                         "margin": round(margin, 6),
                         "w_opt": round(w, 4),
                         "blend_rmsle": round(best, 6),
                         "gain": round(sb - best, 6),
                         "gain_from_margin": round(CONTRIB_K * margin ** 2, 6)}
            print(f"{n:<18} solo {models[n]['val_rmsle']:.6f}  rho {rho:.4f}  "
                  f"MARGIN {margin:+.5f}  w* {w:+.3f}  gain {models[n]['gain']:+.6f}",
                  flush=True)
    else:
        print(f"blend unavailable, missing {missing}", flush=True)

    best_margin = max((m["margin"] for m in models.values()), default=0.0)
    best_gain = max((m["gain"] for m in models.values()), default=0.0)
    out = {"boost_arms": arms,
           "delta_cal_vs_control": deltas,
           "linear_and_tier_models": models,
           "best_margin": round(best_margin, 6),
           "best_gain": round(best_gain, 6),
           "record_margin": RECORD_MARGIN,
           "beats_record_margin": bool(best_margin > RECORD_MARGIN),
           "above_threshold": bool(best_gain > THRESHOLD)}
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "v5_verdict.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print("\n=== RAW JSON ===")
    print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()

"""Verdict for the v10 per-channel-funnel tier: число обучающих срезов: 14 против 20 против всех 23.

Protocol (lesson from v7: compare only against a control on the SAME anchors, never
against historical numbers): twl_v10 and twl_v10ctl share seed, config, anchors and
feature set except USE_V10. The control runs with --no-test, so it has no test preds
and calibrate.py cannot be used on it. Instead both models get the SAME honest
calibration here: cross-fitted binned log-shift on VAL (K folds by user, shifts fitted
on K-1 folds and applied to the held-out one), which is unbiased and comparable.

Output: work/reports/anchors_verdict.json  (+ stdout summary)
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

MODELS = ["twl_a14", "twl_a20", "twl_aall"]
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

    # база — обучение на 14 срезах, как во всех наших моделях; остальные сравниваются с ней
    base = MODELS[0]
    deltas = {}
    for name in MODELS[1:]:
        deltas[name] = {
            "delta_raw": round(res[name]["raw"] - res[base]["raw"], 6),
            "delta_cal": round(res[name]["cal"] - res[base]["cal"], 6),
        }
    best = min(MODELS, key=lambda n: res[n]["cal"])
    d_best = res[best]["cal"] - res[base]["cal"]
    print(f"\nбаза {base} cal {res[base]['cal']:.6f}")
    for n, dd in deltas.items():
        print(f"  {n:<12} cal {res[n]['cal']:.6f}  дельта {dd['delta_cal']:+.6f}")
    out = {
        "models": res,
        "base": base,
        "deltas_vs_base": deltas,
        "best_by_cal": best,
        # порог: тир v7 дал -0.00065 и +0.00014 на двух сидах и был отвергнут как шум,
        # поэтому одиночный эффект меньше 0.001 доказательством не считается
        "above_noise_floor": bool(abs(d_best) > 0.001),
        "verdict": ("БОЛЬШЕ СРЕЗОВ ПОМОГАЕТ" if d_best < -0.001 else
                    "БОЛЬШЕ СРЕЗОВ ВРЕДИТ" if d_best > 0.001 else
                    "ШУМ (|дельта| <= 0.001), нужен второй сид"),
    }
    (REPORTS_DIR / "anchors_verdict.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print("\n=== RAW JSON ===")
    print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()

"""Direct test of the GIFTER hypothesis, before any training.

Subgroup = bottom 20% of the blend prediction on VAL
           (0.536*mlpziln_cal + 0.283*mlpbin_cal + 0.145*gru_final + 0.036*c_xtw_s42).
That is exactly the "model says inactive, but ~16% buy" pocket where our error lives.
Inside it we score every v8 feature by AUC for separating bought / did-not-buy, and
compare against the best existing features (rec_order, ord_days_90) on the SAME rows.

Run: POLARS_MAX_THREADS=3 .venv/bin/python work/scripts/v8_value_check.py
"""
from __future__ import annotations

import os

os.environ.setdefault("POLARS_MAX_THREADS", "3")

import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))
from common import PREDS_DIR, V8_FEATS, VAL_ANCHOR, FEATURES_DIR, rmsle  # noqa: E402

BLEND = {"mlpziln_cal": 0.536, "mlpbin_cal": 0.283, "gru_final": 0.145, "c_xtw_s42": 0.036}
Q = 0.20
BASE_FEATS = ["rec_order", "ord_days_90", "rec_active", "ord_days_30", "ord_days_365"]


def auc(score: np.ndarray, y: np.ndarray) -> float:
    """Rank AUC with tie handling; nan-free input required."""
    n1 = float(y.sum())
    n0 = float(len(y) - n1)
    if n1 < 5 or n0 < 5:
        return float("nan")
    r = np.empty(len(score))
    order = np.argsort(score, kind="mergesort")
    s = score[order]
    ranks = np.arange(1, len(s) + 1, dtype=np.float64)
    # average ranks over ties
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        if j > i:
            ranks[i:j + 1] = ranks[i:j + 1].mean()
        i = j + 1
    r[order] = ranks
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def score_feature(x: np.ndarray, y: np.ndarray) -> dict:
    """AUC on non-null rows and on all rows with nulls pushed to an extreme bucket."""
    ok = np.isfinite(x)
    out = {"cov": round(float(ok.mean()), 4)}
    a = auc(x[ok], y[ok]) if ok.sum() > 20 else float("nan")
    out["auc_nonnull"] = None if not np.isfinite(a) else round(float(max(a, 1 - a)), 4)
    out["dir"] = None if not np.isfinite(a) else (1 if a >= 0.5 else -1)
    if ok.all():
        out["auc_full"] = out["auc_nonnull"]
    else:
        lo = np.nanmin(x[ok]) - 1.0 if ok.any() else 0.0
        xf = np.where(ok, x, lo)
        af = auc(xf, y)
        out["auc_full"] = None if not np.isfinite(af) else round(float(max(af, 1 - af)), 4)
    return out


def main():
    base = pl.read_parquet(FEATURES_DIR / f"anchor={VAL_ANCHOR.isoformat()}.parquet",
                           columns=["user_id", "target"] + BASE_FEATS).sort("user_id")
    v8 = pl.read_parquet(FEATURES_DIR / f"anchor={VAL_ANCHOR.isoformat()}.v8.parquet").sort("user_id")
    df = base.join(v8, on="user_id", how="left")

    p = np.zeros(df.height)
    for name, w in BLEND.items():
        q = pl.read_parquet(PREDS_DIR / f"{name}_val.parquet").sort("user_id")
        assert (q["user_id"].to_numpy() == df["user_id"].to_numpy()).all(), name
        p += w * np.clip(q["pred"].to_numpy().astype(np.float64), 0, None)

    y_all = df["target"].to_numpy().astype(np.float64)
    print(f"[blend] rmsle_val = {rmsle(y_all, p):.6f}  (sanity: should be ~1.66-1.67)")

    thr = float(np.quantile(p, Q))
    sub = p <= thr
    y = (y_all[sub] > 0).astype(np.int64)
    print(f"[subgroup] bottom {Q:.0%} of blend: n={int(sub.sum())} thr_pred={thr:.3f} "
          f"buy_rate={y.mean():.4f} mean_target={y_all[sub].mean():.2f}")
    print(f"[subgroup] share of total val SSE(log) = "
          f"{np.sum((np.log1p(y_all[sub]) - np.log1p(p[sub]))**2) / np.sum((np.log1p(y_all) - np.log1p(p))**2):.4f}")

    rows = []
    for c in BASE_FEATS + V8_FEATS:
        x = df[c].to_numpy().astype(np.float64)[sub]
        r = score_feature(x, y)
        r["feat"] = c
        r["tier"] = "base" if c in BASE_FEATS else ("gifter" if c.startswith("gf_") else "wake")
        rows.append(r)

    rows.sort(key=lambda r: -(r["auc_full"] or 0))
    print(f"\n{'feat':<20}{'tier':<8}{'auc_full':>10}{'auc_nonnull':>13}{'cov':>8}{'dir':>5}")
    for r in rows:
        print(f"{r['feat']:<20}{r['tier']:<8}{str(r['auc_full']):>10}"
              f"{str(r['auc_nonnull']):>13}{r['cov']:>8.3f}{str(r['dir']):>5}")

    best_new = max((r for r in rows if r["tier"] != "base"), key=lambda r: r["auc_full"] or 0)
    best_base = max((r for r in rows if r["tier"] == "base"), key=lambda r: r["auc_full"] or 0)
    rec = next(r for r in rows if r["feat"] == "rec_order")
    o90 = next(r for r in rows if r["feat"] == "ord_days_90")

    out = {
        "n_subgroup": int(sub.sum()),
        "buy_rate_subgroup": round(float(y.mean()), 4),
        "best_new": best_new,
        "best_base": best_base,
        "rec_order": rec,
        "ord_days_90": o90,
        "gate_060": bool((best_new["auc_full"] or 0) > 0.60),
        "all": rows,
    }
    (Path(__file__).parents[1] / "reports" / "v8_value_check.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1))
    print("\n=== RAW JSON ===")
    print(json.dumps({k: v for k, v in out.items() if k != "all"}, ensure_ascii=False))


if __name__ == "__main__":
    main()

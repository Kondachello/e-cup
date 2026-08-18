"""Incremental value of the v8 tier inside the diagnosed subgroup.

Univariate AUC is a poor guide (lesson v6: tsb_p had AUC 0.832 > rec_order 0.802 yet
hurt every model). The honest question is whether a v8 feature adds signal ON TOP OF
what the blend + base features already carry. 4-fold CV logistic regression inside the
bottom-20% blend subgroup on VAL:

  A: log1p(blend)                      -> AUC_A
  B: log1p(blend) + base features      -> AUC_B   (what the model effectively has)
  C: B + one v8 feature                -> AUC_C   (delta = AUC_C - AUC_B)
  D: B + all v8 features               -> AUC_D

Run: POLARS_MAX_THREADS=3 .venv/bin/python work/scripts/v8_incr_check.py
"""
from __future__ import annotations

import os

os.environ.setdefault("POLARS_MAX_THREADS", "3")

import json
import sys
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
from common import FEATURES_DIR, PREDS_DIR, V8_FEATS, VAL_ANCHOR  # noqa: E402
from v8_value_check import BLEND, Q  # noqa: E402

BASE = ["rec_order", "ord_days_30", "ord_days_90", "ord_days_365", "rec_active",
        "gmv_sum_90", "gmv_sum_365", "rec_search", "act_days_90"]
FOLDS = 4
SEED = 42


def prep(X: np.ndarray) -> np.ndarray:
    """median-fill + missing indicator + standardise."""
    cols = []
    for j in range(X.shape[1]):
        x = X[:, j].astype(np.float64)
        ok = np.isfinite(x)
        if ok.sum() == 0:
            continue
        med = np.median(x[ok])
        xf = np.where(ok, x, med)
        sd = xf.std()
        cols.append((xf - xf.mean()) / (sd if sd > 1e-9 else 1.0))
        if ok.mean() < 0.995:
            cols.append(ok.astype(np.float64) - ok.mean())
    return np.stack(cols, axis=1) if cols else np.zeros((X.shape[0], 1))


def cv_auc(X: np.ndarray, y: np.ndarray) -> float:
    Z = prep(X)
    rng = np.random.default_rng(SEED)
    fold = rng.integers(0, FOLDS, len(y))
    oof = np.zeros(len(y))
    for f in range(FOLDS):
        tr, te = fold != f, fold == f
        m = LogisticRegression(max_iter=400, C=1.0)
        m.fit(Z[tr], y[tr])
        oof[te] = m.predict_proba(Z[te])[:, 1]
    return float(roc_auc_score(y, oof))


def main():
    cols = pl.read_parquet(FEATURES_DIR / f"anchor={VAL_ANCHOR.isoformat()}.parquet").columns
    base = [c for c in BASE if c in cols]
    df = pl.read_parquet(FEATURES_DIR / f"anchor={VAL_ANCHOR.isoformat()}.parquet",
                         columns=["user_id", "target"] + base).sort("user_id")
    v8 = pl.read_parquet(FEATURES_DIR / f"anchor={VAL_ANCHOR.isoformat()}.v8.parquet").sort("user_id")
    df = df.join(v8, on="user_id", how="left")

    p = np.zeros(df.height)
    for name, w in BLEND.items():
        q = pl.read_parquet(PREDS_DIR / f"{name}_val.parquet").sort("user_id")
        p += w * np.clip(q["pred"].to_numpy().astype(np.float64), 0, None)
    sub = p <= float(np.quantile(p, Q))
    y = (df["target"].to_numpy()[sub] > 0).astype(np.int64)
    lp = np.log1p(p[sub])[:, None]
    Xb = np.column_stack([lp] + [df[c].to_numpy().astype(np.float64)[sub] for c in base])
    print(f"subgroup n={len(y)} buy_rate={y.mean():.4f} base_feats={base}")

    a_A = cv_auc(lp, y)
    a_B = cv_auc(Xb, y)
    print(f"[A] blend only            AUC = {a_A:.4f}")
    print(f"[B] blend + base feats    AUC = {a_B:.4f}  (+{a_B - a_A:.4f})")

    rows = []
    for c in V8_FEATS:
        x = df[c].to_numpy().astype(np.float64)[sub]
        a_C = cv_auc(np.column_stack([Xb, x]), y)
        rows.append({"feat": c, "auc": round(a_C, 4), "delta": round(a_C - a_B, 4)})
        print(f"    +{c:<20} AUC = {a_C:.4f}  delta = {a_C - a_B:+.4f}")
    Xall = np.column_stack([Xb] + [df[c].to_numpy().astype(np.float64)[sub] for c in V8_FEATS])
    a_D = cv_auc(Xall, y)
    print(f"[D] B + ALL v8            AUC = {a_D:.4f}  delta = {a_D - a_B:+.4f}")

    rows.sort(key=lambda r: -r["delta"])
    out = {"auc_blend_only": round(a_A, 4), "auc_blend_base": round(a_B, 4),
           "auc_blend_base_all_v8": round(a_D, 4), "delta_all_v8": round(a_D - a_B, 4),
           "best_single": rows[0], "per_feature": rows}
    (Path(__file__).parents[1] / "reports" / "v8_incr_check.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1))
    print("\n=== RAW JSON ===")
    print(json.dumps({k: v for k, v in out.items() if k != "per_feature"}, ensure_ascii=False))


if __name__ == "__main__":
    main()

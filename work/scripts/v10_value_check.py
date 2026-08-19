"""Fast pre-training screen of the v10 per-channel funnel tier (before any training).

Three questions, on the VAL anchor (250k users, full population, target>0 as label):
  1. AUC of every v10 feature for separating target>0, vs the best existing features.
  2. Redundancy: max |Spearman| of each v10 feature against all 203 existing features
     (v2/v3/v4 champion set). Rank correlation, because trees only see monotone order.
  3. Incremental value (the v6/v8 lesson: univariate AUC is a poor guide) - 4-fold CV
     logistic, base = top existing features, then base + all v10.

Output: work/reports/v10_value_check.json
Run: POLARS_MAX_THREADS=3 USE_V2=1 USE_V3=1 USE_V4=1 .venv/bin/python work/scripts/v10_value_check.py
"""
from __future__ import annotations

import os

os.environ.setdefault("POLARS_MAX_THREADS", "3")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "3")
os.environ.setdefault("USE_V2", "1")
os.environ.setdefault("USE_V3", "1")
os.environ.setdefault("USE_V4", "1")
os.environ["USE_V10"] = "1"

import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
from scipy.stats import rankdata  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from common import REPORTS_DIR, V10_FEATS, VAL_ANCHOR, feature_cols, load_anchor  # noqa: E402

N_INCR = 120_000   # subsample for the CV-logistic step
FOLDS = 4
SEED = 42
TOP_BASE = 15


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def auc_scores(x: np.ndarray, y: np.ndarray) -> dict:
    """AUC on non-null rows, and on all rows with nulls pushed to an extreme bucket."""
    ok = np.isfinite(x)
    out = {"cov": round(float(ok.mean()), 4)}
    if ok.sum() > 50 and 0 < y[ok].mean() < 1:
        a = roc_auc_score(y[ok], x[ok])
        out["auc_nonnull"] = round(float(max(a, 1 - a)), 4)
        out["dir"] = 1 if a >= 0.5 else -1
    else:
        out["auc_nonnull"], out["dir"] = None, None
    if ok.all():
        out["auc_full"] = out["auc_nonnull"]
    else:
        xf = np.where(ok, x, np.nanmin(x[ok]) - 1.0 if ok.any() else 0.0)
        a = roc_auc_score(y, xf)
        out["auc_full"] = round(float(max(a, 1 - a)), 4)
    return out


def prep(X: np.ndarray) -> np.ndarray:
    """median-fill + missing indicator + standardise (same recipe as v8_incr_check)."""
    cols = []
    for j in range(X.shape[1]):
        x = X[:, j].astype(np.float64)
        ok = np.isfinite(x)
        if ok.sum() == 0:
            continue
        xf = np.where(ok, x, np.median(x[ok]))
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
        m = LogisticRegression(max_iter=300, C=1.0)
        m.fit(Z[tr], y[tr])
        oof[te] = m.predict_proba(Z[te])[:, 1]
    return float(roc_auc_score(y, oof))


def main():
    t0 = time.time()
    df = load_anchor(VAL_ANCHOR)
    allf = feature_cols(df)
    base_feats = [c for c in allf if c not in V10_FEATS]
    y = (df["target"].to_numpy().astype(np.float64) > 0).astype(np.int64)
    log(f"val {df.height} rows, base {len(base_feats)} feats, v10 {len(V10_FEATS)} feats, "
        f"pos_rate {y.mean():.4f}")

    # ---------------------------------------------------------------- 1. AUC
    rows = []
    for c in V10_FEATS:
        r = auc_scores(df[c].to_numpy().astype(np.float64), y)
        r["feat"] = c
        rows.append(r)
    base_rows = []
    for c in base_feats:
        r = auc_scores(df[c].to_numpy().astype(np.float64), y)
        r["feat"] = c
        base_rows.append(r)
    rows.sort(key=lambda r: -(r["auc_full"] or 0))
    base_rows.sort(key=lambda r: -(r["auc_full"] or 0))
    log(f"AUC done in {time.time()-t0:.0f}s; best v10 {rows[0]['feat']} {rows[0]['auc_full']}; "
        f"best base {base_rows[0]['feat']} {base_rows[0]['auc_full']}")

    # ------------------------------------------------- 2. Spearman redundancy
    t1 = time.time()
    order = base_feats + V10_FEATS
    R = np.empty((df.height, len(order)), dtype=np.float32)
    for j, c in enumerate(order):
        x = df[c].to_numpy().astype(np.float64)
        ok = np.isfinite(x)
        if not ok.all():
            x = np.where(ok, x, np.median(x[ok]) if ok.any() else 0.0)
        r = rankdata(x)
        r -= r.mean()
        s = r.std()
        R[:, j] = (r / (s if s > 1e-9 else 1.0)).astype(np.float32)
    nb = len(base_feats)
    C = (R[:, nb:].T.astype(np.float64) @ R[:, :nb].astype(np.float64)) / df.height
    del R
    A = np.abs(C)
    corr = []
    for i, c in enumerate(V10_FEATS):
        j = int(np.argmax(A[i]))
        corr.append({"feat": c, "max_abs_corr": round(float(A[i, j]), 4),
                     "partner": base_feats[j]})
    corr.sort(key=lambda r: -r["max_abs_corr"])
    max_corr = max(r["max_abs_corr"] for r in corr)
    n_dup = sum(1 for r in corr if r["max_abs_corr"] > 0.95)
    log(f"spearman done in {time.time()-t1:.0f}s; max {max_corr:.4f}; "
        f"{n_dup}/{len(V10_FEATS)} above 0.95")

    # ----------------------------------------------------- 3. incremental CV
    t2 = time.time()
    rng = np.random.default_rng(SEED)
    idx = rng.choice(df.height, size=min(N_INCR, df.height), replace=False)
    top_base = [r["feat"] for r in base_rows[:TOP_BASE]]
    Xb = np.column_stack([df[c].to_numpy().astype(np.float64)[idx] for c in top_base])
    ys = y[idx]
    a_B = cv_auc(Xb, ys)
    Xv10 = np.column_stack([df[c].to_numpy().astype(np.float64)[idx] for c in V10_FEATS])
    a_D = cv_auc(np.column_stack([Xb, Xv10]), ys)
    a_v10only = cv_auc(Xv10, ys)
    log(f"incremental: base({TOP_BASE}) {a_B:.4f} -> +all v10 {a_D:.4f} "
        f"(delta {a_D-a_B:+.4f}); v10 alone {a_v10only:.4f}; {time.time()-t2:.0f}s")

    out = {
        "n_features_v10": len(V10_FEATS),
        "n_base_features": len(base_feats),
        "pos_rate": round(float(y.mean()), 4),
        "best_new": rows[0],
        "top10_new": rows[:10],
        "best_base": base_rows[0],
        "top5_base": base_rows[:5],
        "max_corr_with_existing": max_corr,
        "n_new_above_0.95": n_dup,
        "n_new_above_0.90": sum(1 for r in corr if r["max_abs_corr"] > 0.90),
        "median_max_corr": round(float(np.median([r["max_abs_corr"] for r in corr])), 4),
        "least_redundant": corr[-8:][::-1],
        "most_redundant": corr[:8],
        "incr_auc_base": round(a_B, 4),
        "incr_auc_base_plus_v10": round(a_D, 4),
        "incr_delta": round(a_D - a_B, 4),
        "incr_auc_v10_only": round(a_v10only, 4),
        "all_new": rows,
        "all_corr": corr,
    }
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "v10_value_check.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print("\n=== RAW JSON ===")
    print(json.dumps({k: v for k, v in out.items() if k not in ("all_new", "all_corr")},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()

"""Mirror test of the GIFTER hypothesis: does the trait pay off in a GIFT window?

The val window (2026-01-15..02-13) contains NO major event, so testing gifter
features there is a weak test. The test window (02-14..03-15) covers Feb 23 + the
run-up to Mar 8. So we repeat the value check at anchors whose 30d target window IS
covered by a major event, and at matched control anchors whose window is not.

The subgroup ("model says inactive") is defined by a window-agnostic PROXY of the
blend prediction, so the same rule applies at every anchor. Step 1 validates the
proxy against the real blend on VAL (rank correlation + overlap of the bottom 20%).

Run: POLARS_MAX_THREADS=3 .venv/bin/python work/scripts/v8_mirror_check.py
"""
from __future__ import annotations

import os

os.environ.setdefault("POLARS_MAX_THREADS", "3")

import json
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))
from common import FEATURES_DIR, PREDS_DIR, V8_FEATS, VAL_ANCHOR  # noqa: E402
from holiday_cal import group_events, daily_gmv, peak_mask  # noqa: E402
from v8_value_check import BLEND, Q, auc, score_feature  # noqa: E402

PROXY_COLS = ["gmv_sum_90", "gmv_sum_365", "ord_days_90", "ord_days_365", "rec_order"]
BASE_FEATS = ["rec_order", "ord_days_90", "ord_days_365"]
PAD = 5

GIFT = [date(2025, 10, 15), date(2025, 10, 29), date(2025, 11, 5), date(2025, 12, 17)]
NORM = [date(2025, 7, 16), date(2025, 9, 3), date(2025, 9, 10), date(2026, 1, 7), VAL_ANCHOR]


def proxy_score(df: pl.DataFrame) -> np.ndarray:
    """Window-agnostic stand-in for the blend prediction (validated on VAL)."""
    g90 = np.log1p(df["gmv_sum_90"].to_numpy().astype(np.float64))
    g365 = np.log1p(df["gmv_sum_365"].to_numpy().astype(np.float64))
    o90 = np.log1p(df["ord_days_90"].to_numpy().astype(np.float64))
    o365 = np.log1p(df["ord_days_365"].to_numpy().astype(np.float64))
    rec = df["rec_order"].to_numpy().astype(np.float64)
    rec = np.where(np.isfinite(rec), rec, 400.0)
    return 0.45 * g90 + 0.20 * g365 + 0.25 * o90 + 0.10 * o365 - 0.004 * np.minimum(rec, 400.0)


def gift_coverage(a: date, ev) -> float:
    lo, hi = a + timedelta(days=1), a + timedelta(days=30)
    days = {lo + timedelta(days=i) for i in range((hi - lo).days + 1)}
    cov = set()
    for s, e in ev:
        s2, e2 = s - timedelta(days=PAD), e + timedelta(days=PAD)
        cov |= {d for d in days if s2 <= d <= e2}
    return len(cov) / len(days)


def anchor_rows(a: date):
    base = pl.read_parquet(FEATURES_DIR / f"anchor={a.isoformat()}.parquet",
                           columns=["user_id", "target"] + PROXY_COLS).sort("user_id")
    v8 = pl.read_parquet(FEATURES_DIR / f"anchor={a.isoformat()}.v8.parquet").sort("user_id")
    return base.join(v8, on="user_id", how="left")


def run_anchor(a: date, ev, use_blend: bool = False):
    df = anchor_rows(a)
    if use_blend:
        p = np.zeros(df.height)
        for name, w in BLEND.items():
            q = pl.read_parquet(PREDS_DIR / f"{name}_val.parquet").sort("user_id")
            p += w * np.clip(q["pred"].to_numpy().astype(np.float64), 0, None)
        s = np.log1p(p)
    else:
        s = proxy_score(df)
    sub = s <= float(np.quantile(s, Q))
    y = (df["target"].to_numpy() > 0).astype(np.int64)[sub]
    res = {}
    for c in BASE_FEATS + V8_FEATS:
        x = df[c].to_numpy().astype(np.float64)[sub]
        res[c] = score_feature(x, y)
    return {
        "anchor": a.isoformat(),
        "gift_cov": round(gift_coverage(a, ev), 3),
        "n_sub": int(sub.sum()),
        "buy_rate": round(float(y.mean()), 4),
        "auc": res,
    }


def main():
    dates, g = daily_gmv()
    mask, _, _ = peak_mask(g)
    ev = [e for e in group_events(dates, mask) if (e[1] - e[0]).days + 1 >= 3]

    # ---- step 1: is the proxy a faithful stand-in for the blend?
    dfv = anchor_rows(VAL_ANCHOR)
    p = np.zeros(dfv.height)
    for name, w in BLEND.items():
        q = pl.read_parquet(PREDS_DIR / f"{name}_val.parquet").sort("user_id")
        p += w * np.clip(q["pred"].to_numpy().astype(np.float64), 0, None)
    s = proxy_score(dfv)
    rp = np.argsort(np.argsort(p)).astype(np.float64)
    rs = np.argsort(np.argsort(s)).astype(np.float64)
    spear = float(np.corrcoef(rp, rs)[0, 1])
    ov = float((( p <= np.quantile(p, Q)) & (s <= np.quantile(s, Q))).sum() / (Q * len(p)))
    print(f"[proxy] spearman vs blend = {spear:.4f}; bottom-{Q:.0%} overlap = {ov:.4f}")

    out = {"proxy_spearman": round(spear, 4), "proxy_overlap": round(ov, 4), "anchors": []}
    print(f"\n{'anchor':<12}{'giftcov':>8}{'buyrate':>9}   top new features (auc_full)")
    for a in sorted(GIFT + NORM):
        r = run_anchor(a, ev)
        out["anchors"].append(r)
        top = sorted(((k, v["auc_full"]) for k, v in r["auc"].items() if k in V8_FEATS),
                     key=lambda t: -(t[1] or 0))[:4]
        bb = max(((k, v["auc_full"]) for k, v in r["auc"].items() if k in BASE_FEATS),
                 key=lambda t: (t[1] or 0))
        print(f"{r['anchor']:<12}{r['gift_cov']:>8.2f}{r['buy_rate']:>9.4f}   "
              + ", ".join(f"{k}={v}" for k, v in top) + f"  | base_best {bb[0]}={bb[1]}")

    # ---- gift vs normal contrast, feature by feature
    gf = [r for r in out["anchors"] if r["gift_cov"] >= 0.30]
    nf = [r for r in out["anchors"] if r["gift_cov"] <= 0.05]
    print(f"\ncontrast: {len(gf)} gift anchors vs {len(nf)} normal anchors")
    print(f"{'feat':<20}{'gift':>8}{'normal':>9}{'delta':>9}")
    contrast = []
    for c in BASE_FEATS + V8_FEATS:
        ag = np.mean([r["auc"][c]["auc_full"] for r in gf if r["auc"][c]["auc_full"]])
        an = np.mean([r["auc"][c]["auc_full"] for r in nf if r["auc"][c]["auc_full"]])
        contrast.append({"feat": c, "gift": round(float(ag), 4), "normal": round(float(an), 4),
                         "delta": round(float(ag - an), 4)})
    contrast.sort(key=lambda d: -d["delta"])
    for d in contrast:
        print(f"{d['feat']:<20}{d['gift']:>8.4f}{d['normal']:>9.4f}{d['delta']:>+9.4f}")
    out["contrast"] = contrast
    best_gift = max((d for d in contrast if d["feat"] in V8_FEATS), key=lambda d: d["gift"])
    out["best_new_gift_auc"] = best_gift
    (Path(__file__).parents[1] / "reports" / "v8_mirror_check.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1))
    print("\n=== RAW JSON ===")
    print(json.dumps({k: v for k, v in out.items() if k != "anchors"}, ensure_ascii=False))


if __name__ == "__main__":
    main()

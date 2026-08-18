"""Promo-elasticity stage 2: decisive checks (no training).

A) Strong naive base (gmv 30/90/180 + purchase-day counts + recency) residual test
   on promo vs normal windows -> does the promo signal survive a frequency-aware base?
B) Within-window mechanism test on promo windows: corr(lift, residual of PROMO-part
   of target) vs corr(lift, residual of NORMAL-part of target). Window-level effects
   cancel; a real promo-response feature must predict the promo part specifically.
C) Novelty of the test-anchor direction vs measured LB span (subs.MEASURED).
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
from common import DATA_END, DATA_START, TEST_ANCHOR, TRAIN_PARQUET, VAL_ANCHOR, user_universe  # noqa: E402
import subs  # noqa: E402

ROOT = Path("/Users/alexanderkondakov/ozon-cup")
TAU_E, TAU_L = 4.0, 10.0

# ---- calendar (same as stage 1) ----
daily = (
    pl.scan_parquet(TRAIN_PARQUET)
    .group_by("event_date")
    .agg(pl.col("gmv").sum().alias("gmv_day"))
    .sort("event_date")
    .collect()
)
cal = pl.DataFrame({"event_date": pl.date_range(DATA_START, DATA_END, "1d", eager=True)})
daily = cal.join(daily, on="event_date", how="left").fill_null(0.0).sort("event_date")
dates = daily["event_date"].to_list()
g = daily["gmv_day"].to_numpy().astype(np.float64)
n_days = len(g)
med = np.array([np.median(g[max(0, i - 14):min(n_days, i + 14)]) for i in range(n_days)])
ratio = g / np.maximum(med, 1.0)
promo_mask = ratio >= np.quantile(ratio, 0.85)
d_arr = np.array(dates)

ud = (
    pl.scan_parquet(TRAIN_PARQUET)
    .group_by("user_id", "event_date")
    .agg(pl.col("gmv").sum().alias("gmv"))
    .filter(pl.col("gmv") > 0)
    .collect()
    .join(pl.DataFrame({"event_date": dates, "is_promo": promo_mask.tolist()}), on="event_date", how="left")
    .with_columns(pl.col("gmv").log1p().alias("lg"))
)
UNI = user_universe()
N = UNI.height


def align(df: pl.DataFrame, col: str) -> np.ndarray:
    return UNI.join(df, on="user_id", how="left").sort("user_id")[col].fill_null(0.0).to_numpy().astype(np.float64)


def win_lp(a: date, b: date, promo_only: bool | None = None) -> np.ndarray:
    f = ud.filter((pl.col("event_date") >= a) & (pl.col("event_date") <= b))
    if promo_only is True:
        f = f.filter(pl.col("is_promo"))
    elif promo_only is False:
        f = f.filter(~pl.col("is_promo"))
    return np.log1p(align(f.group_by("user_id").agg(pl.col("gmv").sum().alias("s")), "s"))


def strong_base(a: date) -> list[np.ndarray]:
    out = [win_lp(a - timedelta(days=d - 1), a) for d in (30, 90, 180)]
    for d in (30, 90, 180):
        k = ud.filter((pl.col("event_date") > a - timedelta(days=d)) & (pl.col("event_date") <= a)) \
              .group_by("user_id").agg(pl.len().alias("k"))
        out.append(np.log1p(align(k, "k")))
    last = ud.filter(pl.col("event_date") <= a).group_by("user_id").agg(pl.col("event_date").max().alias("d"))
    lastu = UNI.join(last, on="user_id", how="left").sort("user_id")["d"].to_list()
    rec = np.array([(a - x).days if x is not None else 400 for x in lastu], dtype=np.float64)
    out.append(np.log1p(np.minimum(rec, 400)))
    return out


def e_promo_at(anchor: date):
    dmask = d_arr <= np.datetime64(anchor)
    n_p = int(promo_mask[dmask].sum()); n_n = int((~promo_mask[dmask]).sum())
    p0 = n_p / (n_p + n_n)
    agg = (
        ud.filter(pl.col("event_date") <= anchor)
        .group_by("user_id")
        .agg(
            (pl.col("lg") * pl.col("is_promo")).sum().alias("slp_p"),
            (pl.col("lg") * (~pl.col("is_promo"))).sum().alias("slp_n"),
            pl.col("is_promo").sum().alias("k_p"),
            pl.len().alias("k"),
        )
    )
    u = UNI.join(agg, on="user_id", how="left").sort("user_id").fill_null(0)
    k = u["k"].to_numpy().astype(np.float64); k_p = u["k_p"].to_numpy().astype(np.float64)
    e_raw = u["slp_p"].to_numpy() / max(n_p, 1) - u["slp_n"].to_numpy() / max(n_n, 1)
    return e_raw * k / (k + TAU_E), (k_p + p0 * TAU_L) / (k + TAU_L) - p0


def corr(a, b):
    return float(np.corrcoef(a, b)[0, 1])


def ols_resid(y, X):
    A = np.stack([np.ones(len(y))] + X, axis=1)
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return y - A @ coef


WINDOWS = [
    ("promo_1111_2025-10-28", date(2025, 10, 28), "promo"),
    ("promo_dec_2025-11-25", date(2025, 11, 25), "promo"),
    ("normal_2025-09-09", date(2025, 9, 9), "normal"),
    ("normal_2025-06-17", date(2025, 6, 17), "normal"),
    ("val_2026-01-14", VAL_ANCHOR, "mixed"),
]

res = {}
for name, a, kind in WINDOWS:
    t = win_lp(a + timedelta(days=1), a + timedelta(days=30))
    X = strong_base(a)
    e_eb, lift = e_promo_at(a)
    r = ols_resid(t, X)
    wmask = (d_arr > np.datetime64(a)) & (d_arr <= np.datetime64(a + timedelta(days=30)))
    row = {
        "kind": kind,
        "promo_share_tgt": round(float(promo_mask[wmask].mean()), 3),
        "corr_e_strong": round(corr(e_eb, r), 4),
        "corr_lift_strong": round(corr(lift, r), 4),
    }
    # B) within-window split (promo part vs normal part of the target)
    n_p_t = int(promo_mask[wmask].sum())
    if n_p_t >= 2:
        t_p = win_lp(a + timedelta(days=1), a + timedelta(days=30), promo_only=True)
        t_n = win_lp(a + timedelta(days=1), a + timedelta(days=30), promo_only=False)
        r_p = ols_resid(t_p, X)
        r_n = ols_resid(t_n, X)
        row.update(
            n_promo_tgt=n_p_t,
            corr_lift_promopart=round(corr(lift, r_p), 4),
            corr_lift_normpart=round(corr(lift, r_n), 4),
            corr_e_promopart=round(corr(e_eb, r_p), 4),
            corr_e_normpart=round(corr(e_eb, r_n), 4),
        )
    res[name] = row
    print(f"[win2] {name}: {row}", flush=True)

e_t, lift_t = e_promo_at(TEST_ANCHOR)
Sp = subs.span_matrix(subs.MEASURED, N)
for nm, v in (("e_eb", e_t), ("lift", lift_t)):
    z = (v - v.mean()) / v.std()
    nov, _ = subs.novelty(z, Sp)
    print(f"[novelty] {nm}@test: {nov:.3f}")
    res[f"novelty_{nm}"] = round(nov, 3)

print("\n=== STAGE2 JSON ===")
print(json.dumps(res, ensure_ascii=False))

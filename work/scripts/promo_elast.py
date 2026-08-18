"""Promo-elasticity experiment (no training; POLARS_MAX_THREADS=3).

Idea: the test window 14.02-15.03.2026 is promo-heavy. Users differ in how much of
their buying concentrates on global promo days. None of our features encode response
to GLOBAL events. Steps:
  1. Global daily GMV series from train.parquet; promo days = top-15% days by
     ratio of daily GMV to a centered 28d rolling median (data-driven, no external info).
  2. Per-user promo elasticity on history <= anchor, EB-shrunk:
       e_promo = mean log1p(daily gmv) over promo calendar days - over normal days
       lift    = share of purchase-days on promo days - promo share of calendar
  3. Signal check without training: corr(e_promo, residual of naive AR base) on
     promo-covered target windows vs normal windows.
  4. If promo-window corr exceeds normal-window corr by > 0.02: build direction
     h_promo (residualized vs measured LB basis, rms 0.12), save probe

Run:  .venv/bin/python work/scripts/promo_elast.py [--build | --no-build]
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
TAU_E = 4.0     # EB shrink (purchase days) for e_promo
TAU_L = 10.0    # beta prior strength for lift share
TOP_Q = 0.85    # promo days = top 15% by ratio
STEP = 0.45
RMS = 0.12

# ---------------------------------------------------------------- 1. calendar
daily = (
    pl.scan_parquet(TRAIN_PARQUET)
    .group_by("event_date")
    .agg(pl.col("gmv").sum().alias("gmv_day"), pl.col("to_ord").sum().alias("ord_day"))
    .sort("event_date")
    .collect()
)
cal = pl.DataFrame({"event_date": pl.date_range(DATA_START, DATA_END, "1d", eager=True)})
daily = cal.join(daily, on="event_date", how="left").fill_null(0.0).sort("event_date")
dates = daily["event_date"].to_list()
g = daily["gmv_day"].to_numpy().astype(np.float64)
n_days = len(g)
med = np.empty(n_days)
for i in range(n_days):
    lo, hi = max(0, i - 14), min(n_days, i + 14)
    med[i] = np.median(g[lo:hi])
ratio = g / np.maximum(med, 1.0)
thr = float(np.quantile(ratio, TOP_Q))
promo_mask = ratio >= thr
d_arr = np.array(dates)

print(f"[cal] days={n_days} promo_days={int(promo_mask.sum())} thr={thr:.4f}")
runs, cur = [], None
for d, p in zip(dates, promo_mask):
    if p and cur is None:
        cur = [d, d]
    elif p:
        cur[1] = d
    elif cur is not None:
        runs.append(tuple(cur)); cur = None
if cur is not None:
    runs.append(tuple(cur))
print("[cal] promo runs:", "; ".join(f"{a.isoformat()}..{b.isoformat()}" for a, b in runs))

# ------------------------------------------------------- user purchase days
ud = (
    pl.scan_parquet(TRAIN_PARQUET)
    .group_by("user_id", "event_date")
    .agg(pl.col("gmv").sum().alias("gmv"))
    .filter(pl.col("gmv") > 0)
    .collect()
)
ud = ud.join(
    pl.DataFrame({"event_date": dates, "is_promo": promo_mask.tolist()}), on="event_date", how="left"
).with_columns(pl.col("gmv").log1p().alias("lg"))
print(f"[ud] purchase user-days: {ud.height}")

UNI = user_universe()
N = UNI.height


def win_lp(a: date, b: date) -> np.ndarray:
    """log1p of per-user gmv sum over [a, b], aligned to universe."""
    w = (
        ud.filter((pl.col("event_date") >= a) & (pl.col("event_date") <= b))
        .group_by("user_id")
        .agg(pl.col("gmv").sum().alias("s"))
    )
    s = UNI.join(w, on="user_id", how="left").sort("user_id")["s"].fill_null(0.0).to_numpy()
    return np.log1p(s)


def e_promo_at(anchor: date):
    """(e_eb, lift_eb, e_raw, meta) on history <= anchor."""
    dmask = d_arr <= np.datetime64(anchor)
    n_p = int(promo_mask[dmask].sum())
    n_n = int((~promo_mask[dmask]).sum())
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
    k = u["k"].to_numpy().astype(np.float64)
    k_p = u["k_p"].to_numpy().astype(np.float64)
    e_raw = u["slp_p"].to_numpy() / max(n_p, 1) - u["slp_n"].to_numpy() / max(n_n, 1)
    e_eb = e_raw * k / (k + TAU_E)
    lift_eb = (k_p + p0 * TAU_L) / (k + TAU_L) - p0
    return e_eb, lift_eb, e_raw, {"n_promo_hist": n_p, "n_norm_hist": n_n, "p0": round(p0, 4)}


def corr(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.corrcoef(a, b)[0, 1])


def spear(a: np.ndarray, b: np.ndarray) -> float:
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    return corr(ra, rb)


def ols_resid(y: np.ndarray, X: list[np.ndarray]) -> np.ndarray:
    A = np.stack([np.ones(len(y))] + X, axis=1)
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return y - A @ coef


WINDOWS = [
    ("val_2026-01-14", VAL_ANCHOR),
    ("promo_short_2025-02-06", date(2025, 2, 6)),
    ("promo_1111_2025-10-28", date(2025, 10, 28)),
    ("normal_2025-09-09", date(2025, 9, 9)),
    ("normal_2025-06-17", date(2025, 6, 17)),
]

results = {}
for name, a in WINDOWS:
    t = win_lp(a + timedelta(days=1), a + timedelta(days=30))
    base30 = win_lp(a - timedelta(days=29), a)
    e_eb, lift_eb, e_raw, meta = e_promo_at(a)
    wmask = (d_arr > np.datetime64(a)) & (d_arr <= np.datetime64(a + timedelta(days=30)))
    promo_share_tgt = float(promo_mask[wmask].mean())
    r = ols_resid(t, [base30])
    row = {
        **meta,
        "promo_share_target": round(promo_share_tgt, 3),
        "corr_e": round(corr(e_eb, r), 4),
        "corr_lift": round(corr(lift_eb, r), 4),
        "corr_e_raw": round(corr(e_raw, r), 4),
        "spear_e": round(spear(e_eb, r), 4),
    }
    # richer base for long-history anchors (>=200d): controls recency/frequency harder
    if (a - DATA_START).days >= 200:
        base90 = win_lp(a - timedelta(days=89), a)
        base180 = win_lp(a - timedelta(days=179), a)
        r2 = ols_resid(t, [base30, base90, base180])
        row["corr_e_rich"] = round(corr(e_eb, r2), 4)
        row["corr_lift_rich"] = round(corr(lift_eb, r2), 4)
    # our model base on val (c_cand_val = clean base)
    if a == VAL_ANCHOR:
        p = ROOT / "work/preds/c_cand_val.parquet"
        if p.exists():
            lpm = np.log1p(
                np.clip(pl.read_parquet(p).sort("user_id")["pred"].to_numpy().astype(np.float64), 0, None)
            )
            rm = ols_resid(t, [lpm])
            row["corr_e_modelbase"] = round(corr(e_eb, rm), 4)
            row["corr_lift_modelbase"] = round(corr(lift_eb, rm), 4)
    results[name] = row
    print(f"[win] {name}: {row}")

# ------------------------------------------------------------------ verdict
c_promo = results["promo_short_2025-02-06"]["corr_e"]
c_norm = results["val_2026-01-14"]["corr_e"]
c_promo_rich_win = results["promo_1111_2025-10-28"]["corr_e"]
c_norm_ctrl = (results["normal_2025-09-09"]["corr_e"] + results["normal_2025-06-17"]["corr_e"]) / 2
gate_primary = c_promo - c_norm > 0.02
gate_long = c_promo_rich_win - c_norm_ctrl > 0.02
print(f"[gate] primary(promo_short - val) = {c_promo - c_norm:+.4f} -> {gate_primary}")
print(f"[gate] long(promo_1111 - normal_ctrl) = {c_promo_rich_win - c_norm_ctrl:+.4f} -> {gate_long}")

force = "--build" in sys.argv
skip = "--no-build" in sys.argv
build = force or (not skip and (gate_primary or gate_long))

out = {
    "corr_promo_window": c_promo,
    "corr_normal_window": c_norm,
    "verdict": None,
    "file": None,
    "notes": "",
}

if build:
    sign = 1.0 if (c_promo + c_promo_rich_win) >= 0 else -1.0
    e_test, lift_test, _, meta_t = e_promo_at(TEST_ANCHOR)
    z = (e_test - e_test.mean()) / e_test.std()
    d = sign * z
    uidF4, lpF4 = subs.lp("F4_applied.csv")
    Sp = subs.span_matrix(subs.MEASURED, N)
    nov, resid = subs.novelty(d, Sp)
    h = resid / np.sqrt((resid**2).mean()) * RMS
    np.save(ROOT / "work/probes/h_promo.npy", h)
    lp_probe = np.clip(lpF4 + STEP * h, 0, None)
    q = float((h**2).mean())
    print(f"[build] novelty={nov:.3f} q={q:.5f} step={STEP} min_lp={lp_probe.min():.3f} test_meta={meta_t}")
    out["novelty"] = round(nov, 3)

out["all_windows"] = results
print("\n=== RAW JSON ===")
print(json.dumps(out, ensure_ascii=False))

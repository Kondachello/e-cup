"""v8 feature tier: GIFTERS and AWAKENINGS.

Diagnosis this targets: our error concentrates in the subgroup "model says inactive
but 16% buy". Hypothesis: those 16% are GIFTERS - users silent most of the year who
wake up for holidays. The test window (14.02-15.03) covers Feb 23 + the run-up to
Mar 8, i.e. a gift window. The gifter trait is estimated over ALL holiday peaks of
the year (9 multi-day events), not one last-year window -> an order of magnitude
less noisy.

Calendar: data-driven only (holiday_cal.py) - top-15% days by ratio of global daily
GMV to a centered 28d rolling median, consecutive peaks merged into events, MAJOR
events = runs of >= 3 days (the genuine multi-day campaigns; singleton blips are
weekend/payday noise and would push window coverage to 72%).

Features (all Float32, full 250k universe, null where undefined), history <= anchor:

  gifter
    gf_ord_share      share of user's order-days inside holiday windows (+-5d)
    gf_gmv_share      share of user's GMV inside holiday windows
    gf_ord_share_eb   EB-shrunk EXCESS share over the user's own calendar baseline
    gf_lift           log rate-ratio order-days holiday vs non-holiday, shrunk
    gf_gmv_lift       log1p rate-ratio of daily GMV, holiday vs non-holiday, shrunk
    gf_only_flag      1 if share > 0.5 and >= 2 order-days ("buys almost only on holidays")
    gf_n_events       # distinct events with >= 1 order-day (trait robustness)
    gf_n_events_frac  gf_n_events / events available at this anchor
    gf_days_since_ev  days since last order INSIDE an event window
    gf_last_ev_hit    ordered in the most recent fully-observed event window

  awakening (regime-change detection, NOT renewal/hazard - that line is closed)
    wk_r7, wk_r14         log ratio of 7d/14d order rate to personal yearly base rate
    wk_act_r7, wk_act_r14 same on any-activity days
    wk_gap_ratio          current pause / median personal inter-order pause
    wk_gap_vs_max         current pause / max historical pause
    wk_ent90              entropy of the 90d daily activity profile / log(90)
    wk_ent90_sh           same normalised by log(#active days) - pure evenness
    wk_cv_iei             std/mean of inter-order intervals

Output: work/features/anchor=DATE.v8.parquet, joined by common.load_anchor when USE_V8=1.

Usage:
  POLARS_MAX_THREADS=3 .venv/bin/python work/scripts/build_features_v8.py [--anchors a,b]
"""
from __future__ import annotations

import os

_T = os.environ.get("THREADS", "3")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "POLARS_MAX_THREADS"):
    os.environ.setdefault(_v, _T)

import argparse
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))
from common import (  # noqa: E402
    DATA_START, FEATURES_DIR, TEST_ANCHOR, TRAIN_PARQUET, VAL_ANCHOR, user_universe,
)
from exp_lib import available_train_anchors  # noqa: E402
from holiday_cal import daily_gmv, group_events, peak_mask  # noqa: E402

PAD = 5           # +-5 days around each event
MIN_EVENT_LEN = 3  # major events only
TAU_L = 10.0      # beta prior strength for the share
TAU_E = 4.0       # shrink strength (order-days) for the lift
EPS_R = 1.0 / 365.0  # rate floor: one event per year
GAP_CAP = 50.0

FEATS = [
    "gf_ord_share", "gf_gmv_share", "gf_ord_share_eb", "gf_lift", "gf_gmv_lift",
    "gf_only_flag", "gf_n_events", "gf_n_events_frac", "gf_days_since_ev", "gf_last_ev_hit",
    "wk_r7", "wk_r14", "wk_act_r7", "wk_act_r14", "wk_gap_ratio", "wk_gap_vs_max",
    "wk_ent90", "wk_ent90_sh", "wk_cv_iei",
]


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


# --------------------------------------------------------------- calendar
def build_calendar():
    dates, g = daily_gmv()
    mask, ratio, thr = peak_mask(g)
    ev_all = group_events(dates, mask)
    ev = [e for e in ev_all if (e[1] - e[0]).days + 1 >= MIN_EVENT_LEN]
    nd = len(dates)
    hw = np.zeros(nd, dtype=bool)          # any-event window
    ev_id = np.full(nd, -1, dtype=np.int16)  # window -> event index (later event wins)
    d0 = dates[0]
    for j, (a, b) in enumerate(ev):
        i0 = max(0, (a - timedelta(days=PAD) - d0).days)
        i1 = min(nd - 1, (b + timedelta(days=PAD) - d0).days)
        hw[i0:i1 + 1] = True
        ev_id[i0:i1 + 1] = j
    return dates, ev_all, ev, hw, ev_id


# --------------------------------------------------------------- data load
def load_frames(uni: pl.DataFrame):
    """Order-day frame (small, polars) + activity arrays (large, numpy)."""
    uid = uni["user_id"].to_numpy()
    pos = pl.DataFrame({"user_id": uid, "uix": np.arange(len(uid), dtype=np.int32)})

    lf = pl.scan_parquet(TRAIN_PARQUET)
    od = (
        lf.filter(pl.col("to_ord") > 0)
        .select("user_id", "event_date", "gmv")
        .collect(engine="streaming")
        .join(pos, on="user_id", how="inner")
        .with_columns(
            ((pl.col("event_date") - pl.lit(DATA_START)).dt.total_days()).cast(pl.Int32).alias("d"),
            pl.col("gmv").cast(pl.Float32),
        )
        .select("uix", "d", "gmv")
        .sort(["uix", "d"])
    )
    act = (
        lf.select("user_id", "event_date", "searches", "to_cart", "to_ord")
        .collect(engine="streaming")
        .join(pos, on="user_id", how="inner")
        .with_columns(
            ((pl.col("event_date") - pl.lit(DATA_START)).dt.total_days()).cast(pl.Int32).alias("d"),
            (pl.col("searches") + pl.col("to_cart") + pl.col("to_ord"))
            .cast(pl.Float32).clip(0.0, None).alias("w"),
        )
        .select("uix", "d", "w")
    )
    a_uix = act["uix"].to_numpy()
    a_d = act["d"].to_numpy().astype(np.int32)
    a_w = act["w"].to_numpy().astype(np.float32)
    del act
    return od, a_uix, a_d, a_w


def grouped_gap_stats(od: pl.DataFrame, A: int, n: int):
    """median / max inter-order gap and CV of inter-order intervals, per user."""
    g = (
        od.filter(pl.col("d") <= A)
        .select("uix", "d")
        .with_columns((pl.col("d") - pl.col("d").shift(1).over("uix")).alias("gap"))
        .drop_nulls("gap")
        .group_by("uix")
        .agg(
            pl.col("gap").median().alias("gmed"),
            pl.col("gap").max().alias("gmax"),
            pl.col("gap").mean().alias("gmean"),
            pl.col("gap").std().alias("gstd"),
            pl.len().alias("ngap"),
        )
    )
    out = {}
    ix = g["uix"].to_numpy()
    for c in ("gmed", "gmax", "gmean", "gstd", "ngap"):
        v = np.full(n, np.nan)
        v[ix] = g[c].to_numpy().astype(np.float64)
        out[c] = v
    return out


def entropy90(a_uix, a_d, a_w, A: int, n: int):
    """Normalised entropy of the daily activity profile over the last 90 days."""
    m = (a_d <= A) & (a_d > A - 90)
    u, w = a_uix[m], a_w[m].astype(np.float64)
    tot = np.bincount(u, weights=w, minlength=n)
    cnt = np.bincount(u, minlength=n).astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        slogs = np.bincount(u, weights=w * np.log(np.maximum(w, 1e-12)), minlength=n)
        H = np.log(np.maximum(tot, 1e-12)) - slogs / np.maximum(tot, 1e-12)  # = -sum p log p
    H = np.where(tot > 0, np.maximum(H, 0.0), np.nan)
    ent = H / np.log(90.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        ent_sh = np.where(cnt >= 2, H / np.log(np.maximum(cnt, 2.0)), np.nan)
    return ent.astype(np.float64), ent_sh.astype(np.float64)


# --------------------------------------------------------------- per anchor
def build(anchor: date, uni, od, a_uix, a_d, a_w, dates, ev, hw, ev_id, first_day):
    t0 = time.time()
    n = uni.height
    A = (anchor - DATA_START).days
    cum_hw = np.concatenate([[0], np.cumsum(hw)])   # cum_hw[i] = #hw days in [0, i-1]

    # events fully observed at this anchor (window end <= anchor)
    ev_ok = [j for j, (a, b) in enumerate(ev) if (b + timedelta(days=PAD)) <= anchor]
    n_ev_ok = len(ev_ok)
    last_ev = max(ev_ok) if ev_ok else -1

    # ---- per-user calendar baseline over the observed span [first_day, A]
    fd = np.clip(first_day, 0, A)
    span = (A - fd + 1).astype(np.float64)
    n_hw = (cum_hw[A + 1] - cum_hw[fd]).astype(np.float64)
    n_nw = np.maximum(span - n_hw, 1.0)
    n_hw_s = np.maximum(n_hw, 1.0)
    p0 = n_hw / np.maximum(span, 1.0)

    # ---- order-day aggregates
    o = od.filter(pl.col("d") <= A)
    ou = o["uix"].to_numpy()
    odd = o["d"].to_numpy()
    og = o["gmv"].to_numpy().astype(np.float64)
    is_h = hw[odd]
    eid = ev_id[odd]

    k = np.bincount(ou, minlength=n).astype(np.float64)
    k_h = np.bincount(ou[is_h], minlength=n).astype(np.float64)
    s = np.bincount(ou, weights=og, minlength=n)
    s_h = np.bincount(ou[is_h], weights=og[is_h], minlength=n)

    have = k > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        gf_ord_share = np.where(have, k_h / np.maximum(k, 1.0), np.nan)
        gf_gmv_share = np.where(have & (s > 0), s_h / np.maximum(s, 1e-9), np.nan)
    gf_ord_share_eb = (k_h + p0 * TAU_L) / (k + TAU_L) - p0
    shrink = k / (k + TAU_E)
    gf_lift = np.log((k_h / n_hw_s + EPS_R) / ((k - k_h) / n_nw + EPS_R)) * shrink
    gf_gmv_lift = (np.log1p(s_h / n_hw_s) - np.log1p((s - s_h) / n_nw)) * shrink
    gf_only_flag = np.where(k >= 2, (gf_ord_share > 0.5).astype(np.float64), np.nan)

    # distinct events hit
    okset = np.zeros(len(ev) + 1, dtype=bool)
    for j in ev_ok:
        okset[j] = True
    m_ev = is_h & (eid >= 0) & okset[np.maximum(eid, 0)]
    if m_ev.any():
        pair = ou[m_ev].astype(np.int64) * (len(ev) + 1) + eid[m_ev].astype(np.int64)
        uq = np.unique(pair)
        gf_n_events = np.bincount((uq // (len(ev) + 1)).astype(np.int64), minlength=n).astype(np.float64)
        hit_last = np.zeros(n, dtype=np.float64)
        if last_ev >= 0:
            sel = eid[m_ev] == last_ev
            hit_last[np.unique(ou[m_ev][sel])] = 1.0
    else:
        gf_n_events = np.zeros(n)
        hit_last = np.zeros(n)
    gf_n_events_frac = gf_n_events / max(n_ev_ok, 1)
    gf_last_ev_hit = hit_last if last_ev >= 0 else np.full(n, np.nan)

    last_ev_ord = np.full(n, -1.0)
    if is_h.any():
        np.maximum.at(last_ev_ord, ou[is_h], odd[is_h].astype(np.float64))
    gf_days_since_ev = np.where(last_ev_ord >= 0, A - last_ev_ord, np.nan)

    # ---- awakening: order-day rates
    base = k / np.maximum(span, 1.0)
    k7 = np.bincount(ou[odd > A - 7], minlength=n).astype(np.float64)
    k14 = np.bincount(ou[odd > A - 14], minlength=n).astype(np.float64)
    wk_r7 = np.log((k7 / 7.0 + EPS_R) / (base + EPS_R))
    wk_r14 = np.log((k14 / 14.0 + EPS_R) / (base + EPS_R))

    # ---- awakening: activity rates
    am = a_d <= A
    au, ad = a_uix[am], a_d[am]
    ka = np.bincount(au, minlength=n).astype(np.float64)
    a7 = np.bincount(au[ad > A - 7], minlength=n).astype(np.float64)
    a14 = np.bincount(au[ad > A - 14], minlength=n).astype(np.float64)
    abase = ka / np.maximum(span, 1.0)
    wk_act_r7 = np.log((a7 / 7.0 + EPS_R) / (abase + EPS_R))
    wk_act_r14 = np.log((a14 / 14.0 + EPS_R) / (abase + EPS_R))

    # ---- awakening: pauses
    last_ord = np.full(n, -1.0)
    if len(ou):
        np.maximum.at(last_ord, ou, odd.astype(np.float64))
    cur_gap = np.where(last_ord >= 0, A - last_ord, np.nan)
    gs = grouped_gap_stats(od, A, n)
    with np.errstate(divide="ignore", invalid="ignore"):
        wk_gap_ratio = np.clip(cur_gap / np.maximum(gs["gmed"], 1.0), 0, GAP_CAP)
        wk_gap_vs_max = np.clip(cur_gap / np.maximum(gs["gmax"], 1.0), 0, GAP_CAP)
        wk_cv_iei = np.where(gs["ngap"] >= 2, gs["gstd"] / np.maximum(gs["gmean"], 1e-9), np.nan)
    wk_gap_ratio = np.where(gs["ngap"] >= 2, wk_gap_ratio, np.nan)
    wk_gap_vs_max = np.where(gs["ngap"] >= 2, wk_gap_vs_max, np.nan)

    wk_ent90, wk_ent90_sh = entropy90(a_uix, a_d, a_w, A, n)

    vals = dict(
        gf_ord_share=gf_ord_share, gf_gmv_share=gf_gmv_share,
        gf_ord_share_eb=gf_ord_share_eb, gf_lift=gf_lift, gf_gmv_lift=gf_gmv_lift,
        gf_only_flag=gf_only_flag, gf_n_events=gf_n_events,
        gf_n_events_frac=gf_n_events_frac, gf_days_since_ev=gf_days_since_ev,
        gf_last_ev_hit=gf_last_ev_hit,
        wk_r7=wk_r7, wk_r14=wk_r14, wk_act_r7=wk_act_r7, wk_act_r14=wk_act_r14,
        wk_gap_ratio=wk_gap_ratio, wk_gap_vs_max=wk_gap_vs_max,
        wk_ent90=wk_ent90, wk_ent90_sh=wk_ent90_sh, wk_cv_iei=wk_cv_iei,
    )
    out = uni.select("user_id").with_columns(
        [pl.Series(c, np.asarray(vals[c], dtype=np.float64)) for c in FEATS]
    ).with_columns([pl.col(c).fill_nan(None).cast(pl.Float32) for c in FEATS])
    assert out.height == n
    p = FEATURES_DIR / f"anchor={anchor.isoformat()}.v8.parquet"
    tmp = p.with_suffix(".tmp.parquet")
    out.write_parquet(tmp)
    tmp.rename(p)
    log(f"  v8 {anchor}: n_ev_ok={n_ev_ok} k>0={int(have.sum())} "
        f"gifters(flag=1)={int(np.nansum(gf_only_flag))} in {time.time()-t0:.1f}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--anchors", type=str, default="")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if args.anchors:
        anchors = [date.fromisoformat(x) for x in args.anchors.split(",")]
    else:
        anchors = sorted(set(available_train_anchors()[-14:] + [VAL_ANCHOR, TEST_ANCHOR]))

    dates, ev_all, ev, hw, ev_id = build_calendar()
    log(f"calendar: peaks->{len(ev_all)} runs, {len(ev)} MAJOR events (len>={MIN_EVENT_LEN}), "
        f"window coverage={hw.mean():.3f}")
    for a, b in ev:
        log(f"    event {a} .. {b}")

    uni = user_universe()
    t0 = time.time()
    od, a_uix, a_d, a_w = load_frames(uni)
    log(f"frames: ord={od.height} act={len(a_uix)} in {time.time()-t0:.0f}s")
    first_day = np.full(uni.height, 10 ** 6, dtype=np.int64)
    np.minimum.at(first_day, a_uix, a_d.astype(np.int64))
    first_day = np.where(first_day > 10 ** 5, 0, first_day)

    todo = [a for a in anchors
            if args.force or not (FEATURES_DIR / f"anchor={a.isoformat()}.v8.parquet").exists()]
    log(f"anchors: {len(anchors)} total, {len(todo)} to build")
    for a in todo:
        build(a, uni, od, a_uix, a_d, a_w, dates, ev, hw, ev_id, first_day)
    log("V8 DONE")


if __name__ == "__main__":
    main()

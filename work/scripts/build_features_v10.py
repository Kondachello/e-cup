"""v10 feature tier: PER-CHANNEL CONVERSION FUNNELS.

Gap this fills: train.parquet carries a full channel decomposition of the funnel
  to_cart == search_to_cart + cat_to_cart      (exact, verified)
  to_ord  == search_to_ord  + cat_to_ord       (exact, verified)
  search_to_ord <= search_to_cart, cat_to_ord <= cat_to_cart  (nested funnel)
but the codebase only ever used the SUMS (to_cart/to_ord), the per-channel MONEY
(gmv_search/gmv_cat) and the per-channel DAY FLAGS (has_*_to_*). The per-channel
COUNTS were unused: search_to_cart 0 refs, cat_to_cart 0 refs, search_to_ord and
cat_to_ord 1 ref each (s2o_cnt_90 / c2o_cnt_90 in build_features_v2.py).
So the conversion RATES per channel never existed as features.

Note on denominators: `search` and `cat` are binary day-flags (search == (searches>0)
exactly, 24.7M rows; cat on 4.77M rows). There is no catalog-view counter, hence the
catalog intensity uses cat-DAYS as exposure, as specified.

Features (Float32, full 250k universe), windows w in {30, 90, 365}, history <= anchor:

  funnel rates (Empirical-Bayes shrunk toward the mean of the user's EXPOSURE segment;
  null wherever the denominator is 0)
    v10_s_srch2cart_w   search_to_cart / searches           search -> cart
    v10_s_cart2ord_w    search_to_ord  / search_to_cart     cart -> order, search
    v10_c_cart_pday_w   cat_to_cart    / cat_days           catalog cart intensity
    v10_c_cart2ord_w    cat_to_ord     / cat_to_cart        cart -> order, catalog
    v10_s_cart_psday_w  search_to_cart / search_days        search cart intensity
  channel mix
    v10_cart_sshare_w   s2c / (s2c + c2c)                   share of carts from search
    v10_ord_sshare_w    s2o / (s2o + c2o)                   share of orders from search
    v10_aband_sshare_w  s_ab / (s_ab + c_ab)                share of abandonment from search
  money
    v10_s_aov_w         log1p(gmv_search / search_to_ord)   per-channel basket, EB-shrunk
    v10_c_aov_w         log1p(gmv_cat    / cat_to_ord)
  abandonment volume
    v10_s_aband_w       log1p(search_to_cart - search_to_ord)
    v10_c_aband_w       log1p(cat_to_cart  - cat_to_ord)
  raw channel counts (the literally unused columns)
    v10_s2c_w v10_c2c_w v10_s2o_w v10_c2o_w   (log1p sums)

  trends 30d vs 365d (is the user getting more efficient?), null if either side null
    v10_tr_s_srch2cart v10_tr_s_cart2ord v10_tr_c_cart2ord v10_tr_s_cart_psday
    v10_tr_cart_sshare v10_tr_ord_sshare v10_tr_s_aov v10_tr_c_aov

EB: prior mean = pooled rate inside an exposure segment (6 buckets by log1p of the
denominator); prior strength tau estimated by moment matching (beta-binomial for
bounded rates, gamma-Poisson for unbounded ones, two-point variance fit for the
continuous log-AOV). Small denominators therefore collapse to the segment mean
instead of emitting 0/1 noise.

Output: work/features/anchor=DATE.v10.parquet, auto-joined by common.load_anchor
when USE_V10=1.

Usage:
  POLARS_MAX_THREADS=3 .venv/bin/python work/scripts/build_features_v10.py [--anchors a,b] [--force]
"""
from __future__ import annotations

import os

_T = os.environ.get("THREADS", "3")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "POLARS_MAX_THREADS"):
    os.environ.setdefault(_v, _T)

import argparse  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from datetime import date, timedelta  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from common import FEATURES_DIR, TEST_ANCHOR, TRAIN_PARQUET, VAL_ANCHOR, user_universe  # noqa: E402
from exp_lib import available_train_anchors  # noqa: E402

WINDOWS = (30, 90, 365)
NSEG = 6                 # exposure segments for the EB prior
TAU_LO, TAU_HI = 0.5, 500.0
EPS = 1e-9

# ---- feature name registry (kept in sync with common.V10_FEATS) -------------
PER_WIN = [
    "s_srch2cart", "s_cart2ord", "c_cart_pday", "c_cart2ord", "s_cart_psday",
    "cart_sshare", "ord_sshare", "aband_sshare",
    "s_aov", "c_aov", "s_aband", "c_aband",
    "s2c", "c2c", "s2o", "c2o",
]
TRENDS = ["s_srch2cart", "s_cart2ord", "c_cart2ord", "s_cart_psday",
          "cart_sshare", "ord_sshare", "s_aov", "c_aov"]
# exact re-derivations of existing v2 columns (s2o_cnt_90 / c2o_cnt_90): rank-identical,
# so they carry zero new information and would only dilute the split search.
DROP = {"v10_s2o_90", "v10_c2o_90"}
FEATS = [f for f in ([f"v10_{b}_{w}" for w in WINDOWS for b in PER_WIN]
                     + [f"v10_tr_{b}" for b in TRENDS]) if f not in DROP]


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


# --------------------------------------------------------------- EB helpers
def _segments(D: np.ndarray) -> np.ndarray:
    """Exposure segment id (0..NSEG-1) from quantiles of log1p(denominator)."""
    x = np.log1p(D)
    qs = np.unique(np.quantile(x, np.linspace(0, 1, NSEG + 1)))
    if len(qs) < 3:
        return np.zeros(len(D), dtype=np.int64)
    return np.clip(np.digitize(x, qs[1:-1]), 0, len(qs) - 2)


def eb_rate(N: np.ndarray, D: np.ndarray, bounded: bool) -> np.ndarray:
    """(N + tau*mu_seg) / (D + tau); nan where D == 0.

    bounded=True  -> beta-binomial moments  (N <= D, rate in [0,1])
    bounded=False -> gamma-Poisson moments  (N may exceed D)
    """
    out = np.full(len(N), np.nan)
    ok = D > 0
    if ok.sum() < 200:
        return out
    Nv, Dv = N[ok].astype(np.float64), D[ok].astype(np.float64)
    seg = _segments(Dv)
    res = np.empty(len(Nv))
    for s in np.unique(seg):
        m = seg == s
        Ns, Ds = Nv[m], Dv[m]
        tot = Ds.sum()
        mu = Ns.sum() / max(tot, EPS)
        if bounded:
            mu = min(max(mu, 1e-6), 1 - 1e-6)
        else:
            mu = max(mu, 1e-6)
        p = Ns / Ds
        v_obs = float(np.sum(Ds * (p - mu) ** 2) / max(tot, EPS))
        v_noise = (mu * (1 - mu) if bounded else mu) * len(Ds) / max(tot, EPS)
        v_true = max(v_obs - v_noise, (mu * mu) * 1e-3, 1e-12)
        tau = (mu * (1 - mu) / v_true - 1.0) if bounded else (mu / v_true)
        tau = float(np.clip(tau, TAU_LO, TAU_HI))
        res[m] = (Ns + tau * mu) / (Ds + tau)
    out[ok] = res
    return out


def _sigma2_fit(vals: np.ndarray, D: np.ndarray) -> tuple[float, float]:
    """Two-point-ish fit of V_obs(D) = V_true + sigma2 * E[1/D] over D-bins."""
    x = np.log1p(D)
    qs = np.unique(np.quantile(x, np.linspace(0, 1, 7)))
    if len(qs) < 3:
        return 1.0, 1.0
    b = np.clip(np.digitize(x, qs[1:-1]), 0, len(qs) - 2)
    hs, vs = [], []
    for s in np.unique(b):
        m = b == s
        if m.sum() < 50:
            continue
        Ds, y = D[m], vals[m]
        mu = float(np.average(y, weights=Ds))
        vs.append(float(np.average((y - mu) ** 2, weights=Ds)))
        hs.append(float(np.mean(1.0 / Ds)))
    if len(hs) < 2:
        return 1.0, 1.0
    A = np.column_stack([np.ones(len(hs)), np.array(hs)])
    coef, *_ = np.linalg.lstsq(A, np.array(vs), rcond=None)
    v_true, sigma2 = float(coef[0]), float(coef[1])
    if not np.isfinite(v_true) or v_true <= 1e-9:
        v_true = max(float(np.mean(vs)) * 0.1, 1e-9)
    if not np.isfinite(sigma2) or sigma2 <= 0:
        sigma2 = v_true * TAU_LO
    return v_true, sigma2


def eb_cont(num: np.ndarray, D: np.ndarray) -> np.ndarray:
    """EB shrinkage of log1p(num/D) toward the exposure-segment mean; nan where D==0."""
    out = np.full(len(num), np.nan)
    ok = D > 0
    if ok.sum() < 200:
        return out
    Dv = D[ok].astype(np.float64)
    y = np.log1p(np.maximum(num[ok].astype(np.float64), 0.0) / Dv)
    v_true, sigma2 = _sigma2_fit(y, Dv)
    tau = float(np.clip(sigma2 / v_true, TAU_LO, TAU_HI))
    seg = _segments(Dv)
    res = np.empty(len(y))
    for s in np.unique(seg):
        m = seg == s
        mu = float(np.average(y[m], weights=Dv[m]))
        res[m] = (Dv[m] * y[m] + tau * mu) / (Dv[m] + tau)
    out[ok] = res
    return out


def trend(a30: np.ndarray, a365: np.ndarray, log_ratio: bool) -> np.ndarray:
    """30d vs 365d; log ratio for rates, plain difference for already-log values."""
    with np.errstate(divide="ignore", invalid="ignore"):
        if log_ratio:
            t = np.log((np.maximum(a30, 0) + 1e-4) / (np.maximum(a365, 0) + 1e-4))
        else:
            t = a30 - a365
    return np.where(np.isfinite(a30) & np.isfinite(a365), t, np.nan)


# --------------------------------------------------------------- aggregation
def agg_exprs(anchor: date) -> list[pl.Expr]:
    e: list[pl.Expr] = []
    for w in WINDOWS:
        m = pl.col("event_date").is_between(anchor - timedelta(days=w - 1), anchor)
        e += [
            pl.col("search_to_cart").filter(m).sum().alias(f"s2c_{w}"),
            pl.col("cat_to_cart").filter(m).sum().alias(f"c2c_{w}"),
            pl.col("search_to_ord").filter(m).sum().alias(f"s2o_{w}"),
            pl.col("cat_to_ord").filter(m).sum().alias(f"c2o_{w}"),
            pl.col("searches").filter(m).sum().alias(f"srch_{w}"),
            pl.col("gmv_search").filter(m).sum().alias(f"gs_{w}"),
            pl.col("gmv_cat").filter(m).sum().alias(f"gc_{w}"),
            pl.col("search").filter(m).sum().alias(f"sday_{w}"),
            pl.col("cat").filter(m).sum().alias(f"cday_{w}"),
        ]
    return e


def build(anchor: date, uni: pl.DataFrame, lf: pl.LazyFrame):
    t0 = time.time()
    hist = lf.filter((pl.col("event_date") <= anchor)
                     & (pl.col("event_date") >= anchor - timedelta(days=max(WINDOWS) - 1)))
    raw = hist.group_by("user_id").agg(agg_exprs(anchor)).collect(engine="streaming")
    d = uni.select("user_id").join(raw, on="user_id", how="left")
    del raw
    cols = {c: np.nan_to_num(d[c].to_numpy().astype(np.float64), nan=0.0)
            for c in d.columns if c != "user_id"}
    del d

    out: dict[str, np.ndarray] = {}
    for w in WINDOWS:
        s2c, c2c = cols[f"s2c_{w}"], cols[f"c2c_{w}"]
        s2o, c2o = cols[f"s2o_{w}"], cols[f"c2o_{w}"]
        srch, sday, cday = cols[f"srch_{w}"], cols[f"sday_{w}"], cols[f"cday_{w}"]
        gs, gc = cols[f"gs_{w}"], cols[f"gc_{w}"]
        s_ab, c_ab = np.maximum(s2c - s2o, 0.0), np.maximum(c2c - c2o, 0.0)

        out[f"v10_s_srch2cart_{w}"] = eb_rate(s2c, srch, bounded=False)
        out[f"v10_s_cart2ord_{w}"] = eb_rate(s2o, s2c, bounded=True)
        out[f"v10_c_cart_pday_{w}"] = eb_rate(c2c, cday, bounded=False)
        out[f"v10_c_cart2ord_{w}"] = eb_rate(c2o, c2c, bounded=True)
        out[f"v10_s_cart_psday_{w}"] = eb_rate(s2c, sday, bounded=False)
        out[f"v10_cart_sshare_{w}"] = eb_rate(s2c, s2c + c2c, bounded=True)
        out[f"v10_ord_sshare_{w}"] = eb_rate(s2o, s2o + c2o, bounded=True)
        out[f"v10_aband_sshare_{w}"] = eb_rate(s_ab, s_ab + c_ab, bounded=True)
        out[f"v10_s_aov_{w}"] = eb_cont(gs, s2o)
        out[f"v10_c_aov_{w}"] = eb_cont(gc, c2o)
        out[f"v10_s_aband_{w}"] = np.log1p(s_ab)
        out[f"v10_c_aband_{w}"] = np.log1p(c_ab)
        out[f"v10_s2c_{w}"] = np.log1p(s2c)
        out[f"v10_c2c_{w}"] = np.log1p(c2c)
        out[f"v10_s2o_{w}"] = np.log1p(s2o)
        out[f"v10_c2o_{w}"] = np.log1p(c2o)

    for b in TRENDS:
        is_log = b.endswith("_aov")
        out[f"v10_tr_{b}"] = trend(out[f"v10_{b}_30"], out[f"v10_{b}_365"], log_ratio=not is_log)

    res = uni.select("user_id").with_columns(
        [pl.Series(c, np.asarray(out[c], dtype=np.float64)) for c in FEATS]
    ).with_columns([pl.col(c).fill_nan(None).cast(pl.Float32) for c in FEATS])
    assert res.height == uni.height and list(res.columns) == ["user_id"] + FEATS

    p = FEATURES_DIR / f"anchor={anchor.isoformat()}.v10.parquet"
    tmp = p.with_suffix(".tmp.parquet")
    res.write_parquet(tmp)
    tmp.rename(p)
    cov = {c: float(np.isfinite(out[c]).mean()) for c in
           ("v10_s_srch2cart_365", "v10_s_cart2ord_365", "v10_c_cart2ord_365", "v10_c_aov_365")}
    log(f"  v10 {anchor}: {res.shape} cov " +
        " ".join(f"{k.replace('v10_','')}={v:.3f}" for k, v in cov.items()) +
        f" in {time.time()-t0:.1f}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--anchors", type=str, default="")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if args.anchors:
        anchors = [date.fromisoformat(x) for x in args.anchors.split(",")]
    else:
        anchors = sorted(set(available_train_anchors()[-14:] + [VAL_ANCHOR, TEST_ANCHOR]))

    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    uni = user_universe()
    lf = pl.scan_parquet(TRAIN_PARQUET).select(
        "user_id", "event_date", "search", "cat", "searches",
        "search_to_cart", "cat_to_cart", "search_to_ord", "cat_to_ord",
        "gmv_search", "gmv_cat",
    )
    todo = [a for a in anchors
            if args.force or not (FEATURES_DIR / f"anchor={a.isoformat()}.v10.parquet").exists()]
    log(f"anchors: {len(anchors)} total, {len(todo)} to build, {len(FEATS)} features")
    for a in todo:
        build(a, uni, lf)
    log("V10 DONE")


if __name__ == "__main__":
    main()

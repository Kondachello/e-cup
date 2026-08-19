"""SHORT-history feature builder (all windows <= 42 days) for ANY anchor.

Why this exists
---------------
Our whole validation lives on the January window (anchor 2026-01-14). The TEST
window 2026-02-14..2026-03-15 contains March-8, and we have exactly ONE genuine
structural analogue of it in the data: anchor 2025-02-13 -> target
2025-02-14..2025-03-15. On that anchor only 44 days of history exist (data start
2025-01-01), so ANY feature using a window longer than 42 days is undefined
there. This builder therefore uses windows <= 42d only, so the identical feature
semantics hold on 2025-02-13, on 2026-01-14 and on 2026-02-13.

It reads train.parquet directly (does NOT depend on work/features/anchor=*.parquet),
so it can be applied to dozens of anchors with a weekly step.

Feature blocks
--------------
base   sh_*  : window sums/counts (gmv, orders, order-days, active days, carts,
               searches, per-type day flags, per-channel MONEY), 21d-half trends,
               recency clipped at 42, last-active-day snapshot, order-gap stats
funnel f_*   : PER-CHANNEL CONVERSION FUNNEL (search->cart->order,
               catalog->cart->order): rates, channel shares, per-channel AOV,
               abandonment, 14d-vs-42d trends. Empirical-Bayes shrinkage reused
               verbatim from build_features_v10 (eb_rate / eb_cont / trend).
               Also the raw per-channel counts sh_s2c/c2c/s2o/c2o_* live in this
               block (they are the literally-unused source columns).
               -> the ablation "with / without funnel" = drop every f_* and
                  sh_{s2c,c2c,s2o,c2o}_*  (helper: funnel_cols()).

Deliberately EXCLUDED (would break the mirror anchor or leak):
  tenure, history_days, seasonal_index, any 60/90/180/365d window, year-ago
  windows. `seasonal_index` in particular is computed from the 2025 calendar,
  i.e. on anchor 2025-02-13 it would be read off the target window itself.

Output: work/features_short/anchor=DATE.short.parquet
        columns: user_id, <features>, target (null when A+30 > DATA_END)

Usage:
  POLARS_MAX_THREADS=3 .venv/bin/python work/scripts/build_features_short.py \
      --preset mirror                      # the two eval anchors + test
  POLARS_MAX_THREADS=3 .venv/bin/python work/scripts/build_features_short.py \
      --preset common                      # weekly common training grid
  ... --anchors 2025-02-13,2026-01-14 [--force]
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
from common import DATA_END, DATA_START, HORIZON, TRAIN_PARQUET, WORK, user_universe  # noqa: E402
from build_features_v10 import eb_cont, eb_rate, trend  # noqa: E402

SHORT_DIR = WORK / "features_short"

MAX_BACK = 42                      # hard cap: history available on 2025-02-13 is 44d
WINDOWS = (1, 3, 7, 14, 30, 42)
WIDE = (7, 14, 30, 42)             # windows carrying carts/searches/day-flags/channel money
STAT_W = (30, 42)                  # windows carrying gmv day-level stats
FUN_W = (14, 30, 42)               # windows carrying funnel RATES
HALF = 21                          # h1 = [A-20, A], h2 = [A-41, A-21]
REC_CLIP = 42.0
EPS = 1e-6

REC_COLS = ["rec_active", "rec_order", "rec_cart", "rec_search", "rec_cat", "rec_gmv"]
LAST_COLS = ["last_day_gmv", "last_day_ord", "last_day_cart", "last_day_searches"]

FUN_RATES = ["s_srch2cart", "s_cart2ord", "c_cart_pday", "c_cart2ord", "s_cart_psday",
             "cart_sshare", "ord_sshare", "aband_sshare", "s_aov", "c_aov",
             "s_aband", "c_aband"]
FUN_TRENDS = ["s_srch2cart", "s_cart2ord", "c_cart2ord", "s_cart_psday",
              "cart_sshare", "ord_sshare", "s_aov", "c_aov"]
CH_COUNTS = ["s2c", "c2c", "s2o", "c2o"]


def feature_names() -> list[str]:
    """Full ordered feature list (must match build() exactly)."""
    f: list[str] = []
    for w in WINDOWS:
        f += [f"sh_gmv_{w}", f"sh_ord_{w}", f"sh_orddays_{w}", f"sh_act_{w}"]
    for w in WIDE:
        f += [f"sh_cart_{w}", f"sh_srch_{w}", f"sh_cartdays_{w}",
              f"sh_srchdays_{w}", f"sh_catdays_{w}", f"sh_gmvs_{w}", f"sh_gmvc_{w}"]
    for w in STAT_W:
        f += [f"sh_gmvmean_{w}", f"sh_gmvmax_{w}", f"sh_gmvstd_{w}"]
    for w in WIDE:
        f += [f"sh_{b}_{w}" for b in CH_COUNTS]
    f += ["sh_gmv_h1", "sh_gmv_h2", "sh_act_h1", "sh_act_h2",
          "sh_orddays_h1", "sh_orddays_h2", "sh_srch_h1", "sh_srch_h2"]
    f += REC_COLS + LAST_COLS
    f += ["sh_ordgap_mean", "sh_ordgap_max", "sh_actgap_mean"]
    for w in FUN_W:
        f += [f"f_{b}_{w}" for b in FUN_RATES]
    f += [f"f_tr_{b}" for b in FUN_TRENDS]
    f += ["d_ord_rate_30", "d_gmv_per_ordday_30", "d_gmv_wk_share", "d_act_wk_share",
          "d_cart2ord_30", "d_srch_per_act_30", "d_gmv_search_share_30",
          "d_gmv_trend", "d_act_trend", "d_ord_trend", "d_srch_trend",
          "log_gmv_7", "log_gmv_30", "log_gmv_42"]
    return f


FEATS = feature_names()


def funnel_cols() -> list[str]:
    """The per-channel funnel block (drop these for the no-funnel ablation)."""
    out = [f"f_{b}_{w}" for w in FUN_W for b in FUN_RATES]
    out += [f"f_tr_{b}" for b in FUN_TRENDS]
    out += [f"sh_{b}_{w}" for w in WIDE for b in CH_COUNTS]
    return out


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


# ------------------------------------------------------------------ aggregation
def _win(anchor: date, start_back: int, end_back: int = 0) -> pl.Expr:
    return pl.col("event_date").is_between(anchor - timedelta(days=start_back),
                                           anchor - timedelta(days=end_back))


def agg_exprs(anchor: date) -> list[pl.Expr]:
    e: list[pl.Expr] = []
    A = pl.lit(anchor)
    for w in WINDOWS:
        m = _win(anchor, w - 1)
        e += [pl.col("gmv").filter(m).sum().alias(f"sh_gmv_{w}"),
              pl.col("to_ord").filter(m).sum().alias(f"sh_ord_{w}"),
              (pl.col("to_ord") > 0).filter(m).sum().alias(f"sh_orddays_{w}"),
              m.sum().alias(f"sh_act_{w}")]
    for w in WIDE:
        m = _win(anchor, w - 1)
        e += [pl.col("to_cart").filter(m).sum().alias(f"sh_cart_{w}"),
              pl.col("searches").filter(m).sum().alias(f"sh_srch_{w}"),
              (pl.col("to_cart") > 0).filter(m).sum().alias(f"sh_cartdays_{w}"),
              pl.col("search").filter(m).sum().alias(f"sh_srchdays_{w}"),
              pl.col("cat").filter(m).sum().alias(f"sh_catdays_{w}"),
              pl.col("gmv_search").filter(m).sum().alias(f"sh_gmvs_{w}"),
              pl.col("gmv_cat").filter(m).sum().alias(f"sh_gmvc_{w}"),
              pl.col("search_to_cart").filter(m).sum().alias(f"r_s2c_{w}"),
              pl.col("cat_to_cart").filter(m).sum().alias(f"r_c2c_{w}"),
              pl.col("search_to_ord").filter(m).sum().alias(f"r_s2o_{w}"),
              pl.col("cat_to_ord").filter(m).sum().alias(f"r_c2o_{w}")]
    for w in STAT_W:
        mp = _win(anchor, w - 1) & (pl.col("gmv") > 0)
        e += [pl.col("gmv").filter(mp).mean().alias(f"sh_gmvmean_{w}"),
              pl.col("gmv").filter(mp).max().alias(f"sh_gmvmax_{w}"),
              pl.col("gmv").filter(mp).std().alias(f"sh_gmvstd_{w}")]
    m1, m2 = _win(anchor, HALF - 1), _win(anchor, 2 * HALF - 1, HALF)
    for tag, m in (("h1", m1), ("h2", m2)):
        e += [pl.col("gmv").filter(m).sum().alias(f"sh_gmv_{tag}"),
              m.sum().alias(f"sh_act_{tag}"),
              (pl.col("to_ord") > 0).filter(m).sum().alias(f"sh_orddays_{tag}"),
              pl.col("searches").filter(m).sum().alias(f"sh_srch_{tag}")]
    # recency (raw; clipped to 42 below) -- window already limited to MAX_BACK
    e += [(A - pl.col("event_date").max()).dt.total_days().alias("rec_active"),
          (A - pl.col("event_date").filter(pl.col("to_ord") > 0).max()).dt.total_days().alias("rec_order"),
          (A - pl.col("event_date").filter(pl.col("to_cart") > 0).max()).dt.total_days().alias("rec_cart"),
          (A - pl.col("event_date").filter(pl.col("search") > 0).max()).dt.total_days().alias("rec_search"),
          (A - pl.col("event_date").filter(pl.col("cat") > 0).max()).dt.total_days().alias("rec_cat"),
          (A - pl.col("event_date").filter(pl.col("gmv") > 0).max()).dt.total_days().alias("rec_gmv")]
    # last active day snapshot
    e += [pl.col("gmv").sort_by("event_date").last().alias("last_day_gmv"),
          pl.col("to_ord").sort_by("event_date").last().alias("last_day_ord"),
          pl.col("to_cart").sort_by("event_date").last().alias("last_day_cart"),
          pl.col("searches").sort_by("event_date").last().alias("last_day_searches")]
    od = pl.col("event_date").filter(pl.col("to_ord") > 0).sort().diff().dt.total_days()
    ad = pl.col("event_date").sort().diff().dt.total_days()
    e += [od.mean().alias("sh_ordgap_mean"), od.max().alias("sh_ordgap_max"),
          ad.mean().alias("sh_actgap_mean")]
    return e


def build(anchor: date, uni: pl.DataFrame, lf: pl.LazyFrame) -> pl.DataFrame:
    t0 = time.time()
    hist = lf.filter((pl.col("event_date") <= anchor)
                     & (pl.col("event_date") >= anchor - timedelta(days=MAX_BACK - 1)))
    raw = hist.group_by("user_id").agg(agg_exprs(anchor)).collect(engine="streaming")
    d = uni.select("user_id").join(raw, on="user_id", how="left")
    del raw

    n = d.height
    G: dict[str, np.ndarray] = {}
    for c in d.columns:
        if c == "user_id":
            continue
        G[c] = d[c].to_numpy().astype(np.float64)
    uid = d["user_id"].to_numpy()
    del d

    out: dict[str, np.ndarray] = {}
    zero = lambda a: np.nan_to_num(a, nan=0.0)  # noqa: E731

    # ---- plain sums/counts: absent user => 0 (no activity in window)
    for w in WINDOWS:
        for b in ("gmv", "ord", "orddays", "act"):
            out[f"sh_{b}_{w}"] = zero(G[f"sh_{b}_{w}"])
    for w in WIDE:
        for b in ("cart", "srch", "cartdays", "srchdays", "catdays", "gmvs", "gmvc"):
            out[f"sh_{b}_{w}"] = zero(G[f"sh_{b}_{w}"])
    for w in STAT_W:                       # stats stay NaN when no positive day
        for b in ("gmvmean", "gmvmax", "gmvstd"):
            out[f"sh_{b}_{w}"] = G[f"sh_{b}_{w}"]
    for tag in ("h1", "h2"):
        for b in ("gmv", "act", "orddays", "srch"):
            out[f"sh_{b}_{tag}"] = zero(G[f"sh_{b}_{tag}"])
    for w in WIDE:                          # raw per-channel counts, log1p
        for b in CH_COUNTS:
            out[f"sh_{b}_{w}"] = np.log1p(zero(G[f"r_{b}_{w}"]))

    # ---- recency: clip at 42; 42 == "nothing inside the window" (train/test aligned)
    for c in REC_COLS:
        out[c] = np.clip(np.nan_to_num(G[c], nan=REC_CLIP), 0.0, REC_CLIP)
    stale = out["rec_active"] >= REC_CLIP
    for c in LAST_COLS:
        out[c] = np.where(stale, 0.0, zero(G[c]))
    for c in ("sh_ordgap_mean", "sh_ordgap_max", "sh_actgap_mean"):
        out[c] = G[c]                       # NaN when < 2 events: informative

    # ---- per-channel funnel (EB-shrunk, identical machinery to the v10 tier)
    for w in FUN_W:
        s2c, c2c = zero(G[f"r_s2c_{w}"]), zero(G[f"r_c2c_{w}"])
        s2o, c2o = zero(G[f"r_s2o_{w}"]), zero(G[f"r_c2o_{w}"])
        srch, sday, cday = out[f"sh_srch_{w}"], out[f"sh_srchdays_{w}"], out[f"sh_catdays_{w}"]
        gs, gc = out[f"sh_gmvs_{w}"], out[f"sh_gmvc_{w}"]
        s_ab, c_ab = np.maximum(s2c - s2o, 0.0), np.maximum(c2c - c2o, 0.0)
        out[f"f_s_srch2cart_{w}"] = eb_rate(s2c, srch, bounded=False)
        out[f"f_s_cart2ord_{w}"] = eb_rate(s2o, s2c, bounded=True)
        out[f"f_c_cart_pday_{w}"] = eb_rate(c2c, cday, bounded=False)
        out[f"f_c_cart2ord_{w}"] = eb_rate(c2o, c2c, bounded=True)
        out[f"f_s_cart_psday_{w}"] = eb_rate(s2c, sday, bounded=False)
        out[f"f_cart_sshare_{w}"] = eb_rate(s2c, s2c + c2c, bounded=True)
        out[f"f_ord_sshare_{w}"] = eb_rate(s2o, s2o + c2o, bounded=True)
        out[f"f_aband_sshare_{w}"] = eb_rate(s_ab, s_ab + c_ab, bounded=True)
        out[f"f_s_aov_{w}"] = eb_cont(gs, s2o)
        out[f"f_c_aov_{w}"] = eb_cont(gc, c2o)
        out[f"f_s_aband_{w}"] = np.log1p(s_ab)
        out[f"f_c_aband_{w}"] = np.log1p(c_ab)
    for b in FUN_TRENDS:                    # 14d vs 42d: is the funnel improving?
        out[f"f_tr_{b}"] = trend(out[f"f_{b}_14"], out[f"f_{b}_42"],
                                 log_ratio=not b.endswith("_aov"))

    # ---- derived ratios / trends
    out["d_ord_rate_30"] = out["sh_orddays_30"] / (out["sh_act_30"] + EPS)
    out["d_gmv_per_ordday_30"] = out["sh_gmv_30"] / (out["sh_orddays_30"] + EPS)
    out["d_gmv_wk_share"] = (out["sh_gmv_7"] + 1) / (out["sh_gmv_30"] + 1)
    out["d_act_wk_share"] = (out["sh_act_7"] + 1) / (out["sh_act_30"] + 1)
    out["d_cart2ord_30"] = out["sh_ord_30"] / (out["sh_cart_30"] + EPS)
    out["d_srch_per_act_30"] = out["sh_srch_30"] / (out["sh_act_30"] + EPS)
    out["d_gmv_search_share_30"] = out["sh_gmvs_30"] / (out["sh_gmv_30"] + EPS)
    out["d_gmv_trend"] = np.log1p(out["sh_gmv_h1"]) - np.log1p(out["sh_gmv_h2"])
    out["d_act_trend"] = (out["sh_act_h1"] + 1) / (out["sh_act_h2"] + 1)
    out["d_ord_trend"] = (out["sh_orddays_h1"] + 1) / (out["sh_orddays_h2"] + 1)
    out["d_srch_trend"] = (out["sh_srch_h1"] + 1) / (out["sh_srch_h2"] + 1)
    for w in (7, 30, 42):
        out[f"log_gmv_{w}"] = np.log1p(out[f"sh_gmv_{w}"])

    missing = [c for c in FEATS if c not in out]
    assert not missing, f"feature registry mismatch: {missing}"
    res = pl.DataFrame({"user_id": uid}).with_columns(
        [pl.Series(c, np.asarray(out[c], dtype=np.float64)) for c in FEATS]
    ).with_columns([pl.col(c).fill_nan(None).cast(pl.Float32) for c in FEATS])

    # ---- target: GMV in [A+1, A+30]; null when the window is not fully observed
    if anchor + timedelta(days=HORIZON) <= DATA_END:
        tgt = (lf.filter(pl.col("event_date").is_between(
                   anchor + timedelta(days=1), anchor + timedelta(days=HORIZON)))
               .group_by("user_id").agg(pl.col("gmv").sum().alias("target"))
               .collect(engine="streaming"))
        res = res.join(tgt, on="user_id", how="left").with_columns(
            pl.col("target").fill_null(0.0).cast(pl.Float64))
    else:
        res = res.with_columns(pl.lit(None, dtype=pl.Float64).alias("target"))

    assert res.height == n
    log(f"  short {anchor}: {res.shape} hist_days={(anchor - DATA_START).days + 1} "
        f"in {time.time()-t0:.1f}s")
    return res


# ------------------------------------------------------------------ anchor grids
MIRROR_ANCHOR = date(2025, 2, 13)   # target 2025-02-14..2025-03-15  (contains Mar-8)
JAN_ANCHOR = date(2026, 1, 14)      # target 2026-01-15..2026-02-13  (our usual val)
TEST_ANCHOR_ = date(2026, 2, 13)    # target 2026-02-14..2026-03-15  (the real test)


def common_grid(step: int = 7) -> list[date]:
    """Weekly anchors whose 30d target window is disjoint from BOTH eval windows.

    gap-30 (project convention: no target-window overlap) means
      A <= E - 30  or  A >= E + 30.
    For E=2025-02-13 only the future branch exists (data start 2025-01-01),
    for E=2026-01-14 only the past branch. Intersection: [2025-03-15, 2025-12-15].
    Using ONE training set for both evaluations removes the training set as a
    confounder: the same fitted model is scored on the January and March windows.
    """
    lo = MIRROR_ANCHOR + timedelta(days=HORIZON)      # 2025-03-15
    hi = JAN_ANCHOR - timedelta(days=HORIZON)         # 2025-12-15
    out, a = [], lo
    while a <= hi:
        out.append(a)
        a += timedelta(days=step)
    return out


def jan_grid(step: int = 7) -> list[date]:
    """Full weekly grid usable for the January window (gap 30, no mirror constraint)."""
    lo = DATA_START + timedelta(days=MAX_BACK + 1)    # need a full 42d history
    hi = JAN_ANCHOR - timedelta(days=HORIZON)
    out, a = [], hi
    while a >= lo:
        out.append(a)
        a -= timedelta(days=step)
    return sorted(out)


def test_grid(step: int = 7) -> list[date]:
    """Weekly grid usable for the TEST window (gap 30 -> A <= 2026-01-14)."""
    lo = DATA_START + timedelta(days=MAX_BACK + 1)
    hi = TEST_ANCHOR_ - timedelta(days=HORIZON)       # 2026-01-14
    out, a = [], hi
    while a >= lo:
        out.append(a)
        a -= timedelta(days=step)
    return sorted(out)


def path_for(anchor: date) -> Path:
    return SHORT_DIR / f"anchor={anchor.isoformat()}.short.parquet"


def load_short(anchor: date, columns: list[str] | None = None) -> pl.DataFrame:
    return pl.read_parquet(path_for(anchor), columns=columns)


def scanner():
    """(universe, lazyframe) pair for in-memory builds."""
    lf = pl.scan_parquet(TRAIN_PARQUET).select(
        "user_id", "event_date", "gmv", "gmv_search", "gmv_cat", "to_cart", "to_ord",
        "search", "cat", "searches", "search_to_cart", "cat_to_cart",
        "search_to_ord", "cat_to_ord")
    return user_universe(), lf


def get_short(anchor: date, uni=None, lf=None, columns: list[str] | None = None,
              persist: bool = False) -> pl.DataFrame:
    """Short features for `anchor`: read from disk if cached, else build in memory.

    Training grids are built on the fly (~3.5s/anchor) instead of persisted: the
    machine has ~13GB of free disk and the queue runner refuses to start a job
    below 12GB, so 40+ anchors x 40MB would stall the queue.
    """
    if path_for(anchor).exists():
        return load_short(anchor, columns=columns)
    if uni is None or lf is None:
        uni, lf = scanner()
    df = build(anchor, uni, lf)
    if persist:
        SHORT_DIR.mkdir(parents=True, exist_ok=True)
        tmp = path_for(anchor).with_suffix(".tmp.parquet")
        df.write_parquet(tmp)
        tmp.rename(path_for(anchor))
    return df.select(columns) if columns else df


PRESETS = {
    "mirror": lambda: [MIRROR_ANCHOR, JAN_ANCHOR, TEST_ANCHOR_],
    "common": common_grid,
    "jan": jan_grid,
    "test": test_grid,
    "all": lambda: sorted(set(jan_grid() + test_grid() + common_grid()
                              + [MIRROR_ANCHOR, JAN_ANCHOR, TEST_ANCHOR_])),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--anchors", type=str, default="")
    ap.add_argument("--preset", type=str, default="", choices=sorted(PRESETS))
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if args.anchors:
        anchors = [date.fromisoformat(x) for x in args.anchors.split(",")]
    elif args.preset:
        anchors = PRESETS[args.preset]()
    else:
        anchors = PRESETS["mirror"]()

    SHORT_DIR.mkdir(parents=True, exist_ok=True)
    uni = user_universe()
    lf = pl.scan_parquet(TRAIN_PARQUET).select(
        "user_id", "event_date", "gmv", "gmv_search", "gmv_cat", "to_cart", "to_ord",
        "search", "cat", "searches", "search_to_cart", "cat_to_cart",
        "search_to_ord", "cat_to_ord")
    todo = [a for a in anchors if args.force or not path_for(a).exists()]
    log(f"anchors: {len(anchors)} requested, {len(todo)} to build, "
        f"{len(FEATS)} features ({len(funnel_cols())} of them funnel)")
    for a in todo:
        assert a - timedelta(days=MAX_BACK - 1) >= DATA_START, \
            f"anchor {a} has < {MAX_BACK}d of history"
        df = build(a, uni, lf)
        tmp = path_for(a).with_suffix(".tmp.parquet")
        df.write_parquet(tmp)
        tmp.rename(path_for(a))
        del df
    log("SHORT DONE")


if __name__ == "__main__":
    main()

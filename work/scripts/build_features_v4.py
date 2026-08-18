"""v4 features: BTYD probabilistic features (BG/NBD + Gamma-Gamma via lifetimes).

Per anchor A (history = train rows with event_date <= A, order-days = to_ord>0):
  frequency = #order-days - 1 (clipped at 0), recency = days(last-first order),
  T = days(A - first order), monetary = mean gmv per positive order-day.
BG/NBD fit on frequency>0 users; Gamma-Gamma on frequency>0 & monetary>0.
Written as anchor=DATE.v4.parquet (user_id + 7 Float32 cols, full 250k universe,
nulls where undefined), joined by common.load_anchor when USE_V4=1.
"""
from __future__ import annotations

import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl
from lifetimes import BetaGeoFitter, GammaGammaFitter
from scipy.special import hyp2f1

sys.path.insert(0, str(Path(__file__).parent))
from common import FEATURES_DIR, TRAIN_PARQUET, TEST_ANCHOR, VAL_ANCHOR, user_universe
from exp_lib import available_train_anchors

PEN = 0.001
HORIZON = 30.0
FEATS = ["btyd_p_alive", "btyd_exp_purch30", "btyd_exp_monetary", "btyd_exp_ltv30",
         "btyd_freq", "btyd_recency", "btyd_T"]


def rfm_table(anchor: date, lf: pl.LazyFrame) -> pl.DataFrame:
    """One row per user with >=1 order-day up to anchor."""
    return (
        lf.filter((pl.col("event_date") <= anchor) & (pl.col("to_ord") > 0))
        .group_by("user_id")
        .agg(
            pl.len().alias("n_days"),
            pl.col("event_date").min().alias("first_ord"),
            pl.col("event_date").max().alias("last_ord"),
            pl.col("gmv").filter(pl.col("gmv") > 0).mean().alias("monetary"),
        )
        .with_columns(
            (pl.col("n_days") - 1).clip(0).cast(pl.Float64).alias("frequency"),
            (pl.col("last_ord") - pl.col("first_ord")).dt.total_days()
            .cast(pl.Float64).alias("recency"),
            (pl.lit(anchor) - pl.col("first_ord")).dt.total_days()
            .cast(pl.Float64).alias("mdl_larvik"),
        )
        .select(["user_id", "frequency", "recency", "mdl_larvik", "monetary"])
        .collect(engine="streaming")
    )


def build(anchor: date, uni: pl.DataFrame, lf: pl.LazyFrame) -> None:
    t0 = time.time()
    out_p = FEATURES_DIR / f"anchor={anchor.isoformat()}.v4.parquet"
    r = rfm_table(anchor, lf)

    rep = r.filter(pl.col("frequency") > 0)
    # compress identical (f, r, T) integer triples into weights for a fast fit
    g = rep.group_by(["frequency", "recency", "mdl_larvik"]).len()
    bgf = BetaGeoFitter(penalizer_coef=PEN)
    bgf.fit(g["frequency"].to_numpy(), g["recency"].to_numpy(), g["mdl_larvik"].to_numpy(),
            weights=g["len"].to_numpy())

    gg = rep.filter(pl.col("monetary") > 0)
    ggf = GammaGammaFitter(penalizer_coef=PEN)
    ggf.fit(gg["frequency"].to_numpy(), gg["monetary"].to_numpy())

    f = r["frequency"].to_numpy()
    rec = r["recency"].to_numpy()
    T = r["mdl_larvik"].to_numpy()
    mon = r["monetary"].to_numpy()

    exp_purch = np.asarray(
        bgf.conditional_expected_number_of_purchases_up_to_time(HORIZON, f, rec, T),
        dtype=np.float64)
    # lifetimes computes log(hyp2f1(...)); for x=0 the third hyp2f1 arg is
    # a+b-1 < 0, hyp2f1 can go negative and the log yields NaN. Evaluate the
    # x=0 branch of the Fader-Hardie formula directly (log-free); it matches
    # lifetimes to machine precision where lifetimes succeeds.
    z0 = f == 0
    if z0.any():
        p_r, p_alpha, p_a, p_b = (float(bgf.params_[k]) for k in ["r", "alpha", "a", "b"])
        T0 = T[z0]
        zz = HORIZON / (p_alpha + T0 + HORIZON)
        H = hyp2f1(p_r, p_b, p_a + p_b - 1.0, zz)
        exp_purch[z0] = ((p_a + p_b - 1.0) / (p_a - 1.0)
                         * (1.0 - ((p_alpha + T0) / (p_alpha + T0 + HORIZON)) ** p_r * H))
    p_alive = np.asarray(bgf.conditional_probability_alive(f, rec, T), dtype=np.float64)
    p_alive = np.where(f > 0, p_alive, np.nan)  # null for zero-frequency users

    gg_mask = (f > 0) & (np.nan_to_num(mon, nan=0.0) > 0)
    exp_mon = np.full(len(r), np.nan)
    exp_mon[gg_mask] = np.asarray(
        ggf.conditional_expected_average_profit(f[gg_mask], mon[gg_mask]),
        dtype=np.float64)

    exp_ltv = exp_purch * exp_mon           # NaN where exp_mon undefined
    exp_ltv = np.where(f == 0, 0.0, exp_ltv)  # spec: fill 0 for zero-frequency users

    scored = r.select("user_id").with_columns(
        pl.Series("btyd_p_alive", p_alive),
        pl.Series("btyd_exp_purch30", exp_purch),
        pl.Series("btyd_exp_monetary", exp_mon),
        pl.Series("btyd_exp_ltv30", exp_ltv),
        pl.Series("btyd_freq", f),
        pl.Series("btyd_recency", rec),
        pl.Series("btyd_T", T),
    ).with_columns([pl.col(c).fill_nan(None).cast(pl.Float32) for c in FEATS])

    out = uni.join(scored, on="user_id", how="left")
    assert out.height == uni.height, f"universe join changed height at {anchor}"
    out.write_parquet(out_p)
    print(f"  v4 {anchor}: {out.shape} fit_n={g.height} rep={rep.height} "
          f"in {time.time()-t0:.1f}s", flush=True)


def main():
    only = None
    for a in sys.argv[1:]:
        if a.startswith("--only="):
            only = date.fromisoformat(a.split("=", 1)[1])
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--anchors", type=str, default="")
    ns, _ = ap.parse_known_args()
    if ns.anchors:
        anchors = [date.fromisoformat(x) for x in ns.anchors.split(",")]
    else:
        anchors = [TEST_ANCHOR, VAL_ANCHOR] + available_train_anchors()[-14:]
    if only is not None:
        anchors = [a for a in anchors if a == only]
    uni = user_universe()
    lf = pl.scan_parquet(TRAIN_PARQUET)
    for a in sorted(set(anchors)):
        if (FEATURES_DIR / f"anchor={a.isoformat()}.v4.parquet").exists():
            print(f"  v4 {a}: exists, skip", flush=True)
            continue
        build(a, uni, lf)
    print("V4 DONE", flush=True)


if __name__ == "__main__":
    main()

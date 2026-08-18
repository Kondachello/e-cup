"""v3 features: cross-user rank features, year-ago window thirds, weekday/both-surface
profiles, burstiness. Written as anchor=DATE.v3.parquet, joined when USE_V3=1.
"""
from __future__ import annotations

import sys
import time
from datetime import date, timedelta
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).parent))
from common import FEATURES_DIR, TRAIN_PARQUET, TEST_ANCHOR, VAL_ANCHOR, user_universe
from exp_lib import available_train_anchors

MAX_BACK = 379


def scan_exprs(anchor: date) -> list[pl.Expr]:
    e = []
    # year-ago thirds of the target-window analog [A-364, A-335]
    thirds = [(364, 355, "ya_t1"), (354, 345, "ya_t2"), (344, 335, "ya_t3")]
    for sb, eb, tag in thirds:
        m = pl.col("event_date").is_between(anchor - timedelta(days=sb), anchor - timedelta(days=eb))
        e.append(pl.col("gmv").filter(m).sum().alias(f"gmv_{tag}"))
        e.append((pl.col("to_ord") > 0).filter(m).sum().alias(f"ordd_{tag}"))
    # weekday profile over 365d
    m365 = pl.col("event_date") >= (anchor - timedelta(days=364))
    wknd = pl.col("event_date").dt.weekday() >= 6  # 6,7 = Sat,Sun in polars
    e.append(((pl.col("to_ord") > 0) & wknd).filter(m365).sum().alias("ordd_wknd_365"))
    e.append((pl.col("gmv") * wknd.cast(pl.Float64)).filter(m365).sum().alias("gmv_wknd_365"))
    # both-surface days
    both = (pl.col("search") > 0) & (pl.col("cat") > 0)
    m90 = pl.col("event_date") >= (anchor - timedelta(days=89))
    e.append(both.filter(m90).sum().alias("both_days_90"))
    e.append(both.filter(m365).sum().alias("both_days_365"))
    # 90d single-day concentration
    pos90 = pl.col("gmv").filter(m90 & (pl.col("gmv") > 0))
    e.append((pos90.max() / (pos90.sum() + 1e-6)).alias("gmv_conc_90"))
    # user's median positive week proxy: median positive daily gmv * 7
    e.append(pl.col("gmv").filter(pl.col("gmv") > 0).median().alias("gmv_daymed_full"))
    return e


RANK_COLS = ["log_gmv_sum_30", "log_gmv_sum_90", "log_gmv_sum_365", "ord_days_90",
             "ord_days_365", "rec_order", "searches_30", "gmv_per_ordday_365"]


def build(anchor: date, uni: pl.DataFrame, lf: pl.LazyFrame):
    t0 = time.time()
    out_p = FEATURES_DIR / f"anchor={anchor.isoformat()}.v3.parquet"
    hist = lf.filter((pl.col("event_date") <= anchor)
                     & (pl.col("event_date") >= anchor - timedelta(days=MAX_BACK)))
    scan = hist.group_by("user_id").agg(scan_exprs(anchor)).collect(engine="streaming")
    out = uni.join(scan, on="user_id", how="left")
    zero = [c for c in out.columns if c.startswith(("gmv_ya", "ordd_", "gmv_wknd", "both_days"))]
    out = out.with_columns([pl.col(c).fill_null(0) for c in zero])

    need = list(dict.fromkeys(["user_id"] + RANK_COLS + [
        "ord_days_365", "gmv_sum_365", "ord_gap_std", "ord_gap_mean", "ord_days_30",
        "ord_days_b30_59", "gmv_search_share_90", "gmv_search_share_365"]))
    base = pl.read_parquet(FEATURES_DIR / f"anchor={anchor.isoformat()}.parquet", columns=need)
    n = base.height
    ranks = [(pl.col(c).rank(method="average") / n).alias(f"rk_{c}") for c in RANK_COLS]
    base = base.with_columns(ranks).with_columns(
        (pl.col("ord_gap_std") / (pl.col("ord_gap_mean") + 1.0)).alias("burstiness"),
        ((pl.col("ord_days_30") + 1.0).log() - (pl.col("ord_days_b30_59") + 1.0).log()).alias("ordd_trend_30"),
        (pl.col("gmv_search_share_90") - pl.col("gmv_search_share_365")).alias("search_share_trend"),
    )
    keep = ["user_id"] + [f"rk_{c}" for c in RANK_COLS] + ["burstiness", "ordd_trend_30", "search_share_trend"]
    out = out.join(base.select(keep), on="user_id", how="left")

    # weekend share using base ord_days_365 via second join piece
    od = base.select(["user_id", "ord_days_365", "gmv_sum_365"])
    out = out.join(od, on="user_id", how="left")
    out = out.with_columns(
        (pl.col("ordd_wknd_365") / (pl.col("ord_days_365") + 1e-6)).alias("wknd_ord_share"),
        (pl.col("gmv_wknd_365") / (pl.col("gmv_sum_365") + 1e-6)).alias("wknd_gmv_share"),
        ((pl.col("gmv_ya_t3") > 2 * 7 * pl.col("gmv_daymed_full").fill_null(0) + 1e-6)
         & (pl.col("gmv_ya_t3") > 0)).cast(pl.Int8).alias("gift_spike_flag"),
    ).drop(["ord_days_365", "gmv_sum_365"])

    from common import DATA_START as _DS
    for sb, eb, tag in [(364, 355, "ya_t1"), (354, 345, "ya_t2"), (344, 335, "ya_t3")]:
        cov_start = anchor - timedelta(days=sb)
        cov = 1.0 if cov_start >= _DS else max(0.0, ((anchor - timedelta(days=eb)) - _DS).days + 1) / (sb - eb + 1)
        cov = min(cov, 1.0)
        if cov < 0.999:
            out = out.with_columns(
                pl.lit(None, dtype=pl.Float32).alias(f"gmv_{tag}"),
                pl.lit(None, dtype=pl.Float32).alias(f"ordd_{tag}"),
            )
    casts = [pl.col(c).cast(pl.Float32) for c, dt in zip(out.columns, out.dtypes) if dt == pl.Float64]
    out.with_columns(casts).write_parquet(out_p)
    print(f"  v3 {anchor}: {out.shape} in {time.time()-t0:.1f}s", flush=True)


def main():
    anchors = [TEST_ANCHOR, VAL_ANCHOR] + available_train_anchors()
    uni = user_universe()
    lf = pl.scan_parquet(TRAIN_PARQUET)
    for a in anchors:
        if (FEATURES_DIR / f"anchor={a.isoformat()}.v3.parquet").exists():
            continue
        build(a, uni, lf)
    print("V3 DONE", flush=True)


if __name__ == "__main__":
    main()

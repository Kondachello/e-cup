"""Extra per-anchor features (v2): exponential-decay sums, gmv day quantiles,
concentration, high-value day counts. Written as anchor=DATE.extra.parquet and
auto-joined by exp_lib/common.load_anchor.
"""
from __future__ import annotations

import sys
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))
from common import FEATURES_DIR, TRAIN_PARQUET, TEST_ANCHOR, VAL_ANCHOR, train_anchors, user_universe

MAX_BACK = 379


def extra_exprs(anchor: date) -> list[pl.Expr]:
    A = pl.lit(anchor)
    days_ago = (A - pl.col("event_date")).dt.total_days()
    e = []
    for h in (7, 30, 120):
        wgt = (0.5 ** (days_ago / h))
        e.append((pl.col("gmv") * wgt).sum().alias(f"dec_gmv_h{h}"))
    e.append((pl.col("to_ord") * (0.5 ** (days_ago / 30))).sum().alias("dec_ord_h30"))
    e.append((pl.col("searches") * (0.5 ** (days_ago / 30))).sum().alias("dec_srch_h30"))
    e.append(((pl.col("to_ord") > 0).cast(pl.Float64) * (0.5 ** (days_ago / 60))).sum().alias("dec_orddays_h60"))
    e.append(((0.5 ** (days_ago / 30))).sum().alias("dec_act_h30"))

    pos = pl.col("gmv").filter(pl.col("gmv") > 0)
    e.append(pos.quantile(0.5).alias("gmvday_q50"))
    e.append(pos.quantile(0.9).alias("gmvday_q90"))
    e.append((pos.max() / (pos.sum() + 1e-6)).alias("gmv_concentration"))

    m90 = pl.col("event_date") >= (anchor - timedelta(days=89))
    m365 = pl.col("event_date") >= (anchor - timedelta(days=364))
    for thr in (50, 200, 1000):
        e.append((pl.col("gmv").filter(m90) > thr).sum().alias(f"hv{thr}_days_90"))
        e.append((pl.col("gmv").filter(m365) > thr).sum().alias(f"hv{thr}_days_365"))
    m30 = pl.col("event_date") >= (anchor - timedelta(days=29))
    e.append((pl.col("to_cart").filter(m30).sum() - pl.col("to_ord").filter(m30).sum()).alias("cart_minus_ord_30"))
    e.append(pl.col("search_to_ord").filter(m90).sum().alias("s2o_cnt_90"))
    e.append(pl.col("cat_to_ord").filter(m90).sum().alias("c2o_cnt_90"))
    return e


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--anchors", type=str, default=None,
                    help="comma-separated ISO dates (default: TEST+VAL+train_anchors(14))")
    args = ap.parse_args()
    if args.anchors:
        anchors = [date.fromisoformat(s) for s in args.anchors.split(",")]
    else:
        anchors = [TEST_ANCHOR, VAL_ANCHOR] + train_anchors(14)
    universe = user_universe()
    lf = pl.scan_parquet(TRAIN_PARQUET)
    for a in anchors:
        p = FEATURES_DIR / f"anchor={a.isoformat()}.extra.parquet"
        if p.exists():
            print(f"  {a}: exists", flush=True)
            continue
        t0 = time.time()
        hist = lf.filter((pl.col("event_date") <= a)
                         & (pl.col("event_date") >= a - timedelta(days=MAX_BACK)))
        feats = hist.group_by("user_id").agg(extra_exprs(a)).collect(engine="streaming")
        out = universe.join(feats, on="user_id", how="left")
        zero_fill = [c for c in out.columns if c.startswith(("dec_", "hv", "cart_minus", "s2o_cnt", "c2o_cnt"))]
        out = out.with_columns([pl.col(c).fill_null(0) for c in zero_fill])
        casts = [pl.col(c).cast(pl.Float32) for c, dt in zip(out.columns, out.dtypes) if dt == pl.Float64]
        out.with_columns(casts).write_parquet(p)
        print(f"  {a}: {out.shape} in {time.time()-t0:.1f}s", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()

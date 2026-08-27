"""v11 feature tier: NON-SYNCHRONOUS CATALOG ATTRIBUTION (cross-device days).

Fact this encodes: rows with gmv_cat > 0 & cat == 0 are 20.7% of ALL days with
catalog orders (measured on train.parquet: 68 564 of 331 504). The day received
catalog-attributed money while THIS device shows no catalog session — a
cross-device / off-surface purchase trace. The post-hoc segment correction of
INPUT is open (eda3_SYNTHESIS: "zero-cost addition at a planned retrain").

Features (Float32, full 250k universe), windows w in {30, 90, 365}, trailing
[A-(w-1), A], history <= anchor:

  v11_gc_nocat_days_w    # days with gmv_cat>0 & cat==0 in window (0 when none)
  v11_gc_nocat_share_w   days / (cat_days_w + 1); cat_days_w = base-tier
                         definition, i.e. count of days with the binary `cat`
                         day-flag set in the same window
  v11_gmv_gc_nocat_w     log1p(sum of gmv_cat over such days) — money sums are
                         stored log1p as in the v10 tier (base tier keeps raw
                         sums plus separate log_ columns; trees only see ranks)
  v11_rec_gc_nocat       days since the last such day within the last 365 days;
                         null when none (recency convention of the base tier:
                         bounded by the scan span, null where undefined)

Count-like columns are 0-filled (base-tier zero_fill convention for counts and
gmv_cat_ sums); only the recency stays null — same split as build_features.py.

Output: work/features/anchor=DATE.v11.parquet, joined by common.load_anchor when
USE_V11=1 (missing anchors get nulls, schema stays consistent — v8/v10 pattern).

Usage:
  POLARS_MAX_THREADS=3 .venv/bin/python work/scripts/build_features_v11.py [--anchors a,b] [--force]
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
from datetime import date  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from common import (  # noqa: E402
    DATA_START, FEATURES_DIR, TEST_ANCHOR, TRAIN_PARQUET, V11_FEATS, VAL_ANCHOR,
    user_universe,
)
from exp_lib import available_train_anchors  # noqa: E402

WINDOWS = (30, 90, 365)
REC_SPAN = 365          # recency scan span; beyond it -> null (base tier: 379)

FEATS = V11_FEATS       # single source of truth lives in common.py


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def load_frame(uni: pl.DataFrame):
    """One small frame: only days with a catalog session OR catalog money.

    gmv_cat>0 on 331k rows, cat>0 on 4.8M of 30.6M — the rest of the panel is
    irrelevant to this tier. (user_id, event_date) verified unique, `cat` verified
    binary, so bincounts of rows are counts of DAYS.
    """
    uid = uni["user_id"].to_numpy()
    pos = pl.DataFrame({"user_id": uid, "uix": np.arange(len(uid), dtype=np.int32)})
    df = (
        pl.scan_parquet(TRAIN_PARQUET)
        .filter((pl.col("gmv_cat") > 0) | (pl.col("cat") > 0))
        .select("user_id", "event_date", "gmv_cat", "cat")
        .collect(engine="streaming")
        .join(pos, on="user_id", how="inner")
        .with_columns(
            ((pl.col("event_date") - pl.lit(DATA_START)).dt.total_days())
            .cast(pl.Int32).alias("d"))
        .select("uix", "d", "gmv_cat", "cat")
    )
    u = df["uix"].to_numpy().astype(np.int64)
    d = df["d"].to_numpy().astype(np.int32)
    g = df["gmv_cat"].to_numpy().astype(np.float64)
    cat = df["cat"].to_numpy() > 0
    gc = (g > 0) & (~cat)          # the cross-device day: catalog money, no catalog session
    return u, d, g, cat, gc


def build(anchor: date, uni: pl.DataFrame, u, d, g, cat, gc):
    t0 = time.time()
    n = uni.height
    A = (anchor - DATA_START).days

    vals: dict[str, np.ndarray] = {}
    for w in WINDOWS:
        m = (d <= A) & (d >= A - (w - 1))          # trailing window, base-tier style
        mg = m & gc
        days = np.bincount(u[mg], minlength=n).astype(np.float64)
        cat_days = np.bincount(u[m & cat], minlength=n).astype(np.float64)
        vals[f"v11_gc_nocat_days_{w}"] = days
        vals[f"v11_gc_nocat_share_{w}"] = days / (cat_days + 1.0)
        vals[f"v11_gmv_gc_nocat_{w}"] = np.log1p(
            np.bincount(u[mg], weights=g[mg], minlength=n))

    mr = (d <= A) & (d > A - REC_SPAN) & gc
    last = np.full(n, -1.0)
    if mr.any():
        np.maximum.at(last, u[mr], d[mr].astype(np.float64))
    vals["v11_rec_gc_nocat"] = np.where(last >= 0, A - last, np.nan)

    out = uni.select("user_id").with_columns(
        [pl.Series(c, np.asarray(vals[c], dtype=np.float64)) for c in FEATS]
    ).with_columns([pl.col(c).fill_nan(None).cast(pl.Float32) for c in FEATS])
    assert out.height == n and out.columns == ["user_id"] + FEATS

    p = FEATURES_DIR / f"anchor={anchor.isoformat()}.v11.parquet"
    tmp = p.with_suffix(".tmp.parquet")
    out.write_parquet(tmp)
    tmp.rename(p)
    d365 = vals["v11_gc_nocat_days_365"]
    log(f"  v11 {anchor}: users d365>0 = {int((d365 > 0).sum())} "
        f"({(d365 > 0).mean():.3%}), sum days365 = {int(d365.sum())} "
        f"in {time.time()-t0:.1f}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--anchors", type=str, default="")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if args.anchors:
        anchors = [date.fromisoformat(x) for x in args.anchors.split(",")]
    else:
        anchors = sorted(set(available_train_anchors()[-14:] + [VAL_ANCHOR, TEST_ANCHOR]))

    uni = user_universe()
    t0 = time.time()
    u, d, g, cat, gc = load_frame(uni)
    log(f"frame: rows={len(u)} gc_nocat rows={int(gc.sum())} "
        f"catord rows={int((g > 0).sum())} in {time.time()-t0:.0f}s")

    todo = [a for a in anchors
            if args.force or not (FEATURES_DIR / f"anchor={a.isoformat()}.v11.parquet").exists()]
    log(f"anchors: {len(anchors)} total, {len(todo)} to build")
    for a in todo:
        build(a, uni, u, d, g, cat, gc)
    log("V11 DONE")


if __name__ == "__main__":
    main()

"""Materialize dense per-user daily sequences for NN models.

Per anchor: float16 array [250k users (sample_submit order), L=112 days, C=6]
channels: log1p(gmv), log1p(searches), min(to_ord,10), min(to_cart,10), search, cat
day index 0 = anchor-111 ... 111 = anchor day. Saved to work/seq/anchor=DATE.npy
plus targets vector (float32) for labeled anchors: anchor=DATE.target.npy
"""
from __future__ import annotations

import sys
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))
from common import TRAIN_PARQUET, VAL_ANCHOR, TEST_ANCHOR, DATA_END, HORIZON, WORK, train_anchors, user_universe

SEQ_DIR = WORK / "seq"
L = 112


def build(lf: pl.LazyFrame, uni: pl.DataFrame, row_of: dict, anchor: date):
    t0 = time.time()
    win = lf.filter(
        (pl.col("event_date") <= anchor)
        & (pl.col("event_date") > anchor - timedelta(days=L))
    ).select("user_id", "event_date", "gmv", "searches", "to_ord", "to_cart", "search", "cat")
    df = win.collect(engine="streaming")
    uidx = df["user_id"].replace_strict(row_of, return_dtype=pl.Int32).to_numpy()
    days_ago = df.select((pl.lit(anchor) - pl.col("event_date")).dt.total_days().alias("d"))["d"].to_numpy()
    didx = (L - 1) - days_ago
    arr = np.zeros((len(uni), L, 6), dtype=np.float16)
    arr[uidx, didx, 0] = np.log1p(df["gmv"].to_numpy()).astype(np.float16)
    arr[uidx, didx, 1] = np.log1p(df["searches"].to_numpy()).astype(np.float16)
    arr[uidx, didx, 2] = np.minimum(df["to_ord"].to_numpy(), 10).astype(np.float16)
    arr[uidx, didx, 3] = np.minimum(df["to_cart"].to_numpy(), 10).astype(np.float16)
    arr[uidx, didx, 4] = df["search"].to_numpy().astype(np.float16)
    arr[uidx, didx, 5] = df["cat"].to_numpy().astype(np.float16)
    np.save(SEQ_DIR / f"anchor={anchor.isoformat()}.npy", arr)

    if anchor + timedelta(days=HORIZON) <= DATA_END:
        tgt = (
            lf.filter(pl.col("event_date").is_between(
                anchor + timedelta(days=1), anchor + timedelta(days=HORIZON)))
            .group_by("user_id").agg(pl.col("gmv").sum().alias("target"))
            .collect(engine="streaming")
        )
        t = uni.join(tgt, on="user_id", how="left").with_columns(pl.col("target").fill_null(0.0))
        np.save(SEQ_DIR / f"anchor={anchor.isoformat()}.target.npy",
                t["target"].to_numpy().astype(np.float32))
    print(f"  seq {anchor}: {arr.shape} in {time.time()-t0:.1f}s", flush=True)


def main():
    SEQ_DIR.mkdir(exist_ok=True)
    anchors = [TEST_ANCHOR, VAL_ANCHOR] + train_anchors(8)
    uni = user_universe()
    row_of = {u: i for i, u in enumerate(uni["user_id"].to_list())}
    lf = pl.scan_parquet(TRAIN_PARQUET)
    for a in anchors:
        if (SEQ_DIR / f"anchor={a.isoformat()}.npy").exists():
            print(f"  seq {a}: exists", flush=True)
            continue
        build(lf, uni, row_of, a)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()

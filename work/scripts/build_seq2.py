"""Seq tensors v2 for the max sequence-model program.

Per anchor: float16 [250k, L=196, C=8], channels:
0 log1p(gmv_search), 1 log1p(gmv_cat), 2 min(to_ord,10), 3 min(to_cart,20),
4 log1p(searches), 5 search, 6 cat, 7 any_order_flag(has_search_to_ord|has_cat_to_ord)
Saved to work/seq2/anchor=DATE.npy (+ .target.npy with [y30, y7, y14] float32 when observable).
Anchors: 12 recent clean (<= VAL-30d... actually stride-7 recent 12), VAL, TEST.
"""
from __future__ import annotations

import sys
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))
from common import TRAIN_PARQUET, VAL_ANCHOR, TEST_ANCHOR, DATA_END, WORK, user_universe
from exp_lib import available_train_anchors

SEQ_DIR = WORK / "seq2"
L = 196


def build(lf, uni, row_of, anchor: date):
    t0 = time.time()
    win = lf.filter(
        (pl.col("event_date") <= anchor)
        & (pl.col("event_date") > anchor - timedelta(days=L))
    ).select("user_id", "event_date", "gmv_search", "gmv_cat", "to_ord", "to_cart",
             "searches", "search", "cat", "has_search_to_ord", "has_cat_to_ord")
    df = win.collect(engine="streaming")
    uidx = df["user_id"].replace_strict(row_of, return_dtype=pl.Int32).to_numpy()
    days_ago = df.select((pl.lit(anchor) - pl.col("event_date")).dt.total_days().alias("d"))["d"].to_numpy()
    didx = (L - 1) - days_ago
    arr = np.zeros((len(uni), L, 8), dtype=np.float16)
    arr[uidx, didx, 0] = np.log1p(df["gmv_search"].to_numpy()).astype(np.float16)
    arr[uidx, didx, 1] = np.log1p(df["gmv_cat"].to_numpy()).astype(np.float16)
    arr[uidx, didx, 2] = np.minimum(df["to_ord"].to_numpy(), 10).astype(np.float16)
    arr[uidx, didx, 3] = np.minimum(df["to_cart"].to_numpy(), 20).astype(np.float16)
    arr[uidx, didx, 4] = np.log1p(df["searches"].to_numpy()).astype(np.float16)
    arr[uidx, didx, 5] = df["search"].to_numpy().astype(np.float16)
    arr[uidx, didx, 6] = df["cat"].to_numpy().astype(np.float16)
    arr[uidx, didx, 7] = ((df["has_search_to_ord"].to_numpy() + df["has_cat_to_ord"].to_numpy()) > 0).astype(np.float16)
    np.save(SEQ_DIR / f"anchor={anchor.isoformat()}.npy", arr)
    del arr

    if anchor + timedelta(days=30) <= DATA_END:
        tgts = []
        for h in (30, 7, 14):
            t = (lf.filter(pl.col("event_date").is_between(
                    anchor + timedelta(days=1), anchor + timedelta(days=h)))
                 .group_by("user_id").agg(pl.col("gmv").sum().alias("t"))
                 .collect(engine="streaming"))
            v = uni.join(t, on="user_id", how="left").with_columns(pl.col("t").fill_null(0.0))
            tgts.append(v["t"].to_numpy().astype(np.float32))
        np.save(SEQ_DIR / f"anchor={anchor.isoformat()}.target.npy", np.stack(tgts, axis=1))
    print(f"  seq2 {anchor}: done in {time.time()-t0:.1f}s", flush=True)


def main():
    SEQ_DIR.mkdir(exist_ok=True)
    train = available_train_anchors()[-12:]
    anchors = [TEST_ANCHOR, VAL_ANCHOR] + train
    uni = user_universe()
    row_of = {u: i for i, u in enumerate(uni["user_id"].to_list())}
    lf = pl.scan_parquet(TRAIN_PARQUET)
    for a in anchors:
        if (SEQ_DIR / f"anchor={a.isoformat()}.npy").exists():
            print(f"  seq2 {a}: exists", flush=True)
            continue
        build(lf, uni, row_of, a)
    print("SEQ2 DONE", flush=True)


if __name__ == "__main__":
    main()

"""Seq tensors v3: seq2's 8 channels + the 4 per-channel funnel counters.

Gap found 19.08: search_to_cart / search_to_ord / cat_to_cart / cat_to_ord never
reach the sequence models, and fusion (blend weight 0.32) is the model that would
see them per-day.  seq3 adds them as channels 8..11.

Per anchor: uint8 [250k, L=112, C=12] + per-channel dequant scale (work/seq3/quant.json).
  0 log1p(gmv_search)   x20     6 cat flag
  1 log1p(gmv_cat)      x20     7 any_order flag (has_search_to_ord|has_cat_to_ord)
  2 min(to_ord,10)              8 min(search_to_cart,20)   <- new
  3 min(to_cart,20)             9 min(search_to_ord,10)    <- new
  4 log1p(searches)     x20    10 min(cat_to_cart,20)      <- new
  5 search flag                11 min(cat_to_ord,10)       <- new
Channels 0..7 are numerically identical to seq2 up to the 1/20 quantisation step
of the three log channels (all other channels are small integers -> exact).

WHY uint8 (seq2 is float16): the queue runner refuses to start a job when free disk
< 12 GB.  float16 [250k,112,12] = 672 MB/anchor -> 10 anchors = 6.7 GB would push
free space to ~9.9 GB and stall the queue.  uint8 = 336 MB/anchor -> 3.4 GB total.
The strict control (train_fusion3.py --n-ch 8) reads the SAME tensors, so L=112 and
the quantisation are held constant between the two arms and only the 4 new channels
differ.

Anchors: TEST, VAL, and the last --max-train CLEAN train anchors (anchor+30d <= VAL,
i.e. the gap-30 protocol).  The default 8 reproduces exactly the clean-anchor set the
existing fusion runs selected on.  Targets [y30, y7, y14] float32 as in build_seq2.

Run: POLARS_MAX_THREADS=3 .venv/bin/python work/scripts/build_seq3.py [--max-train 8]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))
from common import TRAIN_PARQUET, VAL_ANCHOR, TEST_ANCHOR, DATA_END, WORK, user_universe
from exp_lib import available_train_anchors

SEQ_DIR = WORK / "seq3"
L = 112
C = 12
N_USERS = 250_000
# dequant: value = stored_uint8 * SCALES[c]
SCALES = [0.05, 0.05, 1.0, 1.0, 0.05, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
CH_NAMES = ["log1p_gmv_search", "log1p_gmv_cat", "to_ord_c10", "to_cart_c20",
            "log1p_searches", "search", "cat", "any_ord",
            "search_to_cart_c20", "search_to_ord_c10",
            "cat_to_cart_c20", "cat_to_ord_c10"]
BYTES_PER_TENSOR = N_USERS * L * C  # uint8


def free_gb() -> float:
    st = os.statvfs("/")
    return st.f_bavail * st.f_frsize / 1e9


def qlog(x: np.ndarray) -> np.ndarray:
    """log1p channel -> uint8 at 1/20 resolution (max log1p in data 11.2 -> 224)."""
    return np.clip(np.rint(np.log1p(x) * 20.0), 0, 255).astype(np.uint8)


def qcnt(x: np.ndarray, cap: int) -> np.ndarray:
    """small-integer channel -> uint8, exact."""
    return np.minimum(x, cap).astype(np.uint8)


def build(lf, uni, row_of, anchor: date):
    t0 = time.time()
    win = lf.filter(
        (pl.col("event_date") <= anchor)
        & (pl.col("event_date") > anchor - timedelta(days=L))
    ).select("user_id", "event_date", "gmv_search", "gmv_cat", "to_ord", "to_cart",
             "searches", "search", "cat", "has_search_to_ord", "has_cat_to_ord",
             "search_to_cart", "search_to_ord", "cat_to_cart", "cat_to_ord")
    df = win.collect(engine="streaming")
    uidx = df["user_id"].replace_strict(row_of, return_dtype=pl.Int32).to_numpy()
    days_ago = df.select((pl.lit(anchor) - pl.col("event_date")).dt.total_days().alias("d"))["d"].to_numpy()
    didx = (L - 1) - days_ago
    assert didx.min() >= 0 and didx.max() < L, f"{anchor}: day index out of range"

    arr = np.zeros((N_USERS, L, C), dtype=np.uint8)
    arr[uidx, didx, 0] = qlog(df["gmv_search"].to_numpy())
    arr[uidx, didx, 1] = qlog(df["gmv_cat"].to_numpy())
    arr[uidx, didx, 2] = qcnt(df["to_ord"].to_numpy(), 10)
    arr[uidx, didx, 3] = qcnt(df["to_cart"].to_numpy(), 20)
    arr[uidx, didx, 4] = qlog(df["searches"].to_numpy())
    arr[uidx, didx, 5] = df["search"].to_numpy().astype(np.uint8)
    arr[uidx, didx, 6] = df["cat"].to_numpy().astype(np.uint8)
    arr[uidx, didx, 7] = ((df["has_search_to_ord"].to_numpy()
                           + df["has_cat_to_ord"].to_numpy()) > 0).astype(np.uint8)
    arr[uidx, didx, 8] = qcnt(df["search_to_cart"].to_numpy(), 20)
    arr[uidx, didx, 9] = qcnt(df["search_to_ord"].to_numpy(), 10)
    arr[uidx, didx, 10] = qcnt(df["cat_to_cart"].to_numpy(), 20)
    arr[uidx, didx, 11] = qcnt(df["cat_to_ord"].to_numpy(), 10)
    nz = [int((arr[:, :, c] > 0).sum()) for c in range(C)]
    np.save(SEQ_DIR / f"anchor={anchor.isoformat()}.npy", arr)
    del arr, df

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
    print(f"  seq3 {anchor}: done in {time.time()-t0:.1f}s  nonzero/ch={nz}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-train", type=int, default=8,
                    help="how many CLEAN (gap-30) train anchors, newest first")
    ap.add_argument("--min-free-gb", type=float, default=12.6,
                    help="never let free disk fall below this (queue runner floor 12)")
    args = ap.parse_args()

    SEQ_DIR.mkdir(exist_ok=True)
    (SEQ_DIR / "quant.json").write_text(json.dumps(
        {"L": L, "C": C, "dtype": "uint8", "scales": SCALES, "channels": CH_NAMES}, indent=1))

    clean = [a for a in available_train_anchors() if a + timedelta(days=30) <= VAL_ANCHOR]
    train = clean[-args.max_train:]
    anchors = [TEST_ANCHOR, VAL_ANCHOR] + train[::-1]  # newest-first if truncated
    print(f"seq3 L={L} C={C} uint8 {BYTES_PER_TENSOR/1e9:.3f} GB/anchor; "
          f"{len(anchors)} anchors, free {free_gb():.1f} GB", flush=True)
    print(f"train (clean, gap30): {[a.isoformat() for a in train]}", flush=True)

    uni = user_universe()
    row_of = {u: i for i, u in enumerate(uni["user_id"].to_list())}
    lf = pl.scan_parquet(TRAIN_PARQUET)
    built, skipped = 0, []
    for a in anchors:
        if (SEQ_DIR / f"anchor={a.isoformat()}.npy").exists():
            print(f"  seq3 {a}: exists", flush=True)
            built += 1
            continue
        f = free_gb()
        if f - BYTES_PER_TENSOR / 1e9 < args.min_free_gb:
            print(f"  seq3 {a}: SKIP, free {f:.1f} GB would drop below "
                  f"{args.min_free_gb} GB", flush=True)
            skipped.append(a.isoformat())
            continue
        build(lf, uni, row_of, a)
        built += 1
    print(json.dumps({"built": built, "skipped": skipped,
                      "free_gb": round(free_gb(), 2)}), flush=True)
    print("SEQ3 DONE", flush=True)


if __name__ == "__main__":
    main()

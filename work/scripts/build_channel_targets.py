"""Per-anchor CHANNEL targets: per-user 30d sums of gmv_search / gmv_cat after the anchor.

For each anchor A builds per-user sums over (A, A+30] from train.parquet and writes
  work/features/anchor=DATE.chtgt.parquet  (user_id Int64, tgt_search Float32, tgt_cat Float32)
over the full 250k submit universe (zeros for users without events in the window).
Identity in the data: gmv == gmv_search + gmv_cat, so tgt_search + tgt_cat == target
of the base anchor file (checked per anchor, tolerance for f32 rounding).

Usage:
  build_channel_targets.py                    # default: available_train_anchors()[-14:] + VAL
  build_channel_targets.py --n-back 18        # deeper: champion gap-30 protocol needs 14 train
                                              #  + 4 gap anchors -> [-18:] + VAL
  build_channel_targets.py --anchors 2025-12-03,2025-12-10
  build_channel_targets.py --force            # rebuild even if file exists
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).parent))
from common import DATA_END, FEATURES_DIR, HORIZON, TRAIN_PARQUET, VAL_ANCHOR, user_universe
from exp_lib import available_train_anchors


def chtgt_path(anchor: date) -> Path:
    return FEATURES_DIR / f"anchor={anchor.isoformat()}.chtgt.parquet"


def build_anchor(lf: pl.LazyFrame, universe: pl.DataFrame, anchor: date) -> pl.DataFrame:
    t0 = time.time()
    tgt = (
        lf.filter(pl.col("event_date").is_between(
            anchor + timedelta(days=1), anchor + timedelta(days=HORIZON)))
        .group_by("user_id")
        .agg(
            pl.col("gmv_search").sum().alias("tgt_search"),
            pl.col("gmv_cat").sum().alias("tgt_cat"),
        )
        .collect(engine="streaming")
    )
    out = universe.join(tgt, on="user_id", how="left").with_columns(
        pl.col("tgt_search").fill_null(0.0).cast(pl.Float32),
        pl.col("tgt_cat").fill_null(0.0).cast(pl.Float32),
    )

    # sanity: channel identity against the base anchor target (when labeled)
    msg = ""
    base_p = FEATURES_DIR / f"anchor={anchor.isoformat()}.parquet"
    if base_p.exists():
        base = pl.read_parquet(base_p, columns=["user_id", "target"])
        if base["target"].null_count() == 0:
            j = out.join(base, on="user_id", how="inner")
            diff = float(
                (j["tgt_search"].cast(pl.Float64) + j["tgt_cat"].cast(pl.Float64)
                 - j["target"]).abs().max()
            )
            msg = f" | identity max|s+c-tgt|={diff:.5f}"
            assert diff < 1.0, f"channel identity broken at {anchor}: {diff}"

    nz_s = int((out["tgt_search"] > 0).sum())
    nz_c = int((out["tgt_cat"] > 0).sum())
    print(f"  {anchor}: {out.shape} nz_search={nz_s} nz_cat={nz_c} "
          f"{time.time()-t0:.1f}s{msg}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--anchors", type=str, default=None,
                    help="comma-separated ISO dates; overrides --n-back")
    ap.add_argument("--n-back", type=int, default=14,
                    help="use available_train_anchors()[-N:] + VAL (default 14)")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if args.anchors:
        anchors = [date.fromisoformat(s) for s in args.anchors.split(",")]
    else:
        anchors = available_train_anchors()[-args.n_back:] + [VAL_ANCHOR]

    universe = user_universe()
    lf = pl.scan_parquet(TRAIN_PARQUET)
    print(f"universe={len(universe)}, anchors={len(anchors)}: "
          f"{anchors[0]} .. {anchors[-1]}", flush=True)

    for a in anchors:
        assert a + timedelta(days=HORIZON) <= DATA_END, \
            f"anchor {a} target window not fully observed"
        p = chtgt_path(a)
        if p.exists() and not args.force:
            print(f"  {a}: exists, skip", flush=True)
            continue
        build_anchor(lf, universe, a).write_parquet(p)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()

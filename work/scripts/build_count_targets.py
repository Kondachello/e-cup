"""Per-anchor COUNT x AOV targets: 30d order count and mean check after the anchor.

For each anchor A builds per-user aggregates over (A, A+30] from train.parquet:
  tgt_cnt  Float32  sum of to_ord in the window (0 for users without orders)
  tgt_aov  Float32  gmv_window / orders_window for users with tgt_cnt > 0, null otherwise
and writes work/features/anchor=DATE.cnttgt.parquet over the full 250k submit universe.

Data identity (verified): cnt > 0 <=> gmv > 0, and tgt_cnt * tgt_aov == target of the
base anchor file for buyers (checked per anchor, tolerance for f32 rounding).

Usage:
  build_count_targets.py                     # default: available_train_anchors()[-18:] + VAL
                                             #  (14 train + 4 gap anchors for gap-30 protocol)
  build_count_targets.py --anchors 2025-12-03,2025-12-10
  build_count_targets.py --force             # rebuild even if file exists
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


def cnttgt_path(anchor: date) -> Path:
    return FEATURES_DIR / f"anchor={anchor.isoformat()}.cnttgt.parquet"


def build_anchor(lf: pl.LazyFrame, universe: pl.DataFrame, anchor: date) -> pl.DataFrame:
    t0 = time.time()
    agg = (
        lf.filter(pl.col("event_date").is_between(
            anchor + timedelta(days=1), anchor + timedelta(days=HORIZON)))
        .group_by("user_id")
        .agg(
            pl.col("to_ord").sum().alias("tgt_cnt"),
            pl.col("gmv").sum().alias("_gmv"),
        )
        .collect(engine="streaming")
    )
    out = (
        universe.join(agg, on="user_id", how="left")
        .with_columns(pl.col("tgt_cnt").fill_null(0), pl.col("_gmv").fill_null(0.0))
        .with_columns(
            pl.when(pl.col("tgt_cnt") > 0)
            .then(pl.col("_gmv") / pl.col("tgt_cnt"))
            .otherwise(None)
            .alias("tgt_aov")
        )
    )
    # identity inside the window: buyers by count == buyers by gmv
    bad = out.filter(
        ((pl.col("tgt_cnt") > 0) & (pl.col("_gmv") == 0.0))
        | ((pl.col("tgt_cnt") == 0) & (pl.col("_gmv") > 0.0))
    ).height
    assert bad == 0, f"cnt/gmv positivity mismatch at {anchor}: {bad} users"
    out = out.select(
        "user_id",
        pl.col("tgt_cnt").cast(pl.Float32),
        pl.col("tgt_aov").cast(pl.Float32),
    )

    # sanity: cnt * aov identity against the base anchor target (when labeled)
    msg = ""
    base_p = FEATURES_DIR / f"anchor={anchor.isoformat()}.parquet"
    if base_p.exists():
        base = pl.read_parquet(base_p, columns=["user_id", "target"])
        if base["target"].null_count() == 0:
            j = out.join(base, on="user_id", how="inner").filter(pl.col("tgt_cnt") > 0)
            diff = float(
                (j["tgt_cnt"].cast(pl.Float64) * j["tgt_aov"].cast(pl.Float64)
                 - j["target"]).abs().max()
            )
            msg = f" | identity max|cnt*aov-tgt|={diff:.5f}"
            assert diff < 1.0, f"count*aov identity broken at {anchor}: {diff}"

    nz = int((out["tgt_cnt"] > 0).sum())
    aov = out["tgt_aov"].drop_nulls()
    print(f"  {anchor}: {out.shape} nz_cnt={nz} aov_p50={float(aov.median()):.2f} "
          f"{time.time()-t0:.1f}s{msg}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--anchors", type=str, default=None,
                    help="comma-separated ISO dates; overrides --n-back")
    ap.add_argument("--n-back", type=int, default=18,
                    help="use available_train_anchors()[-N:] + VAL (default 18: "
                         "14 train + 4 gap anchors of the gap-30 protocol)")
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
        p = cnttgt_path(a)
        if p.exists() and not args.force:
            print(f"  {a}: exists, skip", flush=True)
            continue
        build_anchor(lf, universe, a).write_parquet(p)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()

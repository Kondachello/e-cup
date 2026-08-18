"""Per-anchor WEEKLY-HORIZON targets: per-user gmv sums over four sub-windows.

For each anchor A builds per-user sums of gmv over
  w1 (A,    A+7], w2 (A+7,  A+14], w3 (A+14, A+21], w4 (A+21, A+30]
from train.parquet and writes
  work/features/anchor=DATE.hztgt.parquet
  (user_id Int64, tgt_w1..tgt_w4 Float32)
over the full 250k submit universe (zeros for users without events).
Identity: tgt_w1+..+tgt_w4 == target of the base anchor file (checked when the
base anchor is labeled, tolerance for f32 rounding).

Usage:
  build_horizon_targets.py                 # default: available_train_anchors()[-18:] + VAL
  build_horizon_targets.py --n-back 18
  build_horizon_targets.py --anchors 2025-12-03,2025-12-10
  build_horizon_targets.py --force
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

# inclusive day-offset windows from the anchor: (lo, hi] in days -> [A+lo .. A+hi]
WINDOWS = {"w1": (1, 7), "w2": (8, 14), "w3": (15, 21), "w4": (22, 30)}
assert WINDOWS["w4"][1] == HORIZON


def hztgt_path(anchor: date) -> Path:
    return FEATURES_DIR / f"anchor={anchor.isoformat()}.hztgt.parquet"


def build_anchor(lf: pl.LazyFrame, universe: pl.DataFrame, anchor: date) -> pl.DataFrame:
    t0 = time.time()
    aggs = [
        pl.col("gmv").filter(pl.col("event_date").is_between(
            anchor + timedelta(days=lo), anchor + timedelta(days=hi)))
        .sum().alias(f"tgt_{w}")
        for w, (lo, hi) in WINDOWS.items()
    ]
    tgt = (
        lf.filter(pl.col("event_date").is_between(
            anchor + timedelta(days=1), anchor + timedelta(days=HORIZON)))
        .group_by("user_id")
        .agg(aggs)
        .collect(engine="streaming")
    )
    out = universe.join(tgt, on="user_id", how="left").with_columns(
        [pl.col(f"tgt_{w}").fill_null(0.0).cast(pl.Float32) for w in WINDOWS]
    )

    # sanity: weekly identity against the base anchor target (when labeled)
    msg = ""
    base_p = FEATURES_DIR / f"anchor={anchor.isoformat()}.parquet"
    if base_p.exists():
        base = pl.read_parquet(base_p, columns=["user_id", "target"])
        if base["target"].null_count() == 0:
            j = out.join(base, on="user_id", how="inner")
            s = sum(j[f"tgt_{w}"].cast(pl.Float64) for w in WINDOWS)
            diff = float((s - j["target"]).abs().max())
            msg = f" | identity max|sum_w-tgt|={diff:.5f}"
            assert diff < 1.0, f"weekly identity broken at {anchor}: {diff}"

    nz = {w: int((out[f"tgt_{w}"] > 0).sum()) for w in WINDOWS}
    print(f"  {anchor}: {out.shape} nz={nz} {time.time()-t0:.1f}s{msg}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--anchors", type=str, default=None,
                    help="comma-separated ISO dates; overrides --n-back")
    ap.add_argument("--n-back", type=int, default=18,
                    help="use available_train_anchors()[-N:] + VAL (default 18)")
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
        p = hztgt_path(a)
        if p.exists() and not args.force:
            print(f"  {a}: exists, skip", flush=True)
            continue
        build_anchor(lf, universe, a).write_parquet(p)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()

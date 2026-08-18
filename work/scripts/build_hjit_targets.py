"""Per-anchor EXTENDED-HORIZON targets for the snapshot-jitter ("hjit") experiment.

Idea: the test forecast rests on the single 2026-02-13 feature snapshot; models
trained to longer horizons predict the SAME test window (2026-02-14..2026-03-15)
from EARLIER snapshots (2026-02-06 @ h37, 2026-01-30 @ h44), damping single-day
snapshot anomalies when averaged in log1p space.

For each anchor A this script builds per-user gmv sums over (A, A+37] and
(A, A+44] from train.parquet and writes
  work/features/anchor=DATE.hjit.parquet
  (user_id Int64, tgt_h37 Float32, tgt_h44 Float32)
over the full 250k submit universe (zeros for users without events).
A horizon column is NULL when its window is not fully observed (A+H > DATA_END)
so trainers cannot silently consume truncated targets.

Identity checks per anchor against the labeled base anchor file:
  sum gmv over (A, A+30] == base target   (f32 tolerance)
  tgt_h30 <= tgt_h37 <= tgt_h44           (monotone in horizon, where observed)

Horizon-model validation anchors (window ends exactly at DATA_END=2026-02-13):
  h37 -> 2026-01-07,  h44 -> 2025-12-31

Usage:
  build_hjit_targets.py                 # default: available_train_anchors()[-16:]
  build_hjit_targets.py --n-back 20
  build_hjit_targets.py --anchors 2025-12-03,2025-12-10
  build_hjit_targets.py --force
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).parent))
from common import DATA_END, FEATURES_DIR, HORIZON, TRAIN_PARQUET, user_universe
from exp_lib import available_train_anchors

HORIZONS = (37, 44)
H_MAX = max(HORIZONS)


def hjit_path(anchor: date) -> Path:
    return FEATURES_DIR / f"anchor={anchor.isoformat()}.hjit.parquet"


def build_anchor(lf: pl.LazyFrame, universe: pl.DataFrame, anchor: date) -> pl.DataFrame:
    t0 = time.time()
    hs = [HORIZON] + list(HORIZONS)  # h30 computed only for the identity check
    aggs = [
        pl.col("gmv").filter(pl.col("event_date") <= anchor + timedelta(days=h))
        .sum().alias(f"tgt_h{h}")
        for h in hs
    ]
    tgt = (
        lf.filter(pl.col("event_date").is_between(
            anchor + timedelta(days=1), anchor + timedelta(days=H_MAX)))
        .group_by("user_id")
        .agg(aggs)
        .collect(engine="streaming")
    )
    out = universe.join(tgt, on="user_id", how="left").with_columns(
        [pl.col(f"tgt_h{h}").fill_null(0.0).cast(pl.Float64) for h in hs]
    )

    # identity: (A, A+30] sum must equal the base anchor target
    msg = ""
    base_p = FEATURES_DIR / f"anchor={anchor.isoformat()}.parquet"
    if base_p.exists():
        base = pl.read_parquet(base_p, columns=["user_id", "target"])
        if base["target"].null_count() == 0:
            j = out.join(base, on="user_id", how="inner")
            diff = float((j[f"tgt_h{HORIZON}"] - j["target"]).abs().max())
            msg = f" | identity max|h30-tgt|={diff:.5f}"
            assert diff < 1.0, f"h30 identity broken at {anchor}: {diff}"

    # monotone in horizon (before nulling unobserved columns)
    prev = f"tgt_h{HORIZON}"
    for h in HORIZONS:
        n_bad = int((out[f"tgt_h{h}"] < out[prev] - 0.5).sum())
        assert n_bad == 0, f"monotonicity broken at {anchor}: tgt_h{h} < {prev} for {n_bad} users"
        prev = f"tgt_h{h}"

    nz = {f"h{h}": int((out[f"tgt_h{h}"] > 0).sum()) for h in HORIZONS}
    out = out.select(["user_id"] + [f"tgt_h{h}" for h in HORIZONS]).with_columns(
        [pl.col(f"tgt_h{h}").cast(pl.Float32) for h in HORIZONS]
    )
    # null-out horizons whose window is not fully observed
    dropped = []
    for h in HORIZONS:
        if anchor + timedelta(days=h) > DATA_END:
            out = out.with_columns(pl.lit(None, dtype=pl.Float32).alias(f"tgt_h{h}"))
            dropped.append(f"h{h}")
    if dropped:
        msg += f" | unobserved->null: {','.join(dropped)}"
    print(f"  {anchor}: {out.shape} nz={nz} {time.time()-t0:.1f}s{msg}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--anchors", type=str, default=None,
                    help="comma-separated ISO dates; overrides --n-back")
    ap.add_argument("--n-back", type=int, default=16,
                    help="use available_train_anchors()[-N:] (default 16)")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if args.anchors:
        anchors = [date.fromisoformat(s) for s in args.anchors.split(",")]
    else:
        anchors = available_train_anchors()[-args.n_back:]

    universe = user_universe()
    lf = pl.scan_parquet(TRAIN_PARQUET)
    print(f"universe={len(universe)}, anchors={len(anchors)}: "
          f"{anchors[0]} .. {anchors[-1]}", flush=True)

    for a in anchors:
        assert a + timedelta(days=min(HORIZONS)) <= DATA_END, \
            f"anchor {a}: even h{min(HORIZONS)} window not fully observed"
        p = hjit_path(a)
        if p.exists() and not args.force:
            print(f"  {a}: exists, skip", flush=True)
            continue
        build_anchor(lf, universe, a).write_parquet(p)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""season_seg_agg.py — per-user aggregates around a set of anchors.

For each anchor A builds, over the universe of 250k users:
  * features from the 44-day window [A-43, A] (that is ALL the history available at the
    2025-02-13 anchor, since data start at 2025-01-01 — so the same features are
    computable at 2026-02-13 and comparable across years);
  * X = sum gmv over [A-29, A]      (30-day base window)
  * Y = sum gmv over [A+1, A+30]    (30-day target window; absent for the test anchor)

Output: work/features/seasseg_anchor=<A>.parquet

Usage:
  POLARS_MAX_THREADS=3 .venv/bin/python work/scripts/season_seg_agg.py
"""
from __future__ import annotations

import os
import sys
from datetime import date, timedelta

os.environ.setdefault("POLARS_MAX_THREADS", "3")
os.environ.setdefault("OMP_NUM_THREADS", "3")

import polars as pl  # noqa: E402

ROOT = "/Users/alexanderkondakov/ozon-cup"
TRAIN = f"{ROOT}/train.parquet"
OUT = f"{ROOT}/work/features"

FEAT_DAYS = 44   # [A-43, A]
BASE_DAYS = 30   # [A-29, A]
TGT_DAYS = 30    # [A+1, A+30]

ANCHORS = {
    # tag        anchor        has target window
    "seas25":   (date(2025, 2, 13), True),   # X 15.01-13.02 -> Y 14.02-15.03  (сезонный переход)
    "ctl_may":  (date(2025, 5, 14), True),   # X 15.04-14.05 -> Y 15.05-13.06  (контроль из задания)
    "ctl_jul":  (date(2025, 7, 14), True),   # X 15.06-14.07 -> Y 15.07-13.08  (контроль)
    "ctl_sep":  (date(2025, 9, 14), True),   # X 15.08-14.09 -> Y 15.09-14.10  (контроль)
    "val26":    (date(2026, 1, 14), True),   # X 16.12-14.01 -> Y 15.01-13.02  (диагностика)
    "test26":   (date(2026, 2, 13), False),  # применение
}


def agg(anchor: date, has_tgt: bool) -> pl.DataFrame:
    fs = anchor - timedelta(days=FEAT_DAYS - 1)
    xs = anchor - timedelta(days=BASE_DAYS - 1)
    te = anchor + timedelta(days=TGT_DAYS)
    hi = te if has_tgt else anchor

    d = pl.col("event_date")
    lf = (
        pl.scan_parquet(TRAIN)
        .select(["event_date", "user_id", "gmv", "gmv_cat", "to_ord", "to_cart",
                 "searches", "cat_to_ord"])
        .filter((d >= fs) & (d <= hi))
    )
    fm = (d <= anchor).cast(pl.Float64)                    # feature window mask
    xm = ((d >= xs) & (d <= anchor)).cast(pl.Float64)      # base window mask
    ym = (d > anchor).cast(pl.Float64)                     # target window mask
    wknd = (d.dt.weekday() >= 6).cast(pl.Float64)          # 6=Sat, 7=Sun
    g = pl.col("gmv")
    aggs = [
        (g * fm).sum().alias("f_gmv"),
        (pl.col("gmv_cat") * fm).sum().alias("f_gmv_cat"),
        (pl.col("to_ord") * fm).sum().alias("f_ord"),
        (pl.col("cat_to_ord") * fm).sum().alias("f_ord_cat"),
        (pl.col("to_cart") * fm).sum().alias("f_cart"),
        (pl.col("searches") * fm).sum().alias("f_searches"),
        fm.sum().alias("f_days"),
        ((pl.col("to_ord") > 0).cast(pl.Float64) * fm).sum().alias("f_orddays"),
        (((pl.col("to_cart") > 0) & (pl.col("to_ord") == 0)).cast(pl.Float64) * fm)
        .sum().alias("f_cartnoord_days"),
        (g * fm * wknd).sum().alias("f_gmv_wknd"),
        (fm * wknd).sum().alias("f_days_wknd"),
        ((pl.col("to_ord") > 0).cast(pl.Float64) * fm * wknd).sum().alias("f_orddays_wknd"),
        (g * fm).max().alias("f_gmv_maxday"),
        ((g * fm) ** 2).sum().alias("f_gmv_sq"),
        ((g * fm) > 0).cast(pl.Float64).sum().alias("f_gmvdays"),
        # день последнего заказа внутри окна признаков
        pl.when((pl.col("to_ord") > 0) & (d <= anchor)).then(d).otherwise(None)
        .max().alias("f_last_ord_date"),
        pl.when(d <= anchor).then(d).otherwise(None).max().alias("f_last_act_date"),
        (g * xm).sum().alias("x_gmv"),
        ((pl.col("to_ord") > 0).cast(pl.Float64) * xm).sum().alias("x_orddays"),
    ]
    if has_tgt:
        aggs.append((g * ym).sum().alias("y_gmv"))
    out = lf.group_by("user_id").agg(aggs).collect(engine="streaming")

    uni = (pl.read_csv(f"{ROOT}/sample_submit.csv", schema_overrides={"user_id": pl.Int64})
           .select("user_id"))
    out = uni.join(out, on="user_id", how="left").sort("user_id")
    fills = {c: 0.0 for c in out.columns
             if c not in ("user_id", "f_last_ord_date", "f_last_act_date")}
    out = out.with_columns([pl.col(c).fill_null(v) for c, v in fills.items()])
    out = out.with_columns([
        (pl.lit(anchor) - pl.col("f_last_ord_date")).dt.total_days()
        .fill_null(999).alias("f_rec_ord"),
        (pl.lit(anchor) - pl.col("f_last_act_date")).dt.total_days()
        .fill_null(999).alias("f_rec_act"),
    ]).drop(["f_last_ord_date", "f_last_act_date"])
    return out


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    only = sys.argv[1:] or list(ANCHORS)
    for tag in only:
        a, ht = ANCHORS[tag]
        p = f"{OUT}/seasseg_{tag}.parquet"
        if os.path.exists(p):
            print(f"{tag}: уже есть {p}")
            continue
        df = agg(a, ht)
        df.write_parquet(p)
        act = int((df["f_days"] > 0).sum())
        print(f"{tag} anchor={a} rows={df.height} активных(44д)={act} "
              f"mean_lp_x={df['x_gmv'].log1p().mean():.4f}"
              + (f" mean_lp_y={df['y_gmv'].log1p().mean():.4f}" if ht else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

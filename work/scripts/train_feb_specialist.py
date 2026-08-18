"""Feb->March seasonal specialist (March-8 gifting transition), blend component.

Trains ONLY on the three Feb-2025 anchors (2025-02-06/13/20) whose history is
truncated to ~37-51 days by data start, using ONLY short-history features
(windows <= 42d + recency clipped at 42) so feature semantics align with the
TEST anchor 2026-02-13 when we deliberately restrict it to its last 42 days.

Outputs (exp_lib contract): work/preds/febspec_{val,test}.parquet + scores.tsv row.
Val (2026-01-14) is off-season for this model -> expect mediocre val RMSLE;
the value is Feb->Mar seasonal knowledge in the test window.
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))
from common import TEST_ANCHOR, VAL_ANCHOR, load_anchor, rmsle
from exp_lib import log_score, save_preds

os.environ.setdefault("OMP_NUM_THREADS", "3")

NAME = "febspec"
TRAIN_ANCHORS = [date(2025, 2, 6), date(2025, 2, 13), date(2025, 2, 20)]
AUX_ANCHOR = date(2026, 1, 7)  # extra observed holdout, also off-season
REC_CLIP = 42.0
EPS = 1e-6

# base short-history columns present in the feature parquets
SHORT_COLS = (
    [f"gmv_sum_{w}" for w in (1, 3, 7, 14, 30)]
    + [f"ord_cnt_{w}" for w in (1, 3, 7, 14, 30)]
    + [f"ord_days_{w}" for w in (1, 3, 7, 14, 30)]
    + [f"active_days_{w}" for w in (1, 3, 7, 14, 30)]
    + [f"cart_cnt_{w}" for w in (7, 14, 30)]
    + [f"searches_{w}" for w in (7, 14, 30)]
    + ["cart_days_30", "search_days_30", "cat_days_30",
       "gmv_search_30", "gmv_cat_30",
       "gmv_daymean_30", "gmv_daymax_30", "gmv_daystd_30",
       "log_gmv_sum_7", "log_gmv_sum_30"]
)
REC_COLS = ["rec_active", "rec_order", "rec_cart", "rec_search", "rec_cat", "rec_gmv"]
LAST_COLS = ["last_day_gmv", "last_day_ord", "last_day_cart", "last_day_searches"]
DERIVED = ["ord_rate_30", "gmv_per_ordday_30", "gmv_wk_share", "act_wk_share",
           "cart2ord_30", "search_per_act_30", "gmv_search_share_30"]
FEATS = SHORT_COLS + REC_COLS + LAST_COLS + DERIVED


def prep(anchor: date, need_target: bool) -> pl.DataFrame:
    cols = ["user_id"] + SHORT_COLS + REC_COLS + LAST_COLS + (["target"] if need_target else [])
    df = load_anchor(anchor, columns=cols)
    stale = pl.col("rec_active").is_null() | (pl.col("rec_active") > REC_CLIP)
    df = df.with_columns(
        # last-active-day snapshot only meaningful within the 42d window
        *[pl.when(stale).then(0.0).otherwise(pl.col(c)).alias(c) for c in LAST_COLS],
        # recency capped at 42; "42" = nothing within the window (aligns train/test)
        *[pl.col(c).clip(upper_bound=REC_CLIP).fill_null(REC_CLIP).alias(c) for c in REC_COLS],
    )
    df = df.with_columns(
        (pl.col("ord_days_30") / (pl.col("active_days_30") + EPS)).alias("ord_rate_30"),
        (pl.col("gmv_sum_30") / (pl.col("ord_days_30") + EPS)).alias("gmv_per_ordday_30"),
        ((pl.col("gmv_sum_7") + 1) / (pl.col("gmv_sum_30") + 1)).alias("gmv_wk_share"),
        ((pl.col("active_days_7") + 1) / (pl.col("active_days_30") + 1)).alias("act_wk_share"),
        (pl.col("ord_cnt_30") / (pl.col("cart_cnt_30") + EPS)).alias("cart2ord_30"),
        (pl.col("searches_30") / (pl.col("active_days_30") + EPS)).alias("search_per_act_30"),
        (pl.col("gmv_search_30") / (pl.col("gmv_sum_30") + EPS)).alias("gmv_search_share_30"),
    )
    return df


def to_X(df: pl.DataFrame) -> np.ndarray:
    return df.select(FEATS).to_numpy().astype(np.float32)


def main():
    t0 = time.time()
    import lightgbm as lgb

    tr = pl.concat([prep(a, need_target=True) for a in TRAIN_ANCHORS], how="vertical_relaxed")
    X = to_X(tr)
    y = np.log1p(tr["target"].to_numpy().astype(np.float64))
    print(f"train {X.shape} from {[a.isoformat() for a in TRAIN_ANCHORS]}, "
          f"pos_rate={(y > 0).mean():.4f}, load {time.time()-t0:.0f}s", flush=True)
    del tr

    params = dict(
        objective="tweedie", tweedie_variance_power=1.3, metric="rmse",
        learning_rate=0.05, num_leaves=127, min_data_in_leaf=300,
        feature_fraction=0.75, bagging_fraction=0.8, bagging_freq=1,
        lambda_l2=5.0, max_bin=127, num_threads=int(os.environ["OMP_NUM_THREADS"]),
        seed=42, verbosity=-1,
    )
    m = lgb.train(params, lgb.Dataset(X, y, free_raw_data=True), num_boost_round=1500)
    del X
    print(f"trained 1500 iters in {time.time()-t0:.0f}s", flush=True)

    imp = sorted(zip(FEATS, m.feature_importance("gain")), key=lambda t: -t[1])[:15]
    print("top gain:", [(f, round(float(g))) for f, g in imp], flush=True)

    def predict_anchor(anchor: date, need_target: bool):
        df = prep(anchor, need_target)
        p = np.expm1(np.clip(m.predict(to_X(df)), 0, None))
        yt = df["target"].to_numpy().astype(np.float64) if need_target else None
        return df["user_id"].to_numpy(), p, yt

    # honest off-season validation
    uid_v, pv, yv = predict_anchor(VAL_ANCHOR, True)
    val_score = rmsle(yv, pv)
    print(f"VAL {VAL_ANCHOR} rmsle={val_score:.6f} (off-season, for records)", flush=True)

    aux_score = None
    try:
        _, pa, ya = predict_anchor(AUX_ANCHOR, True)
        aux_score = rmsle(ya, pa)
        print(f"AUX {AUX_ANCHOR} rmsle={aux_score:.6f}", flush=True)
    except Exception as e:  # aux anchor optional
        print(f"AUX eval skipped: {e}", flush=True)

    uid_t, pt, _ = predict_anchor(TEST_ANCHOR, False)
    print(f"test pred mean={pt.mean():.2f} p>1 share={(pt > 1).mean():.4f}; "
          f"val pred mean={pv.mean():.2f}", flush=True)

    save_preds(NAME, "val", uid_v, pv)
    save_preds(NAME, "test", uid_t, pt)
    notes = (f"Feb2025->Mar seasonal specialist: 3 anchors 2025-02-06..20, {len(FEATS)} short<=42d feats, "
             f"tweedie1.3 on log1p nl127 mdl300 lr.05 1500it fixed; off-season val; "
             f"aux {AUX_ANCHOR}={aux_score:.6f}" if aux_score is not None else
             f"Feb2025->Mar seasonal specialist, off-season val")
    log_score(NAME, val_score, notes)
    print(json.dumps({"val_rmsle": round(val_score, 6),
                      "aux_rmsle_2026_01_07": round(aux_score, 6) if aux_score else None,
                      "test_saved": True}), flush=True)


if __name__ == "__main__":
    main()

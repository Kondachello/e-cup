"""Censored fresh anchors: use the newest data instead of throwing it away.

Standard training discards every anchor whose 30-day target window is not fully observed.
Those are the FRESHEST rows in the data - closest to the test regime on a platform that
drifts fast (buyer share 0.3673 -> 0.5407 year over year). This trains on them anyway:
target = GMV over the OBSERVED part of the window, with `obs_days` as a numeric feature
(30 for complete rows, 23/16/9 for censored ones). At inference obs_days = 30.

Unlike train_lagdirect.py, the conditioning here is not for decorrelation - that failed,
and it failed because making the model right made it agree with everyone. Here the goal is
STRENGTH in the test direction, which is the class that the exact contribution formula
says actually pays (a strong model needs only a tiny margin; kostya46 at sm/sb=1.0021
contributes 0.000226 off a margin of 0.00137, twice what lagd28 gets off 0.00308).

MIRROR PROTOCOL, and it is the whole point. The real censored anchors (2026-01-21/28,
2026-02-04, censored at DATA_END) have target windows that overlap the VALIDATION window,
So the effect is measured on a mirror shifted back by 30 days:

    real   : censor at 2026-02-13, anchors 21.01/28.01/04.02, predict window from 14.02
    mirror : censor at 2026-01-14, anchors 22.12/29.12/05.01, predict window from 15.01

In the mirror every censored target ends on 2026-01-14, one day before the validation
window opens. Verified in code below, not by argument.

Usage:
  USE_V2=1 USE_V3=1 python work/scripts/train_censored.py --name cens
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))
from common import HORIZON, TRAIN_PARQUET, VAL_ANCHOR, feature_cols, load_anchor, rmsle
from exp_lib import available_train_anchors, log_score, save_preds

MIRROR_CENSOR = VAL_ANCHOR                                   # pretend the data ends here
MIRROR_ANCHORS = [date(2025, 12, 22), date(2025, 12, 29), date(2026, 1, 5)]


def censored_target(anchor: date, censor: date, uid: np.ndarray):
    """GMV over (anchor, min(anchor+30, censor)] and the number of observed days."""
    last = min(anchor + timedelta(days=HORIZON), censor)
    obs = (last - anchor).days
    assert 0 < obs <= HORIZON
    assert last <= censor, "таргет выходит за цензуру"
    tgt = (
        pl.scan_parquet(TRAIN_PARQUET)
        .filter((pl.col("event_date") > anchor) & (pl.col("event_date") <= last))
        .group_by("user_id").agg(pl.col("gmv").sum().alias("t"))
        .collect()
    )
    d = (pl.DataFrame({"user_id": uid}).join(tgt, on="user_id", how="left")
         .with_columns(pl.col("t").fill_null(0.0)))
    return d["t"].to_numpy().astype(np.float64), obs


def stack(anchors, cols, uid, censor, censored: bool):
    """Rows for a set of anchors -> (X with obs_days, log1p target, row weight).

    EXPOSURE, not just a feature. The first version passed obs_days as a plain input and
    the model collapsed: val 1.6899 -> 2.0310, mean log1p 1.32 against a truth of 2.24.
    A tree can only honour a regime variable by SPLITTING on it, so each censoring level
    gets its own subtree - the rows are partitioned instead of shared, which is the exact
    opposite of the point, and the obs_days=30 subtree ends up with less data than the
    control had. A 9-day window is also not a smaller version of a 30-day one: 70% zeros
    against 46%, a different distribution shape.

    So the target is put on a common 30-day scale before log1p (t * 30/obs, the standard
    exposure offset) and the row is weighted by obs/30, because a 9-day observation is a
    noisier estimate of the 30-day quantity than a 23-day one. obs_days stays as a feature
    so the model can still pick up whatever the rescaling fails to remove.
    """
    Xs, ys, ws = [], [], []
    for a in anchors:
        f = load_anchor(a, columns=["user_id", "target"] + cols).sort("user_id")
        assert np.array_equal(f["user_id"].to_numpy(), uid)
        if censored:
            # the stored target spans the FULL 30 days and would reach into the val window;
            # recompute it over the observed part only
            t, obs = censored_target(a, censor, uid)
        else:
            t = np.clip(f["target"].to_numpy().astype(np.float64), 0, None)
            obs = HORIZON
        X = np.column_stack([f.select(cols).to_numpy().astype(np.float32),
                             np.full(len(uid), float(obs), dtype=np.float32)])
        t30 = np.clip(t, 0, None) * (HORIZON / obs)
        Xs.append(X)
        ys.append(np.log1p(t30))
        ws.append(np.full(len(uid), obs / HORIZON, dtype=np.float64))
        print(f"    {a} obs={obs:2d}d  нулей {float((t == 0).mean()):.3f}  "
              f"вес {obs / HORIZON:.2f}  mean_lp {float(np.log1p(t30).mean()):.3f}", flush=True)
        del f
    return np.vstack(Xs), np.concatenate(ys), np.concatenate(ws)


def fit(X, y, Xv, yv, seed, rounds, w=None):
    import lightgbm as lgb
    p = dict(objective="tweedie", tweedie_variance_power=1.45, metric="rmse",
             learning_rate=0.05, num_leaves=255, min_data_in_leaf=300,
             feature_fraction=0.75, bagging_fraction=0.8, bagging_freq=1,
             lambda_l2=5.0, max_bin=127, num_threads=7, seed=seed, verbosity=-1)
    dtr = lgb.Dataset(X, y, weight=w, free_raw_data=True)
    dv = lgb.Dataset(Xv, yv, reference=dtr, free_raw_data=True)
    m = lgb.train(p, dtr, num_boost_round=rounds, valid_sets=[dv],
                  callbacks=[lgb.early_stopping(200, verbose=False), lgb.log_evaluation(400)])
    return m, m.best_iteration


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="cens")
    ap.add_argument("--rounds", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    assert os.environ.get("USE_V2") and os.environ.get("USE_V3")
    t0 = time.time()

    val = load_anchor(VAL_ANCHOR).sort("user_id")
    cols = feature_cols(val)
    uid = val["user_id"].to_numpy()
    yv_raw = np.clip(val["target"].to_numpy().astype(np.float64), 0, None)
    yv = np.log1p(yv_raw)
    Xv = np.column_stack([val.select(cols).to_numpy().astype(np.float32),
                          np.full(len(uid), float(HORIZON), dtype=np.float32)])
    del val

    gap_cut = VAL_ANCHOR - timedelta(days=30)
    base_anchors = [a for a in available_train_anchors()
                    if a <= gap_cut and a not in MIRROR_ANCHORS]
    print(f"обычные якоря (gap30): {base_anchors[0]}..{base_anchors[-1]} ({len(base_anchors)})",
          flush=True)

    # leakage guard, stated in code: no censored target may reach the val window
    for a in MIRROR_ANCHORS:
        last = min(a + timedelta(days=HORIZON), MIRROR_CENSOR)
        assert last < VAL_ANCHOR + timedelta(days=1), f"{a}: таргет достаёт до валидации"
    print(f"зеркало: цензура {MIRROR_CENSOR}, валидация с {VAL_ANCHOR + timedelta(days=1)}",
          flush=True)

    print("  контроль:", flush=True)
    Xb, yb, wb = stack(base_anchors, cols, uid, MIRROR_CENSOR, censored=False)
    mc, itc = fit(Xb, yb, Xv, yv, args.seed, args.rounds, wb)
    pc = np.expm1(np.clip(mc.predict(Xv), 0, None))
    sc = rmsle(yv_raw, pc)
    save_preds(f"{args.name}_ctl", "val", uid, pc)
    log_score(f"{args.name}_ctl", sc, f"censored-anchors control, {len(base_anchors)} якорей, it={itc}")

    print("  + цензурированные свежие:", flush=True)
    Xf, yf, wf = stack(MIRROR_ANCHORS, cols, uid, MIRROR_CENSOR, censored=True)
    Xa = np.vstack([Xb, Xf]); ya = np.concatenate([yb, yf]); wa = np.concatenate([wb, wf])
    del Xb, yb, Xf, yf
    ma, ita = fit(Xa, ya, Xv, yv, args.seed, args.rounds, wa)
    pa = np.expm1(np.clip(ma.predict(Xv), 0, None))
    sa = rmsle(yv_raw, pa)
    save_preds(f"{args.name}_on", "val", uid, pa)
    log_score(f"{args.name}_on", sa,
              f"censored-anchors +{len(MIRROR_ANCHORS)} свежих (obs 23/16/9) c obs_days-признаком, it={ita}")

    print(f"\nконтроль {sc:.6f}  ->  с цензурированными {sa:.6f}   дельта {sa - sc:+.6f}")
    print(f"[DONE] {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

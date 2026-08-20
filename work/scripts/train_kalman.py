"""Kalman predictor: the user's own 30-day sums through a state-space filter.

The most alien mechanism available to us. It uses NO engineered features - not one of
the 203 - only the user's own history of non-overlapping 30-day log-GMV totals. That is
exactly why it is worth building: the hull theorem says a prediction that is a function
of our feature set contributes nothing, and this one provably is not.

The process model is Zhenya's (work/reports/zhenya_report.md), fitted from the decay of
target autocorrelation across non-overlapping 30-day windows:

    lp_t = mu + s_t + e_t ,   s_t = lam*s_{t-1} + eta ,   var(eta) = q(1-lam^2)
    var(mu) = p ,  var(e) = 1-p-q          (normalised so var(lp) = 1)

He fitted p=0.4162 q=0.1796 lam=0.7887 and stopped there: the numbers were a CEILING
estimate, never a predictor. Turning them into one needs a two-state filter per user,
state = [mu, s], because mu is constant while s decays - a single steady-state gain
cannot track both. Users have only ~13 windows, so the transient dominates and the
filter is run explicitly rather than at its steady state.

Prediction for the window after the anchor: lp_hat = mu_hat + lam * s_hat.

Params are REFITTED here on the same windows the filter runs on, by matching the
autocorrelation profile, instead of importing five numbers fitted elsewhere. Pass
--params-from-report to use Zhenya's instead and compare.

Emits val + test under the exp_lib contract.

Usage:
  python work/scripts/train_kalman.py --name kalman
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))
from common import HORIZON, TEST_ANCHOR, TRAIN_PARQUET, VAL_ANCHOR, rmsle, user_universe
from exp_lib import log_score, save_preds

REPORT_PARAMS = (0.4162, 0.1796, 0.7887)      # p, q, lam from zhenya_report.md


def windows(anchor: date, n_win: int) -> pl.DataFrame:
    """Non-overlapping 30-day log-GMV totals per user, most recent window last.

    Window w counts days (anchor - 30*(w+1), anchor - 30*w]; only history, never the
    target window. Users absent from a window get 0, which is a real observation here
    (they bought nothing), not a missing value.
    """
    lo = anchor - timedelta(days=HORIZON * n_win)
    df = (
        pl.scan_parquet(TRAIN_PARQUET)
        .filter((pl.col("event_date") > lo) & (pl.col("event_date") <= anchor))
        .with_columns(
            ((pl.lit(anchor) - pl.col("event_date")).dt.total_days() - 1)
            .floordiv(HORIZON).alias("w")
        )
        .group_by("user_id", "w")
        .agg(pl.col("gmv").sum().alias("g"))
        .collect()
    )
    uni = user_universe()
    grid = uni.join(pl.DataFrame({"w": np.arange(n_win, dtype=np.int64)}), how="cross")
    out = (
        grid.join(df, on=["user_id", "w"], how="left")
        .with_columns(pl.col("g").fill_null(0.0).log1p().alias("lp"))
        .sort("user_id", "w", descending=[False, True])      # oldest window first
    )
    return out


def fit_params(M: np.ndarray) -> tuple[float, float, float]:
    """p, q, lam by matching the autocorrelation profile r(k) = p + q*lam^k."""
    from scipy.optimize import least_squares
    Z = M - M.mean(0)
    sd = Z.std(0)
    sd[sd == 0] = 1.0
    Z = Z / sd
    ks, rs = [], []
    for k in range(1, min(6, M.shape[1])):
        a, b = Z[:, :-k].ravel(), Z[:, k:].ravel()
        ks.append(k); rs.append(float(np.mean(a * b)))
    ks, rs = np.array(ks, float), np.array(rs)
    sol = least_squares(lambda t: t[0] + t[1] * t[2] ** ks - rs, [0.40, 0.16, 0.8],
                        bounds=([0, 0, 0], [1, 1, 0.999]))
    print(f"  r(k) замер: {' '.join(f'{k}:{r:.4f}' for k, r in zip(ks, rs))}")
    return tuple(sol.x)


def kalman(M: np.ndarray, p: float, q: float, lam: float) -> np.ndarray:
    """Two-state filter over users in parallel. Returns predicted next lp (standardised)."""
    R = max(1.0 - p - q, 1e-6)
    Q = q * (1.0 - lam ** 2)
    U, T = M.shape
    x = np.zeros((U, 2))                                  # [mu, s]
    P = np.zeros((U, 2, 2))
    P[:, 0, 0], P[:, 1, 1] = p, q
    for t in range(T):
        # predict: mu stays, s decays
        x[:, 1] *= lam
        P[:, 0, 1] *= lam
        P[:, 1, 0] *= lam
        P[:, 1, 1] = P[:, 1, 1] * lam ** 2 + Q
        # update with observation lp_t = mu + s + e
        S = P[:, 0, 0] + P[:, 0, 1] + P[:, 1, 0] + P[:, 1, 1] + R
        K0 = (P[:, 0, 0] + P[:, 0, 1]) / S
        K1 = (P[:, 1, 0] + P[:, 1, 1]) / S
        innov = M[:, t] - (x[:, 0] + x[:, 1])
        x[:, 0] += K0 * innov
        x[:, 1] += K1 * innov
        H0 = P[:, 0, 0] + P[:, 1, 0]                       # (H P) row for column 0
        H1 = P[:, 0, 1] + P[:, 1, 1]
        P[:, 0, 0] -= K0 * H0
        P[:, 0, 1] -= K0 * H1
        P[:, 1, 0] -= K1 * H0
        P[:, 1, 1] -= K1 * H1
    return x[:, 0] + lam * x[:, 1]


def build(anchor: date, n_win: int, params):
    w = windows(anchor, n_win)
    uid = w["user_id"].unique(maintain_order=True).to_numpy()
    M = w["lp"].to_numpy().reshape(len(uid), n_win).astype(np.float64)
    mean, sd = M.mean(), M.std()
    Z = (M - mean) / sd
    if params is None:
        params = fit_params(M)
        print(f"  подгонка на этих окнах: p={params[0]:.4f} q={params[1]:.4f} lam={params[2]:.4f}")
    pred_z = kalman(Z, *params)
    return uid, np.expm1(np.clip(pred_z * sd + mean, 0, None)), params


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="kalman")
    ap.add_argument("--n-win", type=int, default=12)
    ap.add_argument("--params-from-report", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    params = REPORT_PARAMS if args.params_from_report else None

    print(f"валидация, якорь {VAL_ANCHOR}, окон {args.n_win}")
    uid_v, pv, params = build(VAL_ANCHOR, args.n_win, params)
    tgt = (
        pl.scan_parquet(TRAIN_PARQUET)
        .filter((pl.col("event_date") > VAL_ANCHOR)
                & (pl.col("event_date") <= VAL_ANCHOR + timedelta(days=HORIZON)))
        .group_by("user_id").agg(pl.col("gmv").sum().alias("y")).collect()
    )
    y = (user_universe().join(tgt, on="user_id", how="left")
         .with_columns(pl.col("y").fill_null(0.0)).sort("user_id"))["y"].to_numpy()
    assert np.array_equal(uid_v, np.sort(uid_v))
    s = rmsle(y, pv)
    save_preds(args.name, "val", uid_v, pv)
    note = (f"kalman mu+AR(1) по {args.n_win} окнам x30д, БЕЗ признаков; "
            f"p={params[0]:.4f} q={params[1]:.4f} lam={params[2]:.4f}")
    log_score(args.name, s, note)

    print(f"тест, якорь {TEST_ANCHOR}")
    uid_t, pt, _ = build(TEST_ANCHOR, args.n_win, params)
    save_preds(args.name, "test", uid_t, pt)
    print(f"[DONE] {args.name} val_rmsle={s:.6f} {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

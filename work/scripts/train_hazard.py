"""Daily hazard: a different unit of observation, and three different formulations from it.

WHY THIS AND NOT ANOTHER REGRESSOR. Every model in the project predicts the 30-day sum
from one feature vector per (user, anchor). That is ~3.5M training rows whose targets
overlap 77% between neighbouring anchors, and -- more importantly -- every such model is
a function of the same 203 aggregates, so it lies inside the blend's hull and its margin
is zero by construction (KNOWLEDGE.md, the hull identity).

Here the unit is a (user, DAY) pair and the label is "did this user order on this day".
That is ~90M labels instead of 3.5M, and they do not overlap: each day is its own event.
The 30-day window is then assembled from daily risks rather than predicted directly.

NO GAP IS NEEDED, and this is not an oversight. The trainers drop 30 days because their
labels are 30-day windows that would straddle the validation window. Daily labels do not:
every training day <= anchor lies strictly before the target window. So this model sees
the anchor's own last month, which the tabular models are forbidden to learn from.

THREE FORMULATIONS, ONE FIT (task 4 asks which formulation errs differently, so all three
are produced and measured separately):

  surv  analytic survival. P(no order in the window) = prod_t (1 - h_t) with the state
        "no order yet", which is exact for the first-purchase chain. Expected count and
        the amount head give the level. This is "time to next purchase, integrated".

  pb    Poisson-binomial. Treat the 30 days as conditionally independent given the anchor,
        take the exact distribution of the ORDER COUNT implied by {h_t}, and integrate
        log1p over it. This is "predict the distribution, not the mean".

  mc    Monte-Carlo simulation. Draw each day, reset recency after a simulated order, draw
        amounts from the amount head plus an empirical residual, and average log1p of the
        realised sum over scenarios. This is "generate thirty days many times". It is the
        only head that handles the state dependence honestly.

The three share a fit but not an estimator, so their errors are not the same object --
which is the point of the exercise.

Seasonality enters for free: the hazard is evaluated per calendar day, with last year's
platform-wide order rate on the same weekday as a feature. The test window contains
8 March; the model sees the 2025 analogue of every one of its days rather than a single
global shift.

Contract: exp_lib. Saves work/preds/NAME_{surv,pb,mc}_{val,test}.parquet + scores.tsv rows.

Usage:
  python work/scripts/train_hazard.py --name hz_smoke --days 120 --user-frac 0.15
  python work/scripts/train_hazard.py --name hz_v1
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))
from common import (DATA_START, HORIZON, TEST_ANCHOR, TRAIN_PARQUET, VAL_ANCHOR,
                    load_anchor, rmsle, user_universe)
from exp_lib import log_score, save_preds

REC_CAP = 400.0          # days-since-X is capped here; "never" maps to the cap
YEAR = 364               # 52 weeks: keeps the weekday aligned when looking back a year
ORD_W = [7, 14, 30, 90, 365]
GMV_W = [7, 30, 90, 365]
CART_W = [7, 30]
ACT_W = [7, 30, 90]
# recency values the hazard grid is evaluated at; simulation snaps to the nearest one
REC_GRID = [0, 1, 2, 4, 7, 14, 30, 60, 120, 400]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------------------
# daily matrices


def build_daily(uni: np.ndarray, d0: date, n_days: int):
    """Dense [users, days] matrices. Column j is the day d0 + j."""
    ev = (
        pl.scan_parquet(TRAIN_PARQUET)
        .select(["event_date", "user_id", "gmv", "to_ord", "to_cart"])
        .filter(pl.col("event_date") >= d0)
        .collect()
    )
    rows = np.searchsorted(uni, ev["user_id"].to_numpy())
    assert np.all(uni[np.clip(rows, 0, len(uni) - 1)] == ev["user_id"].to_numpy()), "unknown user_id"
    ed = ev["event_date"].to_numpy().astype("datetime64[D]")
    cols = (ed - np.datetime64(d0, "D")).astype("timedelta64[D]").astype(np.int64)
    keep = (cols >= 0) & (cols < n_days)
    rows, cols = rows[keep], cols[keep]

    U = len(uni)
    gmv = np.zeros((U, n_days), dtype=np.float32)
    ordc = np.zeros((U, n_days), dtype=np.float32)
    cart = np.zeros((U, n_days), dtype=np.float32)
    gmv[rows, cols] = ev["gmv"].to_numpy()[keep]
    ordc[rows, cols] = ev["to_ord"].to_numpy()[keep]
    cart[rows, cols] = ev["to_cart"].to_numpy()[keep]
    act = np.zeros((U, n_days), dtype=np.float32)
    act[rows, cols] = 1.0
    log(f"daily matrices: {U} x {n_days}, {len(rows):,} active cells")
    return gmv, ordc, cart, act


def prefix(mat: np.ndarray) -> np.ndarray:
    """cs[:, d] = sum over days [0, d-1], so a window ending the day BEFORE d is a subtraction."""
    cs = np.empty((mat.shape[0], mat.shape[1] + 1), dtype=np.float32)
    cs[:, 0] = 0.0
    np.cumsum(mat, axis=1, out=cs[:, 1:])
    return cs


def last_index(mask: np.ndarray) -> np.ndarray:
    """li[:, d] = last day index <= d where mask is true, else -1."""
    idx = np.where(mask, np.arange(mask.shape[1], dtype=np.int32)[None, :], np.int32(-1))
    return np.maximum.accumulate(idx, axis=1)


class Panel:
    """Everything needed to produce features for any (user, day) as of the END of day-1."""

    def __init__(self, uni: np.ndarray, d0: date, n_days: int):
        gmv, ordc, cart, act = build_daily(uni, d0, n_days)
        self.d0, self.n_days, self.U = d0, n_days, len(uni)
        self.ord_raw, self.gmv_raw = ordc, gmv
        self.cs_gmv = prefix(gmv)
        self.cs_ordday = prefix((ordc > 0).astype(np.float32))
        self.cs_ordcnt = prefix(ordc)
        self.cs_cartday = prefix((cart > 0).astype(np.float32))
        self.cs_act = prefix(act)
        self.li_ord = last_index(ordc > 0)
        self.li_cart = last_index(cart > 0)
        self.li_act = last_index(act > 0)
        # platform-wide daily order rate, used a year later as a seasonal covariate
        self.day_rate = (ordc > 0).mean(axis=0).astype(np.float32)
        self.first_active = np.argmax(self.cs_act[:, 1:] > 0, axis=1).astype(np.int32)
        del cart, act
        log("prefix sums and recency built")

    def win(self, cs: np.ndarray, d: int, w: int) -> np.ndarray:
        return cs[:, d] - cs[:, max(d - w, 0)]

    def rec(self, li: np.ndarray, d: int) -> np.ndarray:
        prev = li[:, d - 1] if d >= 1 else np.full(self.U, -1, dtype=np.int32)
        r = np.where(prev < 0, REC_CAP, d - prev).astype(np.float32)
        return np.minimum(r, REC_CAP)

    def features(self, d_hist: int, d_cal: int | None = None,
                 rec_ord: np.ndarray | None = None) -> np.ndarray:
        """History as of d_hist (uses days < d_hist), calendar of d_cal.

        The split matters when predicting a window: history must stay FROZEN at the anchor
        (looking at day anchor+15's history would read the future), while the calendar and
        the recency clock advance day by day.
        """
        d = d_hist
        d_cal = d_hist if d_cal is None else d_cal
        cols = []
        for w in ORD_W:
            cols.append(self.win(self.cs_ordday, d, w))
        cols.append(self.win(self.cs_ordcnt, d, 30))
        for w in GMV_W:
            cols.append(np.log1p(self.win(self.cs_gmv, d, w)))
        for w in CART_W:
            cols.append(self.win(self.cs_cartday, d, w))
        for w in ACT_W:
            cols.append(self.win(self.cs_act, d, w))

        ro = self.rec(self.li_ord, d) if rec_ord is None else rec_ord
        cols += [ro, self.rec(self.li_cart, d), self.rec(self.li_act, d)]

        ord365 = cols[ORD_W.index(365)]
        gap = 365.0 / np.maximum(ord365, 1.0)                      # mean days between orders
        cols.append(gap)
        cols.append(ro / gap)                                      # overdueness
        cols.append(np.log1p(self.win(self.cs_gmv, d, 365)) - np.log1p(np.maximum(ord365, 1.0)))
        cols.append((d - self.first_active).astype(np.float32))    # tenure
        # calendar: of the day being predicted, not of the frozen history
        cols.append(np.full(self.U, float(d_cal % 7), dtype=np.float32))
        ly = d_cal - YEAR
        r_ly = self.day_rate[ly] if 0 <= ly < self.n_days else np.nan
        cols.append(np.full(self.U, r_ly, dtype=np.float32))
        return np.column_stack(cols).astype(np.float32)

    @property
    def feature_names(self) -> list[str]:
        n = [f"ord_days_{w}" for w in ORD_W] + ["ord_cnt_30"]
        n += [f"gmv_{w}" for w in GMV_W] + [f"cart_days_{w}" for w in CART_W]
        n += [f"act_days_{w}" for w in ACT_W]
        n += ["rec_ord", "rec_cart", "rec_act", "gap_mean", "overdue", "gmv_per_ordday",
              "tenure", "dow", "season_ly"]
        return n

    def day_index(self, d: date) -> int:
        return (d - self.d0).days


# ---------------------------------------------------------------------------------------
# training


def sample_rows(panel: Panel, last_day: int, n_days_back: int, stride: int, user_frac: float,
                rng: np.random.Generator):
    """(day, user-subset) pairs; features as of the day, label = ordered that day."""
    days = list(range(max(1, last_day - n_days_back + 1), last_day + 1, stride))
    X, y_haz, amt_rows = [], [], []
    n_sub = max(1, int(panel.U * user_frac))
    for d in days:
        sub = rng.choice(panel.U, n_sub, replace=False) if n_sub < panel.U else np.arange(panel.U)
        f = panel.features(d)[sub]
        lab = (panel.ord_raw[sub, d] > 0).astype(np.float32)
        X.append(f)
        y_haz.append(lab)
        pos = lab > 0
        if pos.any():
            amt_rows.append((f[pos], np.log1p(panel.gmv_raw[sub[pos], d]).astype(np.float32)))
    X = np.vstack(X)
    y_haz = np.concatenate(y_haz)
    Xa = np.vstack([a for a, _ in amt_rows])
    ya = np.concatenate([b for _, b in amt_rows])
    log(f"train rows: hazard {len(y_haz):,} (rate {y_haz.mean():.4f}), amount {len(ya):,}")
    return X, y_haz, Xa, ya


def fit_models(X, y, Xa, ya, names, args, rng):
    haz = lgb.LGBMClassifier(
        n_estimators=args.trees, learning_rate=args.lr, num_leaves=args.leaves,
        min_child_samples=200, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
        n_jobs=args.threads, random_state=args.seed, verbose=-1)
    haz.fit(X, y)
    amt = lgb.LGBMRegressor(
        n_estimators=max(200, args.trees // 3), learning_rate=args.lr, num_leaves=args.leaves,
        min_child_samples=100, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
        n_jobs=args.threads, random_state=args.seed, verbose=-1)
    amt.fit(Xa, ya)
    resid = ya - amt.predict(Xa)
    top = sorted(zip(names, haz.feature_importances_), key=lambda t: -t[1])[:5]
    log(f"models fitted; amount residual sd={resid.std():.3f}; "
        f"top hazard features: {', '.join(n for n, _ in top)}")
    return haz, amt, resid.astype(np.float32)


# ---------------------------------------------------------------------------------------
# window assembly


def window_hazards(panel: Panel, haz, anchor_idx: int, horizon: int, threads: int):
    """h[grid, day] for every user: hazard on each window day at each recency on the grid.

    History is frozen at the anchor and only the calendar and the recency clock advance.
    Verified directly: with d_hist fixed, features() output changes in exactly two columns
    (dow, season_ly) as d_cal moves across the window.
    """
    U = panel.U
    grid = np.array(REC_GRID, dtype=np.float32)
    out = np.empty((len(grid), horizon, U), dtype=np.float32)
    base_rec = panel.rec(panel.li_ord, anchor_idx + 1)
    for j, r in enumerate(grid):
        for t in range(horizon):
            d = anchor_idx + 1 + t
            # recency the "no order yet" chain would have, or the grid value after an order
            # grid slot REC_CAP carries the "no order yet in the window" chain
            # BUGFIX 22.08: non-cap slots must be evaluated AT the grid value. head_mc
            # re-picks the slot every day from its own `since` clock (line ~305), so the
            # old `r + t` double-counted elapsed days: after an in-window order the hazard
            # was read at recency r + day-number (up to +29), understating repeat orders.
            rec = np.minimum(base_rec + t, REC_CAP) if r == REC_CAP else np.full(U, r, np.float32)
            f = panel.features(anchor_idx + 1, d_cal=d, rec_ord=np.minimum(rec, REC_CAP))
            out[j, t] = haz.predict_proba(f, num_threads=threads)[:, 1]
    return out, base_rec


def head_surv(h_nostate: np.ndarray, mu_amt: np.ndarray) -> np.ndarray:
    """P(at least one order) x expected log-size of the window's spend."""
    surv = np.clip(1.0 - h_nostate, 1e-9, 1.0)
    p_buy = 1.0 - np.prod(surv, axis=0)
    n_exp = h_nostate.sum(axis=0)
    n_cond = n_exp / np.maximum(p_buy, 1e-9)                 # expected orders GIVEN >= 1
    return p_buy * np.log1p(np.expm1(mu_amt) * n_cond)


def head_pb(h_nostate: np.ndarray, mu_amt: np.ndarray, kmax: int = 40) -> np.ndarray:
    """Exact order-count distribution under conditional independence, log1p integrated over it."""
    U = h_nostate.shape[1]
    pk = np.zeros((kmax + 1, U), dtype=np.float64)
    pk[0] = 1.0
    for t in range(h_nostate.shape[0]):
        h = h_nostate[t].astype(np.float64)
        pk[1:] = pk[1:] * (1 - h) + pk[:-1] * h
        pk[0] = pk[0] * (1 - h)
    pk /= np.maximum(pk.sum(axis=0, keepdims=True), 1e-12)   # tail beyond kmax is truncated
    a = np.expm1(mu_amt)
    ks = np.arange(kmax + 1)[:, None]
    return (pk * np.log1p(a[None, :] * ks)).sum(axis=0)


def head_mc(h_grid: np.ndarray, base_rec: np.ndarray, mu_amt: np.ndarray, resid: np.ndarray,
            scenarios: int, rng: np.random.Generator) -> np.ndarray:
    """Simulate the window scenario by scenario, resetting recency after every drawn order."""
    grid = np.array(REC_GRID, dtype=np.float32)
    n_grid, horizon, U = h_grid.shape
    # since-days -> nearest grid slot, precomputed (since can only be 0..horizon-1)
    slot_of = np.abs(grid[None, :-1] - np.arange(horizon + 1, dtype=np.float32)[:, None]).argmin(1)
    acc = np.zeros(U, dtype=np.float64)
    for _ in range(scenarios):
        total = np.zeros(U, dtype=np.float64)
        since = np.full(U, -1, dtype=np.int32)               # days since the last drawn order
        for t in range(horizon):
            # -1 means "no order yet in the window" -> the last grid slot (the no-state chain)
            slot = np.where(since < 0, n_grid - 1, slot_of[np.clip(since, 0, horizon)])
            h = h_grid[slot, t, np.arange(U)]
            hit = rng.random(U) < h
            if hit.any():
                eps = resid[rng.integers(0, len(resid), int(hit.sum()))]
                total[hit] += np.expm1(mu_amt[hit] + eps)
                since[hit] = 0
            since[~hit & (since >= 0)] += 1
        acc += np.log1p(np.clip(total, 0, None))
    return acc / scenarios


# ---------------------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="hz_v1")
    ap.add_argument("--days", type=int, default=300, help="training days back from the anchor")
    ap.add_argument("--stride", type=int, default=2, help="use every Nth day")
    ap.add_argument("--user-frac", type=float, default=0.35)
    ap.add_argument("--trees", type=int, default=600)
    ap.add_argument("--leaves", type=int, default=127)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--scenarios", type=int, default=48)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val-only", action="store_true", help="skip the test anchor")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    uni = user_universe()["user_id"].to_numpy()
    assert np.all(np.diff(uni) > 0), "user universe must be sorted"
    n_days = (TEST_ANCHOR - DATA_START).days + 1
    panel = Panel(uni, DATA_START, n_days)
    names = panel.feature_names

    anchors = [VAL_ANCHOR] if args.val_only else [VAL_ANCHOR, TEST_ANCHOR]
    preds = {}
    for anchor in anchors:
        ai = panel.day_index(anchor)
        log(f"=== anchor {anchor} (day {ai}); training on days <= {anchor} ===")
        X, y, Xa, ya = sample_rows(panel, ai, args.days, args.stride, args.user_frac, rng)
        haz, amt, resid = fit_models(X, y, Xa, ya, names, args, rng)
        del X, y, Xa, ya

        t0 = time.time()
        h_grid, base_rec = window_hazards(panel, haz, ai, HORIZON, args.threads)
        log(f"window hazards in {time.time() - t0:.0f}s; mean daily risk={h_grid[-1].mean():.5f}")
        mu_amt = amt.predict(panel.features(ai + 1, d_cal=ai + 1),
                             num_threads=args.threads).astype(np.float64)

        h_ns = h_grid[-1]                                     # the "no order yet" chain
        lp = {"surv": head_surv(h_ns, mu_amt),
              "pb": head_pb(h_ns, mu_amt),
              "mc": head_mc(h_grid, base_rec, mu_amt, resid, args.scenarios, rng)}
        preds[anchor] = {k: np.expm1(np.clip(v, 0, None)) for k, v in lp.items()}
        for k, v in lp.items():
            log(f"  head {k}: mean log1p pred={v.mean():.4f}")

    val = load_anchor(VAL_ANCHOR, columns=["user_id", "target"]).sort("user_id")
    assert np.array_equal(val["user_id"].to_numpy(), uni)
    y_val = val["target"].to_numpy().astype(np.float64)
    for head in ("surv", "pb", "mc"):
        nm = f"{args.name}_{head}"
        r = rmsle(y_val, preds[VAL_ANCHOR][head])
        save_preds(nm, "val", uni, preds[VAL_ANCHOR][head])
        if TEST_ANCHOR in preds:
            save_preds(nm, "test", uni, preds[TEST_ANCHOR][head])
        log_score(nm, r, f"daily hazard {head}: days={args.days} stride={args.stride} "
                        f"uf={args.user_frac} trees={args.trees} scen={args.scenarios}")
    print(f"\nnext: calibrate.py --pred {args.name}_mc --bins 24, then err_corr.py {args.name}_mc_cal")


if __name__ == "__main__":
    main()

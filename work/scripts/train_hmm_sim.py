"""Generative user simulator (hmmsim): per-segment discrete HMM over day types
+ lognormal buy-size emission with empirical-Bayes per-user mean.

Mechanism instead of regression: learn day-level behavior dynamics, then
Monte-Carlo the 30d future to get E[log1p(sum gmv)] exactly (the metric's
per-user optimum), rather than approximating it with a supervised regressor.

Day observation o_t in {0: inactive (no row), 1: active w/o purchase, 2: buy (gmv>0)}.
Hidden state s_t in K states (dormant .. buyer). 4 user segments by buy-days in
last 90d (0 / 1 / 2-4 / 5+); each segment gets its own (A, B, pi) fitted by EM
on the last WIN=120 days <= anchor. Per-user initial state = filtered forward
distribution at the anchor day. Buy size: v = log1p(gmv_day) ~ N(mu_u, sig_u),
mu_u = empirical-Bayes blend of personal mean check and segment mean (K0=5),
sig_u = blend of within-user and pooled segment std with the same weight.

No supervised training on targets; all parameters use only data <= anchor,
so no gap-30 is required. Honest val = forecast from VAL_ANCHOR vs observed.

Usage:
  smoke: THREADS=3 .venv/bin/python work/scripts/train_hmm_sim.py --smoke \
             --users 50000 --states 3 --sims 200
  full : THREADS=6 .venv/bin/python work/scripts/train_hmm_sim.py \
             --name hmmsim --states 3 --sims 500 --splits val,test --em-cap 25000
"""
from __future__ import annotations

import os

_T = os.environ.get("THREADS", "3")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, _T)

import argparse
import json
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))
from common import (  # noqa: E402
    TEST_ANCHOR, TRAIN_PARQUET, VAL_ANCHOR, WORK, load_anchor, rmsle, user_universe,
)
from exp_lib import PREDS_DIR, log_score, save_preds  # noqa: E402

HORIZON = 30
N_SYM = 3
SEG_EDGES = [1, 2, 5]  # buy_days_90 bins: 0 | 1 | 2-4 | 5+
VCAP = 12.5            # cap on sampled log1p day-gmv (max observed 73830 -> 11.2)


def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------- data prep

def build_inputs(anchor: date, users: np.ndarray, win: int, k0: float):
    """Day-type matrix O (N, win) for [anchor-win+1 .. anchor], segment ids and
    per-user buy-value params from full history <= anchor."""
    start_win = anchor - timedelta(days=win - 1)
    lf = pl.scan_parquet(TRAIN_PARQUET).select("user_id", "event_date", "gmv")
    lf = lf.filter(pl.col("event_date") <= anchor)
    lf = lf.filter((pl.col("gmv") > 0) | (pl.col("event_date") >= start_win))
    df = lf.collect(engine="streaming")
    df = df.join(pl.DataFrame({"user_id": users}), on="user_id", how="semi")
    df = df.with_columns(
        (pl.col("event_date") - pl.lit(start_win)).dt.total_days().alias("d")
    )
    uid = df["user_id"].to_numpy()
    d = df["d"].to_numpy().astype(np.int64)
    gmv = df["gmv"].to_numpy()
    ridx = np.searchsorted(users, uid)

    N = len(users)
    O = np.zeros((N, win), dtype=np.int8)
    m = d >= 0
    O[ridx[m], d[m]] = 1 + (gmv[m] > 0).astype(np.int8)

    buy = gmv > 0
    rb = ridx[buy]
    v = np.log1p(gmv[buy])
    n_u = np.bincount(rb, minlength=N).astype(np.float64)
    s_u = np.bincount(rb, weights=v, minlength=N)
    ss_u = np.bincount(rb, weights=v * v, minlength=N)

    bd90 = np.bincount(ridx[buy & (d >= win - 90)], minlength=N)
    seg = np.digitize(bd90, SEG_EDGES)

    # global fallbacks
    ng_all = n_u.sum()
    mu_all = s_u.sum() / max(ng_all, 1.0)
    var_all = max(ss_u.sum() / max(ng_all, 1.0) - mu_all**2, 0.35**2)

    mu_u = np.empty(N)
    sig_u = np.empty(N)
    seg_val = {}
    for g in range(4):
        mask = seg == g
        ng = n_u[mask].sum()
        if ng < 50:
            mu_g, var_pool, var_within = mu_all, var_all, var_all
        else:
            mu_g = s_u[mask].sum() / ng
            var_pool = max(ss_u[mask].sum() / ng - mu_g**2, 0.35**2)
            m2 = mask & (n_u >= 2)
            dof = (n_u[m2] - 1).sum()
            if dof < 50:
                var_within = var_pool
            else:
                rss = (ss_u[m2] - s_u[m2] ** 2 / n_u[m2]).sum()
                var_within = min(max(rss / dof, 0.35**2), var_pool)
        w = n_u[mask] / (n_u[mask] + k0)
        mu_u[mask] = (s_u[mask] + k0 * mu_g) / (n_u[mask] + k0)
        sig_u[mask] = np.sqrt(w * var_within + (1.0 - w) * var_pool)
        seg_val[g] = (float(mu_g), float(np.sqrt(var_pool)), float(np.sqrt(var_within)))
    return O, seg, mu_u, sig_u, seg_val


# ---------------------------------------------------------------- discrete HMM

def _init_params(K: int):
    if K == 3:
        B = np.array([[0.96, 0.037, 0.003],
                      [0.35, 0.620, 0.030],
                      [0.20, 0.550, 0.250]])
    else:
        B = np.array([[0.97, 0.028, 0.002],
                      [0.60, 0.385, 0.015],
                      [0.20, 0.750, 0.030],
                      [0.15, 0.550, 0.300]])[:K]
    A = np.full((K, K), 0.1 / (K - 1))
    np.fill_diagonal(A, 0.9)
    pi = np.full(K, 1.0 / K)
    return A, B, pi


def em_fit(O: np.ndarray, K: int, iters: int = 40, tol: float = 1e-5,
           batch: int = 8192, tag: str = ""):
    N, T = O.shape
    A, B, pi = _init_params(K)
    prev = -np.inf
    for it in range(iters):
        trans = np.zeros((K, K))
        emis = np.zeros((K, N_SYM))
        pi_n = np.zeros(K)
        ll = 0.0
        for b0 in range(0, N, batch):
            Ob = O[b0:b0 + batch]
            nb = Ob.shape[0]
            alphas = np.empty((T, nb, K))
            c = np.empty((T, nb))
            a = pi[None, :] * B[:, Ob[:, 0]].T
            c[0] = a.sum(1)
            alphas[0] = a / c[0][:, None]
            for t in range(1, T):
                a = (alphas[t - 1] @ A) * B[:, Ob[:, t]].T
                c[t] = a.sum(1)
                alphas[t] = a / c[t][:, None]
            ll += float(np.log(c).sum())
            beta = np.ones((nb, K))
            gam = alphas[T - 1] * beta
            for s in range(N_SYM):
                msk = Ob[:, T - 1] == s
                if msk.any():
                    emis[:, s] += gam[msk].sum(0)
            for t in range(T - 2, -1, -1):
                w = (B[:, Ob[:, t + 1]].T * beta) / c[t + 1][:, None]
                trans += A * (alphas[t].T @ w)
                beta = w @ A.T
                gam = alphas[t] * beta
                for s in range(N_SYM):
                    msk = Ob[:, t] == s
                    if msk.any():
                        emis[:, s] += gam[msk].sum(0)
            pi_n += gam.sum(0)
        A = trans + 1e-8
        A /= A.sum(1, keepdims=True)
        B = emis + 1e-6
        B /= B.sum(1, keepdims=True)
        pi = pi_n + 1e-8
        pi /= pi.sum()
        per_obs = ll / (N * T)
        if it >= 2 and per_obs - prev < tol:
            prev = per_obs
            break
        prev = per_obs
    # canonical state order: by buy-emission prob ascending
    order = np.argsort(B[:, 2])
    A = A[order][:, order]
    B = B[order]
    pi = pi[order]
    log(f"    EM[{tag}] N={N} iters<={it + 1} ll/obs={prev:.5f} "
        f"p_buy_by_state={np.round(B[:, 2], 4).tolist()}")
    return A, B, pi


def filter_last(O: np.ndarray, A, B, pi, batch: int = 16384) -> np.ndarray:
    """Forward filtering; returns normalized state distribution at the last day."""
    N, T = O.shape
    K = A.shape[0]
    out = np.empty((N, K))
    for b0 in range(0, N, batch):
        Ob = O[b0:b0 + batch]
        a = pi[None, :] * B[:, Ob[:, 0]].T
        a /= a.sum(1, keepdims=True)
        for t in range(1, T):
            a = (a @ A) * B[:, Ob[:, t]].T
            a /= a.sum(1, keepdims=True)
        out[b0:b0 + batch] = a
    return out


# ---------------------------------------------------------------- simulation

def simulate(filt: np.ndarray, A: np.ndarray, p_buy: np.ndarray,
             mu_u: np.ndarray, sig_u: np.ndarray, S: int, rng,
             batch_users: int = 2048):
    """Sample S 30-day trajectories per user; return (E[log1p(sum)], P[sum==0])."""
    N = filt.shape[0]
    cumA = A.cumsum(1)
    cumF = filt.cumsum(1)
    pred = np.empty(N)
    pz = np.empty(N)
    for b0 in range(0, N, batch_users):
        b1 = min(b0 + batch_users, N)
        nb = b1 - b0
        M = nb * S
        rep = np.repeat(np.arange(b0, b1), S)
        state = (rng.random(M)[:, None] > cumF[rep]).sum(1)
        mu_rep = mu_u[rep]
        sig_rep = sig_u[rep]
        sums = np.zeros(M)
        for _ in range(HORIZON):
            state = (rng.random(M)[:, None] > cumA[state]).sum(1)
            buy = rng.random(M) < p_buy[state]
            k = int(buy.sum())
            if k:
                v = rng.normal(mu_rep[buy], sig_rep[buy])
                np.clip(v, 0.0, VCAP, out=v)
                sums[buy] += np.expm1(v)
        sums = sums.reshape(nb, S)
        pred[b0:b1] = np.log1p(sums).mean(1)
        pz[b0:b1] = (sums == 0).mean(1)
    return pred, pz


# ---------------------------------------------------------------- split runner

def run_split(anchor: date, users: np.ndarray, K: int, S: int, win: int,
              k0: float, em_cap: int, seed: int):
    t0 = time.time()
    O, seg, mu_u, sig_u, seg_val = build_inputs(anchor, users, win, k0)
    log(f"  inputs built: N={len(users)} win={win} "
        f"seg sizes={np.bincount(seg, minlength=4).tolist()} "
        f"({time.time() - t0:.0f}s)")
    for g in range(4):
        mg, sp, sw = seg_val[g]
        log(f"    seg{g} value: mu={mg:.3f} sig_pool={sp:.3f} sig_within={sw:.3f}")
    rng_em = np.random.default_rng(seed)
    rng_sim = np.random.default_rng(seed + 1000)
    pred = np.empty(len(users))
    pz = np.empty(len(users))
    for g in range(4):
        idx = np.nonzero(seg == g)[0]
        Og = O[idx]
        sub = idx if len(idx) <= em_cap else rng_em.choice(idx, em_cap, replace=False)
        A, B, pi = em_fit(O[sub], K, tag=f"seg{g}")
        filt = filter_last(Og, A, B, pi)
        p, z = simulate(filt, A, B[:, 2], mu_u[idx], sig_u[idx], S, rng_sim)
        pred[idx] = p
        pz[idx] = z
        log(f"    seg{g} simulated: mean_elog={p.mean():.3f} mean_pzero={z.mean():.3f} "
            f"({time.time() - t0:.0f}s cum)")
    log(f"  split {anchor} done in {time.time() - t0:.0f}s")
    return pred, pz, time.time() - t0


# ---------------------------------------------------------------- entrypoints

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="hmmsim")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--users", type=int, default=50000, help="smoke subsample size")
    ap.add_argument("--states", type=int, default=3)
    ap.add_argument("--sims", type=int, default=200)
    ap.add_argument("--win", type=int, default=120)
    ap.add_argument("--k0", type=float, default=5.0)
    ap.add_argument("--em-cap", type=int, default=15000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--splits", default="val")
    args = ap.parse_args()

    uni = user_universe()["user_id"].to_numpy()
    if args.smoke:
        rng = np.random.default_rng(42)
        users = np.sort(rng.choice(uni, args.users, replace=False))
    else:
        users = uni

    log(f"start name={args.name} smoke={args.smoke} N={len(users)} K={args.states} "
        f"S={args.sims} win={args.win} threads={_T}")

    out = {}
    if "val" in args.splits:
        log("VAL split")
        pred, pz, dt = run_split(VAL_ANCHOR, users, args.states, args.sims,
                                 args.win, args.k0, args.em_cap, args.seed)
        tgt = (load_anchor(VAL_ANCHOR, ["user_id", "target"])
               .join(pl.DataFrame({"user_id": users}), on="user_id", how="semi")
               .sort("user_id"))
        assert (tgt["user_id"].to_numpy() == users).all()
        y = tgt["target"].to_numpy()
        val_rmsle = rmsle(y, np.expm1(pred))
        zero_sim = float(pz.mean())
        zero_real = float((y == 0).mean())
        log(f"VAL rmsle={val_rmsle:.4f} zero_sim={zero_sim:.4f} zero_real={zero_real:.4f} "
            f"mean_elog={pred.mean():.4f} mean_real_log={np.log1p(y).mean():.4f}")
        out.update(val_rmsle=val_rmsle, zero_share_sim=zero_sim,
                   zero_share_real=zero_real, mean_elog=float(pred.mean()),
                   val_seconds=round(dt))

        if args.smoke:
            vp = pl.read_parquet(WORK / "preds_pack" / "val_preds.parquet",
                                 columns=["user_id", "mlpziln_cal"]) \
                   .join(pl.DataFrame({"user_id": users}), on="user_id", how="semi") \
                   .sort("user_id")
            assert (vp["user_id"].to_numpy() == users).all()
            ly = np.log1p(y)
            e_hmm = pred - ly
            e_z = np.log1p(np.clip(vp["mlpziln_cal"].to_numpy(), 0, None)) - ly
            out["err_corr_mlpziln_cal"] = float(np.corrcoef(e_hmm, e_z)[0, 1])
            log(f"err corr vs mlpziln_cal = {out['err_corr_mlpziln_cal']:.4f}")
        else:
            save_preds(args.name, "val", users, np.expm1(pred))
            pl.DataFrame({"user_id": users.astype(np.int64), "e_log": pred,
                          "p_zero": pz}).write_parquet(
                PREDS_DIR / f"{args.name}_aux_val.parquet")
            log_score(args.name, val_rmsle,
                      f"generative HMM sim K={args.states} S={args.sims} win={args.win} "
                      f"no-target-training zero_sim={zero_sim:.3f}")

    if "test" in args.splits:
        log("TEST split")
        pred, pz, dt = run_split(TEST_ANCHOR, users, args.states, args.sims,
                                 args.win, args.k0, args.em_cap, args.seed)
        save_preds(args.name, "test", users, np.expm1(pred))
        pl.DataFrame({"user_id": users.astype(np.int64), "e_log": pred,
                      "p_zero": pz}).write_parquet(
            PREDS_DIR / f"{args.name}_aux_test.parquet")
        out.update(test_mean_elog=float(pred.mean()), test_seconds=round(dt))

    print("RESULT_JSON " + json.dumps(out))


if __name__ == "__main__":
    main()

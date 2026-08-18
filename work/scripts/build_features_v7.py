"""v7 feature tier: HMM-simulator Monte-Carlo signals per anchor.

For each anchor slice runs the generative HMM simulator (same mechanism as
train_hmm_sim.py: per-segment discrete HMM over day types + lognormal buy
emission, EM-fit on data <= anchor only, no target training) and stores three
per-user columns that trees CANNOT reconstruct from tabular windows (they are
MC integrals over 300 latent trajectories):

  hmm_elog    E[log1p(sum gmv 30d)]  - the metric's per-user optimum under the model
  hmm_p_zero  P[sum gmv 30d == 0]    - mechanism-based zero probability
  hmm_sim_std std[log1p(sum gmv)]    - trajectory spread (aleatoric uncertainty)

Output: work/features/anchor=DATE.v7.parquet (user_id + 3 float32 cols).
Resumable: existing files are skipped.

Usage:
  THREADS=3 .venv/bin/python work/scripts/build_features_v7.py \
      --anchors 2026-02-13,2026-01-14 --sims 300 --states 4 --em-cap 15000
"""
from __future__ import annotations

import os

_T = os.environ.get("THREADS", "3")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "POLARS_MAX_THREADS"):
    os.environ.setdefault(_v, _T)

import argparse
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))
from common import FEATURES_DIR, user_universe  # noqa: E402
from train_hmm_sim import HORIZON, VCAP, build_inputs, em_fit, filter_last, log  # noqa: E402


def simulate_std(filt, A, p_buy, mu_u, sig_u, S, rng, batch_users=2048):
    """Like train_hmm_sim.simulate but also returns std of log1p across trajectories."""
    N = filt.shape[0]
    cumA = A.cumsum(1)
    cumF = filt.cumsum(1)
    pred = np.empty(N)
    pz = np.empty(N)
    sd = np.empty(N)
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
        ls = np.log1p(sums.reshape(nb, S))
        pred[b0:b1] = ls.mean(1)
        sd[b0:b1] = ls.std(1)
        pz[b0:b1] = (sums.reshape(nb, S) == 0).mean(1)
    return pred, pz, sd


def run_anchor(anchor: date, users: np.ndarray, K: int, S: int, win: int,
               k0: float, em_cap: int, seed: int) -> pl.DataFrame:
    t0 = time.time()
    O, seg, mu_u, sig_u, _ = build_inputs(anchor, users, win, k0)
    log(f"  [{anchor}] inputs: N={len(users)} seg={np.bincount(seg, minlength=4).tolist()} "
        f"({time.time() - t0:.0f}s)")
    rng_em = np.random.default_rng(seed)
    rng_sim = np.random.default_rng(seed + 1000)
    pred = np.empty(len(users))
    pz = np.empty(len(users))
    sd = np.empty(len(users))
    for g in range(4):
        idx = np.nonzero(seg == g)[0]
        sub = idx if len(idx) <= em_cap else rng_em.choice(idx, em_cap, replace=False)
        A, B, pi = em_fit(O[sub], K, tag=f"{anchor} seg{g}")
        filt = filter_last(O[idx], A, B, pi)
        p, z, s_ = simulate_std(filt, A, B[:, 2], mu_u[idx], sig_u[idx], S, rng_sim)
        pred[idx] = p
        pz[idx] = z
        sd[idx] = s_
    log(f"  [{anchor}] done in {time.time() - t0:.0f}s "
        f"mean_elog={pred.mean():.3f} mean_pz={pz.mean():.3f} mean_std={sd.mean():.3f}")
    return pl.DataFrame({
        "user_id": users.astype(np.int64),
        "hmm_elog": pred.astype(np.float32),
        "hmm_p_zero": pz.astype(np.float32),
        "hmm_sim_std": sd.astype(np.float32),
    })


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--anchors", required=True, help="comma-separated ISO dates")
    ap.add_argument("--states", type=int, default=4)
    ap.add_argument("--sims", type=int, default=300)
    ap.add_argument("--win", type=int, default=120)
    ap.add_argument("--k0", type=float, default=5.0)
    ap.add_argument("--em-cap", type=int, default=15000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    anchors = [date.fromisoformat(a) for a in args.anchors.split(",")]
    users = user_universe()["user_id"].to_numpy()
    log(f"v7 builder: {len(anchors)} anchors, K={args.states} S={args.sims} "
        f"em_cap={args.em_cap} threads={_T}")
    for a in anchors:
        out = FEATURES_DIR / f"anchor={a.isoformat()}.v7.parquet"
        if out.exists():
            log(f"  [{a}] exists, skip")
            continue
        df = run_anchor(a, users, args.states, args.sims, args.win,
                        args.k0, args.em_cap, args.seed)
        tmp = out.with_suffix(".tmp.parquet")
        df.write_parquet(tmp)
        tmp.rename(out)
    log("v7 builder DONE")


if __name__ == "__main__":
    main()

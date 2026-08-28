"""Deterministic 30-day forecast from the TPP hazard model (quadrature, no MC noise).

For each user, roll the state 30 days forward from the anchor:
  - exogenous browsing (srch/cart/s2c/active) is mean-field: constant inflow at the
    user's recent per-day rate (mean over the last 56 days), expected-dsl recursion;
  - self-excitation is mean-field: the day's expected purchase h_k and expected
    log-check mu_k feed back into the buy-EMAs, nbuys, dsl_buy, amount memory.
The rollout yields deterministic sequences {h_k, mu_k}. Then exact first-buy-day
quadrature: q_k = h_k * prod_{j<k}(1-h_j);
  E[log1p(S)] = sum_k q_k * log1p( expm1(mu_k) + n_tail_k * (exp(mu_bar+s^2/2)-1) ),
n_tail_k = sum_{j>k} h_j  (expected extra purchases), mu_bar = amount forecast mean
after the first buy. A single purchase is exact: log1p(expm1(mu)) = mu.

Usage: python3 tpp_infer.py --model ktpp_val_s1 --anchor 379 --out ktpp_val_s1_pred.parquet
"""
import argparse, json, time
import numpy as np
import polars as pl
import tpp_common as C

EXO_WIN = 56


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--anchor", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--chunk", type=int, default=125000)
    ap.add_argument("--selfexc", type=float, default=1.0,
                    help="scale of mean-field self-excitation feedback")
    ap.add_argument("--inflow-hl", type=float, default=0.0,
                    help="half-life (days) of exogenous inflow cooling; 0 = constant")
    args = ap.parse_args()

    import torch
    torch.set_num_threads(2)
    cfg = json.load(open(f"{args.model}.json"))
    net = C.build_mlp(cfg["seed"], cfg.get("h1", 128), cfg.get("h2", 96))
    net.load_state_dict(torch.load(f"{args.model}.pt", map_location="cpu"))
    net.eval()
    mu_n = np.array(cfg["mu"], dtype=np.float32)
    sd_n = np.array(cfg["sd"], dtype=np.float32)
    sigma = float(cfg["sigma_amt"])
    A = args.anchor

    cube = C.load_cube()
    t0 = time.time()
    _, _, st = C.scan_collect(cube, A, None, state_out_day=A)
    print(f"scan to {A}: {time.time()-t0:.0f}s")
    n = st.n

    # exogenous mean-field rates over the transform used in updates
    lo = A - EXO_WIN
    r_srch = np.log1p(cube["srch"][:, lo:A].astype(np.float32)).mean(1)
    r_cart = np.log1p(cube["cart"][:, lo:A].astype(np.float32)).mean(1)
    r_s2c = np.log1p(cube["s2c"][:, lo:A].astype(np.float32)).mean(1)
    r_act = (cube["active"][:, lo:A] > 0).astype(np.float32).mean(1)
    p_ev = {  # daily probability of "some event" per dsl channel
        1: (cube["cart"][:, lo:A] > 0).mean(1).astype(np.float32),      # cart
        2: r_act.copy(),                                                # active
        3: (cube["s2c"][:, lo:A] > 0).mean(1).astype(np.float32),       # s2c
    }
    inflow = np.zeros((n, C.NC), dtype=np.float32)
    inflow[:, 2] = r_srch; inflow[:, 3] = r_cart
    inflow[:, 4] = r_s2c; inflow[:, 5] = r_act

    # continuous dsl state (float) for the rollout
    dslf = np.minimum(A - st.last, 500).astype(np.float32)
    dslf[st.last < 0] = 500.0

    H = np.zeros((n, 30), dtype=np.float32)
    MU = np.zeros((n, 30), dtype=np.float32)
    f = np.empty((n, C.NF), dtype=np.float32)

    for k in range(30):
        t = A + k
        # ---- features from current soft state ----
        f[:, :C.N_EMA] = np.log1p(st.E.reshape(n, C.N_EMA))
        f[:, C.N_EMA:C.N_EMA + 4] = np.log1p(dslf)
        f[:, C.N_EMA + 4:C.N_EMA + 8] = (st.last < 0) & (dslf >= 499.5)
        i = C.N_EMA + 8
        nb = st.nbuys
        multi = np.clip(nb - 1.0, 0.0, 1.0)          # soft version of nbuys>=2
        mean_int = np.where(nb >= 2, st.sum_int / np.maximum(nb - 1, 1e-3), 0.0)
        phase = np.where(nb >= 2, np.minimum(dslf[:, 0] / np.maximum(mean_int, 1.0), 8.0), 0.0)
        f[:, i] = np.log1p(nb)
        f[:, i + 1] = np.log1p(mean_int)
        f[:, i + 2] = np.log1p(st.last_int)
        f[:, i + 3] = phase
        f[:, i + 4] = multi
        ever = np.clip(nb, 0.0, 1.0)
        f[:, i + 5] = st.last_amt
        f[:, i + 6] = np.where(nb > 0, st.sum_amt / np.maximum(nb, 1e-3), 0.0)
        f[:, i + 7] = ever
        i += 8
        f[:, i:i + 7] = 0.0
        f[:, i + C.DOW[t]] = 1.0
        i += 7
        f[:, i] = np.sin(2 * np.pi * C.DOM[t] / 30.44)
        f[:, i + 1] = np.cos(2 * np.pi * C.DOM[t] / 30.44)
        f[:, i + 2] = np.sin(2 * np.pi * C.DOY[t] / 365.25)
        f[:, i + 3] = np.cos(2 * np.pi * C.DOY[t] / 365.25)
        f[:, i + 4] = t / 408.0

        with torch.no_grad():
            for a in range(0, n, args.chunk):
                b = min(a + args.chunk, n)
                xb = torch.from_numpy((f[a:b] - mu_n) / sd_n)
                lg, am = net(xb)
                H[a:b, k] = torch.sigmoid(lg).numpy()
                MU[a:b, k] = am.numpy()

        # ---- mean-field update with expected events ----
        h = H[:, k] * args.selfexc; m = MU[:, k]
        cool = 0.5 ** (k / args.inflow_hl) if args.inflow_hl > 0 else 1.0
        st.E *= C.DECAYS[None, None, :]
        st.E[:, 0, :] += h[:, None]
        st.E[:, 1, :] += (h * m)[:, None]
        st.E[:, 2:, :] += cool * inflow[:, 2:, None]
        # dsl: buy via expected recursion; exo channels likewise
        dslf[:, 0] = (1 - h) * (dslf[:, 0] + 1)
        for j, p in p_ev.items():
            dslf[:, j] = (1 - p) * (dslf[:, j] + 1)
        st.last_amt = (1 - h) * st.last_amt + h * m
        st.sum_amt += h * m
        # intervals: dsl at the moment of an (expected) buy is the realized gap
        had = st.nbuys > 0
        st.sum_int += np.where(had, h * dslf[:, 0], 0.0)
        st.last_int = (1 - h) * st.last_int + h * np.where(had, dslf[:, 0], 0.0)
        st.nbuys += h
        if k % 10 == 0:
            print(f"day {k}: h mean {h.mean():.4f} max {h.max():.3f} "
                  f"mu mean {m.mean():.2f} ({time.time()-t0:.0f}s)", flush=True)

    # ---- first-buy-day quadrature ----
    logs = np.log(np.clip(1 - H, 1e-9, 1))
    cum = np.cumsum(logs, axis=1)
    surv_before = np.exp(np.concatenate([np.zeros((n, 1), np.float32), cum[:, :-1]], 1))
    q = H * surv_before                               # (n,30) first-buy-day probs
    p0 = np.exp(cum[:, -1])
    tail_after = np.concatenate(
        [np.cumsum(H[:, ::-1], axis=1)[:, ::-1][:, 1:], np.zeros((n, 1), np.float32)], 1)
    w = q / np.maximum(q.sum(1, keepdims=True), 1e-12)
    mu_bar = (w * MU).sum(1)                          # amount level after first buy
    # plug-in scale e^{mu_bar} (median), NOT e^{mu+s^2/2}: sigma~2.4 would let the
    # lognormal right tail dominate a term that sits INSIDE log1p; calibration
    # fixes the residual level structure by quantile anyway
    exp_amt = np.expm1(np.clip(mu_bar, 0, 12.0))
    contrib = np.log1p(np.expm1(np.clip(MU, 0, 12.0)) +
                       tail_after * exp_amt[:, None])
    elog = (q * contrib).sum(1)                       # + p0 * 0
    pred = np.expm1(np.clip(elog, 0, 12.0))

    users = pl.read_parquet("/root/work/users_order.parquet")["user_id"]
    pl.DataFrame({"user_id": users, "pred": pred.astype(np.float64)}
                 ).sort("user_id").write_parquet(args.out)
    print(f"saved {args.out}; P(buy30) mean {(1-p0).mean():.4f}; "
          f"elog mean {elog.mean():.4f} max {elog.max():.2f}")


if __name__ == "__main__":
    main()

"""Train the TPP v0 hazard+amount MLP on observed user-days (teacher forcing).

Honest gap discipline (matches work_kostya models): the val-variant sees target
days t <= 373 only (= last day covered by an anchor-344 window, 35-day gap to the
val anchor 379); the test-variant sees t <= 403 (gap 35 to the test anchor 409).

Usage: python3 tpp_train.py --tmax 373 --seed 1 --out ktpp_val_s1
"""
import argparse, json, time
import numpy as np
import tpp_common as C


def sample_rows(cube, tmax, seed, neg_target):
    nord = cube["nord"]
    active = cube["active"]
    n, T = nord.shape
    first_act = np.where(active.any(1), active.argmax(1), 10**6)
    ub, tb = np.nonzero(nord[:, : tmax + 1] > 0)
    keep = (tb >= 21) & (tb > first_act[ub])
    ub, tb = ub[keep], tb[keep]
    n_buy = len(ub)
    lo = np.maximum(21, first_act + 1)
    elig = np.maximum(0, tmax - lo + 1)
    elig_neg_total = int(elig.sum()) - n_buy
    rng = np.random.default_rng(seed)
    got_u, got_t = [], []
    got = 0
    while got < neg_target:
        m = int((neg_target - got) * 1.6) + 1000
        uu = rng.integers(0, n, m)
        tt = rng.integers(21, tmax + 1, m)
        ok = (tt > first_act[uu]) & (nord[uu, tt] == 0)
        uu, tt = uu[ok], tt[ok]
        got_u.append(uu); got_t.append(tt); got += len(uu)
    uu = np.concatenate(got_u)[:neg_target]
    tt = np.concatenate(got_t)[:neg_target]
    w_neg = elig_neg_total / len(uu)
    u = np.concatenate([ub, uu]).astype(np.int64)
    t = np.concatenate([tb, tt]).astype(np.int64)
    y = np.zeros(len(u), dtype=np.float32); y[:n_buy] = 1.0
    amt = cube["lgmv"][u, t].astype(np.float32)
    w = np.ones(len(u), dtype=np.float32); w[n_buy:] = w_neg
    print(f"rows: {n_buy} buys + {len(uu)} negs (w_neg={w_neg:.2f}, "
          f"elig_neg={elig_neg_total})")
    return u, t, y, amt, w


def collect(cube, u, t, tmax):
    order = np.argsort(t, kind="stable")
    rows_by_t = {}
    tu = t[order]
    bounds = np.searchsorted(tu, np.arange(tmax + 2))
    for d in range(tmax + 1):
        a, b = bounds[d], bounds[d + 1]
        if b > a:
            rows_by_t[d] = u[order[a:b]]
    t0 = time.time()
    X, keys, _ = C.scan_collect(cube, tmax + 1, rows_by_t)
    print(f"scan+collect {time.time()-t0:.0f}s -> {X.shape}")
    return X, order


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tmax", type=int, required=True)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--neg", type=int, default=4_500_000)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--h1", type=int, default=128)
    ap.add_argument("--h2", type=int, default=96)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import torch
    torch.set_num_threads(2)
    cube = C.load_cube()
    u, t, y, amt, w = sample_rows(cube, args.tmax, args.seed, args.neg)
    X, order = collect(cube, u, t, args.tmax)
    y, amt, w = y[order], amt[order], w[order]

    rng = np.random.default_rng(args.seed + 7)
    dev = rng.random(len(y)) < 0.03
    tr = ~dev
    Xd = torch.from_numpy(X[dev].astype(np.float32))
    yd = torch.from_numpy(y[dev]); ad = torch.from_numpy(amt[dev])
    wd = torch.from_numpy(w[dev])
    idx_tr = np.flatnonzero(tr)

    # standardize features on a sample (helps the MLP; stats go to json)
    smp = rng.choice(len(X), 400_000, replace=False)
    mu = X[smp].astype(np.float32).mean(0)
    sd = X[smp].astype(np.float32).std(0) + 1e-3

    net = C.build_mlp(args.seed, args.h1, args.h2)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-5)
    B = 65536
    steps_ep = len(idx_tr) // B
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs * steps_ep)
    bce = torch.nn.BCEWithLogitsLoss(reduction="none")
    mses = torch.nn.MSELoss(reduction="none")
    tmu = torch.from_numpy(mu); tsd = torch.from_numpy(sd)

    def norm(a):
        return (a - tmu) / tsd

    t0 = time.time()
    for ep in range(args.epochs):
        perm = rng.permutation(idx_tr)
        tot = totn = 0.0
        for s in range(steps_ep):
            ix = perm[s * B:(s + 1) * B]
            xb = norm(torch.from_numpy(X[ix].astype(np.float32)))
            yb = torch.from_numpy(y[ix]); ab = torch.from_numpy(amt[ix])
            wb = torch.from_numpy(w[ix])
            lg, am = net(xb)
            l_b = (bce(lg, yb) * wb).sum() / wb.sum()
            mask = yb > 0
            l_a = (mses(am[mask], ab[mask])).mean() if mask.any() else lg.new_zeros(())
            loss = l_b + 0.5 * l_a
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            tot += float(loss) * len(ix); totn += len(ix)
        with torch.no_grad():
            lg, am = net(norm(Xd))
            dl_b = float((bce(lg, yd) * wd).sum() / wd.sum())
            m = yd > 0
            dl_a = float(mses(am[m], ad[m]).mean())
        print(f"ep{ep} train {tot/totn:.5f} dev bce {dl_b:.5f} amt {dl_a:.4f} "
              f"({time.time()-t0:.0f}s)", flush=True)

    with torch.no_grad():
        lg, am = net(norm(Xd))
        m = yd > 0
        sigma = float(torch.sqrt(mses(am[m], ad[m]).mean()))
    torch.save(net.state_dict(), f"{args.out}.pt")
    json.dump({"seed": args.seed, "tmax": args.tmax, "neg": args.neg,
               "epochs": args.epochs, "lr": args.lr, "h1": args.h1, "h2": args.h2,
               "sigma_amt": sigma,
               "mu": mu.tolist(), "sd": sd.tolist(),
               "feat_names": C.FEAT_NAMES,
               "cmd": "tpp_train.py " + json.dumps(vars(args))},
              open(f"{args.out}.json", "w"))
    print(f"saved {args.out}.pt sigma_amt={sigma:.3f}")


if __name__ == "__main__":
    main()

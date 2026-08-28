"""Program 3 (compressed): attention-over-features transformer (FT/SAINT-lite)
on the kostya46 feature space. Own implementation (no third-party FT libs, no
license questions), CPU-sized: features are grouped into tokens of 5, 2 attention
layers, d=24. Same anchors/targets/gap discipline as train_model2.py (10 slices
281..344, recency weights, val anchor 379). Heads: direct z=log1p(gmv30) + aux buy.

Usage: python3 ft_train.py --seed 1 --out ftt_s1 [--tmax-anchor 344 --val-day 379]
       python3 ft_train.py --seed 1 --out ftt_t_s1 --test  (anchors 311..374, no val)
"""
import argparse, json, time
import numpy as np

WORK = "/root/work"


def build_rows(train_days, grid_day, seed, keep_neg=0.35):
    import sys
    sys.path.insert(0, WORK)
    from features import build_features
    if grid_day == 379:
        cube = np.load(f"{WORK}/cube_val.npy", mmap_mode="r")
        gmv_mat = np.load(f"{WORK}/gmv_mat.npy")
        buy_mat = np.load(f"{WORK}/buy_mat.npy")
        anchor_days = np.load(f"{WORK}/anchor_days.npy").tolist()
    else:
        cube = np.load(f"{WORK}/cube_test.npy", mmap_mode="r")
        gmv_mat = np.load(f"{WORK}/gmv_mat_testgrid.npy")
        buy_mat = (gmv_mat > 0).astype(np.float32)
        anchor_days = json.load(open(f"{WORK}/testgrid_days.json"))
    day_to_col = {int(d): i for i, d in enumerate(anchor_days)}
    rng = np.random.default_rng(seed)
    Xs, zs, ys, ws = [], [], [], []
    for d in train_days:
        X, names = build_features(d, cube, grid_day)
        c = day_to_col[d]
        y = buy_mat[:, c] > 0
        z = np.log1p(gmv_mat[:, c]).astype(np.float32)
        keep = y | (rng.random(len(y)) < keep_neg)
        w = np.where(y, 1.0, 1.0 / keep_neg).astype(np.float32)
        wk_back = (grid_day - d) / 7.0
        w *= 0.5 ** (wk_back / 26.0)
        Xs.append(np.clip(X[keep], -6e4, 6e4).astype(np.float16))
        zs.append(z[keep]); ys.append(y[keep].astype(np.float32))
        ws.append(w[keep])
        print(f"anchor {d}: kept {int(keep.sum())}", flush=True)
    X = np.concatenate(Xs); del Xs
    return X, np.concatenate(zs), np.concatenate(ys), np.concatenate(ws), names


def build_net(seed, nf, gsize=5, d=24, heads=3, layers=2):
    import torch
    import torch.nn as nn
    torch.manual_seed(seed)
    ntok = (nf + gsize - 1) // gsize

    class FT(nn.Module):
        def __init__(self):
            super().__init__()
            self.pad = ntok * gsize - nf
            self.emb = nn.Parameter(torch.randn(ntok, gsize, d) * 0.2)
            self.bias = nn.Parameter(torch.zeros(ntok, d))
            self.cls = nn.Parameter(torch.zeros(1, 1, d))
            enc = nn.TransformerEncoderLayer(
                d_model=d, nhead=heads, dim_feedforward=2 * d, dropout=0.0,
                activation="gelu", batch_first=True, norm_first=True)
            self.enc = nn.TransformerEncoder(enc, num_layers=layers)
            self.h_z = nn.Linear(d, 1)
            self.h_b = nn.Linear(d, 1)

        def forward(self, x):                      # x: (B, nf) standardized
            b = x.shape[0]
            if self.pad:
                x = torch.nn.functional.pad(x, (0, self.pad))
            g = x.view(b, ntok, gsize)
            tok = torch.einsum("btg,tgd->btd", g, self.emb) + self.bias
            tok = torch.cat([self.cls.expand(b, 1, d), tok], dim=1)
            z = self.enc(tok)[:, 0]
            return self.h_z(z).squeeze(-1), self.h_b(z).squeeze(-1)

    return FT(), ntok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", required=True)
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--keep-neg", type=float, default=0.35)
    args = ap.parse_args()
    import torch
    torch.set_num_threads(2)

    if args.test:
        train_days = list(range(311, 375, 7))      # 311..374, grid day 409
        grid = 409
    else:
        train_days = list(range(281, 345, 7))      # 281..344, grid day 379
        grid = 379

    X, z, y, w, names = build_rows(train_days, grid, args.seed, args.keep_neg)
    print("rows", X.shape, flush=True)
    rng = np.random.default_rng(args.seed + 11)
    smp = rng.choice(len(X), min(300_000, len(X)), replace=False)
    mu = X[smp].astype(np.float32).mean(0)
    sd = X[smp].astype(np.float32).std(0) + 1e-3

    net, ntok = build_net(args.seed, X.shape[1])
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    steps_ep = len(X) // args.batch
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs * steps_ep)
    tmu, tsd = torch.from_numpy(mu), torch.from_numpy(sd)
    bce = torch.nn.BCEWithLogitsLoss(reduction="none")
    t0 = time.time()
    for ep in range(args.epochs):
        perm = rng.permutation(len(X))
        tot = totw = 0.0
        for s in range(steps_ep):
            ix = perm[s * args.batch:(s + 1) * args.batch]
            xb = (torch.from_numpy(X[ix].astype(np.float32)) - tmu) / tsd
            zb = torch.from_numpy(z[ix]); yb = torch.from_numpy(y[ix])
            wb = torch.from_numpy(w[ix])
            pz, pb = net(xb)
            loss = ((pz - zb) ** 2 * wb).sum() / wb.sum() \
                + 0.3 * (bce(pb, yb) * wb).sum() / wb.sum()
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            tot += float(loss.detach()) * float(wb.sum()); totw += float(wb.sum())
            if s % 100 == 0:
                print(f"ep{ep} s{s}/{steps_ep} loss {tot/max(totw,1):.4f} "
                      f"({time.time()-t0:.0f}s)", flush=True)
    del X

    # predict on the grid anchor
    Xv = np.load(f"{WORK}/Xval3_379.npy" if not args.test else f"{WORK}/Xtest_409.npy",
                 mmap_mode="r")
    preds = np.empty(Xv.shape[0], dtype=np.float32)
    net.eval()
    with torch.no_grad():
        for a in range(0, Xv.shape[0], 50000):
            b = min(a + 50000, Xv.shape[0])
            xb = (torch.from_numpy(np.clip(np.asarray(Xv[a:b], dtype=np.float32),
                                           -6e4, 6e4)) - tmu) / tsd
            pz, _ = net(xb)
            preds[a:b] = pz.numpy()
    import polars as pl
    users = pl.read_parquet(f"{WORK}/users_order.parquet")["user_id"]
    pl.DataFrame({"user_id": users,
                  "pred": np.expm1(np.clip(preds, 0, 12)).astype(np.float64)}
                 ).sort("user_id").write_parquet(f"{args.out}_pred.parquet")
    torch.save(net.state_dict(), f"{args.out}.pt")
    json.dump({"seed": args.seed, "epochs": args.epochs, "batch": args.batch,
               "lr": args.lr, "keep_neg": args.keep_neg, "arch":
               {"gsize": 5, "d": 24, "heads": 3, "layers": 2, "ntok": ntok},
               "train_days": train_days, "grid": grid,
               "license": "own implementation (torch nn.TransformerEncoder), no 3rd-party FT code",
               "mu": mu.tolist(), "sd": sd.tolist()},
              open(f"{args.out}.json", "w"))
    print(f"saved {args.out}_pred.parquet ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()

"""PyTorch MLP trainer following the exp_lib contract (mirrors train_gbdt.py flow).

Preprocessing (fit on train anchors only, applied unchanged to val/test):
  median-impute NaN -> clip to train [p1, p99] -> standardize. Target: log1p.
Arch: 512-256-128 MLP with LayerNorm + GELU + Dropout(0.15).
AdamW + cosine schedule, MSE on log1p target, early stop on val RMSLE.
pred = expm1(clip(out, 0, None)).

Flow: train on anchors < VAL with early stopping on VAL rmsle -> save val preds +
log score; then retrain on train+val for the stopped epoch count -> save test preds.
Multi-seed: --seeds 42,1337 trains one model per seed and averages predictions.
Memory: train+val features live in ONE preallocated C-order array (no duplication),
so the retrain phase reuses the same buffer via views.

Examples:
  train_mlp.py --name mlp_a --n-anchors 6 --no-test
  train_mlp.py --name mlp_final --seeds 42,1337
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "3")
os.environ.setdefault("USE_V2", "1")

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from common import (FEATURES_DIR, TEST_ANCHOR, VAL_ANCHOR, feature_cols,
                    load_anchor, rmsle)
from exp_lib import available_train_anchors, log_score, save_preds

STATS_MAX_ROWS = 750_000   # row-subsample size for percentile/mean/std estimation
BLOCK = 262_144            # rows per block for in-place transform


def fit_stats(X: np.ndarray) -> dict:
    """Estimate impute/clip/standardize stats from a row-subsample of train."""
    step = max(1, int(np.ceil(X.shape[0] / STATS_MAX_ROWS)))
    S = np.ascontiguousarray(X[::step])
    q = np.nanpercentile(S, [1.0, 50.0, 99.0], axis=0)
    med = np.where(np.isfinite(q[1]), q[1], 0.0).astype(np.float32)
    lo = np.where(np.isfinite(q[0]), q[0], med).astype(np.float32)
    hi = np.where(np.isfinite(q[2]), q[2], med).astype(np.float32)
    np.copyto(S, np.broadcast_to(med, S.shape), where=np.isnan(S))
    np.clip(S, lo, hi, out=S)
    mean = S.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = S.std(axis=0, dtype=np.float64).astype(np.float32)
    std[~np.isfinite(std) | (std < 1e-7)] = 1.0
    del S
    return dict(med=med, lo=lo, hi=hi, mean=mean, std=std)


def apply_stats(X: np.ndarray, s: dict) -> None:
    """Blockwise in-place: median-impute -> clip [p1,p99] -> standardize."""
    for i in range(0, X.shape[0], BLOCK):
        B = X[i:i + BLOCK]
        np.copyto(B, np.broadcast_to(s["med"], B.shape), where=np.isnan(B))
        np.clip(B, s["lo"], s["hi"], out=B)
        B -= s["mean"]
        B /= s["std"]


def anchor_heights(anchors) -> list[int]:
    import polars as pl
    return [
        pl.scan_parquet(FEATURES_DIR / f"anchor={a.isoformat()}.parquet")
        .select(pl.len()).collect().item()
        for a in anchors
    ]


def build_model(d_in: int, hidden: list[int], dropout: float):
    import torch.nn as nn
    layers = []
    prev = d_in
    for h in hidden:
        layers += [nn.Linear(prev, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(dropout)]
        prev = h
    layers.append(nn.Linear(prev, 1))
    return nn.Sequential(*layers)


def predict_log(model, X: np.ndarray, device: str, bs: int = 65536) -> np.ndarray:
    import torch
    model.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, X.shape[0], bs):
            xb = torch.from_numpy(np.ascontiguousarray(X[i:i + bs])).to(device)
            outs.append(model(xb).squeeze(1).float().cpu().numpy())
    return np.concatenate(outs)


def train_one(X, ylog, w, Xv, ylv, cfg, seed, device, epochs, tag=""):
    """One MLP fit. With Xv: early stop on val rmsle, return model at best epoch.
    Without Xv: fixed `epochs` run (cosine schedule compressed to that length)."""
    import torch
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = build_model(X.shape[1], cfg["hidden"], cfg["dropout"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=epochs, eta_min=cfg["lr"] * 0.01)
    n, bs = X.shape[0], cfg["bs"]
    best, best_epoch, bad, best_state = np.inf, 0, 0, None
    for ep in range(1, epochs + 1):
        model.train()
        perm = rng.permutation(n)
        loss_sum = torch.zeros((), device=device)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            xb = torch.from_numpy(X[idx]).to(device)
            yb = torch.from_numpy(ylog[idx]).to(device)
            out = model(xb).squeeze(1)
            if w is not None:
                wb = torch.from_numpy(w[idx]).to(device)
                loss = (wb * (out - yb) ** 2).sum() / wb.sum()
            else:
                loss = torch.nn.functional.mse_loss(out, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            loss_sum += loss.detach() * len(idx)
        sched.step()
        tr_mse = float(loss_sum) / n
        if Xv is None:
            print(f"{tag}ep {ep}/{epochs} train_mse {tr_mse:.5f}", flush=True)
            continue
        out_v = predict_log(model, Xv, device)
        score = float(np.sqrt(np.mean((np.clip(out_v, 0, None) - ylv) ** 2)))
        mark = ""
        if score < best - 1e-5:
            best, best_epoch, bad = score, ep, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            mark = " *"
        else:
            bad += 1
        print(f"{tag}ep {ep} train_mse {tr_mse:.5f} val_rmsle {score:.5f}{mark}", flush=True)
        if bad >= cfg["patience"]:
            print(f"{tag}early stop at ep {ep} (best {best:.5f} @ ep {best_epoch})", flush=True)
            break
    if Xv is None:
        return model, epochs, None
    model.load_state_dict(best_state)
    return model, best_epoch, best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--n-anchors", type=int, default=0)
    ap.add_argument("--weight-tau", type=float, default=0.0)
    ap.add_argument("--drop-cols", type=str, default="")
    ap.add_argument("--seeds", type=str, default="42")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=8192)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--dropout", type=float, default=0.15)
    ap.add_argument("--hidden", type=str, default="512,256,128")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--device", type=str, default="")
    ap.add_argument("--no-test", action="store_true")
    ap.add_argument("--notes", type=str, default="")
    args = ap.parse_args()
    if args.threads:
        os.environ["OMP_NUM_THREADS"] = str(args.threads)
    import polars as pl
    import torch
    torch.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))
    device = args.device or ("mps" if torch.backends.mps.is_available() else "cpu")
    seeds = [int(s) for s in args.seeds.split(",")]
    cfg = dict(hidden=[int(h) for h in args.hidden.split(",")], dropout=args.dropout,
               lr=args.lr, wd=args.wd, bs=args.batch_size, patience=args.patience)
    print(f"device={device} seeds={seeds} cfg={cfg}", flush=True)

    t0 = time.time()
    tr_anchors = available_train_anchors()
    if args.n_anchors:
        tr_anchors = tr_anchors[-args.n_anchors:]
    print(f"train anchors: {[a.isoformat() for a in tr_anchors]}", flush=True)

    val = load_anchor(VAL_ANCHOR)
    cols = feature_cols(val)
    if args.drop_cols:
        drop = set(args.drop_cols.split(","))
        cols = [c for c in cols if c not in drop]
    print(f"{len(cols)} features", flush=True)
    f32 = [pl.col(c).cast(pl.Float32) for c in cols]

    heights = anchor_heights(tr_anchors)
    n, nv, d = sum(heights), val.height, len(cols)
    Xfull = np.empty((n + nv, d), np.float32)      # train rows then val rows
    ylog_full = np.empty(n + nv, np.float32)
    days = np.empty(n, np.float32)
    pos = 0
    for a, h in zip(tr_anchors, heights):
        df = load_anchor(a, ["target"] + cols)
        assert df.height == h, f"height mismatch for {a}"
        Xfull[pos:pos + h] = df.select(f32).to_numpy()
        ylog_full[pos:pos + h] = np.log1p(
            df["target"].to_numpy().astype(np.float64)).astype(np.float32)
        days[pos:pos + h] = (VAL_ANCHOR - a).days
        pos += h
        del df
    Xfull[n:] = val.select(f32).to_numpy()
    yv_raw = val["target"].to_numpy().astype(np.float64)
    ylog_full[n:] = np.log1p(yv_raw).astype(np.float32)
    uid_val = val["user_id"].to_numpy()
    del val
    w = None
    if args.weight_tau:
        w = np.exp(-days / args.weight_tau).astype(np.float32)  # val-adjacent ~ 1.0
    del days
    print(f"X {(n, d)}, Xv {(nv, d)}, load {time.time()-t0:.0f}s", flush=True)

    stats = fit_stats(Xfull[:n])                   # train-only stats
    apply_stats(Xfull, stats)                      # transform train+val in place
    print(f"preprocess done {time.time()-t0:.0f}s", flush=True)

    X, Xv = Xfull[:n], Xfull[n:]
    ylog, ylv = ylog_full[:n], ylog_full[n:].astype(np.float64)

    val_preds, best_epochs = [], []
    for seed in seeds:
        m, be, _ = train_one(X, ylog, w, Xv, ylv, cfg, seed, device,
                             args.epochs, tag=f"[s{seed}] ")
        pv = np.expm1(np.clip(predict_log(m, Xv, device), 0, None))
        print(f"[s{seed}] best_epoch={be} val_rmsle={rmsle(yv_raw, pv):.6f}", flush=True)
        val_preds.append(pv)
        best_epochs.append(be)
        del m
    pv_avg = np.mean(val_preds, axis=0)
    score = rmsle(yv_raw, pv_avg)
    save_preds(args.name, "val", uid_val, pv_avg)
    notes = args.notes or (f"mlp {args.hidden} do{args.dropout} lr{args.lr} bs{args.batch_size} "
                           f"tau{args.weight_tau} seeds={args.seeds} {len(tr_anchors)}anch "
                           f"ep={best_epochs}")
    log_score(args.name, score, notes)

    if args.no_test:
        print(f"[DONE] {args.name} val_rmsle={score:.6f} total {time.time()-t0:.0f}s", flush=True)
        return

    # retrain on train+val (same buffer, no copy) for the stopped epoch count
    wall = np.concatenate([w, np.ones(nv, np.float32)]) if w is not None else None
    test = load_anchor(TEST_ANCHOR)
    Xt = test.select(f32).to_numpy()
    uid_t = test["user_id"].to_numpy()
    del test
    apply_stats(Xt, stats)

    test_preds = []
    for seed, be in zip(seeds, best_epochs):
        m, _, _ = train_one(Xfull, ylog_full, wall, None, None, cfg, seed, device,
                            max(1, be), tag=f"[s{seed} full] ")
        test_preds.append(np.expm1(np.clip(predict_log(m, Xt, device), 0, None)))
        del m
    save_preds(args.name, "test", uid_t, np.mean(test_preds, axis=0))
    print(f"[DONE] {args.name} val_rmsle={score:.6f} total {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

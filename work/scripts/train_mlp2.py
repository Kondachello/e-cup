"""Hurdle MLP trainer (two-head) following the exp_lib contract.

Shared trunk on tabular features -> two heads:
  (a) logit head: P(y>0), BCE loss;
  (b) magnitude head: mu = E[log1p(y) | y>0], MSE on positive rows only.
Prediction: pred_log = sigmoid(logit) * clip(mu, 0, None);
            pred     = expm1(clip(pred_log, 0, None)).
Loss = bce_w * BCE + MSE_pos  (bce_w default 0.7).

Trunk: Linear(F->512) GELU LayerNorm Dropout(0.15) -> 256 (same block) -> heads.
Preprocessing (fit on train anchors only; saved and reused for val/test):
  median-impute NaN -> clip to train [p1, p99] -> standardize. Stats saved to
  work/models/NAME_stats.npz on non-smoke runs.

Anchors: like train_gbdt --gap-days (default 30): selection trains only on
anchors ending >= gap days before VAL (no target-window overlap with val);
the test retrain adds the gap anchors + VAL and runs for the stopped epochs.
Early stop on VAL RMSLE, patience 4. USE_V2/USE_V3 feature tiers are joined
automatically by common.load_anchor (both default to on here).

Memory: train(+gap)+val features live in ONE preallocated C-order float32
array; selection/retrain use views of it (no duplication).

--smoke: single seed, batch <= 2048, hard cap of 200 optimizer steps, forces
--no-test, does not write preds/scores/stats — just prints the val RMSLE.

Examples:
  train_mlp2.py --name mlp2_smoke --smoke --n-anchors 1 --threads 2
  train_mlp2.py --name mlp2_a --n-anchors 8 --no-test
  train_mlp2.py --name mlp2_final --seeds 42,1337
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import timedelta
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("USE_V2", "1")
os.environ.setdefault("USE_V3", "1")

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from common import (WORK, TEST_ANCHOR, VAL_ANCHOR, feature_cols, load_anchor,
                    rmsle)
from exp_lib import FEATURES_DIR, available_train_anchors, log_score, save_preds
from model_io import save_meta, save_torch

MODELS_DIR = WORK / "models"
STATS_MAX_ROWS = 750_000   # row-subsample size for percentile/mean/std estimation
BLOCK = 262_144            # rows per block for in-place transform
SMOKE_MAX_STEPS = 200
SMOKE_MAX_BATCH = 2048


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

    class HurdleMLP(nn.Module):
        def __init__(self):
            super().__init__()
            layers, prev = [], d_in
            for h in hidden:
                layers += [nn.Linear(prev, h), nn.GELU(), nn.LayerNorm(h),
                           nn.Dropout(dropout)]
                prev = h
            self.trunk = nn.Sequential(*layers)
            self.head_logit = nn.Linear(prev, 1)   # P(y>0)
            self.head_mu = nn.Linear(prev, 1)      # E[log1p(y) | y>0]

        def forward(self, x):
            z = self.trunk(x)
            return self.head_logit(z).squeeze(1), self.head_mu(z).squeeze(1)

    return HurdleMLP()


def predict_log(model, X: np.ndarray, device: str, bs: int = 65536) -> np.ndarray:
    """sigmoid(logit) * clip(mu, 0) per row, in log1p space (always >= 0)."""
    import torch
    model.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, X.shape[0], bs):
            xb = torch.from_numpy(np.ascontiguousarray(X[i:i + bs])).to(device)
            logit, mu = model(xb)
            pl_ = torch.sigmoid(logit) * torch.clamp(mu, min=0)
            outs.append(pl_.float().cpu().numpy())
    return np.concatenate(outs)


def train_one(X, ylog, Xv, ylv, cfg, seed, device, epochs, max_steps=None, tag=""):
    """One hurdle-MLP fit. With Xv: early stop on val rmsle, return best-epoch
    model. Without Xv: fixed `epochs` run (cosine compressed to that length)."""
    import torch
    import torch.nn.functional as F
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = build_model(X.shape[1], cfg["hidden"], cfg["dropout"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=epochs, eta_min=cfg["lr"] * 0.01)
    n, bs, bce_w = X.shape[0], cfg["bs"], cfg["bce_w"]
    best, best_epoch, bad, best_state = np.inf, 0, 0, None
    steps = 0
    for ep in range(1, epochs + 1):
        model.train()
        perm = rng.permutation(n)
        bce_sum = torch.zeros((), device=device)
        mse_sum = torch.zeros((), device=device)
        seen = 0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            xb = torch.from_numpy(X[idx]).to(device)
            yb = torch.from_numpy(ylog[idx]).to(device)
            pos = yb > 0
            logit, mu = model(xb)
            bce = F.binary_cross_entropy_with_logits(logit, pos.float())
            if pos.any():
                mse = F.mse_loss(mu[pos], yb[pos])
            else:
                mse = torch.zeros((), device=device)
            loss = bce_w * bce + mse
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            bce_sum += bce.detach() * len(idx)
            mse_sum += mse.detach() * len(idx)
            seen += len(idx)
            steps += 1
            if max_steps is not None and steps >= max_steps:
                break
        sched.step()
        tr_bce, tr_mse = float(bce_sum) / seen, float(mse_sum) / seen
        out_of_budget = max_steps is not None and steps >= max_steps
        if Xv is None:
            print(f"{tag}ep {ep}/{epochs} bce {tr_bce:.5f} mse_pos {tr_mse:.5f}",
                  flush=True)
            if out_of_budget:
                break
            continue
        pred_log = predict_log(model, Xv, device).astype(np.float64)
        score = float(np.sqrt(np.mean((np.clip(pred_log, 0, None) - ylv) ** 2)))
        mark = ""
        if score < best - 1e-5:
            best, best_epoch, bad = score, ep, 0
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
            mark = " *"
        else:
            bad += 1
        print(f"{tag}ep {ep} bce {tr_bce:.5f} mse_pos {tr_mse:.5f} "
              f"val_rmsle {score:.5f}{mark}", flush=True)
        if out_of_budget:
            print(f"{tag}step budget {max_steps} reached at ep {ep}", flush=True)
            break
        if bad >= cfg["patience"]:
            print(f"{tag}early stop at ep {ep} (best {best:.5f} @ ep {best_epoch})",
                  flush=True)
            break
    if Xv is None:
        return model, epochs, None
    model.load_state_dict(best_state)
    return model, best_epoch, best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--n-anchors", type=int, default=0)
    ap.add_argument("--gap-days", type=int, default=30,
                    help="selection uses only anchors ending >= GAP days before "
                         "VAL; test retrain adds the gap anchors + val")
    ap.add_argument("--seeds", type=str, default="42,1337")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--batch", type=int, default=8192)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--dropout", type=float, default=0.15)
    ap.add_argument("--bce-w", type=float, default=0.7)
    ap.add_argument("--hidden", type=str, default="512,256")
    ap.add_argument("--drop-cols", type=str, default="")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--device", type=str, default="")
    ap.add_argument("--no-test", action="store_true")
    ap.add_argument("--smoke", action="store_true",
                    help="sanity run: 1 seed, batch<=2048, <=200 optimizer "
                         "steps, no test, nothing written")
    ap.add_argument("--notes", type=str, default="")
    args = ap.parse_args()
    if args.threads:
        os.environ["OMP_NUM_THREADS"] = str(args.threads)
    max_steps = None
    seeds = [int(s) for s in args.seeds.split(",")]
    if args.smoke:
        args.no_test = True
        args.batch = min(args.batch, SMOKE_MAX_BATCH)
        max_steps = SMOKE_MAX_STEPS
        seeds = seeds[:1]
    import polars as pl
    import torch
    torch.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))
    device = args.device or ("mps" if torch.backends.mps.is_available() else "cpu")
    cfg = dict(hidden=[int(h) for h in args.hidden.split(",")], dropout=args.dropout,
               lr=args.lr, wd=args.wd, bs=args.batch, patience=args.patience,
               bce_w=args.bce_w)
    print(f"device={device} seeds={seeds} smoke={args.smoke} cfg={cfg}", flush=True)

    t0 = time.time()
    cutoff = VAL_ANCHOR - timedelta(days=args.gap_days)
    tr_anchors = [a for a in available_train_anchors() if a <= cutoff]
    if args.n_anchors:
        tr_anchors = tr_anchors[-args.n_anchors:]
    gap_anchors = []
    if not args.no_test and args.gap_days:
        gap_anchors = [a for a in available_train_anchors() if cutoff < a < VAL_ANCHOR]
    print(f"train anchors: {[a.isoformat() for a in tr_anchors]}", flush=True)
    if gap_anchors:
        print(f"gap anchors (retrain only): "
              f"{[a.isoformat() for a in gap_anchors]}", flush=True)

    val = load_anchor(VAL_ANCHOR)
    cols = feature_cols(val)
    if args.drop_cols:
        drop = set(args.drop_cols.split(","))
        cols = [c for c in cols if c not in drop]
    print(f"{len(cols)} features", flush=True)
    f32 = [pl.col(c).cast(pl.Float32) for c in cols]

    # ONE buffer: [train | gap | val] rows; selection/retrain are views of it.
    heights = anchor_heights(tr_anchors)
    gap_heights = anchor_heights(gap_anchors)
    n_tr, n_gap, nv, d = sum(heights), sum(gap_heights), val.height, len(cols)
    Xfull = np.empty((n_tr + n_gap + nv, d), np.float32)
    ylog_full = np.empty(n_tr + n_gap + nv, np.float32)
    pos_ = 0
    for a, h in zip(tr_anchors + gap_anchors, heights + gap_heights):
        df = load_anchor(a, ["target"] + cols)
        assert df.height == h, f"height mismatch for {a}"
        Xfull[pos_:pos_ + h] = df.select(f32).to_numpy()
        ylog_full[pos_:pos_ + h] = np.log1p(
            df["target"].to_numpy().astype(np.float64)).astype(np.float32)
        pos_ += h
        del df
    Xfull[n_tr + n_gap:] = val.select(f32).to_numpy()
    yv_raw = val["target"].to_numpy().astype(np.float64)
    ylog_full[n_tr + n_gap:] = np.log1p(yv_raw).astype(np.float32)
    uid_val = val["user_id"].to_numpy()
    del val
    print(f"X {(n_tr, d)}, Xgap {(n_gap, d)}, Xv {(nv, d)}, "
          f"load {time.time()-t0:.0f}s", flush=True)

    stats = fit_stats(Xfull[:n_tr])                # train-only stats
    apply_stats(Xfull, stats)                      # transform all rows in place
    print(f"preprocess done {time.time()-t0:.0f}s", flush=True)

    X, Xv = Xfull[:n_tr], Xfull[n_tr + n_gap:]
    ylog, ylv = ylog_full[:n_tr], ylog_full[n_tr + n_gap:].astype(np.float64)
    pos_rate = float((ylog > 0).mean())
    print(f"train pos_rate={pos_rate:.4f}", flush=True)

    val_preds, best_epochs = [], []
    for seed in seeds:
        m, be, _ = train_one(X, ylog, Xv, ylv, cfg, seed, device, args.epochs,
                             max_steps=max_steps, tag=f"[s{seed}] ")
        pv = np.expm1(np.clip(predict_log(m, Xv, device), 0, None))
        print(f"[s{seed}] best_epoch={be} val_rmsle={rmsle(yv_raw, pv):.6f}",
              flush=True)
        val_preds.append(pv)
        best_epochs.append(be)
        del m
    pv_avg = np.mean(val_preds, axis=0)
    score = rmsle(yv_raw, pv_avg)

    if args.smoke:
        print(f"[SMOKE] {args.name} val_rmsle={score:.6f} "
              f"total {time.time()-t0:.0f}s", flush=True)
        return

    save_preds(args.name, "val", uid_val, pv_avg)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(MODELS_DIR / f"{args.name}_stats.npz", **stats)
    # freeze: what inference needs to rebuild this model besides the weights
    save_meta(args.name, kind="mlp2", feature_cols=cols, cfg=cfg, seeds=seeds,
              best_epochs=best_epochs, d_in=d, device=device,
              gap_days=args.gap_days, val_rmsle=float(score),
              stats_npz=f"{args.name}_stats.npz",
              weights=[f"{args.name}_seed{s}.pt" for s in seeds])
    notes = args.notes or (
        f"hurdle-mlp {args.hidden} bce_w{args.bce_w} do{args.dropout} "
        f"lr{args.lr} bs{args.batch} gap{args.gap_days} seeds={args.seeds} "
        f"{len(tr_anchors)}anch ep={best_epochs}")
    log_score(args.name, score, notes)

    if args.no_test:
        print(f"[DONE] {args.name} val_rmsle={score:.6f} "
              f"total {time.time()-t0:.0f}s", flush=True)
        return

    # retrain on train+gap+val (same buffer, no copy) for the stopped epochs
    test = load_anchor(TEST_ANCHOR)
    Xt = test.select(f32).to_numpy()
    uid_t = test["user_id"].to_numpy()
    del test
    apply_stats(Xt, stats)

    test_preds = []
    for seed, be in zip(seeds, best_epochs):
        m, _, _ = train_one(Xfull, ylog_full, None, None, cfg, seed, device,
                            max(1, be), tag=f"[s{seed} full] ")
        test_preds.append(np.expm1(np.clip(predict_log(m, Xt, device), 0, None)))
        save_torch(args.name, m, seed)   # retrain weights -> work/models/
        del m
    save_preds(args.name, "test", uid_t, np.mean(test_preds, axis=0))
    print(f"[DONE] {args.name} val_rmsle={score:.6f} "
          f"total {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

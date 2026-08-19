"""ZILN MLP trainer (zero-inflated lognormal head) following the exp_lib contract.

Same trunk/data plumbing as train_mlp2.py, but the two hurdle heads are
replaced by a ZILN head (Google lifetime_value, arXiv:1912.07753):
  three outputs from the trunk: logit p (zero-inflation gate), mu (linear),
  sigma = softplus(raw) + 1e-3 (clamped to <= 10).

Loss per row (batch mean):
  bce_w * BCE(logit, 1[y>0]) + 1[y>0] * lognormal NLL of y,
  NLL = log(sigma) + 0.5*((log(y) - mu)/sigma)^2  (+ const dropped),
  using log(y) (NOT log1p) on positive rows only; log(y) clamped to [-20, 30].

Prediction targets RMSLE directly:
  E[log1p(y)] = p * E[log1p(exp(Z))], Z ~ N(mu, sigma^2), computed with
  20-point Gauss-Hermite quadrature:
    E[f(Z)] = (1/sqrt(pi)) * sum_i w_i * f(mu + sqrt(2)*sigma*x_i),
  f(z) = log1p(exp(z)) = softplus(z) (stable). pred = expm1(clip(p*Equad, 0)).

Preprocessing / anchors / gap-days / early stopping / seed averaging are
identical to train_mlp2.py. Stats npz additionally stores per-seed val
percentiles of (p, mu, sigma) as calibration info. Preds contract unchanged:
single `pred` column in raw GMV scale.

--smoke: single seed, batch <= 2048, hard cap of 200 optimizer steps, forces
--no-test, does not write preds/scores/stats — just prints the val RMSLE.

Examples:
  train_mlpziln.py --name ziln_smoke --smoke --n-anchors 1 --threads 2
  train_mlpziln.py --name ziln_a --n-anchors 8 --no-test
  train_mlpziln.py --name ziln_final --seeds 42,1337
"""
from __future__ import annotations

import argparse
import math
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
LOGY_MIN, LOGY_MAX = -20.0, 30.0   # clamp for log(y) NLL inputs
SIGMA_MAX = 10.0
GH_X, GH_W = np.polynomial.hermite.hermgauss(20)   # Gauss-Hermite nodes/weights
CAL_PCTS = [1.0, 5.0, 25.0, 50.0, 75.0, 95.0, 99.0]


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
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class ZILNMLP(nn.Module):
        def __init__(self):
            super().__init__()
            layers, prev = [], d_in
            for h in hidden:
                layers += [nn.Linear(prev, h), nn.GELU(), nn.LayerNorm(h),
                           nn.Dropout(dropout)]
                prev = h
            self.trunk = nn.Sequential(*layers)
            self.head_logit = nn.Linear(prev, 1)   # zero-inflation gate P(y>0)
            self.head_mu = nn.Linear(prev, 1)      # lognormal mu (of log y)
            self.head_sigma = nn.Linear(prev, 1)   # raw -> softplus -> sigma

        def forward(self, x):
            z = self.trunk(x)
            logit = self.head_logit(z).squeeze(1)
            mu = self.head_mu(z).squeeze(1)
            sigma = torch.clamp(F.softplus(self.head_sigma(z).squeeze(1)) + 1e-3,
                                max=SIGMA_MAX)
            return logit, mu, sigma

    return ZILNMLP()


def predict_log(model, X: np.ndarray, device: str, bs: int = 65536,
                collect_heads: bool = False):
    """E[log1p(y)] = sigmoid(logit) * GH-quadrature E[softplus(Z)], Z~N(mu,s^2).

    Returns per-row predictions in log1p space (always >= 0). With
    collect_heads=True also returns (p, mu, sigma) arrays for calibration."""
    import torch
    import torch.nn.functional as F
    model.eval()
    gx = torch.as_tensor(GH_X, dtype=torch.float32, device=device)
    gw = torch.as_tensor(GH_W, dtype=torch.float32, device=device)
    sqrt2, inv_sqrt_pi = math.sqrt(2.0), 1.0 / math.sqrt(math.pi)
    outs, ps, mus, sgs = [], [], [], []
    with torch.no_grad():
        for i in range(0, X.shape[0], bs):
            xb = torch.from_numpy(np.ascontiguousarray(X[i:i + bs])).to(device)
            logit, mu, sigma = model(xb)
            zq = mu.unsqueeze(1) + sqrt2 * sigma.unsqueeze(1) * gx   # (b, 20)
            equad = (F.softplus(zq) * gw).sum(dim=1) * inv_sqrt_pi
            p = torch.sigmoid(logit)
            outs.append((p * equad).float().cpu().numpy())
            if collect_heads:
                ps.append(p.float().cpu().numpy())
                mus.append(mu.float().cpu().numpy())
                sgs.append(sigma.float().cpu().numpy())
    pred = np.concatenate(outs)
    if collect_heads:
        return pred, (np.concatenate(ps), np.concatenate(mus),
                      np.concatenate(sgs))
    return pred


def train_one(X, ylog, logy, Xv, ylv, cfg, seed, device, epochs,
              max_steps=None, tag=""):
    """One ZILN-MLP fit. With Xv: early stop on val rmsle, return best-epoch
    model. Without Xv: fixed `epochs` run (cosine compressed to that length).
    ylog = log1p(y) (pos mask + val metric); logy = clamped log(y) (NLL)."""
    import torch
    import torch.nn.functional as F
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = build_model(X.shape[1], cfg["hidden"], cfg["dropout"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=epochs, eta_min=cfg["lr"] * 0.01)
    n, bs, bce_w, gclip = X.shape[0], cfg["bs"], cfg["bce_w"], cfg["grad_clip"]
    best, best_epoch, bad, best_state = np.inf, 0, 0, None
    steps = 0
    for ep in range(1, epochs + 1):
        model.train()
        perm = rng.permutation(n)
        bce_sum = torch.zeros((), device=device)
        nll_sum = torch.zeros((), device=device)
        seen, pos_seen = 0, 0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            xb = torch.from_numpy(X[idx]).to(device)
            yb = torch.from_numpy(ylog[idx]).to(device)
            lb = torch.from_numpy(logy[idx]).to(device)
            pos = yb > 0
            logit, mu, sigma = model(xb)
            bce_row = F.binary_cross_entropy_with_logits(
                logit, pos.float(), reduction="none")
            nll_all = torch.log(sigma) + 0.5 * ((lb - mu) / sigma) ** 2
            nll_row = torch.where(pos, nll_all, torch.zeros_like(nll_all))
            loss = (bce_w * bce_row + nll_row).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if gclip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gclip)
            opt.step()
            bce_sum += bce_row.detach().sum()
            nll_sum += nll_row.detach().sum()
            seen += len(idx)
            pos_seen += int(pos.sum())
            steps += 1
            if max_steps is not None and steps >= max_steps:
                break
        sched.step()
        tr_bce = float(bce_sum) / max(seen, 1)
        tr_nll = float(nll_sum) / max(pos_seen, 1)
        out_of_budget = max_steps is not None and steps >= max_steps
        if Xv is None:
            print(f"{tag}ep {ep}/{epochs} bce {tr_bce:.5f} nll_pos {tr_nll:.5f}",
                  flush=True)
            if out_of_budget:
                break
            continue
        pred_log = predict_log(model, Xv, device).astype(np.float64)
        score = float(np.sqrt(np.mean((np.clip(pred_log, 0, None) - ylv) ** 2)))
        mark = ""
        if not np.isfinite(score):
            print(f"{tag}ep {ep} non-finite val score, stopping", flush=True)
            break
        if score < best - 1e-5:
            best, best_epoch, bad = score, ep, 0
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
            mark = " *"
        else:
            bad += 1
        print(f"{tag}ep {ep} bce {tr_bce:.5f} nll_pos {tr_nll:.5f} "
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
    assert best_state is not None, "no finite val score reached"
    model.load_state_dict(best_state)
    return model, best_epoch, best


def head_summary(p: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> str:
    return (f"p_mean={p.mean():.4f} mu_p50={np.percentile(mu, 50):.3f} "
            f"sigma_p50={np.percentile(sigma, 50):.3f} "
            f"sigma_max={sigma.max():.3f}")


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
    ap.add_argument("--bce-w", type=float, default=1.0)
    ap.add_argument("--grad-clip", type=float, default=5.0,
                    help="max grad norm (0 disables)")
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
               bce_w=args.bce_w, grad_clip=args.grad_clip)
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

    def targets_of(y_raw: np.ndarray):
        """(log1p(y), clamped log(y) with zeros on y<=0) as float32."""
        ylog = np.log1p(y_raw).astype(np.float32)
        logy = np.zeros_like(ylog)
        posm = y_raw > 0
        logy[posm] = np.clip(np.log(y_raw[posm]), LOGY_MIN, LOGY_MAX
                             ).astype(np.float32)
        return ylog, logy

    # ONE buffer: [train | gap | val] rows; selection/retrain are views of it.
    heights = anchor_heights(tr_anchors)
    gap_heights = anchor_heights(gap_anchors)
    n_tr, n_gap, nv, d = sum(heights), sum(gap_heights), val.height, len(cols)
    Xfull = np.empty((n_tr + n_gap + nv, d), np.float32)
    ylog_full = np.empty(n_tr + n_gap + nv, np.float32)
    logy_full = np.empty(n_tr + n_gap + nv, np.float32)
    pos_ = 0
    for a, h in zip(tr_anchors + gap_anchors, heights + gap_heights):
        df = load_anchor(a, ["target"] + cols)
        assert df.height == h, f"height mismatch for {a}"
        Xfull[pos_:pos_ + h] = df.select(f32).to_numpy()
        y_raw = df["target"].to_numpy().astype(np.float64)
        ylog_full[pos_:pos_ + h], logy_full[pos_:pos_ + h] = targets_of(y_raw)
        pos_ += h
        del df
    Xfull[n_tr + n_gap:] = val.select(f32).to_numpy()
    yv_raw = val["target"].to_numpy().astype(np.float64)
    ylog_full[n_tr + n_gap:], logy_full[n_tr + n_gap:] = targets_of(yv_raw)
    uid_val = val["user_id"].to_numpy()
    del val
    print(f"X {(n_tr, d)}, Xgap {(n_gap, d)}, Xv {(nv, d)}, "
          f"load {time.time()-t0:.0f}s", flush=True)

    stats = fit_stats(Xfull[:n_tr])                # train-only stats
    apply_stats(Xfull, stats)                      # transform all rows in place
    print(f"preprocess done {time.time()-t0:.0f}s", flush=True)

    X, Xv = Xfull[:n_tr], Xfull[n_tr + n_gap:]
    ylog, logy = ylog_full[:n_tr], logy_full[:n_tr]
    ylv = ylog_full[n_tr + n_gap:].astype(np.float64)
    pos_rate = float((ylog > 0).mean())
    print(f"train pos_rate={pos_rate:.4f}", flush=True)

    val_preds, best_epochs = [], []
    cal_p, cal_mu, cal_sigma = [], [], []
    for seed in seeds:
        m, be, _ = train_one(X, ylog, logy, Xv, ylv, cfg, seed, device,
                             args.epochs, max_steps=max_steps, tag=f"[s{seed}] ")
        pv_log, (hp, hmu, hsg) = predict_log(m, Xv, device, collect_heads=True)
        pv = np.expm1(np.clip(pv_log, 0, None))
        print(f"[s{seed}] best_epoch={be} val_rmsle={rmsle(yv_raw, pv):.6f} "
              f"{head_summary(hp, hmu, hsg)}", flush=True)
        val_preds.append(pv)
        best_epochs.append(be)
        cal_p.append(np.percentile(hp, CAL_PCTS))
        cal_mu.append(np.percentile(hmu, CAL_PCTS))
        cal_sigma.append(np.percentile(hsg, CAL_PCTS))
        del m, hp, hmu, hsg
    pv_avg = np.mean(val_preds, axis=0)
    score = rmsle(yv_raw, pv_avg)

    if args.smoke:
        print(f"[SMOKE] {args.name} val_rmsle={score:.6f} "
              f"total {time.time()-t0:.0f}s", flush=True)
        return

    save_preds(args.name, "val", uid_val, pv_avg)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(MODELS_DIR / f"{args.name}_stats.npz", **stats,
             ziln_cal_pcts=np.asarray(CAL_PCTS, np.float32),
             ziln_val_p=np.stack(cal_p).astype(np.float32),
             ziln_val_mu=np.stack(cal_mu).astype(np.float32),
             ziln_val_sigma=np.stack(cal_sigma).astype(np.float32))
    # freeze: what inference needs to rebuild this model besides the weights
    save_meta(args.name, kind="mlpziln", feature_cols=cols, cfg=cfg,
              seeds=seeds, best_epochs=best_epochs, d_in=d, device=device,
              gap_days=args.gap_days, gh_points=len(GH_X),
              sigma_max=SIGMA_MAX, val_rmsle=float(score),
              stats_npz=f"{args.name}_stats.npz",
              weights=[f"{args.name}_seed{s}.pt" for s in seeds])
    notes = args.notes or (
        f"ziln-mlp {args.hidden} bce_w{args.bce_w} do{args.dropout} "
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
        m, _, _ = train_one(Xfull, ylog_full, logy_full, None, None, cfg, seed,
                            device, max(1, be), tag=f"[s{seed} full] ")
        test_preds.append(np.expm1(np.clip(predict_log(m, Xt, device), 0, None)))
        save_torch(args.name, m, seed)   # retrain weights -> work/models/
        del m
    save_preds(args.name, "test", uid_t, np.mean(test_preds, axis=0))
    print(f"[DONE] {args.name} val_rmsle={score:.6f} "
          f"total {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

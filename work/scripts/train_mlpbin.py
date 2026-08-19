"""Binned-classification MLP trainer (regression-as-classification), exp_lib contract.

Shared trunk on tabular features -> ONE softmax head over K+1 bins:
  bin 0            = exactly-zero target;
  bins 1..K (K=31) = quantile bins of log1p(y) over POSITIVE targets, edges
                     computed on the selection-train rows only.
Loss: cross-entropy (optional label smoothing via --label-smoothing, e.g. 0.05;
note eps>0 slightly biases the expectation decode upward on zero rows).
Decode: E[log1p(y)] = sum_k p_k * c_k with c_0 = 0 and c_k = mean log1p(y) of
selection-train rows in bin k;  pred = expm1(clip(E, 0, None)).
Seed averaging happens on the decoded E[log1p] (before expm1).

Trunk: Linear(F->512) GELU LayerNorm Dropout(0.15) -> 256 (same block) -> head
Linear(256 -> K+1). Preprocessing (fit on train anchors only; reused for
val/test): median-impute NaN -> clip to train [p1, p99] -> standardize. Stats
plus bin edges/centers saved to work/models/NAME_stats.npz on non-smoke runs.

Anchors: like train_gbdt --gap-days (default 30): selection trains only on
anchors ending >= gap days before VAL (no target-window overlap with val);
the test retrain adds the gap anchors + VAL and runs for the stopped epochs
(bin edges/centers stay fixed from the selection-train fit). Early stop on
VAL RMSLE, patience 4. USE_V2/USE_V3 feature tiers are joined automatically
by common.load_anchor (both default to on here).

Memory: train(+gap)+val features live in ONE preallocated C-order float32
array; selection/retrain use views of it (no duplication). Bin labels are a
uint8 array over the same rows.

--smoke: single seed, batch <= 2048, hard cap of 200 optimizer steps, forces
--no-test, does not write preds/scores/stats — just prints the val RMSLE.

Examples:
  train_mlpbin.py --name mlpbin_smoke --smoke --n-anchors 1 --threads 2
  train_mlpbin.py --name mlpbin_a --n-anchors 8 --no-test
  train_mlpbin.py --name mlpbin_final --seeds 42,1337
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
from calibrate import apply_shifts, fit_shifts
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


def assign_bins(ylog: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """0 for exactly-zero rows; 1..K for positives via the K-1 internal edges."""
    out = np.zeros(ylog.shape[0], np.uint8)
    m = ylog > 0
    out[m] = (1 + np.searchsorted(edges, ylog[m].astype(np.float64),
                                  side="right")).astype(np.uint8)
    return out


def fit_bins(ylog_tr: np.ndarray, k: int):
    """Quantile bin edges over positive log1p targets of the selection-train
    rows, plus per-bin centers c_k = mean log1p(y) (c_0 = 0). Returns
    (edges[k-1], centers[k+1], counts[k+1])."""
    assert k + 1 <= 255, "uint8 labels support at most 254 positive bins"
    pos = ylog_tr[ylog_tr > 0].astype(np.float64)
    assert pos.size >= k, f"only {pos.size} positive train rows for {k} bins"
    edges = np.quantile(pos, np.linspace(0.0, 1.0, k + 1)[1:-1])
    b = assign_bins(ylog_tr, edges)
    cnt = np.bincount(b, minlength=k + 1).astype(np.int64)
    sm = np.bincount(b, weights=ylog_tr.astype(np.float64), minlength=k + 1)
    centers = np.zeros(k + 1, np.float64)
    nz = cnt > 0
    centers[nz] = sm[nz] / cnt[nz]
    centers[0] = 0.0
    for j in range(1, k + 1):  # empty bin (duplicate quantile edge): edge value
        if cnt[j] == 0:
            centers[j] = edges[min(j - 1, edges.size - 1)]
    return edges, centers, cnt


def anchor_heights(anchors) -> list[int]:
    import polars as pl
    return [
        pl.scan_parquet(FEATURES_DIR / f"anchor={a.isoformat()}.parquet")
        .select(pl.len()).collect().item()
        for a in anchors
    ]


def cal_holdout(lp: np.ndarray, ly: np.ndarray, y_raw: np.ndarray,
                half: np.ndarray, bins: int = 24):
    """Honest binned log-shift calibration: fit on half the users, score the rest.

    Every model is calibrated before blending (KNOWLEDGE.md), so the calibrated
    number is the decision-relevant one; raw scores mostly reflect the ~0.25 log
    level bias that calibration removes anyway. Returns (raw_holdout, cal_holdout).
    """
    lp = np.clip(lp, 0, None)
    c, s = fit_shifts(lp[half], ly[half], bins)
    return (float(rmsle(y_raw[~half], np.expm1(lp[~half]))),
            float(rmsle(y_raw[~half], np.expm1(apply_shifts(lp[~half], c, s)))))


def cal_rmsle_2fold(pred_log: np.ndarray, ly: np.ndarray, y_raw: np.ndarray,
                    half: np.ndarray, bins: int):
    """Honest calibrated val RMSLE of a checkpoint -> (pooled, single_fold).

    Copy of train_fusion3.cal_rmsle_2fold.  Same transform as calibrate.py
    (fit_shifts / apply_shifts imported from it), but the shift table is never
    applied to the rows it was fitted on: half A fits the shifts that score half B
    and vice versa, so no row is calibrated by itself.  `pooled` scores all rows
    that way (the criterion actually used); `single_fold` is only the B half, i.e.
    exactly the number cal_holdout / calibrate.py print, kept for eyeballing the
    two against each other in the logs.  Both folds are honest, so using both is
    the same estimator with half the variance — which matters here, because the
    differences being resolved are ~1e-4.
    """
    lp = np.clip(np.asarray(pred_log, dtype=np.float64), 0, None)
    out = np.empty_like(lp)
    c_a, s_a = fit_shifts(lp[half], ly[half], bins)
    out[~half] = apply_shifts(lp[~half], c_a, s_a)
    c_b, s_b = fit_shifts(lp[~half], ly[~half], bins)
    out[half] = apply_shifts(lp[half], c_b, s_b)
    return (rmsle(y_raw, np.expm1(out)),
            rmsle(y_raw[~half], np.expm1(out[~half])))


def build_model(d_in: int, hidden: list[int], dropout: float, n_classes: int,
                norm: str = "layer"):
    import torch.nn as nn

    def make_norm(h):
        if norm == "layer":
            return [nn.LayerNorm(h)]
        if norm == "batch":
            return [nn.BatchNorm1d(h)]
        if norm == "none":
            return []
        raise ValueError(f"unknown norm {norm!r}")

    class BinnedMLP(nn.Module):
        def __init__(self):
            super().__init__()
            layers, prev = [], d_in
            for h in hidden:
                layers += [nn.Linear(prev, h), nn.GELU(), *make_norm(h),
                           nn.Dropout(dropout)]
                prev = h
            self.trunk = nn.Sequential(*layers)
            self.head = nn.Linear(prev, n_classes)  # softmax over K+1 bins

        def forward(self, x):
            return self.head(self.trunk(x))

    return BinnedMLP()


def predict_elog(model, X: np.ndarray, centers_t, device: str,
                 bs: int = 65536) -> np.ndarray:
    """E[log1p(y)] per row = softmax(logits) @ centers (>= 0: centers >= 0)."""
    import torch
    model.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, X.shape[0], bs):
            xb = torch.from_numpy(np.ascontiguousarray(X[i:i + bs])).to(device)
            p = torch.softmax(model(xb), dim=1)
            outs.append((p @ centers_t).float().cpu().numpy())
    return np.concatenate(outs)


def train_one(X, yb, Xv, ylv, centers_t, cfg, seed, device, epochs,
              max_steps=None, tag="", es_cal=None):
    """One binned-MLP fit. With Xv: early stop on val rmsle, return best-epoch
    model. Without Xv: fixed `epochs` run (cosine compressed to that length).

    es_cal (from --es-metric cal) swaps the early-stopping CRITERION for the
    honest calibrated val RMSLE; it never touches training itself — same seeds,
    same batches, same steps, same RNG streams, it only decides which epoch's
    checkpoint is kept and when patience runs out."""
    import math

    import torch
    import torch.nn.functional as F
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = build_model(X.shape[1], cfg["hidden"], cfg["dropout"],
                        centers_t.shape[0], cfg.get("norm", "layer")).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    n, bs, ls = X.shape[0], cfg["bs"], cfg["ls"]
    warm = cfg.get("warmup", 0.0)
    per_step_sched = warm > 0
    if per_step_sched:
        # linear warmup -> cosine over the PLANNED step budget (epochs * steps/ep,
        # capped by max_steps); floor at 1% of lr like the epoch-wise cosine.
        # sched_epochs (= --epochs) keeps the LR trajectory identical in the test
        # retrain, which runs the SAME schedule but stops at best_epoch.
        total = cfg.get("sched_epochs", epochs) * int(np.ceil(n / bs))
        if max_steps is not None:
            total = min(total, max_steps)
        wsteps = max(1, int(round(warm * total)))

        def lr_lambda(s):
            if s < wsteps:
                return (s + 1) / wsteps
            t = (s - wsteps) / max(1, total - wsteps)
            return 0.01 + 0.99 * 0.5 * (1.0 + math.cos(math.pi * min(1.0, t)))

        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    else:
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=epochs, eta_min=cfg["lr"] * 0.01)
    best, best_epoch, bad, best_state = np.inf, 0, 0, None
    steps = 0
    for ep in range(1, epochs + 1):
        model.train()
        perm = rng.permutation(n)
        ce_sum = torch.zeros((), device=device)
        seen = 0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            if len(idx) < 2:
                continue                      # BatchNorm needs >1 row
            xb = torch.from_numpy(X[idx]).to(device)
            tb = torch.from_numpy(yb[idx].astype(np.int64)).to(device)
            logits = model(xb)
            loss = F.cross_entropy(logits, tb, label_smoothing=ls)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            ce_sum += loss.detach() * len(idx)
            seen += len(idx)
            steps += 1
            if per_step_sched:
                sched.step()
            if max_steps is not None and steps >= max_steps:
                break
        if not per_step_sched:
            sched.step()
        tr_ce = float(ce_sum) / seen
        out_of_budget = max_steps is not None and steps >= max_steps
        if Xv is None:
            print(f"{tag}ep {ep}/{epochs} ce {tr_ce:.5f}", flush=True)
            if out_of_budget:
                break
            continue
        elog = predict_elog(model, Xv, centers_t, device).astype(np.float64)
        score = float(np.sqrt(np.mean((np.clip(elog, 0, None) - ylv) ** 2)))
        crit, extra = score, ""
        if es_cal is not None:
            tc = time.time()
            vc, vc1 = cal_rmsle_2fold(elog, es_cal["ly"], es_cal["y"],
                                      es_cal["half"], es_cal["bins"])
            dt = time.time() - tc
            es_cal["secs"].append(dt)
            crit = vc
            extra = f" | CAL {vc:.5f} (holdout {vc1:.5f}, +{dt:.2f}s)"
        mark = ""
        if crit < best - 1e-5:
            best, best_epoch, bad = crit, ep, 0
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
            mark = " *"
        else:
            bad += 1
        print(f"{tag}ep {ep} ce {tr_ce:.5f} val_rmsle {score:.5f}{extra}{mark}",
              flush=True)
        if out_of_budget:
            print(f"{tag}step budget {max_steps} reached at ep {ep}", flush=True)
            break
        if bad >= cfg["patience"]:
            print(f"{tag}early stop at ep {ep} (best {best:.5f} @ ep {best_epoch})",
                  flush=True)
            break
    else:
        if Xv is not None:
            print(f"{tag}БЮДЖЕТ ЭПОХ ИСЧЕРПАН: прошли все {epochs} эпох, ранняя "
                  f"остановка не сработала (best {best:.5f} @ ep {best_epoch}) — "
                  f"оптимум может лежать дальше, нужен больший --epochs", flush=True)
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
    ap.add_argument("--k-bins", type=int, default=31,
                    help="positive-target quantile bins (classes = K+1)")
    ap.add_argument("--label-smoothing", type=float, default=0.0,
                    help="CE label smoothing eps (e.g. 0.05); eps>0 slightly "
                         "biases the expectation decode up on zero rows")
    ap.add_argument("--hidden", type=str, default="512,256")
    ap.add_argument("--norm", type=str, default="layer",
                    choices=["layer", "batch", "none"],
                    help="normalization inside each trunk block")
    ap.add_argument("--warmup", type=float, default=0.0,
                    help="fraction of the planned step budget spent on linear "
                         "warmup; >0 switches the cosine schedule from per-epoch "
                         "(T_max=--epochs) to per-step over the planned budget")
    ap.add_argument("--drop-cols", type=str, default="")
    ap.add_argument("--es-metric", choices=("raw", "cal"), default="raw",
                    help="early-stopping criterion: raw val RMSLE (default, keeps "
                         "historical behaviour bit-for-bit) or the honest calibrated "
                         "one (calibrate.py binned log-shift, 2-fold over users). "
                         "Every prediction file is calibrated before it reaches a "
                         "blend, and that rewrites the LEVEL of the forecast, so the "
                         "raw criterion spends its checkpoint choice on a level that "
                         "is about to be overwritten for free and pays in RANKING, "
                         "which calibration preserves")
    ap.add_argument("--es-bins", type=int, default=24,
                    help="quantile bins of the --es-metric cal calibration "
                         "(calibrate.py default is 24; keep them equal)")
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
               ls=args.label_smoothing, norm=args.norm, warmup=args.warmup,
               sched_epochs=args.epochs)
    print(f"device={device} seeds={seeds} smoke={args.smoke} "
          f"k={args.k_bins} cfg={cfg}", flush=True)

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

    # bins/centers from selection-train rows only; labels for ALL rows so the
    # test retrain (train+gap+val) reuses the same fixed edges/centers.
    edges, centers, cnt = fit_bins(ylog, args.k_bins)
    bins_full = assign_bins(ylog_full, edges)
    bins = bins_full[:n_tr]
    print(f"train pos_rate={pos_rate:.4f} bins: k={args.k_bins} "
          f"edges[{edges[0]:.3f}..{edges[-1]:.3f}] "
          f"centers[{centers[1]:.3f}..{centers[-1]:.3f}] "
          f"empty={int((cnt[1:] == 0).sum())}", flush=True)
    centers_t = torch.tensor(centers, dtype=torch.float32, device=device)

    # honest 2-fold user split, shared by --es-metric cal and the report below
    half = np.random.default_rng(0).permutation(nv) < nv // 2
    es_cal = None
    if args.es_metric == "cal":
        es_cal = {"ly": np.log1p(np.clip(yv_raw, 0, None)), "y": yv_raw,
                  "half": half, "bins": args.es_bins, "secs": []}
        print(f"es_metric=cal: early stop on the honest calibrated val RMSLE "
              f"({args.es_bins} bins, 2-fold over users)", flush=True)

    val_elogs, best_epochs = [], []
    for seed in seeds:
        m, be, _ = train_one(X, bins, Xv, ylv, centers_t, cfg, seed, device,
                             args.epochs, max_steps=max_steps, tag=f"[s{seed}] ",
                             es_cal=es_cal)
        ev = predict_elog(m, Xv, centers_t, device).astype(np.float64)
        pv = np.expm1(np.clip(ev, 0, None))
        print(f"[s{seed}] best_epoch={be} val_rmsle={rmsle(yv_raw, pv):.6f}",
              flush=True)
        val_elogs.append(ev)
        best_epochs.append(be)
        del m
    # seed averaging in decoded E[log1p] space, THEN expm1
    elog_avg = np.mean(val_elogs, axis=0)
    pv_avg = np.expm1(np.clip(elog_avg, 0, None))
    score = rmsle(yv_raw, pv_avg)
    # decision-relevant number: every model is calibrated before blending, so
    # compare configs AFTER calibration (raw gaps are mostly level artifacts).
    raw_h, cal_h = cal_holdout(elog_avg, ylv, yv_raw, half)
    if es_cal is not None and es_cal["secs"]:
        print(f"es_metric=cal cost: {len(es_cal['secs'])} calibrations, mean "
              f"{np.mean(es_cal['secs']):.3f}s, total {sum(es_cal['secs']):.1f}s",
              flush=True)

    if args.smoke:
        print(f"[SMOKE] {args.name} val_rmsle={score:.6f} "
              f"holdout raw={raw_h:.6f} cal={cal_h:.6f} "
              f"total {time.time()-t0:.0f}s", flush=True)
        return

    save_preds(args.name, "val", uid_val, pv_avg)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(MODELS_DIR / f"{args.name}_stats.npz", **stats,
             edges=edges, centers=centers)
    # freeze: what inference needs to rebuild this model besides the weights
    # (bin edges/centers live in the stats npz above, they decode the logits)
    save_meta(args.name, kind="mlpbin", feature_cols=cols, cfg=cfg, seeds=seeds,
              best_epochs=best_epochs, d_in=d, n_classes=int(len(centers)),
              k_bins=args.k_bins, device=device, gap_days=args.gap_days,
              val_rmsle=float(score), stats_npz=f"{args.name}_stats.npz",
              weights=[f"{args.name}_seed{s}.pt" for s in seeds])
    notes = args.notes or (
        f"binned-mlp k{args.k_bins} ls{args.label_smoothing} {args.hidden} "
        f"do{args.dropout} norm={args.norm} warmup={args.warmup} lr{args.lr} "
        f"bs{args.batch} gap{args.gap_days} seeds={args.seeds} "
        f"{len(tr_anchors)}anch ep={best_epochs}")
    if args.es_metric != "raw":
        notes = f"{notes}; es={args.es_metric}"
    notes = f"{notes}; holdout raw={raw_h:.6f} cal={cal_h:.6f}"
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

    test_elogs = []
    for seed, be in zip(seeds, best_epochs):
        m, _, _ = train_one(Xfull, bins_full, None, None, centers_t, cfg, seed,
                            device, max(1, be), tag=f"[s{seed} full] ")
        test_elogs.append(predict_elog(m, Xt, centers_t, device).astype(np.float64))
        save_torch(args.name, m, seed)   # retrain weights -> work/models/
        del m
    pt = np.expm1(np.clip(np.mean(test_elogs, axis=0), 0, None))
    save_preds(args.name, "test", uid_t, pt)
    print(f"[DONE] {args.name} val_rmsle={score:.6f} "
          f"total {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

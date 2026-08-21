"""Ordinal-CDF MLP trainer (CREAD / RQ-Reg class) following the exp_lib contract.

Same trunk/data plumbing as train_mlpziln.py, but the parametric ZILN head is
replaced by a discretized-CDF head (idea #5 of the research report; on
heavy-tailed spend the ordinal-CDF class beats ZILN in the CREAD paper):

  z = log1p(y) is discretized into --n-bins B=64 ORDERED buckets by quantiles
  of the TRAIN distribution.  The zero bucket is separate (~46% of targets are
  exactly 0): edge e_0 = 0, and e_1..e_{B-1} are the j/(B-1) quantiles of the
  POSITIVE train z.  Ties are deduped, so the effective threshold count K can
  be < B-1 (it is logged).

  The head is one Linear with K = B-1 outputs: logits of the survival function
  s_k = P(z > e_k), k = 0..K-1 (s_0 is exactly the zero gate P(y > 0)).
  Loss = BCE over all K sigmoids with label smoothing --label-smooth 0.05:
  t_k = 1[z > e_k] -> t_k*(1-eps) + eps/2, mean over rows x thresholds.

Point prediction targets RMSLE through the CDF identity
  E[z] = integral_0^zmax P(z > t) dt  ~=  sum_k (e_{k+1} - e_k) * s_k,
i.e. bucket widths weighted by the cumulative (survival) probabilities.  At
decode the sigmoids are forced monotone (running cummin over k -- independent
sigmoids do not guarantee a valid survival curve), then pred = expm1(E[z]).
--desmooth optionally inverts the label smoothing at decode
(s -> (s - eps/2)/(1-eps), clipped to [0,1]); off by default -- the binned
log-shift calibration every blend member gets absorbs the smoothing bias.

Preprocessing / anchors / gap-days / early stopping / seed averaging /
--feat-prep / --es-metric are identical to train_mlpziln.py.  Stats npz stores
the bucket edges plus per-seed val percentiles of (s_0, E[z]) as calibration
info.  Preds contract unchanged: single `pred` column in raw GMV scale.

--smoke: single seed, batch <= 2048, hard cap of 200 optimizer steps, forces
--no-test, does not write preds/scores/stats -- just prints the val RMSLE.

Examples:
  train_mlpcdf.py --name cdf_smoke --smoke --n-anchors 1 --threads 2
  train_mlpcdf.py --name mlpcdf --n-anchors 14 --seeds 42 --epochs 40
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
from featprep import MODES as PREP_MODES
from featprep import apply_stats, fit_stats
from model_io import save_meta, save_torch

MODELS_DIR = WORK / "models"
SMOKE_MAX_STEPS = 200
SMOKE_MAX_BATCH = 2048
CAL_PCTS = [1.0, 5.0, 25.0, 50.0, 75.0, 95.0, 99.0]


def anchor_heights(anchors) -> list[int]:
    import polars as pl
    return [
        pl.scan_parquet(FEATURES_DIR / f"anchor={a.isoformat()}.parquet")
        .select(pl.len()).collect().item()
        for a in anchors
    ]


def cal_rmsle_2fold(pred_log: np.ndarray, ly: np.ndarray, y_raw: np.ndarray,
                    half: np.ndarray, bins: int):
    """Honest calibrated val RMSLE of a checkpoint -> (pooled, single_fold).

    Copy of train_mlpziln.cal_rmsle_2fold (itself a copy of train_fusion3's):
    half A fits the calibrate.py shifts that score half B and vice versa, so no
    row is ever calibrated by itself."""
    lp = np.clip(np.asarray(pred_log, dtype=np.float64), 0, None)
    out = np.empty_like(lp)
    c_a, s_a = fit_shifts(lp[half], ly[half], bins)
    out[~half] = apply_shifts(lp[~half], c_a, s_a)
    c_b, s_b = fit_shifts(lp[~half], ly[~half], bins)
    out[half] = apply_shifts(lp[half], c_b, s_b)
    return (rmsle(y_raw, np.expm1(out)),
            rmsle(y_raw[~half], np.expm1(out[~half])))


def fit_edges(ylog_tr: np.ndarray, n_bins: int) -> np.ndarray:
    """Bucket edges from TRAIN z=log1p(y) only: e_0=0 (separate zero bucket),
    e_j = quantile(z_pos, j/(n_bins-1)), j=1..n_bins-1 (last = train max).
    Returns strictly increasing float64 edges, len <= n_bins (ties deduped)."""
    zpos = ylog_tr[ylog_tr > 0].astype(np.float64)
    assert zpos.size, "no positive targets in train"
    qs = np.arange(1, n_bins) / (n_bins - 1)
    edges = np.concatenate([[0.0], np.quantile(zpos, qs)])
    edges = np.unique(np.maximum.accumulate(edges))
    if len(edges) < n_bins:
        print(f"[edges] {n_bins - len(edges)} tied quantile edges deduped -> "
              f"{len(edges) - 1} thresholds", flush=True)
    return edges


def build_model(d_in: int, hidden: list[int], dropout: float, k_out: int):
    import torch.nn as nn

    class CDFMLP(nn.Module):
        def __init__(self):
            super().__init__()
            layers, prev = [], d_in
            for h in hidden:
                layers += [nn.Linear(prev, h), nn.GELU(), nn.LayerNorm(h),
                           nn.Dropout(dropout)]
                prev = h
            self.trunk = nn.Sequential(*layers)
            self.head = nn.Linear(prev, k_out)   # survival logits P(z > e_k)

        def forward(self, x):
            return self.head(self.trunk(x))

    return CDFMLP()


def predict_log(model, X: np.ndarray, device: str, widths, desmooth: float,
                bs: int = 65536, collect_heads: bool = False):
    """E[log1p(y)] = sum_k width_k * s_k with s forced monotone (cummin).

    widths: (K,) torch tensor on `device`.  Returns per-row predictions in
    log1p space (always >= 0).  With collect_heads=True also returns
    (s_0 = P(y>0), E[z], monotonicity-violation row rate)."""
    import torch
    model.eval()
    outs, s0s, viol_rows, n_rows = [], [], 0, 0
    with torch.no_grad():
        for i in range(0, X.shape[0], bs):
            xb = torch.from_numpy(np.ascontiguousarray(X[i:i + bs])).to(device)
            s = torch.sigmoid(model(xb))                     # (b, K)
            if desmooth:
                s = ((s - desmooth / 2) / (1.0 - desmooth)).clamp(0.0, 1.0)
            if collect_heads:
                viol_rows += int((s[:, 1:] > s[:, :-1] + 1e-6).any(dim=1).sum())
                n_rows += s.shape[0]
            s = torch.cummin(s, dim=1).values                # valid survival
            ez = (s * widths).sum(dim=1)
            outs.append(ez.float().cpu().numpy())
            if collect_heads:
                s0s.append(s[:, 0].float().cpu().numpy())
    pred = np.concatenate(outs)
    if collect_heads:
        return pred, (np.concatenate(s0s), pred.copy(),
                      viol_rows / max(n_rows, 1))
    return pred


def train_one(X, ylog, Xv, ylv, cfg, edges: np.ndarray, seed, device, epochs,
              max_steps=None, tag="", es_cal=None):
    """One CDF-MLP fit. With Xv: early stop on val rmsle (log space), return
    best-epoch model. Without Xv: fixed `epochs` run (cosine compressed).
    ylog = log1p(y); the K binary labels 1[z > e_k] are built on the fly.

    es_cal (from --es-metric cal) swaps the early-stopping CRITERION for the
    honest calibrated val RMSLE; it never touches training itself."""
    import torch
    import torch.nn.functional as F
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    thr = torch.as_tensor(edges[:-1], dtype=torch.float32, device=device)   # (K,)
    widths = torch.as_tensor(np.diff(edges), dtype=torch.float32, device=device)
    k_out = int(thr.shape[0])
    eps = cfg["label_smooth"]
    model = build_model(X.shape[1], cfg["hidden"], cfg["dropout"], k_out).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=epochs, eta_min=cfg["lr"] * 0.01)
    n, bs, gclip = X.shape[0], cfg["bs"], cfg["grad_clip"]
    best, best_epoch, bad, best_state = np.inf, 0, 0, None
    steps = 0
    for ep in range(1, epochs + 1):
        model.train()
        perm = rng.permutation(n)
        bce_sum = torch.zeros((), device=device)
        seen = 0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            xb = torch.from_numpy(X[idx]).to(device)
            zb = torch.from_numpy(ylog[idx]).to(device)
            t = (zb.unsqueeze(1) > thr).float()              # (b, K)
            if eps:
                t = t * (1.0 - eps) + 0.5 * eps
            logits = model(xb)
            loss = F.binary_cross_entropy_with_logits(logits, t)  # mean b*K
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if gclip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gclip)
            opt.step()
            bce_sum += loss.detach() * len(idx)
            seen += len(idx)
            steps += 1
            if max_steps is not None and steps >= max_steps:
                break
        sched.step()
        tr_bce = float(bce_sum) / max(seen, 1)
        out_of_budget = max_steps is not None and steps >= max_steps
        if Xv is None:
            print(f"{tag}ep {ep}/{epochs} bce {tr_bce:.5f}", flush=True)
            if out_of_budget:
                break
            continue
        pred_log = predict_log(model, Xv, device, widths,
                               cfg["desmooth"]).astype(np.float64)
        score = float(np.sqrt(np.mean((np.clip(pred_log, 0, None) - ylv) ** 2)))
        mark = ""
        if not np.isfinite(score):
            print(f"{tag}ep {ep} non-finite val score, stopping", flush=True)
            break
        crit, extra = score, ""
        if es_cal is not None:
            tc = time.time()
            vc, vc1 = cal_rmsle_2fold(pred_log, es_cal["ly"], es_cal["y"],
                                      es_cal["half"], es_cal["bins"])
            dt = time.time() - tc
            es_cal["secs"].append(dt)
            crit = vc
            extra = f" | CAL {vc:.5f} (holdout {vc1:.5f}, +{dt:.2f}s)"
        if crit < best - 1e-5:
            best, best_epoch, bad = crit, ep, 0
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
            mark = " *"
        else:
            bad += 1
        print(f"{tag}ep {ep} bce {tr_bce:.5f} val_rmsle {score:.5f}{extra}{mark}",
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
    assert best_state is not None, "no finite val score reached"
    model.load_state_dict(best_state)
    return model, best_epoch, best


def head_summary(s0: np.ndarray, ez: np.ndarray, viol: float) -> str:
    return (f"s0_mean={s0.mean():.4f} ez_p50={np.percentile(ez, 50):.3f} "
            f"ez_p99={np.percentile(ez, 99):.3f} mono_viol={viol:.3f}")


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
    ap.add_argument("--n-bins", type=int, default=64,
                    help="total ordered buckets incl. the separate zero bucket; "
                         "the head has n_bins-1 survival sigmoids (fewer if "
                         "quantile edges tie)")
    ap.add_argument("--label-smooth", type=float, default=0.05,
                    help="binary label smoothing eps: targets eps/2 / 1-eps/2")
    ap.add_argument("--desmooth", action="store_true",
                    help="invert label smoothing at decode: "
                         "s -> (s-eps/2)/(1-eps) clipped to [0,1]. Off by "
                         "default: the blend calibration absorbs the level "
                         "bias smoothing introduces")
    ap.add_argument("--grad-clip", type=float, default=5.0,
                    help="max grad norm (0 disables)")
    ap.add_argument("--hidden", type=str, default="512,256")
    ap.add_argument("--drop-cols", type=str, default="")
    ap.add_argument("--feat-prep", choices=PREP_MODES, default="clip99",
                    help="feature preprocessing (featprep.py); clip99 is the "
                         "historical median-impute -> clip [p1,p99] -> "
                         "standardize path bit-for-bit")
    ap.add_argument("--es-metric", choices=("raw", "cal"), default="raw",
                    help="early-stopping criterion: raw val RMSLE (default) or "
                         "the honest calibrated one (calibrate.py binned "
                         "log-shift, 2-fold over users)")
    ap.add_argument("--es-bins", type=int, default=24,
                    help="quantile bins of the --es-metric cal calibration")
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
    assert args.n_bins >= 4, "--n-bins must be >= 4"
    assert 0.0 <= args.label_smooth < 1.0
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
               grad_clip=args.grad_clip, n_bins=args.n_bins,
               label_smooth=args.label_smooth, desmooth=(args.label_smooth
                                                         if args.desmooth else 0.0))
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

    stats = fit_stats(Xfull[:n_tr], args.feat_prep)   # train-only stats
    apply_stats(Xfull, stats)                         # transform all rows in place
    prep_tag = "" if args.feat_prep == "clip99" else f" [{args.feat_prep}]"
    print(f"preprocess done{prep_tag} {time.time()-t0:.0f}s", flush=True)

    X, Xv = Xfull[:n_tr], Xfull[n_tr + n_gap:]
    ylog = ylog_full[:n_tr]
    ylv = ylog_full[n_tr + n_gap:].astype(np.float64)
    pos_rate = float((ylog > 0).mean())

    # bucket edges from the TRAIN distribution only (used for retrain too)
    edges = fit_edges(ylog, args.n_bins)
    k_out = len(edges) - 1
    print(f"train pos_rate={pos_rate:.4f} | {k_out} thresholds, "
          f"e1={edges[1]:.3f} e_med={edges[len(edges)//2]:.3f} "
          f"e_max={edges[-1]:.3f}", flush=True)

    es_cal = None
    if args.es_metric == "cal":
        es_cal = {"ly": np.log1p(np.clip(yv_raw, 0, None)), "y": yv_raw,
                  "half": np.random.default_rng(0).permutation(nv) < nv // 2,
                  "bins": args.es_bins, "secs": []}
        print(f"es_metric=cal: early stop on the honest calibrated val RMSLE "
              f"({args.es_bins} bins, 2-fold over users)", flush=True)

    import torch as _torch
    widths_dev = _torch.as_tensor(np.diff(edges), dtype=_torch.float32,
                                  device=device)
    val_preds, best_epochs = [], []
    cal_s0, cal_ez = [], []
    for seed in seeds:
        m, be, _ = train_one(X, ylog, Xv, ylv, cfg, edges, seed, device,
                             args.epochs, max_steps=max_steps, tag=f"[s{seed}] ",
                             es_cal=es_cal)
        pv_log, (hs0, hez, viol) = predict_log(m, Xv, device, widths_dev,
                                               cfg["desmooth"],
                                               collect_heads=True)
        pv = np.expm1(np.clip(pv_log, 0, None))
        print(f"[s{seed}] best_epoch={be} val_rmsle={rmsle(yv_raw, pv):.6f} "
              f"{head_summary(hs0, hez, viol)}", flush=True)
        val_preds.append(pv)
        best_epochs.append(be)
        cal_s0.append(np.percentile(hs0, CAL_PCTS))
        cal_ez.append(np.percentile(hez, CAL_PCTS))
        del m, hs0, hez
    pv_avg = np.mean(val_preds, axis=0)
    score = rmsle(yv_raw, pv_avg)
    if es_cal is not None and es_cal["secs"]:
        print(f"es_metric=cal cost: {len(es_cal['secs'])} calibrations, mean "
              f"{np.mean(es_cal['secs']):.3f}s, total {sum(es_cal['secs']):.1f}s",
              flush=True)

    if args.smoke:
        print(f"[SMOKE] {args.name} val_rmsle={score:.6f} "
              f"total {time.time()-t0:.0f}s", flush=True)
        return

    save_preds(args.name, "val", uid_val, pv_avg)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(MODELS_DIR / f"{args.name}_stats.npz", **stats,
             cdf_edges=edges.astype(np.float64),
             cdf_cal_pcts=np.asarray(CAL_PCTS, np.float32),
             cdf_val_s0=np.stack(cal_s0).astype(np.float32),
             cdf_val_ez=np.stack(cal_ez).astype(np.float32))
    # freeze: what inference needs to rebuild this model besides the weights
    save_meta(args.name, kind="mlpcdf", feature_cols=cols, cfg=cfg,
              seeds=seeds, best_epochs=best_epochs, d_in=d, device=device,
              gap_days=args.gap_days, n_bins=args.n_bins, k_thresholds=k_out,
              label_smooth=args.label_smooth, desmooth=bool(args.desmooth),
              feat_prep=args.feat_prep, val_rmsle=float(score),
              stats_npz=f"{args.name}_stats.npz",
              weights=[f"{args.name}_seed{s}.pt" for s in seeds])
    notes = args.notes or (
        f"cdf-mlp B{args.n_bins}(K{k_out}) ls{args.label_smooth} "
        f"{args.hidden} do{args.dropout} lr{args.lr} bs{args.batch} "
        f"gap{args.gap_days} seeds={args.seeds} {len(tr_anchors)}anch "
        f"ep={best_epochs}")
    if args.desmooth:
        notes = f"{notes}; desmooth"
    if args.es_metric != "raw":
        notes = f"{notes}; es={args.es_metric}"
    if args.feat_prep != "clip99":
        notes = f"{notes}; prep={args.feat_prep}"
    log_score(args.name, score, notes)

    if args.no_test:
        print(f"[DONE] {args.name} val_rmsle={score:.6f} "
              f"total {time.time()-t0:.0f}s", flush=True)
        return

    # retrain on train+gap+val (same buffer, no copy) for the stopped epochs;
    # bucket edges stay the train-only ones (frozen discretization)
    test = load_anchor(TEST_ANCHOR)
    Xt = test.select(f32).to_numpy()
    uid_t = test["user_id"].to_numpy()
    del test
    apply_stats(Xt, stats)

    test_preds = []
    for seed, be in zip(seeds, best_epochs):
        m, _, _ = train_one(Xfull, ylog_full, None, None, cfg, edges, seed,
                            device, max(1, be), tag=f"[s{seed} full] ")
        test_preds.append(np.expm1(np.clip(
            predict_log(m, Xt, device, widths_dev, cfg["desmooth"]), 0, None)))
        save_torch(args.name, m, seed)   # retrain weights -> work/models/
        del m
    save_preds(args.name, "test", uid_t, np.mean(test_preds, axis=0))
    print(f"[DONE] {args.name} val_rmsle={score:.6f} "
          f"total {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

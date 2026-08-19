"""FUSION v3: tabular features + daily-sequence transformer on the seq3 tensors.

Same architecture and protocol as train_fusion.py.  Differences: the sequence input is
work/seq3 (uint8 [250k, 112, 12], dequantised per channel; channels 8..11 are the
per-channel funnel counters that never reached the tensors before), and --n-ch
truncates the tensor to its first N channels.  --n-ch 12 = the new arm, --n-ch 8 =
STRICT CONTROL: same anchors, same L=112, same quantisation, only the funnel channels
removed, so the delta isolates the new channels rather than L=112-vs-196.

Per user, two inputs:
  (a) tabular: work/features anchors (USE_V2=1 USE_V3=1, ~194 cols), preprocessing
      exactly like train_mlp2 — median impute -> clip [p1,p99] -> standardize,
      stats fit on the SELECTION train anchors only, reused for val/test/final.
      Stored standardized as float16 (cast to f32 per batch).
  (b) sequence: work/seq3/anchor=DATE.npy uint8 [250k, 112 days, 12 ch] * SCALES,
      np.load(mmap_mode='r'); row order == sample_submit sorted user_id ==
      features-parquet row order (asserted per anchor).

Model:
  seq encoder : Conv1d(n_ch->96, k=7, s=7) + pos emb -> 2-layer TransformerEncoder
                (4 heads, ff 192, norm_first) -> concat(mean, last) = 192
  tab encoder : Linear(F->256) GELU LayerNorm
  fusion trunk: concat(448) -> [Linear 384 GELU LN Drop(0.15)] -> [Linear 256
                GELU LN Drop(0.15)]  (mlp2-style blocks)
  heads       : hurdle like train_mlp2 — logit P(y30>0) (BCE) + mu =
                E[log1p(y30)|y>0] (MSE on positives); aux y7/y14 log1p-MSE
                heads, weight --aux-w (0.3). Targets from .target.npy [y30,y7,y14].
  prediction  : expm1(clip(sigmoid(logit) * clip(mu, 0), 0)).

Phases (train_seq2 contract):
  selection: train on CLEAN anchors only (target window ends before VAL, i.e.
    <= 2025-12-10), early stop on VAL RMSLE (eval every --eval-every steps,
    patience 3 evals); non-smoke saves val preds (exp_lib.save_preds) +
    log_score under NAME + models/NAME_stats.npz.

Early-stopping criterion (--es-metric):
  raw (default, historical behaviour)  plain val RMSLE of the checkpoint.
  cal                                  the honest CALIBRATED val RMSLE — the same
    binned log-shift calibrate.py applies before this model ever reaches a blend.
    Every prediction file is calibrated (KNOWLEDGE.md "ПОРЯДОК ОПЕРАЦИЙ"), and the
    calibration rewrites the LEVEL of the forecast; so stopping on the raw score
    picks the checkpoint with the best level, a gain calibration would have handed
    over for free, while giving up RANKING, which calibration preserves.  Measured:
    finer eval (246 vs 984 steps) moved the 5-seed fusion_v3 average from raw
    1.681560 -> 1.677117 but calibrated 1.668594 -> 1.669033, i.e. the sharper
    optimum of the raw metric was the WORSE model.
    Honest cut: shifts are fitted on one half of the validation users and scored on
    table was fitted on.  Cost is ~1% of one evaluation (the forward pass over the
    250k val rows dominates), so it is computed at every eval, no throttling.
    --es-metric never touches training itself: same seeds, same batches, same steps.
    It only decides WHICH checkpoint is kept, so --final test preds (retrained with
    no early stopping) are bit-identical between the two arms.
  --final: additionally retrain on ALL seq3 anchors + VAL for --epochs
    epochs (no early stop), save TEST preds. Multi-seed: raw-GMV averaging.

Smoke: --smoke = 1 train anchor, <=200 optimizer steps, batch <=1024, val
rows ::5, single seed, NOTHING written; prints one JSON line with val_rmsle.

Usage:
  .venv/bin/python work/scripts/train_fusion3.py --name f3_smoke --smoke --threads 2
  .venv/bin/python work/scripts/train_fusion3.py --name fusion_v3 --final --epochs 3
  .venv/bin/python work/scripts/train_fusion3.py --name fusion_v3ctl --final --epochs 3 --n-ch 8
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from datetime import date, timedelta
from pathlib import Path

# Cap BLAS/OMP/polars threads before numpy/torch load (external env still wins).
_thr = "2"
if "--threads" in sys.argv[1:]:
    try:
        _thr = sys.argv[sys.argv.index("--threads") + 1]
    except IndexError:
        pass
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
           "POLARS_MAX_THREADS"):
    os.environ.setdefault(_v, _thr)
os.environ.setdefault("USE_V2", "1")
os.environ.setdefault("USE_V3", "1")

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from calibrate import apply_shifts, fit_shifts  # noqa: E402
from common import (TEST_ANCHOR, VAL_ANCHOR, WORK, feature_cols,  # noqa: E402
                    load_anchor, rmsle, user_universe)
from exp_lib import log_score, save_preds  # noqa: E402

SEQ_DIR = WORK / "seq3"
MODELS_DIR = WORK / "models"
N_USERS = 250_000
_Q = json.loads((SEQ_DIR / "quant.json").read_text())
L, C_ALL = int(_Q["L"]), int(_Q["C"])
assert (L, C_ALL, _Q["dtype"]) == (112, 12, "uint8"), _Q
assert L % 7 == 0, "conv stride 7 needs L divisible by 7"
SCALES_ALL = np.asarray(_Q["scales"], dtype=np.float32)
NCH = C_ALL            # overwritten from --n-ch in main()
SCALE = SCALES_ALL     # dequant scale of the first NCH channels
STATS_MAX_ROWS = 750_000
SMOKE_MAX_STEPS = 200
SMOKE_MAX_BATCH = 1024


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--name", default=None)
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--batch", type=int, default=1024)
    p.add_argument("--eval-batch", type=int, default=4096)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=1e-5)
    p.add_argument("--dropout", type=float, default=0.15)
    p.add_argument("--tab-dim", type=int, default=256,
                   help="width of the tabular encoder")
    p.add_argument("--trunk", default="384,256",
                   help="fusion trunk block widths (last one feeds the heads)")
    p.add_argument("--bce-w", type=float, default=0.7)
    p.add_argument("--aux-w", type=float, default=0.3)
    p.add_argument("--seeds", default="42", help='e.g. "42,1337"')
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--n-ch", type=int, default=12,
                   help="use the first N sequence channels (12 = v3, 8 = seq2-equivalent control)")
    p.add_argument("--device", default="", help="cpu|mps (default: auto)")
    p.add_argument("--smoke", action="store_true",
                   help="1 anchor, <=200 steps, batch<=1024, val ::5, no saves")
    p.add_argument("--final", action="store_true",
                   help="also retrain on ALL 12 anchors + VAL, save test preds")
    p.add_argument("--eval-every", type=int, default=2000)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--es-metric", choices=("raw", "cal"), default="raw",
                   help="early-stopping criterion: raw val RMSLE (default, keeps "
                        "historical behaviour bit-for-bit) or the honest calibrated "
                        "one (calibrate.py binned log-shift, 2-fold over users)")
    p.add_argument("--retrain-from-best", action="store_true",
                   help="масштабировать длину переобучения от найденной точки "
                        "остановки (как делает бустинг), а не гнать фиксированное "
                        "число эпох; по умолчанию выключено, поведение прежнее")
    p.add_argument("--es-bins", type=int, default=24,
                   help="quantile bins of the --es-metric cal calibration "
                        "(calibrate.py default is 24; keep them equal)")
    p.add_argument("--notes", default="")
    return p.parse_args()


# ---------------- data ----------------

def seq3_train_anchors() -> list[date]:
    out = []
    for p in sorted(SEQ_DIR.glob("anchor=*.npy")):
        if p.name.endswith(".target.npy"):
            continue
        a = date.fromisoformat(p.stem.split("=")[1])
        if a < VAL_ANCHOR and (SEQ_DIR / f"anchor={a.isoformat()}.target.npy").exists():
            out.append(a)
    return sorted(out)


def open_x(a: date):
    x = np.load(SEQ_DIR / f"anchor={a.isoformat()}.npy", mmap_mode="r")
    assert x.shape == (N_USERS, L, C_ALL), f"{a}: bad seq shape {x.shape}"
    assert x.dtype == np.uint8, f"{a}: bad seq dtype {x.dtype}"
    return x


def take_seq(x_mm, rows) -> np.ndarray:
    """Gather rows of a uint8 seq memmap -> float32 [B, L, NCH] in seq2 units."""
    a = x_mm[rows]
    if NCH != C_ALL:
        a = a[:, :, :NCH]
    return a.astype(np.float32) * SCALE


def load_y(a: date):
    """-> (ylog [N,3] f32 = log1p([y30,y7,y14]), ybuy [N] f32, y30_raw [N] f64)."""
    y = np.load(SEQ_DIR / f"anchor={a.isoformat()}.target.npy")
    assert y.shape == (N_USERS, 3), f"{a}: bad target shape {y.shape}"
    ylog = np.log1p(np.clip(y, 0, None)).astype(np.float32)
    ybuy = (y[:, 0] > 0).astype(np.float32)
    return ylog, ybuy, y[:, 0].astype(np.float64)


def load_tab_raw(a: date, cols, f32_exprs, uids, check_target: bool) -> np.ndarray:
    """Features matrix [N, F] float32 in universe row order; alignment asserted."""
    need = ["user_id"] + (["target"] if check_target else []) + cols
    df = load_anchor(a, need)
    assert df.height == N_USERS, f"{a}: parquet height {df.height} != {N_USERS}"
    assert np.array_equal(df["user_id"].to_numpy(), uids), f"{a}: user_id order mismatch"
    if check_target:
        y = np.load(SEQ_DIR / f"anchor={a.isoformat()}.target.npy")
        assert np.allclose(df["target"].to_numpy().astype(np.float64), y[:, 0],
                           rtol=1e-4, atol=1e-2), f"{a}: parquet target != npy y30"
    X = df.select(f32_exprs).to_numpy()
    del df
    if not X.flags.c_contiguous:
        X = np.ascontiguousarray(X)
    return X


# ---- preprocessing: exactly train_mlp2 style ----

def fit_stats(X: np.ndarray) -> dict:
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


def apply_stats_f16(X: np.ndarray, s: dict) -> np.ndarray:
    """In-place impute/clip/standardize, then downcast to float16 (values are
    standardized + clipped to train [p1,p99], safely inside f16 range)."""
    np.copyto(X, np.broadcast_to(s["med"], X.shape), where=np.isnan(X))
    np.clip(X, s["lo"], s["hi"], out=X)
    X -= s["mean"]
    X /= s["std"]
    return X.astype(np.float16)


# ---------------- model ----------------

def build_model(d_tab: int, dropout: float, device, tab_dim: int = 256,
                trunk: tuple[int, ...] = (384, 256)):
    import torch
    import torch.nn as nn

    class FusionNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv1d(NCH, 96, kernel_size=7, stride=7)  # 112 -> 16 tokens
            self.pos = nn.Parameter(torch.zeros(1, L // 7, 96))
            nn.init.trunc_normal_(self.pos, std=0.02)
            layer = nn.TransformerEncoderLayer(
                d_model=96, nhead=4, dim_feedforward=192, dropout=0.1,
                activation="gelu", batch_first=True, norm_first=True)
            self.enc = nn.TransformerEncoder(
                layer, num_layers=2, norm=nn.LayerNorm(96), enable_nested_tensor=False)
            self.tab = nn.Sequential(nn.Linear(d_tab, tab_dim), nn.GELU(),
                                     nn.LayerNorm(tab_dim))
            blocks, prev = [], 192 + tab_dim
            for h in trunk:
                blocks += [nn.Linear(prev, h), nn.GELU(), nn.LayerNorm(h),
                           nn.Dropout(dropout)]
                prev = h
            self.trunk = nn.Sequential(*blocks)
            self.head_logit = nn.Linear(prev, 1)   # P(y30>0)
            self.head_mu = nn.Linear(prev, 1)      # E[log1p(y30) | y30>0]
            self.head_aux = nn.Linear(prev, 2)     # log1p(y7), log1p(y14)

        def forward(self, xseq, xtab):            # [B,112,NCH] f32, [B,F] f32
            h = self.conv(xseq.transpose(1, 2)).transpose(1, 2) + self.pos  # [B,16,96]
            h = self.enc(h)
            zs = torch.cat([h.mean(dim=1), h[:, -1]], dim=1)                # [B,192]
            z = self.trunk(torch.cat([zs, self.tab(xtab)], dim=1))          # [B,256]
            return (self.head_logit(z).squeeze(1), self.head_mu(z).squeeze(1),
                    self.head_aux(z))

    model = FusionNet().to(device)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"model=fusion3 d_tab={d_tab} n_ch={NCH} L={L} params={n_par:,}", flush=True)
    return model


def predict_log(model, x_mm, xtab, idx, device, eval_batch):
    """Hurdle log-space preds sigmoid(logit)*clip(mu,0) over rows idx."""
    import torch
    model.eval()
    out = np.empty(len(idx), dtype=np.float32)
    with torch.no_grad():
        for s in range(0, len(idx), eval_batch):
            rows = idx[s:s + eval_batch]
            xs_b = torch.from_numpy(take_seq(x_mm, rows)).to(device)
            xt_b = torch.from_numpy(xtab[rows].astype(np.float32)).to(device)
            logit, mu, _ = model(xs_b, xt_b)
            pl_ = torch.sigmoid(logit) * torch.clamp(mu, min=0)
            out[s:s + eval_batch] = pl_.float().cpu().numpy()
    model.train()
    return out


def cal_rmsle_2fold(pred_log: np.ndarray, ly: np.ndarray, y_raw: np.ndarray,
                    half: np.ndarray, bins: int):
    """Honest calibrated val RMSLE of a checkpoint -> (pooled, single_fold).

    Same transform as calibrate.py (fit_shifts / apply_shifts imported from it), but
    the shift table is never applied to the rows it was fitted on: half A fits the
    shifts that score half B and vice versa, so no row is calibrated by itself.
    `pooled` scores all rows this way (the criterion actually used); `single_fold` is
    only the B half, i.e. exactly the number calibrate.py prints as `holdout`, kept
    for eyeballing the two against each other in the logs.

    Both folds are honest, so using both is the same estimator with half the variance
    — which matters here, because the differences being resolved are ~1e-4.
    """
    lp = np.clip(np.asarray(pred_log, dtype=np.float64), 0, None)
    out = np.empty_like(lp)
    c_a, s_a = fit_shifts(lp[half], ly[half], bins)
    out[~half] = apply_shifts(lp[~half], c_a, s_a)
    c_b, s_b = fit_shifts(lp[~half], ly[~half], bins)
    out[half] = apply_shifts(lp[half], c_b, s_b)
    return (rmsle(y_raw, np.expm1(out)),
            rmsle(y_raw[~half], np.expm1(out[~half])))


# ---------------- training ----------------

def run_train(args, device, seed, d_tab, xs, tabs, ylogs, ybuys, max_steps, epochs,
              val=None, eval_every=0, patience=3, label="p1", cal_secs=None):
    """val = (val_x_mm, val_tab, val_y30_raw, idx) -> early stop on VAL RMSLE.
    Returns (model, best_state, best_rmsle, best_step, steps_done)."""
    import torch
    import torch.nn.functional as F
    torch.manual_seed(seed)
    model = build_model(d_tab, args.dropout, device, args.tab_dim,
                        tuple(int(h) for h in args.trunk.split(",")))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    warmup = max(1, min(500, max_steps // 20))

    def lr_lambda(s):
        if s < warmup:
            return (s + 1) / warmup
        t = (s - warmup) / max(1, max_steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, t)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    rng = np.random.default_rng(seed)
    best_state, best_rmsle, best_step = None, float("inf"), 0
    bad, step, stop, ema = 0, 0, False, None
    last_eval_step = -1
    t0 = time.time()

    # Calibrated criterion: prepared once, and ONLY when asked for — the raw path
    # must not touch a single extra RNG draw (default_rng(0) here is its own stream
    # and independent of `rng`/torch either way).
    es_cal = val is not None and getattr(args, "es_metric", "raw") == "cal"
    es_y = es_ly = es_half = None
    if es_cal:
        es_y = val[2][val[3]]
        es_ly = np.log1p(np.clip(es_y, 0, None))
        es_half = np.random.default_rng(0).permutation(len(es_y)) < len(es_y) // 2

    def do_eval():
        nonlocal best_state, best_rmsle, best_step, bad, stop, last_eval_step
        val_x, val_tab, vy_raw, idx = val
        pred_log = predict_log(model, val_x, val_tab, idx, device, args.eval_batch)
        vr = rmsle(vy_raw[idx], np.expm1(np.clip(pred_log, 0, None)))
        last_eval_step = step
        crit, extra = vr, ""
        if es_cal:
            tc = time.time()
            vc, vc1 = cal_rmsle_2fold(pred_log, es_ly, es_y, es_half, args.es_bins)
            dt = time.time() - tc
            if cal_secs is not None:
                cal_secs.append(dt)
            crit = vc
            extra = f" | CAL {vc:.5f} (holdout {vc1:.5f}, +{dt:.2f}s)"
        if crit < best_rmsle - 1e-5:
            best_rmsle, best_step, bad = crit, step, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            mark = " *"
        else:
            bad += 1
            mark = ""
        print(f"  [{label} s{seed}] EVAL step {step}: val_rmsle {vr:.5f}{extra}{mark} "
              f"(best {best_rmsle:.5f} @ {best_step}, bad {bad}/{patience})", flush=True)
        if bad >= patience:
            stop = True

    for epoch in range(epochs):
        if stop or step >= max_steps:
            break
        for ai in rng.permutation(len(xs)):
            if stop or step >= max_steps:
                break
            perm = rng.permutation(N_USERS)
            for s0 in range(0, N_USERS, args.batch):
                rows = np.sort(perm[s0:s0 + args.batch])
                xb = torch.from_numpy(take_seq(xs[ai], rows)).to(device)
                tb = torch.from_numpy(tabs[ai][rows].astype(np.float32)).to(device)
                yb = torch.from_numpy(ylogs[ai][rows]).to(device)
                bb = torch.from_numpy(ybuys[ai][rows]).to(device)
                pos = bb > 0.5
                opt.zero_grad(set_to_none=True)
                logit, mu, aux = model(xb, tb)
                bce = F.binary_cross_entropy_with_logits(logit, bb)
                if bool(pos.any()):
                    mse_pos = F.mse_loss(mu[pos], yb[pos, 0])
                else:
                    mse_pos = torch.zeros((), device=device)
                l7 = F.mse_loss(aux[:, 0], yb[:, 1])
                l14 = F.mse_loss(aux[:, 1], yb[:, 2])
                loss = args.bce_w * bce + mse_pos + args.aux_w * (l7 + l14)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
                step += 1
                lv = float(loss.item())
                ema = lv if ema is None else 0.98 * ema + 0.02 * lv
                if step % 50 == 0:
                    print(f"  [{label} s{seed}] ep{epoch} a{int(ai)} step {step}/{max_steps} "
                          f"loss {lv:.4f} ema {ema:.4f} bce {float(bce.item()):.4f} "
                          f"mse_pos {float(mse_pos.item()):.4f} "
                          f"lr {sched.get_last_lr()[0]:.2e} {time.time() - t0:.0f}s", flush=True)
                if val is not None and eval_every and step % eval_every == 0:
                    do_eval()
                if stop or step >= max_steps:
                    break

    if val is not None and last_eval_step != step:
        do_eval()  # guarantee at least one eval / catch the final state
    if val is None:
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        best_step = step
    return model, best_state, best_rmsle, best_step, step


def main():
    args = parse_args()
    seeds = [int(t) for t in args.seeds.replace(",", " ").split()]
    if args.smoke:
        seeds = seeds[:1]
        args.batch = min(args.batch, SMOKE_MAX_BATCH)
        args.final = False
    name = args.name or "fusion3"
    global NCH, SCALE
    assert 1 <= args.n_ch <= C_ALL, f"--n-ch must be 1..{C_ALL}"
    NCH = args.n_ch
    SCALE = SCALES_ALL[:NCH]

    import torch
    torch.set_num_threads(args.threads)
    device = args.device or ("mps" if torch.backends.mps.is_available() else "cpu")

    all_train = seq3_train_anchors()
    clean = [a for a in all_train if a + timedelta(days=30) <= VAL_ANCHOR]
    p1_anchors = clean[-1:] if args.smoke else clean
    print(f"device={device} name={name} seeds={seeds} smoke={args.smoke} "
          f"seq={SEQ_DIR.name} L={L} n_ch={NCH}/{C_ALL} "
          f"threads={args.threads} bs={args.batch} lr={args.lr:g} wd={args.wd:g} "
          f"bce_w={args.bce_w} aux_w={args.aux_w} do={args.dropout}", flush=True)
    print(f"all train anchors ({len(all_train)}): "
          f"{[a.isoformat() for a in all_train]}", flush=True)
    print(f"selection anchors ({len(p1_anchors)}, clean<=VAL-30d): "
          f"{[a.isoformat() for a in p1_anchors]}  VAL={VAL_ANCHOR} TEST={TEST_ANCHOR}",
          flush=True)

    # Row-order contract: preds follow sample_submit user_id sorted (== tensor rows).
    uids = user_universe()["user_id"].to_numpy()
    assert uids.shape[0] == N_USERS, f"sample_submit rows {uids.shape[0]} != {N_USERS}"
    assert bool(np.all(np.diff(uids) > 0)), "sample_submit user_id not strictly increasing"

    t0 = time.time()
    import polars as pl

    # Feature list discovered mlp2-style from the full VAL frame (also gives VAL tab).
    val_df = load_anchor(VAL_ANCHOR)
    cols = feature_cols(val_df)
    d_tab = len(cols)
    f32 = [pl.col(c).cast(pl.Float32) for c in cols]
    assert val_df.height == N_USERS
    assert np.array_equal(val_df["user_id"].to_numpy(), uids), "VAL user_id order mismatch"
    Xv_raw = val_df.select(f32).to_numpy()
    if not Xv_raw.flags.c_contiguous:
        Xv_raw = np.ascontiguousarray(Xv_raw)
    del val_df
    print(f"{d_tab} tabular features, val loaded {time.time() - t0:.0f}s", flush=True)

    ylog_v, _, val_y30_raw = load_y(VAL_ANCHOR)
    del ylog_v

    # Selection tabular: load raw f32, fit mlp2-style stats on selection anchors
    # only, then standardize everything and keep as float16 (cast per batch).
    raw = [load_tab_raw(a, cols, f32, uids, check_target=True) for a in p1_anchors]
    per = max(1, STATS_MAX_ROWS // len(raw))
    pool = np.concatenate([R[::max(1, int(np.ceil(N_USERS / per)))] for R in raw])
    stats = fit_stats(pool)
    del pool
    tabs1 = []
    for _ in range(len(raw)):
        tabs1.append(apply_stats_f16(raw.pop(0), stats))
    val_tab = apply_stats_f16(Xv_raw, stats)
    del Xv_raw
    print(f"tabular preprocessed (f16) {time.time() - t0:.0f}s", flush=True)

    xs1 = [open_x(a) for a in p1_anchors]           # seq memmaps, lazy
    ylogs1, ybuys1 = [], []
    for a in p1_anchors:
        yl, yb, _ = load_y(a)
        ylogs1.append(yl)
        ybuys1.append(yb)
    val_x = open_x(VAL_ANCHOR)

    steps_per_epoch = len(p1_anchors) * math.ceil(N_USERS / args.batch)
    max_steps = args.epochs * steps_per_epoch
    eval_every, row_step = args.eval_every, 1
    if args.smoke:
        max_steps = min(max_steps, SMOKE_MAX_STEPS)
        eval_every = min(eval_every, 100)
        row_step = 5
    val_idx = np.arange(0, N_USERS, row_step)
    print(f"steps/epoch={steps_per_epoch} max_steps={max_steps} "
          f"eval_every={eval_every} val_rows={len(val_idx)}", flush=True)

    # ---- Phase 1: selection train + early stop on VAL ----
    val_pred_sum, per_seed_rmsle, best_steps = None, [], []
    cal_secs: list[float] = []
    for seed in seeds:
        model, best_state, _, best_step, steps_done = run_train(
            args, device, seed, d_tab, xs1, tabs1, ylogs1, ybuys1,
            max_steps=max_steps, epochs=args.epochs,
            val=(val_x, val_tab, val_y30_raw, val_idx),
            eval_every=eval_every, patience=args.patience, label="p1",
            cal_secs=cal_secs)
        model.load_state_dict(best_state)
        pred_log = predict_log(model, val_x, val_tab, val_idx, device, args.eval_batch)
        pred_raw = np.expm1(np.clip(pred_log, 0, None)).astype(np.float64)
        r = rmsle(val_y30_raw[val_idx], pred_raw)
        per_seed_rmsle.append(r)
        best_steps.append(int(best_step))
        val_pred_sum = pred_raw if val_pred_sum is None else val_pred_sum + pred_raw
        print(f"[p1 seed {seed}] val_rmsle {r:.5f} best_step {best_step} "
              f"steps_done {steps_done}", flush=True)
        del model
    ens_val = val_pred_sum / len(seeds)
    ens_rmsle = rmsle(val_y30_raw[val_idx], ens_val)
    print(f"ENSEMBLE({len(seeds)} seeds) val_rmsle {ens_rmsle:.5f} "
          f"per_seed {[round(r, 5) for r in per_seed_rmsle]}", flush=True)
    cal_cost = float(np.mean(cal_secs)) if cal_secs else 0.0
    if cal_secs:
        print(f"es_metric=cal cost: {len(cal_secs)} calibrations, "
              f"mean {cal_cost:.3f}s, total {sum(cal_secs):.1f}s", flush=True)

    if args.smoke:
        import resource
        rss_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30
        print(json.dumps({"name": name, "smoke": True, "n_ch": NCH, "L": L,
                          "val_rmsle": round(float(ens_rmsle), 6),
                          "best_steps": best_steps,
                          "peak_rss_gb": round(rss_gb, 2),
                          "secs": round(time.time() - t0)}), flush=True)
        return

    save_preds(name, "val", uids, ens_val)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(MODELS_DIR / f"{name}_stats.npz", **stats)
    notes = (f"fusion3 seq3 L{L} ch{NCH} conv7-tr2 + tab{d_tab}->{args.tab_dim} "
             f"trunk{args.trunk} "
             f"hurdle bce_w{args.bce_w} aux{args.aux_w} "
             f"do{args.dropout} b{args.batch} lr{args.lr:g} wd{args.wd:g} ep{args.epochs} "
             f"eval_every={eval_every} es={args.es_metric} "
             f"seeds={','.join(map(str, seeds))} best_steps={best_steps} "
             f"per_seed={[round(r, 5) for r in per_seed_rmsle]} {args.notes}").strip()
    log_score(name, float(ens_rmsle), notes)

    # ---- Phase 2 (--final): retrain on ALL 12 anchors + VAL, save TEST preds ----
    test_saved = False
    if args.final:
        f_anchors = all_train + [VAL_ANCHOR]
        print(f"FINAL: retrain on {len(f_anchors)} anchors x {args.epochs} epochs", flush=True)
        tab_by_anchor = dict(zip(p1_anchors, tabs1))
        tab_by_anchor[VAL_ANCHOR] = val_tab
        tabs2 = []
        for a in f_anchors:
            if a not in tab_by_anchor:
                tab_by_anchor[a] = apply_stats_f16(
                    load_tab_raw(a, cols, f32, uids, check_target=True), stats)
            tabs2.append(tab_by_anchor[a])
        xs2 = [open_x(a) for a in f_anchors]
        ylogs2, ybuys2 = [], []
        for a in f_anchors:
            yl, yb, _ = load_y(a)
            ylogs2.append(yl)
            ybuys2.append(yb)
        test_tab = apply_stats_f16(
            load_tab_raw(TEST_ANCHOR, cols, f32, uids, check_target=False), stats)
        test_x = open_x(TEST_ANCHOR)
        test_idx = np.arange(N_USERS)
        max2 = args.epochs * len(f_anchors) * math.ceil(N_USERS / args.batch)
        # Переобучение без ранней остановки шло ФИКСИРОВАННОЙ длины и игнорировало
        # найденную на первой фазе точку остановки: при best_step 492-1968 оно
        # прогоняло 3321 шаг, то есть тестовая модель обучалась в 1.7-6.7 раза
        # дольше валидационно-оптимальной. Бустинг у нас так не делает — там число
        # итераций масштабируется от найденного оптимума тем же множителем.
        row_ratio = len(f_anchors) / max(len(p1_anchors), 1)
        iter_mult = 1.0 + 0.7 * (row_ratio - 1.0)
        test_sum = None
        for si, seed in enumerate(seeds):
            steps_cap = max2
            if args.retrain_from_best and best_steps:
                bs = best_steps[si] if si < len(best_steps) else best_steps[-1]
                steps_cap = min(max2, max(1, int(round(bs * iter_mult))))
                print(f"[p2 seed {seed}] шагов {steps_cap} вместо {max2} "
                      f"(best_step {bs} x {iter_mult:.4f})", flush=True)
            model2, _, _, _, steps2 = run_train(
                args, device, seed, d_tab, xs2, tabs2, ylogs2, ybuys2,
                max_steps=steps_cap, epochs=args.epochs, val=None, label="p2")
            t_log = predict_log(model2, test_x, test_tab, test_idx, device, args.eval_batch)
            t_raw = np.expm1(np.clip(t_log, 0, None)).astype(np.float64)
            test_sum = t_raw if test_sum is None else test_sum + t_raw
            print(f"[p2 seed {seed}] steps {steps2} test mean {t_raw.mean():.2f}", flush=True)
            del model2
        ens_test = test_sum / len(seeds)
        save_preds(name, "test", uids, ens_test)
        print(f"saved test preds: mean {ens_test.mean():.2f} "
              f"nonzero>1 {(ens_test > 1).mean():.4f}", flush=True)
        test_saved = True

    print(json.dumps({"name": name, "val_rmsle": round(float(ens_rmsle), 6),
                      "per_seed": [round(float(r), 6) for r in per_seed_rmsle],
                      "best_steps": best_steps, "test_saved": test_saved,
                      "es_metric": args.es_metric,
                      "cal_secs_per_eval": round(cal_cost, 3)}), flush=True)


if __name__ == "__main__":
    main()

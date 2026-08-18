"""Seq2 sequence models (GRU / Conv-Transformer) for 30d GMV (LTV), multi-task heads.

Data: work/seq2/anchor=DATE.npy  float16 [250k users, 196 days, 8 ch]
  ch: log1p(gmv_search), log1p(gmv_cat), min(to_ord,10), min(to_cart,20),
      log1p(searches), search_flag, cat_flag, any_order_flag
  companion anchor=DATE.target.npy float32 [250k, 3] = [y30, y7, y14] (raw GMV;
  absent for TEST anchor). Row order = sample_submit.csv sorted by user_id.

Heads: y30 log1p-MSE (main) + 0.3*(y7 + y14 log1p-MSE) + 0.3*BCE(buy30 = y30>0).
Selection (default --clean-only): train on anchors with target window ending
before VAL (<= 2025-12-10), early stop on VAL RMSLE of main head
(eval every --eval-every steps, patience 3 evals), save val preds + log_score.
--final: additionally retrain on ALL train anchors + VAL for --epochs epochs
(no early stop), save TEST preds. Multi-seed: preds averaged in raw GMV space.
All big tensors are np.load(mmap_mode='r'); batches gather rows -> float32.

Usage:
  .venv/bin/python work/scripts/train_seq2.py --name seq2_gru --arch gru --epochs 6
  .venv/bin/python work/scripts/train_seq2.py --name seq2_gru --arch gru --epochs 6 --final
  smoke: --smoke --arch tr --threads 2   (1 anchor, <=200 steps, val rows ::5, no saves)
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

# Cap BLAS/OMP threads before numpy/torch load (external env still wins).
_thr = "2"
if "--threads" in sys.argv[1:]:
    try:
        _thr = sys.argv[sys.argv.index("--threads") + 1]
    except IndexError:
        pass
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, _thr)

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import TEST_ANCHOR, VAL_ANCHOR, WORK, rmsle, user_universe  # noqa: E402
from exp_lib import log_score, save_preds  # noqa: E402

SEQ_DIR = WORK / "seq2"
N_USERS = 250_000
L, C = 196, 8


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--name", default=None)
    p.add_argument("--arch", choices=["gru", "tr"], default="gru")
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--batch", type=int, default=1024)
    p.add_argument("--eval-batch", type=int, default=4096)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=1e-5)
    p.add_argument("--seeds", default="42", help='e.g. "42,43,44"')
    p.add_argument("--smoke", action="store_true",
                   help="1 train anchor, 200 optimizer-step cap, val ::5, no saves")
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--clean-only", action=argparse.BooleanOptionalAction, default=True,
                   help="phase-1 train only anchors with target window ending "
                        "before VAL (<= 2025-12-10); leak-free selection")
    p.add_argument("--final", action="store_true",
                   help="also retrain on ALL train anchors + VAL for --epochs, save test preds")
    p.add_argument("--eval-every", type=int, default=2000)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--notes", default="")
    return p.parse_args()


def seq2_train_anchors() -> list[date]:
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
    assert x.shape == (N_USERS, L, C), f"{a}: bad shape {x.shape}"
    return x


def load_y(a: date):
    """-> (ylog [N,3] f32, ybuy [N] f32, y30_raw [N] f64). Small (~7MB), loaded fully."""
    y = np.load(SEQ_DIR / f"anchor={a.isoformat()}.target.npy")
    assert y.shape == (N_USERS, 3), f"{a}: bad target shape {y.shape}"
    ylog = np.log1p(np.clip(y, 0, None)).astype(np.float32)
    ybuy = (y[:, 0] > 0).astype(np.float32)
    return ylog, ybuy, y[:, 0].astype(np.float64)


def build_model(arch: str, device):
    import torch
    import torch.nn as nn

    class GRUNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.gru = nn.GRU(C, 128, num_layers=2, dropout=0.1, batch_first=True)
            self.proj = nn.Sequential(nn.Linear(3 * 128, 256), nn.GELU())
            self.heads = nn.Linear(256, 4)

        def forward(self, x):                       # x [B,196,8] f32
            mask = (x != 0).any(dim=2)              # [B,196] active days
            out, _ = self.gru(x)                    # [B,196,128]
            last = out[:, -1]
            m = mask.unsqueeze(-1).to(out.dtype)
            cnt = m.sum(dim=1)                      # [B,1]
            mean = (out * m).sum(dim=1) / cnt.clamp(min=1.0)
            mx = out.masked_fill(~mask.unsqueeze(-1), -1e4).amax(dim=1)
            mx = torch.where(cnt > 0, mx, torch.zeros_like(mx))
            return self.heads(self.proj(torch.cat([last, mean, mx], dim=1)))

    class TrNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv1d(C, 96, kernel_size=7, stride=7)   # 196 -> 28 tokens
            self.pos = nn.Parameter(torch.zeros(1, L // 7, 96))
            nn.init.trunc_normal_(self.pos, std=0.02)
            layer = nn.TransformerEncoderLayer(
                d_model=96, nhead=4, dim_feedforward=192, dropout=0.1,
                activation="gelu", batch_first=True, norm_first=True)
            self.enc = nn.TransformerEncoder(
                layer, num_layers=3, norm=nn.LayerNorm(96), enable_nested_tensor=False)
            self.proj = nn.Sequential(nn.Linear(2 * 96, 256), nn.GELU())
            self.heads = nn.Linear(256, 4)

        def forward(self, x):                       # x [B,196,8] f32
            h = self.conv(x.transpose(1, 2)).transpose(1, 2) + self.pos  # [B,28,96]
            h = self.enc(h)
            return self.heads(self.proj(torch.cat([h.mean(dim=1), h[:, -1]], dim=1)))

    model = (GRUNet() if arch == "gru" else TrNet()).to(device)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"model={arch} params={n_par:,}", flush=True)
    return model


def predict_main(model, x_mm, device, eval_batch, row_step=1):
    """Main-head (y30) log-space preds over memmap rows [::row_step] -> (idx, preds)."""
    import torch
    model.eval()
    idx = np.arange(0, x_mm.shape[0], row_step)
    out = np.empty(len(idx), dtype=np.float32)
    with torch.no_grad():
        for s in range(0, len(idx), eval_batch):
            rows = idx[s:s + eval_batch]
            xb = torch.from_numpy(x_mm[rows].astype(np.float32)).to(device)
            out[s:s + eval_batch] = model(xb)[:, 0].float().cpu().numpy()
    model.train()
    return idx, out


def run_train(args, device, seed, arch, xs, ylogs, ybuys, max_steps, epochs,
              val=None, eval_every=0, patience=3, label="p1"):
    """Train up to max_steps (optimizer steps) / epochs, whichever first.
    Loop: per epoch, iterate anchors (shuffled order), shuffle user rows per anchor.
    val=(val_x_mm, val_y30_raw, row_step) -> early stop on VAL RMSLE (main head).
    Returns (model, best_state, best_rmsle, best_step, steps_done)."""
    import torch
    import torch.nn.functional as F
    torch.manual_seed(seed)
    model = build_model(arch, device)
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

    def do_eval():
        nonlocal best_state, best_rmsle, best_step, bad, stop, last_eval_step
        val_x, vy_raw, row_step = val
        idx, pred_log = predict_main(model, val_x, device, args.eval_batch, row_step)
        vr = rmsle(vy_raw[idx], np.expm1(np.clip(pred_log, 0, None)))
        last_eval_step = step
        if vr < best_rmsle - 1e-5:
            best_rmsle, best_step, bad = vr, step, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            mark = " *"
        else:
            bad += 1
            mark = ""
        print(f"  [{label} s{seed}] EVAL step {step}: val_rmsle {vr:.5f}{mark} "
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
                xb = torch.from_numpy(xs[ai][rows].astype(np.float32)).to(device)
                yb = torch.from_numpy(ylogs[ai][rows]).to(device)
                bb = torch.from_numpy(ybuys[ai][rows]).to(device)
                opt.zero_grad(set_to_none=True)
                out = model(xb)
                l30 = F.mse_loss(out[:, 0], yb[:, 0])
                l7 = F.mse_loss(out[:, 1], yb[:, 1])
                l14 = F.mse_loss(out[:, 2], yb[:, 2])
                lbuy = F.binary_cross_entropy_with_logits(out[:, 3], bb)
                loss = l30 + 0.3 * (l7 + l14) + 0.3 * lbuy
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
                step += 1
                lv = float(loss.item())
                ema = lv if ema is None else 0.98 * ema + 0.02 * lv
                if step % 50 == 0:
                    print(f"  [{label} s{seed}] ep{epoch} a{int(ai)} step {step}/{max_steps} "
                          f"loss {lv:.4f} ema {ema:.4f} l30 {float(l30.item()):.4f} "
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
        args.batch = min(args.batch, 1024)
    name = args.name or f"seq2_{args.arch}"

    import torch
    torch.set_num_threads(args.threads)
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    all_train = seq2_train_anchors()
    clean = [a for a in all_train if a + timedelta(days=30) <= VAL_ANCHOR]
    p1_anchors = (clean if args.clean_only else all_train)
    if args.smoke:
        p1_anchors = p1_anchors[-1:]
    print(f"device={device} arch={args.arch} name={name} seeds={seeds} "
          f"smoke={args.smoke} threads={args.threads}", flush=True)
    print(f"train anchors ({len(all_train)}): {[a.isoformat() for a in all_train]}", flush=True)
    print(f"phase1 anchors ({len(p1_anchors)}, clean_only={args.clean_only}): "
          f"{[a.isoformat() for a in p1_anchors]}  VAL={VAL_ANCHOR} TEST={TEST_ANCHOR}",
          flush=True)

    # Row-order contract: preds follow sample_submit user_id sorted (== tensor rows).
    uids = user_universe()["user_id"].to_numpy()
    assert uids.shape[0] == N_USERS, f"sample_submit rows {uids.shape[0]} != {N_USERS}"
    assert bool(np.all(np.diff(uids) > 0)), "sample_submit user_id not strictly increasing"

    xs1 = [open_x(a) for a in p1_anchors]           # memmaps, lazy
    ylogs1, ybuys1 = [], []
    for a in p1_anchors:
        yl, yb, _ = load_y(a)
        ylogs1.append(yl)
        ybuys1.append(yb)
    val_x = open_x(VAL_ANCHOR)
    _, _, val_y30_raw = load_y(VAL_ANCHOR)

    steps_per_epoch = len(p1_anchors) * math.ceil(N_USERS / args.batch)
    max_steps = args.epochs * steps_per_epoch
    eval_every, row_step = args.eval_every, 1
    if args.smoke:
        max_steps = min(max_steps, 200)
        eval_every = min(eval_every, 100)
        row_step = 5
    print(f"steps/epoch={steps_per_epoch} max_steps={max_steps} "
          f"eval_every={eval_every} batch={args.batch} lr={args.lr:g}", flush=True)

    # ---- Phase 1: selection train + early stop on VAL ----
    val_pred_sum, per_seed_rmsle, best_steps, idx = None, [], [], None
    for seed in seeds:
        model, best_state, _, best_step, steps_done = run_train(
            args, device, seed, args.arch, xs1, ylogs1, ybuys1,
            max_steps=max_steps, epochs=args.epochs,
            val=(val_x, val_y30_raw, row_step),
            eval_every=eval_every, patience=args.patience, label="p1")
        model.load_state_dict(best_state)
        idx, pred_log = predict_main(model, val_x, device, args.eval_batch, row_step)
        pred_raw = np.expm1(np.clip(pred_log, 0, None)).astype(np.float64)
        r = rmsle(val_y30_raw[idx], pred_raw)
        per_seed_rmsle.append(r)
        best_steps.append(int(best_step))
        val_pred_sum = pred_raw if val_pred_sum is None else val_pred_sum + pred_raw
        print(f"[p1 seed {seed}] val_rmsle {r:.5f} best_step {best_step} "
              f"steps_done {steps_done}", flush=True)
        del model
    ens_val = val_pred_sum / len(seeds)
    ens_rmsle = rmsle(val_y30_raw[idx], ens_val)
    print(f"ENSEMBLE({len(seeds)} seeds) val_rmsle {ens_rmsle:.5f} "
          f"per_seed {[round(r, 5) for r in per_seed_rmsle]}", flush=True)

    if args.smoke:
        print(json.dumps({"name": name, "arch": args.arch, "smoke": True,
                          "val_rmsle": round(float(ens_rmsle), 6),
                          "best_steps": best_steps}), flush=True)
        return

    save_preds(name, "val", uids, ens_val)
    notes = (f"seq2 {args.arch} b{args.batch} lr{args.lr:g} ep{args.epochs} "
             f"clean={int(args.clean_only)} seeds={','.join(map(str, seeds))} "
             f"best_steps={best_steps} per_seed={[round(r, 5) for r in per_seed_rmsle]} "
             f"{args.notes}").strip()
    log_score(name, float(ens_rmsle), notes)

    # ---- Phase 2 (--final): retrain on ALL train anchors + VAL, save TEST preds ----
    test_saved = False
    if args.final:
        f_anchors = all_train + [VAL_ANCHOR]
        print(f"FINAL: retrain on {len(f_anchors)} anchors x {args.epochs} epochs", flush=True)
        xs2 = [open_x(a) for a in f_anchors]
        ylogs2, ybuys2 = [], []
        for a in f_anchors:
            yl, yb, _ = load_y(a)
            ylogs2.append(yl)
            ybuys2.append(yb)
        max2 = args.epochs * len(f_anchors) * math.ceil(N_USERS / args.batch)
        test_x = open_x(TEST_ANCHOR)
        test_sum = None
        for seed in seeds:
            model2, _, _, _, steps2 = run_train(
                args, device, seed, args.arch, xs2, ylogs2, ybuys2,
                max_steps=max2, epochs=args.epochs, val=None, label="p2")
            _, t_log = predict_main(model2, test_x, device, args.eval_batch, 1)
            t_raw = np.expm1(np.clip(t_log, 0, None)).astype(np.float64)
            test_sum = t_raw if test_sum is None else test_sum + t_raw
            print(f"[p2 seed {seed}] steps {steps2} test mean {t_raw.mean():.2f}", flush=True)
            del model2
        ens_test = test_sum / len(seeds)
        save_preds(name, "test", uids, ens_test)
        print(f"saved test preds: mean {ens_test.mean():.2f} "
              f"nonzero>1 {(ens_test > 1).mean():.4f}", flush=True)
        test_saved = True

    print(json.dumps({"name": name, "arch": args.arch,
                      "val_rmsle": round(float(ens_rmsle), 6),
                      "per_seed": [round(float(r), 6) for r in per_seed_rmsle],
                      "best_steps": best_steps, "test_saved": test_saved}), flush=True)


if __name__ == "__main__":
    main()

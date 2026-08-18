"""GRU over daily behavior sequences (work/seq tensors) for 30d GMV (LTV) prediction.

Data: work/seq/anchor=DATE.npy  float16 [250k users x 112 days x 6 ch]
  channels: log1p(gmv), log1p(searches), min(to_ord,10), min(to_cart,10), search, cat
  row order = sample_submit user_id sorted (same as features parquet).
Model: GRU(6->hidden, 2 layers, dropout) -> concat(last, mean, max) -> Linear -> GELU -> Linear(1)
Loss: MSE on log1p(target). Early stop on VAL RMSLE every half-epoch, patience N evals.
Contract (exp_lib): save val preds from best early-stopped model + log_score; then
retrain on train+VAL anchors for the same number of steps and save test preds.
Preds saved raw scale: expm1(clip(pred_log, 0, None)).

Usage:
  .venv/bin/python work/scripts/train_gru.py --name gru_final
  smoke: --name gru_smoke --n-anchors 2 --row-frac 0.2 --epochs 2 --no-test
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from common import VAL_ANCHOR, TEST_ANCHOR, WORK, rmsle, user_universe  # noqa: E402
from exp_lib import log_score, save_preds  # noqa: E402

SEQ_DIR = WORK / "seq"
N_USERS = 250_000


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--name", default="gru_final")
    p.add_argument("--n-anchors", type=int, default=8, help="newest N labeled train anchors")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch", type=int, default=4096)
    p.add_argument("--eval-batch", type=int, default=8192)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--wd", type=float, default=1e-5)
    p.add_argument("--hidden", type=int, default=96)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--patience", type=int, default=3, help="early-stop patience in evals")
    p.add_argument("--evals-per-epoch", type=int, default=2)
    p.add_argument("--row-frac", type=float, default=1.0, help="subsample train rows (smoke)")
    p.add_argument("--weight-tau", type=float, default=0.0,
                   help=">0: per-sample weight exp(-(VAL-anchor)/tau) days")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--threads", type=int, default=3)
    p.add_argument("--no-test", action="store_true")
    p.add_argument("--notes", default="")
    return p.parse_args()


def seq_train_anchors(n: int) -> list[date]:
    out = []
    for p in sorted(SEQ_DIR.glob("anchor=*.npy")):
        if p.name.endswith(".target.npy"):
            continue
        a = date.fromisoformat(p.stem.split("=")[1])
        if a < VAL_ANCHOR and (SEQ_DIR / f"anchor={a.isoformat()}.target.npy").exists():
            out.append(a)
    return sorted(out)[-n:]


def open_anchor(a: date):
    x = np.load(SEQ_DIR / f"anchor={a.isoformat()}.npy", mmap_mode="r")
    tp = SEQ_DIR / f"anchor={a.isoformat()}.target.npy"
    y = np.log1p(np.load(tp)).astype(np.float32) if tp.exists() else None
    return x, y


def build_model(args, device):
    import torch.nn as nn

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.gru = nn.GRU(6, args.hidden, num_layers=args.layers,
                              batch_first=True, dropout=args.dropout)
            self.head = nn.Sequential(
                nn.Linear(3 * args.hidden, 64), nn.GELU(), nn.Linear(64, 1))

        def forward(self, x):
            out, _ = self.gru(x)
            z = torch.cat([out[:, -1], out.mean(dim=1), out.amax(dim=1)], dim=1)
            return self.head(z).squeeze(1)

    import torch
    return Net().to(device)


def predict(model, x_mm, device, eval_batch):
    """Sequential prediction over a memmapped [N,112,6] fp16 array -> log-space preds."""
    import torch
    model.eval()
    out = np.empty(x_mm.shape[0], dtype=np.float32)
    with torch.no_grad():
        for s in range(0, x_mm.shape[0], eval_batch):
            xb = torch.from_numpy(np.asarray(x_mm[s:s + eval_batch])).to(device).float()
            out[s:s + eval_batch] = model(xb).float().cpu().numpy()
    model.train()
    return out


def make_epoch_batches(rng, rows_per_anchor, n_anchors, batch):
    """Global shuffle over (anchor, row); yields per-batch [(a, rows_sorted), ...] groups."""
    total = sum(len(r) for r in rows_per_anchor)
    g_anchor = np.concatenate([np.full(len(r), i, dtype=np.int8)
                               for i, r in enumerate(rows_per_anchor)])
    g_row = np.concatenate(rows_per_anchor)
    perm = rng.permutation(total)
    g_anchor, g_row = g_anchor[perm], g_row[perm]
    batches = []
    for s in range(0, total, batch):
        ba, br = g_anchor[s:s + batch], g_row[s:s + batch]
        groups = []
        for a in np.unique(ba):
            groups.append((int(a), np.sort(br[ba == a])))
        batches.append(groups)
    return batches


def gather_batch(groups, xs, ys, ws):
    xb = np.concatenate([np.asarray(xs[a][rows]) for a, rows in groups])
    yb = np.concatenate([ys[a][rows] for a, rows in groups])
    wb = None
    if ws is not None:
        wb = np.concatenate([np.full(len(rows), ws[a], dtype=np.float32)
                             for a, rows in groups])
    return xb, yb, wb


def train_phase(args, device, anchors, xs, ys, ws, max_steps, total_sched_steps,
                val_x=None, val_y_raw=None, eval_every=None, label="train"):
    """Train up to max_steps. If val given: early stop, return (best_state, best_rmsle, best_step).
    Else return (final_state, None, steps_done)."""
    import torch
    torch.manual_seed(args.seed)
    model = build_model(args, device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_sched_steps)
    rng = np.random.default_rng(args.seed)
    rows_per_anchor = [np.arange(N_USERS, dtype=np.int32) for _ in anchors]
    if args.row_frac < 1.0:
        k = int(N_USERS * args.row_frac)
        rows_per_anchor = [np.sort(rng.choice(N_USERS, size=k, replace=False)).astype(np.int32)
                           for _ in anchors]

    best_state, best_rmsle, best_step = None, np.inf, 0
    bad_evals, step, stop = 0, 0, False
    t0 = time.time()
    while step < max_steps and not stop:
        for groups in make_epoch_batches(rng, rows_per_anchor, len(anchors), args.batch):
            xb, yb, wb = gather_batch(groups, xs, ys, ws)
            xt = torch.from_numpy(xb).to(device).float()
            yt = torch.from_numpy(yb).to(device)
            opt.zero_grad(set_to_none=True)
            pred = model(xt)
            if wb is not None:
                wt = torch.from_numpy(wb).to(device)
                loss = ((pred - yt) ** 2 * wt).sum() / wt.sum()
            else:
                loss = torch.nn.functional.mse_loss(pred, yt)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if step < total_sched_steps:
                sched.step()
            step += 1
            if step % 50 == 0:
                print(f"  [{label}] step {step}/{max_steps} loss {loss.item():.4f} "
                      f"lr {sched.get_last_lr()[0]:.2e} {time.time()-t0:.0f}s", flush=True)
            if val_x is not None and eval_every and step % eval_every == 0:
                pl_log = predict(model, val_x, device, args.eval_batch)
                vr = rmsle(val_y_raw, np.expm1(np.clip(pl_log, 0, None)))
                mark = ""
                if vr < best_rmsle - 1e-5:
                    best_rmsle, best_step, bad_evals = vr, step, 0
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                    mark = " *"
                else:
                    bad_evals += 1
                print(f"  [{label}] EVAL step {step}: val_rmsle {vr:.5f}{mark} "
                      f"(best {best_rmsle:.5f} @ {best_step}, bad {bad_evals})", flush=True)
                if bad_evals >= args.patience:
                    stop = True
            if step >= max_steps or stop:
                break
    if val_x is None:
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        return model, best_state, None, step
    return model, best_state, best_rmsle, best_step


def main():
    args = parse_args()
    import torch
    torch.set_num_threads(args.threads)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    anchors = seq_train_anchors(args.n_anchors)
    print(f"device={device} anchors={[a.isoformat() for a in anchors]} "
          f"VAL={VAL_ANCHOR} TEST={TEST_ANCHOR}", flush=True)

    xs, ys = [], []
    for a in anchors:
        x, y = open_anchor(a)
        xs.append(x); ys.append(y)
    ws = None
    if args.weight_tau > 0:
        ws = [np.exp(-((VAL_ANCHOR - a).days) / args.weight_tau) for a in anchors]
        print(f"anchor weights: {[round(w, 3) for w in ws]}", flush=True)
    val_x, val_y_log = open_anchor(VAL_ANCHOR)
    val_y_raw = np.expm1(val_y_log.astype(np.float64))

    n_rows = int(N_USERS * args.row_frac) * len(anchors)
    steps_per_epoch = int(np.ceil(n_rows / args.batch))
    total_steps = steps_per_epoch * args.epochs
    eval_every = max(1, steps_per_epoch // args.evals_per_epoch)
    print(f"steps/epoch={steps_per_epoch} total={total_steps} eval_every={eval_every}", flush=True)

    model, best_state, best_rmsle, best_step = train_phase(
        args, device, anchors, xs, ys, ws, total_steps, total_steps,
        val_x=val_x, val_y_raw=val_y_raw, eval_every=eval_every, label="p1")
    print(f"PHASE1 done: best val_rmsle {best_rmsle:.5f} @ step {best_step}", flush=True)

    model.load_state_dict(best_state)
    val_pred_log = predict(model, val_x, device, args.eval_batch)
    val_pred = np.expm1(np.clip(val_pred_log, 0, None))
    uids = user_universe()["user_id"].to_numpy()
    save_preds(args.name, "val", uids, val_pred)
    notes = (args.notes or
             f"gru h{args.hidden}x{args.layers} do{args.dropout} b{args.batch} lr{args.lr} "
             f"{len(anchors)}anch best_step{best_step}"
             + (f" tau{args.weight_tau:.0f}" if args.weight_tau > 0 else ""))
    log_score(args.name, float(best_rmsle), notes)

    if args.no_test:
        print(json.dumps({"name": args.name, "val_rmsle": round(float(best_rmsle), 6),
                          "best_step": int(best_step)}), flush=True)
        return

    # Phase 2: retrain on train+VAL anchors for best_step steps, predict TEST.
    print(f"PHASE2: retrain on {len(anchors)+1} anchors for {best_step} steps", flush=True)
    xs2, ys2 = xs + [val_x], ys + [val_y_log]
    ws2 = None
    if args.weight_tau > 0:
        ws2 = [np.exp(-((TEST_ANCHOR - a).days) / args.weight_tau)
               for a in anchors + [VAL_ANCHOR]]
    anchors2 = anchors + [VAL_ANCHOR]
    model2, _, _, steps2 = train_phase(
        args, device, anchors2, xs2, ys2, ws2, best_step, best_step, label="p2")
    test_x, _ = open_anchor(TEST_ANCHOR)
    test_pred = np.expm1(np.clip(predict(model2, test_x, device, args.eval_batch), 0, None))
    save_preds(args.name, "test", uids, test_pred)
    print(f"saved test preds ({steps2} steps). test_pred mean {test_pred.mean():.2f} "
          f"nonzero {(test_pred > 1).mean():.3f}", flush=True)
    print(json.dumps({"name": args.name, "val_rmsle": round(float(best_rmsle), 6),
                      "best_step": int(best_step), "test_saved": True}), flush=True)


if __name__ == "__main__":
    main()

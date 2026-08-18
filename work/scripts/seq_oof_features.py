"""Seq-transformer OOF stacking features (AMEX-style NN-preds-into-GBDT).

Trains train_seq2's conv-transformer (arch tr, 1 seed, reduced budget:
4 epochs, batch 2048) ONLY on early seq2 anchors <= 2025-11-12, then predicts
the main head E[log1p(y30)] for every later seq2 anchor
(2025-11-19 .. 2026-01-07) + VAL + TEST. Those anchors were never seen in
training -> honest-ish (time-split) OOF features for GBDT stacking.

Writes work/features/anchor=DATE.seqoof.parquet (user_id, seqoof_pred f32),
row order = sample_submit user_id sorted (same contract as seq2 tensors).
Early anchors get NO file by design; common.load_anchor with USE_SEQOOF=1
fills a null seqoof_pred column for them so the GBDT schema stays consistent.

Env knobs:
  SMOKE=1        1 train anchor, 100 optimizer steps, predict VAL only,
                 write to $SMOKE_OUT (default /tmp/seqoof_smoke), device cpu
  SMOKE_OUT=DIR  smoke output dir
  SEQOOF_DEVICE  force torch device (cpu/mps); default: cpu if SMOKE else auto
  OMP_NUM_THREADS  BLAS/torch threads (default 4)

Usage: OMP_NUM_THREADS=4 .venv/bin/python work/scripts/seq_oof_features.py
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from datetime import date
from pathlib import Path

# Cap BLAS threads before numpy/torch load (external env wins).
_thr = os.environ.get("OMP_NUM_THREADS", "4")
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, _thr)

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import FEATURES_DIR, VAL_ANCHOR, TEST_ANCHOR, rmsle, user_universe  # noqa: E402
from train_seq2 import N_USERS, SEQ_DIR, load_y, open_x, predict_main, run_train  # noqa: E402

TRAIN_CUT = date(2025, 11, 12)  # transformer trains on anchors <= this only
ARCH, SEED, EPOCHS, BATCH = "tr", 42, 4, 2048
SMOKE = os.environ.get("SMOKE") == "1"
SMOKE_STEPS = 100


def seq2_x_anchors() -> list[date]:
    out = []
    for p in sorted(SEQ_DIR.glob("anchor=*.npy")):
        if not p.name.endswith(".target.npy"):
            out.append(date.fromisoformat(p.stem.split("=")[1]))
    return sorted(out)


def main():
    t0 = time.time()
    import argparse
    import torch

    args = argparse.Namespace(batch=BATCH, lr=1e-3, wd=1e-5, eval_batch=4096)
    threads = int(os.environ.get("OMP_NUM_THREADS", "4"))
    torch.set_num_threads(threads)
    device = os.environ.get("SEQOOF_DEVICE") or (
        "cpu" if SMOKE else ("mps" if torch.backends.mps.is_available() else "cpu"))

    all_x = seq2_x_anchors()
    train_a = [a for a in all_x if a <= TRAIN_CUT
               and (SEQ_DIR / f"anchor={a.isoformat()}.target.npy").exists()]
    pred_a = [a for a in all_x if a > TRAIN_CUT]  # late train anchors + VAL + TEST
    out_dir = FEATURES_DIR
    if SMOKE:
        train_a = train_a[-1:]
        pred_a = [VAL_ANCHOR]
        out_dir = Path(os.environ.get("SMOKE_OUT", "/tmp/seqoof_smoke"))
    out_dir.mkdir(parents=True, exist_ok=True)
    assert train_a, "no seq2 training anchors <= TRAIN_CUT with targets"
    if not SMOKE:
        assert VAL_ANCHOR in pred_a and TEST_ANCHOR in pred_a, pred_a

    print(f"device={device} threads={threads} smoke={SMOKE} out={out_dir}", flush=True)
    print(f"train anchors ({len(train_a)}, <= {TRAIN_CUT}): "
          f"{[a.isoformat() for a in train_a]}", flush=True)
    print(f"predict anchors ({len(pred_a)}): {[a.isoformat() for a in pred_a]}", flush=True)

    # Row-order contract: tensor rows == sample_submit user_id sorted.
    uids = user_universe()["user_id"].to_numpy()
    assert uids.shape[0] == N_USERS, f"sample_submit rows {uids.shape[0]} != {N_USERS}"
    assert bool(np.all(np.diff(uids) > 0)), "user_id not strictly increasing"

    xs = [open_x(a) for a in train_a]
    ylogs, ybuys = [], []
    for a in train_a:
        yl, yb, _ = load_y(a)
        ylogs.append(yl)
        ybuys.append(yb)

    steps_per_epoch = len(xs) * math.ceil(N_USERS / args.batch)
    max_steps = EPOCHS * steps_per_epoch
    if SMOKE:
        max_steps = min(max_steps, SMOKE_STEPS)
    print(f"steps/epoch={steps_per_epoch} max_steps={max_steps} batch={args.batch} "
          f"lr={args.lr:g} seed={SEED}", flush=True)

    model, _, _, _, steps_done = run_train(
        args, device, SEED, ARCH, xs, ylogs, ybuys,
        max_steps=max_steps, epochs=EPOCHS, val=None, label="oof")
    print(f"trained {steps_done} steps in {time.time() - t0:.0f}s", flush=True)

    written = []
    for a in pred_a:
        x = open_x(a)
        idx, pred_log = predict_main(model, x, device, args.eval_batch, row_step=1)
        assert len(idx) == N_USERS and len(pred_log) == N_USERS
        p = out_dir / f"anchor={a.isoformat()}.seqoof.parquet"
        pl.DataFrame({"user_id": uids.astype(np.int64),
                      "seqoof_pred": pred_log.astype(np.float32)}).write_parquet(p)
        msg = (f"  wrote {p.name} mean_log {float(pred_log.mean()):.4f} "
               f"pos_frac {float((pred_log > 0.1).mean()):.4f}")
        if (SEQ_DIR / f"anchor={a.isoformat()}.target.npy").exists():
            _, _, y_raw = load_y(a)
            msg += f" rmsle {rmsle(y_raw, np.expm1(np.clip(pred_log, 0, None))):.5f}"
        print(msg, flush=True)
        written.append(p.name)
        del x

    print(json.dumps({
        "smoke": SMOKE, "device": device, "steps": int(steps_done),
        "train_anchors": [a.isoformat() for a in train_a],
        "written": written, "seconds": round(time.time() - t0)}), flush=True)


if __name__ == "__main__":
    main()

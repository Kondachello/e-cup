"""Experiment harness: standard data loading / scoring / saving for model agents.

Contract for every model experiment `NAME`:
  1. Train on anchors < VAL_ANCHOR, validate on VAL_ANCHOR (early stopping allowed).
  2. Save validation predictions -> work/preds/NAME_val.parquet  (user_id, pred)
  3. Retrain on train+val anchors (best_iter * 1.07 if early stopped), predict
     TEST_ANCHOR -> work/preds/NAME_test.parquet (user_id, pred)
  4. Append one line to work/reports/scores.tsv: NAME\tval_rmsle\tnotes
Predictions are raw GMV scale (>=0), NOT log.
"""
from __future__ import annotations

import fcntl
import sys
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))
from common import (  # noqa: F401
    FEATURES_DIR, PREDS_DIR, REPORTS_DIR, TEST_ANCHOR, VAL_ANCHOR,
    feature_cols, load_anchor, rmsle, train_anchors,
)

PREDS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def available_train_anchors() -> list[date]:
    out = []
    for p in sorted(FEATURES_DIR.glob("anchor=*.parquet")):
        if p.stem.endswith(".extra"):
            continue
        a = date.fromisoformat(p.stem.split("=")[1])
        if a < VAL_ANCHOR:
            out.append(a)
    return out


def load_matrix(anchors: list[date], columns: list[str] | None = None) -> pl.DataFrame:
    dfs = [load_anchor(a, columns) for a in anchors]
    return pl.concat(dfs, how="vertical_relaxed")


def to_xy(df: pl.DataFrame, cols: list[str]):
    X = df.select(cols).to_numpy().astype(np.float32)
    y = df["target"].to_numpy().astype(np.float64)
    return X, y


def save_preds(name: str, split: str, user_ids: np.ndarray, preds: np.ndarray):
    preds = np.clip(np.asarray(preds, dtype=np.float64), 0, None)
    pl.DataFrame({"user_id": user_ids.astype(np.int64), "pred": preds}).write_parquet(
        PREDS_DIR / f"{name}_{split}.parquet"
    )


def log_score(name: str, val_rmsle: float, notes: str = ""):
    path = REPORTS_DIR / "scores.tsv"
    with open(path, "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.write(f"{name}\t{val_rmsle:.6f}\t{notes}\n")
        fcntl.flock(f, fcntl.LOCK_UN)
    print(f"[SCORE] {name}: {val_rmsle:.6f} ({notes})", flush=True)

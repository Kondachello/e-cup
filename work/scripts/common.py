"""Shared utilities for the Ozon E-CUP LTV competition."""
from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path("/Users/alexanderkondakov/ozon-cup")
TRAIN_PARQUET = ROOT / "train.parquet"
SAMPLE_SUBMIT = ROOT / "sample_submit.csv"
WORK = ROOT / "work"
FEATURES_DIR = WORK / "features"
PREDS_DIR = WORK / "preds"
REPORTS_DIR = WORK / "reports"

DATA_START = date(2025, 1, 1)
DATA_END = date(2026, 2, 13)
HORIZON = 30

TEST_ANCHOR = date(2026, 2, 13)   # predict 2026-02-14 .. 2026-03-15
VAL_ANCHOR = date(2026, 1, 14)    # target 2026-01-15 .. 2026-02-13 (observed)


def train_anchors(n: int = 14, stride: int = 14) -> list[date]:
    """Training anchors strictly before VAL_ANCHOR, newest first in time order."""
    out = [VAL_ANCHOR - timedelta(days=stride * i) for i in range(1, n + 1)]
    return sorted(out)


def all_label_anchors(n_train: int = 14, stride: int = 14) -> list[date]:
    return train_anchors(n_train, stride) + [VAL_ANCHOR]


def rmsle(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    lt = np.log1p(np.clip(np.asarray(y_true, dtype=np.float64), 0, None))
    lp = np.log1p(np.clip(np.asarray(y_pred, dtype=np.float64), 0, None))
    return float(np.sqrt(np.mean((lt - lp) ** 2)))


def user_universe() -> pl.DataFrame:
    return (
        pl.read_csv(SAMPLE_SUBMIT, schema_overrides={"user_id": pl.Int64})
        .select("user_id")
        .sort("user_id")
    )


def load_anchor(anchor: date, columns: list[str] | None = None) -> pl.DataFrame:
    import os
    p = FEATURES_DIR / f"anchor={anchor.isoformat()}.parquet"
    extra = FEATURES_DIR / f"anchor={anchor.isoformat()}.extra.parquet"
    v3 = FEATURES_DIR / f"anchor={anchor.isoformat()}.v3.parquet"
    v4 = FEATURES_DIR / f"anchor={anchor.isoformat()}.v4.parquet"
    use2 = extra.exists() and os.environ.get("USE_V2")
    use3 = v3.exists() and os.environ.get("USE_V3")
    use4 = v4.exists() and os.environ.get("USE_V4")
    if not (use2 or use3 or use4):
        return pl.read_parquet(p, columns=columns)
    df = pl.read_parquet(p)
    if use2:
        df = df.join(pl.read_parquet(extra), on="user_id", how="left")
    if use3:
        df = df.join(pl.read_parquet(v3), on="user_id", how="left")
    if use4:
        df = df.join(pl.read_parquet(v4), on="user_id", how="left")
    return df.select(columns) if columns else df


def feature_cols(df: pl.DataFrame) -> list[str]:
    drop = {"user_id", "anchor_date", "target"}
    return [c for c in df.columns if c not in drop]

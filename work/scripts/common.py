"""Shared utilities for the Ozon E-CUP LTV competition."""
from __future__ import annotations

import os
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl

# Repo root: OZON_ROOT env var, else two levels up from this file (work/scripts/ -> repo).
# Must contain train.parquet + sample_submit.csv; work/ subtree is created next to them.
ROOT = Path(os.environ.get("OZON_ROOT", str(Path(__file__).resolve().parents[2])))
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

# v8 tier column list (kept here so load_anchor can null-fill uncovered anchors
# without importing build_features_v8 -> circular import)
V8_FEATS = [
    "gf_ord_share", "gf_gmv_share", "gf_ord_share_eb", "gf_lift", "gf_gmv_lift",
    "gf_only_flag", "gf_n_events", "gf_n_events_frac", "gf_days_since_ev", "gf_last_ev_hit",
    "wk_r7", "wk_r14", "wk_act_r7", "wk_act_r14", "wk_gap_ratio", "wk_gap_vs_max",
    "wk_ent90", "wk_ent90_sh", "wk_cv_iei",
]

# v10 tier = per-channel conversion funnels (build_features_v10.py). Kept here so
# load_anchor can null-fill uncovered anchors without importing the builder.
V10_WINDOWS = (30, 90, 365)
V10_PER_WIN = [
    "s_srch2cart", "s_cart2ord", "c_cart_pday", "c_cart2ord", "s_cart_psday",
    "cart_sshare", "ord_sshare", "aband_sshare",
    "s_aov", "c_aov", "s_aband", "c_aband",
    "s2c", "c2c", "s2o", "c2o",
]
V10_TRENDS = ["s_srch2cart", "s_cart2ord", "c_cart2ord", "s_cart_psday",
              "cart_sshare", "ord_sshare", "s_aov", "c_aov"]
# v10_s2o_90 / v10_c2o_90 dropped: rank-identical to v2's s2o_cnt_90 / c2o_cnt_90
V10_DROP = {"v10_s2o_90", "v10_c2o_90"}
V10_FEATS = [f for f in ([f"v10_{b}_{w}" for w in V10_WINDOWS for b in V10_PER_WIN]
                         + [f"v10_tr_{b}" for b in V10_TRENDS]) if f not in V10_DROP]

# v5 tier = joint low-rank factors of the user x week matrices (build_features_v5.py).
# One frozen basis for every anchor, so factor j means the same thing at every anchor.
# USE_V5=N takes the first N components (they are ordered by singular value); N <= 48.
V5_MAX_COMPS = 48


def v5_cols(n: int = V5_MAX_COMPS) -> list[str]:
    return [f"v5_j{j:02d}" for j in range(n)]


V5_FEATS = v5_cols()


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
    v6 = FEATURES_DIR / f"anchor={anchor.isoformat()}.v6.parquet"
    v7 = FEATURES_DIR / f"anchor={anchor.isoformat()}.v7.parquet"
    v8 = FEATURES_DIR / f"anchor={anchor.isoformat()}.v8.parquet"
    v10 = FEATURES_DIR / f"anchor={anchor.isoformat()}.v10.parquet"
    v5 = FEATURES_DIR / f"anchor={anchor.isoformat()}.v5.parquet"
    seqoof = FEATURES_DIR / f"anchor={anchor.isoformat()}.seqoof.parquet"
    use2 = extra.exists() and os.environ.get("USE_V2")
    use3 = v3.exists() and os.environ.get("USE_V3")
    use4 = v4.exists() and os.environ.get("USE_V4")
    use6 = v6.exists() and os.environ.get("USE_V6")
    use7 = os.environ.get("USE_V7")
    use8 = os.environ.get("USE_V8")
    use10 = os.environ.get("USE_V10")
    use5 = os.environ.get("USE_V5")
    use5s = os.environ.get("USE_V5S")
    use5cap = os.environ.get("USE_V5CAP")
    useoof = os.environ.get("USE_SEQOOF")
    if not (use2 or use3 or use4 or use6 or use7 or use8 or use10 or use5
            or use5s or use5cap or useoof):
        return pl.read_parquet(p, columns=columns)
    df = pl.read_parquet(p)
    if use2:
        df = df.join(pl.read_parquet(extra), on="user_id", how="left")
    if use3:
        df = df.join(pl.read_parquet(v3), on="user_id", how="left")
    if use4:
        df = df.join(pl.read_parquet(v4), on="user_id", how="left")
    if use6:
        df = df.join(pl.read_parquet(v6), on="user_id", how="left")
    if use7:
        if v7.exists():
            df = df.join(pl.read_parquet(v7), on="user_id", how="left")
        else:
            # v7 = HMM-sim MC tier; anchors without coverage get nulls (schema stays consistent)
            df = df.with_columns([pl.lit(None, dtype=pl.Float32).alias(c)
                                  for c in ("hmm_elog", "hmm_p_zero", "hmm_sim_std")])
    if use8:
        # v8 = gifter/awakening tier (holiday calendar mined from global daily GMV)
        if v8.exists():
            df = df.join(pl.read_parquet(v8), on="user_id", how="left")
        else:
            df = df.with_columns([pl.lit(None, dtype=pl.Float32).alias(c)
                                  for c in V8_FEATS])
    if use10:
        # v10 = per-channel conversion funnel tier; uncovered anchors get nulls
        if v10.exists():
            df = df.join(pl.read_parquet(v10), on="user_id", how="left")
        else:
            df = df.with_columns([pl.lit(None, dtype=pl.Float32).alias(c)
                                  for c in V10_FEATS])
    if use5:
        # v5 = joint low-rank weekly factors; USE_V5=N keeps the first N components
        want = v5_cols(max(1, min(int(use5), V5_MAX_COMPS)))
        if v5.exists():
            df = df.join(pl.read_parquet(v5, columns=["user_id"] + want),
                         on="user_id", how="left")
        else:
            df = df.with_columns([pl.lit(None, dtype=pl.Int16).alias(c) for c in want])
    if use5s:
        # v5s = 2 SUPERVISED weekly columns from train_wklin.py --emit-tier (leave-one-anchor-out).
        # The concentrated form of the weekly signal: the booster cannot mine 180 weak linear
        # columns, but it can use two strong ones.
        v5s = FEATURES_DIR / f"anchor={anchor.isoformat()}.v5s.parquet"
        if v5s.exists():
            df = df.join(pl.read_parquet(v5s), on="user_id", how="left")
        else:
            df = df.with_columns([pl.lit(None, dtype=pl.Float32).alias(c)
                                  for c in ("v5s_lin", "v5s_orth")])
    if use5cap:
        # Equal-capacity control for v5: USE_V5CAP=N adds N columns that carry NO new
        # information -- a fixed random orthogonal mixture of within-anchor percentile
        # ranks of N existing features. Same count, same dense-continuous character, same
        # extra parameters for the booster to spend, zero extra information. This is the
        # control the v6/v8/v10 post-mortems demand ("compare against equal capacity, not
        # against fewer features"). Built on the fly: it costs no disk.
        df = add_capacity_control(df, int(use5cap))
    if useoof:
        if seqoof.exists():
            df = df.join(pl.read_parquet(seqoof), on="user_id", how="left")
        else:
            # keep schema consistent: anchors without OOF coverage get nulls
            df = df.with_columns(pl.lit(None, dtype=pl.Float32).alias("seqoof_pred"))
    return df.select(columns) if columns else df


def add_capacity_control(df: pl.DataFrame, n: int) -> pl.DataFrame:
    """N columns of pure capacity: rotate percentile ranks of N existing features."""
    src = sorted(c for c in df.columns
                 if c not in ("user_id", "anchor_date", "target") and not c.startswith("v5cap_"))
    src = src[:n]
    R = np.linalg.qr(np.random.default_rng(20260819).normal(size=(len(src), n)))[0]
    A = np.empty((df.height, len(src)), dtype=np.float64)
    for j, c in enumerate(src):
        v = df[c].to_numpy().astype(np.float64)
        v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
        A[:, j] = np.argsort(np.argsort(v)) / max(len(v) - 1, 1)
    M = A @ R
    return df.with_columns([pl.Series(f"v5cap_{j:02d}", M[:, j].astype(np.float32))
                            for j in range(n)])


def feature_cols(df: pl.DataFrame) -> list[str]:
    drop = {"user_id", "anchor_date", "target"}
    return [c for c in df.columns if c not in drop]

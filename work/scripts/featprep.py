"""Feature preprocessing for the tabular MLP trainers (mlp2 / mlpbin / mlpziln).

WHY THIS EXISTS.  All three trainers carried a byte-identical copy of

    median-impute NaN -> clip to train [p1, p99] -> standardize

which was written once, in the first version, and never questioned.  The p99 clip
is the suspicious half: on money features with a heavy right tail it throws away
exactly the information that separates big spenders, while the target is itself
heavy-tailed and the metric lives in logs.  `--feat-prep` turns that choice into a
measurable one.  `clip99` is the historical path, bit-for-bit (same nanpercentile
call, same copyto, same clip, same mean/std), and stays the default.

Modes
  clip99   impute -> clip [p1, p99]     -> standardize   (historical default)
  clip999  impute -> clip [p0.1, p99.9] -> standardize   (keep more of the tail)
  noclip   impute                       -> standardize   (keep all of it)
  signlog  impute -> sign(x)*log1p(|x|) -> standardize   (compress, never discard)
  rank     impute -> rank-gauss via train quantile knots (tail shape dropped,
           ordering fully kept; ties share the mean z of their knot run)

fit_stats() estimates on a row-subsample of the SELECTION-train rows only, exactly
as before; apply_stats() transforms any matrix in place, blockwise, so nothing is
duplicated in memory.  The returned dict is np.savez-able and carries `mode`, so a
stats npz written before this module (no `mode` key) still reads back as clip99.
"""
from __future__ import annotations

import numpy as np

STATS_MAX_ROWS = 750_000   # row-subsample size for percentile/mean/std estimation
BLOCK = 262_144            # rows per block for in-place transform
SL_BLOCK = 65_536          # rows per block for the signlog pass (bool mask memory)
RANK_KNOTS = 2048          # quantile knots per feature for mode=rank
MODES = ("clip99", "clip999", "noclip", "signlog", "rank")
_CLIP_PCT = {"clip99": (1.0, 99.0), "clip999": (0.1, 99.9)}


def mode_of(s) -> str:
    """Mode stored in a stats dict / npz; absent (pre-flag file) means clip99."""
    if "mode" not in s:
        return "clip99"
    return str(np.asarray(s["mode"]).item())


def _signlog_inplace(A: np.ndarray) -> None:
    """A := sign(A) * log1p(|A|), in place, without a full-size float temp."""
    for i in range(0, A.shape[0], SL_BLOCK):
        B = A[i:i + SL_BLOCK]
        neg = B < 0
        np.abs(B, out=B)
        np.log1p(B, out=B)
        np.negative(B, out=B, where=neg)


def _rank_knots(S: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-feature (values, z, length) knot tables from an ALREADY-IMPUTED sample.

    Knots sit at quantile levels (i+0.5)/K, so the extremes map to finite z
    (+-3.49 at K=2048).  Equal knot values are collapsed to one knot carrying the
    MEAN z of the run — the usual mid-rank treatment of ties, which matters here
    because most features have a large mass of exact zeros.
    """
    from scipy.special import ndtri
    n, d = S.shape
    lv = (np.arange(RANK_KNOTS) + 0.5) / RANK_KNOTS
    zl = ndtri(lv)
    pos = lv * (n - 1)
    grid = np.arange(n, dtype=np.float64)
    vals = np.zeros((d, RANK_KNOTS), np.float32)
    zs = np.zeros((d, RANK_KNOTS), np.float32)
    lens = np.zeros(d, np.int32)
    for j in range(d):
        col = np.sort(S[:, j].astype(np.float64))
        kn = np.interp(pos, grid, col)
        u, inv = np.unique(kn, return_inverse=True)
        zavg = np.bincount(inv, weights=zl) / np.bincount(inv)
        m = u.size
        vals[j, :m] = u
        zs[j, :m] = zavg
        lens[j] = m
    return vals, zs, lens


def _pre(B: np.ndarray, s: dict) -> None:
    """Everything except the final standardize, in place on one block."""
    np.copyto(B, np.broadcast_to(s["med"], B.shape), where=np.isnan(B))
    mode = mode_of(s)
    if mode in _CLIP_PCT:
        np.clip(B, s["lo"], s["hi"], out=B)
    elif mode == "signlog":
        _signlog_inplace(B)
    elif mode == "rank":
        vals, zs, lens = s["rank_vals"], s["rank_z"], s["rank_len"]
        for j in range(B.shape[1]):
            m = int(lens[j])
            B[:, j] = np.interp(B[:, j].astype(np.float64),
                                vals[j, :m], zs[j, :m])


def fit_stats(X: np.ndarray, mode: str = "clip99") -> dict:
    """Estimate the transform from a row-subsample of train.

    mode="clip99" reproduces the historical function bit-for-bit.
    """
    assert mode in MODES, f"unknown --feat-prep {mode!r}, pick from {MODES}"
    step = max(1, int(np.ceil(X.shape[0] / STATS_MAX_ROWS)))
    S = np.ascontiguousarray(X[::step])
    lo_p, hi_p = _CLIP_PCT.get(mode, (1.0, 99.0))
    q = np.nanpercentile(S, [lo_p, 50.0, hi_p], axis=0)
    med = np.where(np.isfinite(q[1]), q[1], 0.0).astype(np.float32)
    lo = np.where(np.isfinite(q[0]), q[0], med).astype(np.float32)
    hi = np.where(np.isfinite(q[2]), q[2], med).astype(np.float32)
    s = dict(mode=np.array(mode), med=med, lo=lo, hi=hi)
    if mode not in _CLIP_PCT:      # lo/hi kept only so the npz layout is stable
        s["lo"] = np.full_like(med, -np.inf)
        s["hi"] = np.full_like(med, np.inf)
    if mode == "rank":
        np.copyto(S, np.broadcast_to(med, S.shape), where=np.isnan(S))
        s["rank_vals"], s["rank_z"], s["rank_len"] = _rank_knots(S)
        _pre(S, s)                 # re-imputing a NaN-free block is a no-op
    else:
        _pre(S, s)
    mean = S.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = S.std(axis=0, dtype=np.float64).astype(np.float32)
    std[~np.isfinite(std) | (std < 1e-7)] = 1.0
    del S
    s["mean"], s["std"] = mean, std
    return s


def apply_stats(X: np.ndarray, s: dict) -> None:
    """Blockwise in-place: impute -> (clip | signlog | rank) -> standardize."""
    mean, std = s["mean"], s["std"]
    for i in range(0, X.shape[0], BLOCK):
        B = X[i:i + BLOCK]
        _pre(B, s)
        B -= mean
        B /= std

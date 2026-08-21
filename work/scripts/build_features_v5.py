"""v5 feature tier: LOW-RANK WEEKLY FACTORS (user x week matrices, one shared decomposition).

Why this tier exists
--------------------
Measured identity (the CONSTANT is dead - buried 20-21.08, use margin.py's exact algebra;
the identity for delta itself still holds):
a model's contribution to the blend is ~7.1*delta^2, where delta is the
share of the model OUTSIDE the blend's linear hull, and the blend residual is NOT predictable
from our 203 aggregate features. Only a representation that is not a function of that feature
set can help. Screening found exactly one: the low-rank structure of the user x week matrix
(work/scripts/screen_repr.py, work/scripts/screen_v5.py).

Composition, chosen by screen_v5.py (out-of-sample R^2 of the blend residual, half the users
fit / half measured, Gaussian placebo of the same width in parentheses):

    sep 32 comps x 5 matrices, 54 weeks   dim 160   +0.001361 (-0.000934)
    sep 32 comps x 5 matrices, 36 weeks   dim 160   +0.000997 (-0.001223)
    sep 16 comps x 5 matrices, 36 weeks   dim  80   +0.001265 (-0.000617)
    JOINT 32 comps, 36 weeks              dim  32   +0.001155 (-0.000110)   <-- shipped
    joint 48 / 64 / 96                    dim 48/64/96  +0.001081/+0.001116/+0.000891
    singles (act/srch/cart/ord/gmv) x32   dim  32   +0.000889/+0.000855/+0.000756/
                                                    +0.000658/+0.000628

The five matrices overlap almost completely, so ONE decomposition of the five stacked side by
side reaches the same R^2 as 160 separate factors in 32 columns, with a placebo an order of
magnitude smaller (less fitting noise) and a fifth of the disk. This tier ships the joint
basis at 48 components; components 0..31 are the screened winner and are a plain prefix, so
USE_V5=32 and USE_V5=48 are both available from one file.

COMPARABILITY BETWEEN ANCHORS -- the thing that makes or breaks this tier
------------------------------------------------------------------------
SVD components are defined up to rotation and sign, so decompositions fitted SEPARATELY per
anchor are mutually incomparable and a model trained across anchors would learn noise.
Therefore:
  * weeks are indexed RELATIVE to the anchor: week 1 = [anchor-6, anchor], week 2 =
    [anchor-13, anchor-7], ..., week W = [anchor-7W+1, anchor-7W+7];
  * ONE decomposition is fitted on the row-union of the TRAINING anchors, and every anchor is
    then PROJECTED onto those frozen components.
The fit is exact and streaming: for uncentered X the right singular vectors are the
eigenvectors of the Gram matrix, so G = sum_a Xa^T Xa (180 x 180) is accumulated over training
anchors and eigendecomposed once. Signs are pinned (largest-|.| entry of each component
positive), so the basis is deterministic. --report prints the per-anchor mean/std of every
factor, and for contrast the alignment of bases fitted per anchor, which is what the naive
approach would have produced.

W = 36 weeks (252 days): the longest window with FULL coverage at every anchor we build --
the earliest training anchor 2025-09-10 has 253 days of history (data starts 2025-01-01).
54 weeks screens better (+0.001361 vs +0.000997 raw; +0.0023 vs +0.0022 after subtracting the
placebo, i.e. the truncation is nearly free once fitting noise is accounted for) but is only
covered from 2025-12-30 onwards, and the 30-day gap forces training anchors to end 2025-12-15.
Zero-padding the missing weeks would make early anchors structurally different, which is
exactly the artefact this tier must not have.

NO LEAKAGE: anchor A uses only events with event_date <= A. The decomposition is fitted on
training anchors ONLY -- never the validation anchor, never the test anchor, and not on the
4 gap anchors either (they enter only the test-model retrain).

Storage. Factors are stored as Int16 on a GLOBAL per-column grid (scale from the training-anchor
projections, identical at every anchor, ~4095 levels over the robust range, outliers preserved
rather than clipped). Trees are invariant to a fixed monotone affine map and LightGBM re-bins
into max_bin=127 anyway, so this is lossless in practice and cuts the tier to a third. True
scales live in v5_components.npz for any consumer that wants floats.

Output
  work/features/anchor=DATE.v5.parquet   48 joint factors v5_j00..v5_j47
  work/features/v5_components.npz        frozen basis + scales + provenance
  work/reports/v5_build.json             comparability table, variance explained, screening
Loaded by common.load_anchor when USE_V5=32 (or any 1..48); the value is how many to use.

Usage
  POLARS_MAX_THREADS=3 .venv/bin/python work/scripts/build_features_v5.py
        [--comps 48] [--report] [--screen] [--smoke 5000] [--force]
"""
from __future__ import annotations

import os

_T = os.environ.get("THREADS", "3")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "POLARS_MAX_THREADS"):
    os.environ.setdefault(_v, _T)

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from datetime import date, timedelta  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from common import (  # noqa: E402
    FEATURES_DIR, REPORTS_DIR, TEST_ANCHOR, TRAIN_PARQUET, VAL_ANCHOR,
    V5_MAX_COMPS, v5_cols, user_universe,
)
from exp_lib import available_train_anchors  # noqa: E402

W_WEEKS = 36                 # 252 days: full coverage at every anchor we build
# (tag, source column). `act` counts rows, i.e. days with any activity in the week.
QTY = (("act", None), ("cart", "to_cart"), ("ord", "to_ord"),
       ("srch", "searches"), ("gmv", "gmv"))
GAP_DAYS = 30
N_TRAIN_ANCHORS = 14
QLEVELS = 4095               # int16 grid over the robust range (outliers not clipped)
MIN_FREE_GB = 12.6           # queue runner stalls below 12 GB; keep a margin
CHUNK = 50_000               # rows per float64 Gram chunk


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def free_gb() -> float:
    st = os.statvfs("/")
    return st.f_bavail * st.f_frsize / 1e9


# ------------------------------------------------------------------ anchor plan
def anchor_plan() -> dict:
    """The exact anchors the champion protocol touches, split by role."""
    avail = available_train_anchors()
    cutoff = VAL_ANCHOR - timedelta(days=GAP_DAYS)
    return {"fit": [a for a in avail if a <= cutoff][-N_TRAIN_ANCHORS:],
            "gap": [a for a in avail if cutoff < a < VAL_ANCHOR],
            "val": [VAL_ANCHOR], "test": [TEST_ANCHOR]}


# ------------------------------------------------------------------ weekly matrices
def weekly_dense(ref: date, n_weeks: int, uid: np.ndarray, strict: bool = True) -> dict[str, np.ndarray]:
    """user x week (log1p of the weekly sum) per quantity; week 0 = [ref-6, ref].

    One streaming pass over train.parquet reading only event_date <= ref, so a grid built
    for `ref` can never see the future of any anchor <= ref.
    """
    t0 = time.time()
    lo = ref - timedelta(days=7 * n_weeks - 1)
    aggs = [pl.len().cast(pl.Float32).alias("act")]
    aggs += [pl.col(c).sum().cast(pl.Float32).alias(t) for t, c in QTY if c]
    df = (pl.scan_parquet(TRAIN_PARQUET)
          .select("user_id", "event_date", "to_cart", "to_ord", "searches", "gmv")
          .filter((pl.col("event_date") <= ref) & (pl.col("event_date") >= lo))
          .with_columns(((pl.lit(ref) - pl.col("event_date")).dt.total_days() // 7)
                        .cast(pl.Int16).alias("wk"))
          .group_by(["user_id", "wk"]).agg(aggs)
          .collect(engine="streaming"))
    u = df["user_id"].to_numpy()
    pos = np.clip(np.searchsorted(uid, u), 0, len(uid) - 1)
    keep = uid[pos] == u
    assert keep.all() or not strict, "event user_id outside the submission universe"
    rows, wk = pos[keep], df["wk"].to_numpy().astype(np.int64)[keep]
    out = {}
    for tag, _ in QTY:
        M = np.zeros((len(uid), n_weeks), dtype=np.float32)
        M[rows, wk] = np.log1p(np.maximum(df[tag].to_numpy()[keep], 0.0))
        out[tag] = M
    log(f"  weekly grid ref={ref} weeks={n_weeks}: {df.height} cells in {time.time()-t0:.0f}s")
    return out


def joint_block(grids: dict[str, np.ndarray], offset_weeks: int) -> np.ndarray:
    """[act | cart | ord | srch | gmv] over anchor-relative weeks 1..W, 5*W columns."""
    return np.hstack([grids[t][:, offset_weeks:offset_weeks + W_WEEKS] for t, _ in QTY])


# ------------------------------------------------------------------ decomposition
def gram(X: np.ndarray) -> np.ndarray:
    """X^T X accumulated in float64 in row chunks (float32 sgemm would lose precision)."""
    G = np.zeros((X.shape[1], X.shape[1]), dtype=np.float64)
    for i in range(0, X.shape[0], CHUNK):
        C = X[i:i + CHUNK].astype(np.float64)
        G += C.T @ C
    return G


def basis_from_gram(G: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ev, V = np.linalg.eigh(G)
    order = np.argsort(ev)[::-1]
    ev, V = ev[order], V[:, order]
    for j in range(V.shape[1]):                 # pin the sign: largest |entry| positive
        if V[np.argmax(np.abs(V[:, j])), j] < 0:
            V[:, j] *= -1.0
    return V, np.sqrt(np.maximum(ev, 0.0))


# ------------------------------------------------------------------ quantisation
def make_scales(F: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lo = np.percentile(F, 0.005, axis=0)
    hi = np.percentile(F, 99.995, axis=0)
    return (hi + lo) / 2.0, np.maximum((hi - lo) / QLEVELS, 1e-12)


def quantise(F: np.ndarray, mid: np.ndarray, step: np.ndarray) -> np.ndarray:
    return np.clip(np.round((F - mid) / step), -32767, 32767).astype(np.int16)


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--comps", type=int, default=V5_MAX_COMPS)
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--screen", action="store_true")
    ap.add_argument("--smoke", type=int, default=0, help="mechanics check on N users, no writes")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    assert 1 <= args.comps <= 5 * W_WEEKS

    t0 = time.time()
    plan = anchor_plan()
    fit_anchors, gap_anchors = plan["fit"], plan["gap"]
    all_anchors = fit_anchors + gap_anchors + [VAL_ANCHOR, TEST_ANCHOR]
    log(f"fit anchors ({len(fit_anchors)}): {[a.isoformat() for a in fit_anchors]}")
    log(f"gap anchors ({len(gap_anchors)}): {[a.isoformat() for a in gap_anchors]}")
    log(f"projecting {len(all_anchors)} anchors, W={W_WEEKS}, comps={args.comps}")

    uid = user_universe()["user_id"].to_numpy()
    assert (np.diff(uid) > 0).all(), "universe must be sorted"
    if args.smoke:
        uid = uid[:args.smoke]

    hist_days = (min(all_anchors) - date(2025, 1, 1)).days + 1
    assert hist_days >= 7 * W_WEEKS, f"only {hist_days} days at {min(all_anchors)}"
    log(f"coverage ok: earliest anchor has {hist_days} days >= {7*W_WEEKS}")

    # Every anchor except TEST sits on the VAL 7-day grid -> one pass serves all 19.
    grid_anchors = fit_anchors + gap_anchors + [VAL_ANCHOR]
    max_off = max((VAL_ANCHOR - a).days // 7 for a in grid_anchors)
    Gv = weekly_dense(VAL_ANCHOR, max_off + W_WEEKS, uid, strict=not args.smoke)
    off = {a: (VAL_ANCHOR - a).days // 7 for a in grid_anchors}

    # ---- one frozen basis, fitted on TRAINING anchors only
    G = np.zeros((5 * W_WEEKS, 5 * W_WEEKS), dtype=np.float64)
    for a in fit_anchors:
        G += gram(joint_block(Gv, off[a]))
    V, sv = basis_from_gram(G)
    B = V[:, :args.comps]
    ex = float((sv[:args.comps] ** 2).sum() / max((sv ** 2).sum(), 1e-12))
    log(f"basis: {args.comps}/{5*W_WEEKS} comps, variance explained {ex:.4f}")

    Ftr = np.vstack([joint_block(Gv, off[a]) @ B for a in fit_anchors])
    mid, step = make_scales(Ftr)
    del Ftr

    if args.smoke:
        F = joint_block(Gv, off[VAL_ANCHOR]) @ B
        log(f"SMOKE ok: F {F.shape} mean {F.mean():.4f} std {F.std():.4f} "
            f"all-zero rows {(np.abs(F).sum(1) == 0).mean():.4f}; nothing written")
        return

    # ---- project + write
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    names = v5_cols(args.comps)
    stats, sizes = {}, {}
    Gt = None
    for a in all_anchors:
        if a == TEST_ANCHOR:
            Gt = weekly_dense(TEST_ANCHOR, W_WEEKS, uid)
            X = joint_block(Gt, 0)
        else:
            X = joint_block(Gv, off[a])
        F = (X @ B).astype(np.float64)
        stats[a.isoformat()] = {"mean": np.round(F.mean(0), 5).tolist(),
                                "std": np.round(F.std(0), 5).tolist()}
        p = FEATURES_DIR / f"anchor={a.isoformat()}.v5.parquet"
        if args.force or not p.exists():
            if free_gb() < MIN_FREE_GB:
                log(f"ABORT: only {free_gb():.1f} GB free (< {MIN_FREE_GB})")
                sys.exit(3)
            Q = quantise(F, mid, step)
            df = pl.DataFrame({"user_id": uid.astype(np.int64)}
                              | {c: Q[:, j] for j, c in enumerate(names)})
            tmp = p.with_suffix(".tmp.parquet")
            df.write_parquet(tmp, compression="zstd", compression_level=12)
            tmp.rename(p)
            sizes[a.isoformat()] = round(p.stat().st_size / 1e6, 1)
            log(f"  {p.name}: {df.shape} {sizes[a.isoformat()]} MB")
            del Q, df
        del F, X

    np.savez(FEATURES_DIR / "v5_components.npz", V=B, sv=sv, mid=mid, step=step,
             weeks=np.int32(W_WEEKS), comps=np.int32(args.comps),
             quantities=np.array([t for t, _ in QTY]),
             fit_anchors=np.array([a.isoformat() for a in fit_anchors]))

    out = {"weeks": W_WEEKS, "comps": args.comps, "quantities": [t for t, _ in QTY],
           "construction": "joint SVD of [act|cart|ord|srch|gmv] stacked, frozen basis",
           "fit_anchors": [a.isoformat() for a in fit_anchors],
           "projected_anchors": [a.isoformat() for a in all_anchors],
           "variance_explained": round(ex, 5),
           "mb_per_anchor": sizes, "mb_total": round(sum(sizes.values()), 1),
           "free_gb_after": round(free_gb(), 2)}
    if args.report:
        out["comparability"] = comparability(stats, Gv, off, fit_anchors)
    if args.screen:
        out["screen"] = screen(joint_block(Gv, off[VAL_ANCHOR]) @ B)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "v5_build.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    log(f"V5 DONE in {time.time()-t0:.0f}s, free disk {free_gb():.1f} GB")
    print("\n=== v5_build.json (without the full comparability table) ===")
    print(json.dumps({k: v for k, v in out.items() if k != "comparability"},
                     ensure_ascii=False))
    if args.report:
        c = out["comparability"]
        print(f"comparability ok={c['ok']} max_drift={c['max_drift']:.4f} "
              f"sd_ratio_max={c['sd_ratio_max']:.3f} sign_flips={c['mean_sign_flips']}")


# ------------------------------------------------------------------ comparability
def comparability(stats: dict, Gv, off, fit_anchors) -> dict:
    """Is factor j the same quantity at every anchor?

    By construction yes: one frozen basis, projection only. This is the empirical check --
    per-anchor mean/std of every factor. Anchors legitimately differ in LEVEL (the platform
    roughly doubled through 2025), so the criterion is not "identical means" but "no factor
    drifts by more than a fraction of its own spread, no factor flips sign, no factor changes
    scale by more than 2x".

    The contrast row shows what per-anchor decompositions would have given: the diagonal of
    |V_first^T V_last| for bases fitted separately on the first and last training anchor.
    Values well below 1 mean the axes are rotated relative to each other -- proof that the
    naive per-anchor approach really is incomparable.
    """
    anchors = sorted(stats)
    mu = np.array([stats[a]["mean"] for a in anchors])
    sd = np.array([stats[a]["std"] for a in anchors])
    pooled = sd.mean(0)
    drift = mu.std(0) / np.maximum(pooled, 1e-9)
    sd_ratio = sd.max(0) / np.maximum(sd.min(0), 1e-9)
    flips = int(((mu > 0).any(0) & (mu < 0).any(0)).sum())
    V0, _ = basis_from_gram(gram(joint_block(Gv, off[fit_anchors[0]])))
    V1, _ = basis_from_gram(gram(joint_block(Gv, off[fit_anchors[-1]])))
    align = np.abs(np.diag(V0[:, :8].T @ V1[:, :8]))
    return {"anchors": anchors,
            "mean_drift_over_pooled_sd": np.round(drift, 4).tolist(),
            "max_drift": float(drift.max()),
            "sd_ratio_per_factor": np.round(sd_ratio, 3).tolist(),
            "sd_ratio_max": float(sd_ratio.max()),
            "mean_sign_flips": flips,
            "per_anchor_mean_f0": np.round(mu[:, 0], 4).tolist(),
            "per_anchor_mean_f1": np.round(mu[:, 1], 4).tolist(),
            "per_anchor_basis_alignment_top8": np.round(align, 4).tolist(),
            "ok": bool(drift.max() < 0.5 and sd_ratio.max() < 2.0 and flips == 0)}


# ------------------------------------------------------------------ screening
def screen(F: np.ndarray) -> dict:
    """Out-of-sample R^2 of the blend residual on the shipped factors vs a same-size placebo."""
    from sklearn.linear_model import Ridge
    from screen_repr import blend_residual
    _, e = blend_residual()
    sb = float(np.sqrt((e ** 2).mean()))
    rng = np.random.default_rng(0)
    m = rng.random(len(e)) < 0.5

    def r2(A):
        A = np.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0)
        A = (A - A.mean(0)) / (A.std(0) + 1e-9)
        p = Ridge(alpha=10.0).fit(A[m], e[m]).predict(A[~m])
        return float(1 - ((e[~m] - p) ** 2).sum() / ((e[~m] - e[~m].mean()) ** 2).sum())

    out = {}
    for k in (16, 32, F.shape[1]):
        if k > F.shape[1]:
            continue
        R, P = r2(F[:, :k]), r2(rng.normal(size=(F.shape[0], k)))
        out[f"comps{k}"] = {"r2": round(R, 6), "placebo": round(P, 6),
                            "gain_rmsle": round(sb - sb * float(np.sqrt(max(1 - max(R, 0.), 0.))), 6)}
        print(f"  screen comps{k}: mdl_flint {R:+.6f} placebo {P:+.6f}", flush=True)
    return out


if __name__ == "__main__":
    main()

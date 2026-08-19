"""Honest gain from ADDING a set of candidates to the live blend.

margin.py answers "is this one model outside the hull". It cannot answer the question
that decides a whole direction: several stale vintages are decorrelated from the blend
AND from each other, so the set may be worth more than any member. The team's
contribution shorthand (7.1 * margin^2) is per-model and does not compose.

So measure it directly, in the currency of the leaderboard:

  fit non-negative weights for [blend, cand1, cand2, ...] in log1p space on one half

Cross-fitting is not optional here: weights fitted and scored on the same users buy a
gain that does not exist (the lesson of caruana.md, where val 1.6240 came from 94%
contaminated members). Candidates are calibrated first, by the same honest cross-fit
margin.py uses, because the team's rule is that raw predictions rank models wrongly.

Usage:
  python work/scripts/joint_gain.py lagd28 lagd42 lagd56
  python work/scripts/joint_gain.py --each lagd28 lagd42 lagd56   # also one-at-a-time
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))
from common import PREDS_DIR, ROOT
from margin import calibrate_honest, score

NOISE = 0.000022


def nnls_weights(A: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Non-negative least squares, exact (active-set). Weights are free, no sum-to-1.

    Negative weights are legal in the team's Gram mixing, but here they would let the
    fit exploit noise structure that does not transfer; the honest half-split would
    catch it, and forbidding them keeps the number conservative.

    Solved on the normal equations rather than the 125k x k design: identical solution,
    and it keeps the columns (which are highly collinear, r > 0.99) in float64.
    """
    from scipy.optimize import nnls
    G = A.T @ A
    b = A.T @ y
    # Cholesky factor of the Gram matrix turns min||Aw-y|| into an equivalent small NNLS
    L = np.linalg.cholesky(G + 1e-10 * np.eye(len(G)))
    return nnls(L.T, np.linalg.solve(L, b))[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="+")
    ap.add_argument("--pack", type=Path, default=ROOT / "work" / "preds_pack")
    ap.add_argument("--each", action="store_true", help="also report each candidate alone")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    pack = pl.read_parquet(args.pack / "val_preds.parquet").sort("user_id")
    uid = pack["user_id"].to_numpy()
    ly = np.log1p(np.clip(pack["target"].to_numpy().astype(np.float64), 0, None))
    lb = pack["blend"].to_numpy().astype(np.float64)
    sb = score(lb, ly)
    print(f"эталон: бленд из пакета, скор {sb:.6f}, n={len(ly)}\n")

    cand = {}
    for n in args.names:
        p = PREDS_DIR / f"{n}_val.parquet"
        if not p.exists():
            print(f"{n}: НЕТ ФАЙЛА {p}")
            continue
        df = pl.read_parquet(p).sort("user_id")
        assert np.array_equal(df["user_id"].to_numpy(), uid), f"порядок user_id не совпал: {p}"
        raw = np.log1p(np.clip(df["pred"].to_numpy().astype(np.float64), 0, None))
        cand[n] = calibrate_honest(raw, ly, 24, args.seed)

    rng = np.random.default_rng(args.seed)
    half = rng.permutation(len(ly)) < len(ly) // 2

    def gain(names):
        A = np.column_stack([lb] + [cand[n] for n in names])
        gs, ws = [], []
        for m in (half, ~half):                      # fit on m, score on ~m, then swap
            w = nnls_weights(A[m], ly[m])
            gs.append(sb - score(A[~m] @ w, ly[~m]))
            ws.append(w)
        return float(np.mean(gs)), np.mean(ws, axis=0)

    rows = ([( [n], n) for n in cand] if args.each else []) + \
           ([(list(cand), "ВСЕ ВМЕСТЕ")] if len(cand) > 1 else [])
    hdr = f"{'набор':<28}{'выигрыш':>11}  веса"
    print(hdr); print("-" * (len(hdr) + 20))
    for names, label in rows:
        g, w = gain(names)
        verdict = "ГОДИТСЯ" if g >= 0.0003 else "слабо, но не шум" if g >= 2 * NOISE else "шум"
        wtxt = " ".join(f"{n}={v:.3f}" for n, v in zip(["бленд"] + names, w))
        print(f"{label:<28}{g:+11.6f}  {wtxt}")
        print(f"{'':<28}{'':>11}  -> {verdict}")


if __name__ == "__main__":
    main()

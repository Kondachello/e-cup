"""Greedy set selection over the whole prediction library, with an honest floor.

Why this exists: acceptance in this project is per-model (margin vs the blend), and we
measured that the rule is structurally wrong - lagd28 and hz_v1_surv are each worthless
some may be alive as a SET. The library holds 65 val files; this sweeps them.

Two traps this design avoids, both already paid for by the team:

1. caruana.md: a set chosen and scored on the same users buys a gain that does not
   transfer (val 1.6240 -> public 1.6755). So users are split ONCE into DEV and EVAL;
   every selection decision and every weight is fitted on DEV, and the reported number
   comes from EVAL, which selection never touched.
2. Rule 4 of the team protocol: four arbitrary existing models added to the blend give
   +0.00011 by themselves. A positive gain is therefore meaningless without a floor, so
   the same pipeline is run on RANDOM sets of the same size. Greedy has to beat the
   95th percentile of random, not zero.

Usage:
  python work/scripts/library_sweep.py                 # full sweep
  python work/scripts/library_sweep.py --max-k 8 --random-sets 200
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))
from common import PREDS_DIR, ROOT
from margin import calibrate_split, score

# A single model cannot beat a 30-model blend; anything that does is contaminated

SANITY_FLOOR = 1.66

# Models trained BEFORE the 30-day gap was introduced: their val score is inflated because
# they partly memorised the validation users. gru_final is named as such in err_corr.py and
# was sitting in the core of every greedy pick, which makes every number that rests on it
# incomparable. Excluded by default; --allow-dirty puts them back for a controlled check.
# Evidence, not association: gru_final is named in err_corr.py as pre-gap30, and the other
# three are the fake scores of H3. seq2tr_f was on this list by association and is removed -
# Sasha's rebuilt pack drops gru_final and KEEPS seq2tr_f_cal, which settles it.
CONTAMINATED = {"gru_final", "blend_w1a", "twlog_probe", "ts2_a"}


def nnls_weights(A: np.ndarray, y: np.ndarray) -> np.ndarray:
    from scipy.optimize import nnls
    G = A.T @ A
    b = A.T @ y
    # The pack ships near-duplicate columns (mlpziln_cal vs mlpziln_cal_avg_cal, and random
    # sets can draw two of them), so the Gram matrix is singular and a fixed 1e-10 ridge is
    # not enough. Scale the ridge to the matrix and grow it until the factorisation holds.
    eps = 1e-10 * max(np.trace(G) / len(G), 1.0)
    for _ in range(12):
        try:
            L = np.linalg.cholesky(G + eps * np.eye(len(G)))
            return nnls(L.T, np.linalg.solve(L, b))[0]
        except np.linalg.LinAlgError:
            eps *= 100.0
    return np.linalg.lstsq(A, y, rcond=None)[0].clip(0)


def fit_eval(A_dev, y_dev, A_ev, y_ev, sb_ev):
    """Weights from DEV, gain measured on EVAL. The only number that counts."""
    w = nnls_weights(A_dev, y_dev)
    return sb_ev - score(A_ev @ w, y_ev), w


def inner_gain(A_dev, y_dev, half):
    """Cross-fit gain INSIDE dev - used only to rank candidates during selection."""
    gs = []
    for m in (half, ~half):
        w = nnls_weights(A_dev[m], y_dev[m])
        gs.append(score(A_dev[~m][:, 0], y_dev[~m]) - score(A_dev[~m] @ w, y_dev[~m]))
    return float(np.mean(gs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", type=Path, default=ROOT / "work" / "preds_pack")
    ap.add_argument("--max-k", type=int, default=10)
    ap.add_argument("--random-sets", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--allow-dirty", action="store_true",
                    help="keep pre-gap30 models (gru_final etc.) in the library")
    args = ap.parse_args()

    pack = pl.read_parquet(args.pack / "val_preds.parquet").sort("user_id")
    uid = pack["user_id"].to_numpy()
    ly = np.log1p(np.clip(pack["target"].to_numpy().astype(np.float64), 0, None))
    lb = pack["blend"].to_numpy().astype(np.float64)
    print(f"эталон: бленд, скор {score(lb, ly):.6f}, n={len(ly)}")

    rng = np.random.default_rng(args.seed)
    dev = rng.permutation(len(ly)) < len(ly) // 2      # отбор и веса живут здесь
    ev = ~dev                                          # это EVAL, отбор его не видел

    names, cols = [], []
    for f in sorted(PREDS_DIR.glob("*_val.parquet")):
        n = f.name[: -len("_val.parquet")]
        if n == "blend":
            continue
        if n in CONTAMINATED and not args.allow_dirty:
            print(f"  ИСКЛЮЧЁН {n}: обучен до зазора 30 дней, val-скор завышен")
            continue
        d = pl.read_parquet(f).sort("user_id")
        if d.height != len(uid) or not np.array_equal(d["user_id"].to_numpy(), uid):
            print(f"  пропуск {n}: чужой юниверс")
            continue
        lp = calibrate_split(
            np.log1p(np.clip(d["pred"].to_numpy().astype(np.float64), 0, None)), ly, dev, 24)
        s = score(lp, ly)
        if s < SANITY_FLOOR:
            print(f"  ПРОПУСК {n}: скор {s:.4f} < {SANITY_FLOOR} — признак контаминации")
            continue
        names.append(n); cols.append(lp)
    X = np.column_stack(cols)
    print(f"кандидатов в библиотеке: {len(names)}\n")

    half = rng.permutation(dev.sum()) < dev.sum() // 2

    bd, be = lb[dev], lb[ev]
    yd, ye = ly[dev], ly[ev]
    sb_ev = score(be, ye)
    Xd, Xe = X[dev], X[ev]

    chosen, curve = [], []
    for k in range(1, args.max_k + 1):
        best = (-9, None)
        for j in range(len(names)):
            if j in chosen:
                continue
            A = np.column_stack([bd] + [Xd[:, i] for i in chosen + [j]])
            g = inner_gain(A, yd, half)
            if g > best[0]:
                best = (g, j)
        chosen.append(best[1])
        Ad = np.column_stack([bd] + [Xd[:, i] for i in chosen])
        Ae = np.column_stack([be] + [Xe[:, i] for i in chosen])
        g_ev, w = fit_eval(Ad, yd, Ae, ye, sb_ev)
        curve.append((k, names[best[1]], best[0], g_ev))
        print(f"k={k:2d}  +{names[best[1]]:<22} отбор(dev)={best[0]:+.6f}  ЧЕСТНО(eval)={g_ev:+.6f}")

    print("\n--- пол: случайные наборы той же ёмкости (правило 4) ---")
    floors = {}
    for k in range(1, args.max_k + 1):
        gs = []
        for _ in range(args.random_sets):
            idx = rng.choice(len(names), size=k, replace=False)
            Ad = np.column_stack([bd] + [Xd[:, i] for i in idx])
            Ae = np.column_stack([be] + [Xe[:, i] for i in idx])
            gs.append(fit_eval(Ad, yd, Ae, ye, sb_ev)[0])
        floors[k] = (float(np.mean(gs)), float(np.percentile(gs, 95)))
        print(f"k={k:2d}  случайный набор: среднее {floors[k][0]:+.6f}  95-й перцентиль {floors[k][1]:+.6f}")

    print("\n--- ВЕРДИКТ ---")
    print(f"{'k':>3}{'жадный (eval)':>16}{'пол (95%)':>14}  бьёт пол?")
    for k, nm, gd, ge in curve:
        print(f"{k:>3}{ge:>16.6f}{floors[k][1]:>14.6f}  {'ДА' if ge > floors[k][1] else 'нет'}")


if __name__ == "__main__":
    main()

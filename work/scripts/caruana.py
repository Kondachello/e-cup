"""Bagged Caruana ensemble selection (with replacement) over the val-pred library.

Optimizes RMSLE in log1p space. Library = every work/preds/*_val.parquet with a
matching *_test.parquet, excluding smoke/blend/base_best/cand/c_cand/A prefixes,
plus mlp2_final_cal explicitly. Guard: drop models with solo val RMSLE < 1.60
(contaminated-protocol leftovers).

Algorithm: B=20 bags; each bag samples 50% of library models (no replacement)
and 80% of val users (row bagging); greedy forward selection WITH replacement
for 40 steps minimizing bag-RMSLE of the running mean of log1p preds.
Final weights = average of per-bag pick frequencies.

Compares vs plain hill-climb on full val (blend.py logic) and the manual mix
0.85*mlp2_final_cal + 0.15*c_xtw_s42. Saves caruana_v1 if it beats the manual
mix on full val.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "POLARS_MAX_THREADS"):
    os.environ.setdefault(_v, "2")

import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))
from common import PREDS_DIR, REPORTS_DIR, VAL_ANCHOR, load_anchor, rmsle
from exp_lib import log_score, save_preds

EXCLUDE_PREFIXES = ("smoke", "blend", "base_best", "cand", "c_cand", "A", "caruana")
FORCE_INCLUDE = ["mlp2_final_cal"]
GUARD_RMSLE = 1.60
B_BAGS = 20
MODEL_FRAC = 0.5
ROW_FRAC = 0.8
STEPS = 40
SEED = 42
HILL_ITERS = 60
NAME = "caruana_v1"
MANUAL = {"mlp2_final_cal": 0.85, "c_xtw_s42": 0.15}


def discover() -> list[str]:
    names = []
    for p in sorted(PREDS_DIR.glob("*_val.parquet")):
        n = p.name[: -len("_val.parquet")]
        if not (PREDS_DIR / f"{n}_test.parquet").exists():
            continue
        if n.startswith(EXCLUDE_PREFIXES) and n not in FORCE_INCLUDE:
            print(f"  [excl prefix] {n}")
            continue
        names.append(n)
    for n in FORCE_INCLUDE:
        if n not in names and (PREDS_DIR / f"{n}_val.parquet").exists() \
                and (PREDS_DIR / f"{n}_test.parquet").exists():
            names.append(n)
    return sorted(set(names))


def greedy_gram(G: np.ndarray, b: np.ndarray, yy: float, steps: int,
                stop_no_improve: bool = False) -> np.ndarray:
    """Forward selection with replacement minimizing mean((ly - mean_of_picks)^2).

    G[i,j] = mean(p_i * p_j) on the (bagged) rows, b[i] = mean(p_i * ly),
    yy = mean(ly^2). Returns pick counts per model.
    All in exact algebra: for counts c (k picks), running mean m = (1/k) sum c_i p_i,
    mse = yy - 2/k * c.b + 1/k^2 * c'Gc.
    """
    m = G.shape[0]
    counts = np.zeros(m, dtype=np.int64)
    g = np.zeros(m)          # (G @ c)
    bc = 0.0                 # c . b
    cGc = 0.0                # c' G c
    diag = np.diag(G).copy()
    best_prev = np.sqrt(yy)  # score of empty ensemble (cur = 0), as in blend.py
    for _ in range(steps):
        k = counts.sum()
        mse = yy - 2.0 * (bc + b) / (k + 1) + (cGc + 2.0 * g + diag) / (k + 1) ** 2
        j = int(np.argmin(mse))
        s = float(np.sqrt(max(mse[j], 0.0)))
        if stop_no_improve and k > 0 and s >= best_prev - 1e-7:
            break
        counts[j] += 1
        cGc += 2.0 * g[j] + diag[j]
        bc += b[j]
        g += G[:, j]
        best_prev = s
    return counts


def main():
    rng = np.random.default_rng(SEED)

    val = load_anchor(VAL_ANCHOR, columns=["user_id", "target"]).sort("user_id")
    y = val["target"].to_numpy().astype(np.float64)
    uid = val["user_id"].to_numpy()
    ly = np.log1p(np.clip(y, 0, None))
    N = len(uid)

    names_all = discover()
    print(f"discovered {len(names_all)} candidates: {names_all}")

    P, T, solo = {}, {}, {}
    uid_t = None
    for n in names_all:
        dv = pl.read_parquet(PREDS_DIR / f"{n}_val.parquet").sort("user_id")
        assert (dv["user_id"].to_numpy() == uid).all(), f"uid mismatch {n}"
        P[n] = np.log1p(np.clip(dv["pred"].to_numpy().astype(np.float64), 0, None))
        dt = pl.read_parquet(PREDS_DIR / f"{n}_test.parquet").sort("user_id")
        if uid_t is None:
            uid_t = dt["user_id"].to_numpy()
        else:
            assert (dt["user_id"].to_numpy() == uid_t).all(), f"test uid mismatch {n}"
        T[n] = np.log1p(np.clip(dt["pred"].to_numpy().astype(np.float64), 0, None))
        solo[n] = rmsle(y, np.expm1(P[n]))

    guard_dropped = sorted(n for n in names_all if solo[n] < GUARD_RMSLE)
    names = [n for n in names_all if n not in guard_dropped]
    print("\nsolo val RMSLE:")
    for n in sorted(names_all, key=lambda x: solo[x]):
        flag = "  <-- GUARD-EXCLUDED (<1.60, contaminated-protocol leftover)" \
            if n in guard_dropped else ""
        print(f"  {n}: {solo[n]:.6f}{flag}")
    print(f"\nlibrary after guard: {len(names)} models")

    def bagged_caruana(lib: list[str], rng: np.random.Generator, verbose: bool = True):
        Ml = np.stack([P[n] for n in lib])
        ml = len(lib)
        n_sel = max(2, int(round(MODEL_FRAC * ml)))
        n_rows = int(round(ROW_FRAC * N))
        freq = np.zeros(ml)
        bag_scores = []
        for bag in range(B_BAGS):
            sel = np.sort(rng.choice(ml, size=n_sel, replace=False))
            rows = rng.choice(N, size=n_rows, replace=False)
            Pb = Ml[np.ix_(sel, rows)]
            lyb = ly[rows]
            G = (Pb @ Pb.T) / n_rows
            b = (Pb @ lyb) / n_rows
            yy = float(np.mean(lyb ** 2))
            counts = greedy_gram(G, b, yy, STEPS, stop_no_improve=False)
            freq[sel] += counts / counts.sum()
            wb = counts / counts.sum()
            mse = yy - 2.0 * wb @ b + wb @ G @ wb
            bag_scores.append(float(np.sqrt(max(mse, 0.0))))
            if verbose:
                picked = {lib[sel[i]]: int(counts[i]) for i in range(n_sel) if counts[i]}
                print(f"  bag {bag:02d}: rmsle={bag_scores[-1]:.6f} picks={picked}")
        w = freq / B_BAGS
        wd = {lib[i]: float(w[i]) for i in range(ml) if w[i] > 0}
        assert abs(sum(wd.values()) - 1.0) < 1e-9
        lv = w @ Ml
        return wd, lv, float(np.mean(bag_scores))

    M = np.stack([P[n] for n in names])            # (m, N) float64 log1p preds
    m = len(names)
    yy_full = float(np.mean(ly ** 2))

    # ---------- bagged Caruana (full library) ----------
    weights, lv_car, mean_bag = bagged_caruana(names, rng)
    caruana_val = rmsle(y, np.expm1(lv_car))
    print(f"\nbagged-Caruana weights ({len(weights)} nonzero):")
    for n, wi in sorted(weights.items(), key=lambda kv: -kv[1]):
        print(f"  {n}: {wi:.6f}")
    print(f"bagged-Caruana full-val RMSLE: {caruana_val:.6f}")
    print(f"mean bag rmsle: {mean_bag:.6f}")

    # ---------- sensitivity: clean-protocol-only library (report-only) ----------
    # Pre-gap30-era models: trained with anchors adjacent to the val window
    # (no 30d gap), so their val scores are optimistic vs CLEAN retrains
    # (same archs with gap30 score ~1.69, these show ~1.63).
    OLD_ERA = {"lgblog_final", "xgblog_final", "mlp_final", "gru_final"}
    clean_names = [n for n in names if n not in OLD_ERA]
    w_clean, lv_clean, _ = bagged_caruana(clean_names, np.random.default_rng(SEED + 1),
                                          verbose=False)
    clean_val = rmsle(y, np.expm1(lv_clean))
    print(f"\n[sensitivity] clean-only ({len(clean_names)} models) bagged-Caruana "
          f"val {clean_val:.6f}, weights {dict((k, round(v, 4)) for k, v in sorted(w_clean.items(), key=lambda kv: -kv[1]))}")

    # ---------- plain hill-climb on full val (blend.py logic) ----------
    G_full = (M @ M.T) / N
    b_full = (M @ ly) / N
    counts_hc = greedy_gram(G_full, b_full, yy_full, HILL_ITERS, stop_no_improve=True)
    w_hc = counts_hc / counts_hc.sum()
    lv_hc = w_hc @ M
    hillclimb_val = rmsle(y, np.expm1(lv_hc))
    hc_weights = {names[i]: float(w_hc[i]) for i in range(m) if w_hc[i] > 0}
    print(f"\nhill-climb (blend.py logic, {int(counts_hc.sum())} picks) weights: {hc_weights}")
    print(f"hill-climb full-val RMSLE: {hillclimb_val:.6f}")

    # ---------- manual mix ----------
    for n in MANUAL:
        assert n in P, f"manual-mix model {n} missing"
    lv_man = sum(P[n] * wi for n, wi in MANUAL.items())
    manual_val = rmsle(y, np.expm1(lv_man))
    man_raw = rmsle(y, sum(np.expm1(P[n]) * wi for n, wi in MANUAL.items()))
    print(f"\nmanual mix {MANUAL}: log-space val {manual_val:.6f} (raw-space {man_raw:.6f})")

    # ---------- save if caruana beats manual ----------
    saved = bool(caruana_val < manual_val)
    if saved:
        lt_car = sum(T[n] * wi for n, wi in weights.items())
        save_preds(NAME, "val", uid, np.expm1(np.clip(lv_car, 0, None)))
        save_preds(NAME, "test", uid_t, np.expm1(np.clip(lt_car, 0, None)))
        top = sorted(weights.items(), key=lambda kv: -kv[1])[:6]
        line_sig = f"{NAME}\t{caruana_val:.6f}"
        scores_p = REPORTS_DIR / "scores.tsv"
        already = scores_p.exists() and any(
            ln.startswith(line_sig) for ln in scores_p.read_text().splitlines())
        if already:
            print(f"[SCORE] {NAME}: {caruana_val:.6f} (already logged, skip)")
        else:
            log_score(NAME, caruana_val,
                      f"bagged Caruana B={B_BAGS} steps={STEPS} lib={m} top={dict((k, round(v, 3)) for k, v in top)}")
    else:
        print(f"\nNOT saved: caruana {caruana_val:.6f} >= manual {manual_val:.6f}")

    out = {
        "n_models": m,
        "caruana_val": round(caruana_val, 6),
        "hillclimb_val": round(hillclimb_val, 6),
        "manual_val": round(manual_val, 6),
        "saved": saved,
        "weights": {k: round(v, 6) for k, v in sorted(weights.items(), key=lambda kv: -kv[1])},
        "notes": (f"guard-dropped {guard_dropped} (<{GUARD_RMSLE}); "
                  f"CAVEAT: gain driven by pre-gap30-era models {sorted(OLD_ERA & set(weights))} "
                  f"whose val is likely optimistic (CLEAN gap30 retrains of same archs ~1.69); "
                  f"clean-only sensitivity: {len(clean_names)} models val {clean_val:.6f} "
                  f"({'beats' if clean_val < manual_val else 'does not beat'} manual {manual_val:.6f}); "
                  f"mean bag rmsle {mean_bag:.4f}; "
                  f"hillclimb weights {dict((k, round(v, 3)) for k, v in hc_weights.items())}; "
                  f"manual raw-space {man_raw:.6f}"),
    }
    (REPORTS_DIR / "caruana.json").write_text(json.dumps(out, indent=1))
    print("\nJSON_RESULT " + json.dumps(out))
    return out


if __name__ == "__main__":
    main()

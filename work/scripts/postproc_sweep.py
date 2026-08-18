"""Systematic sweep of cheap post-processors on validation predictions.

Protocol (honest 2-fold by users, same split convention as calibrate.py):
  split users into halves via rng(0); fit transform on one half, evaluate

Families:
  1. zero_floor : pred < t -> 0, grid t in {0.1..3.0}
  2. affine_log : lp' = a*lp + b (OLS in log1p space), plus a-only / b-only probes
  3. isotonic   : lp -> ly via 50 quantile bins + sklearn IsotonicRegression
  4. power      : lp' = c * lp**gamma (grid gamma, closed-form c)
  5. combo      : top-2 families composed (order chosen on train fold)

Usage: postproc_sweep.py  (operates on my26 and mlp2_big_cal)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.isotonic import IsotonicRegression

sys.path.insert(0, str(Path(__file__).parent))
from common import PREDS_DIR, REPORTS_DIR, VAL_ANCHOR, load_anchor, rmsle
from exp_lib import save_preds, log_score

GAIN_USEFUL = 0.0005
GAIN_APPLY = 0.0008


def lrmse(ly: np.ndarray, lp: np.ndarray) -> float:
    """RMSLE given both sides already in log1p space (lp clipped >= 0)."""
    return float(np.sqrt(np.mean((ly - np.clip(lp, 0, None)) ** 2)))


# ---------------------------------------------------------------- families
# each family: fit(lp, ly) -> params ;  apply(lp, params) -> lp'

T_GRID = np.round(np.arange(0.1, 3.01, 0.1), 2)


def fit_zero_floor(lp, ly):
    best_t, best_s = 0.0, lrmse(ly, lp)
    for t in T_GRID:
        s = lrmse(ly, np.where(lp < np.log1p(t), 0.0, lp))
        if s < best_s:
            best_t, best_s = float(t), s
    return {"t": best_t}


def apply_zero_floor(lp, p):
    if p["t"] <= 0:
        return lp.copy()
    return np.where(lp < np.log1p(p["t"]), 0.0, lp)


def fit_affine(lp, ly):
    A = np.vstack([lp, np.ones_like(lp)]).T
    (a, b), *_ = np.linalg.lstsq(A, ly, rcond=None)
    return {"a": float(a), "b": float(b)}


def apply_affine(lp, p):
    return p["a"] * lp + p["b"]


def fit_affine_a_only(lp, ly):  # b=0: pure log-scale slope (shape)
    return {"a": float(np.dot(lp, ly) / np.dot(lp, lp)), "b": 0.0}


def fit_affine_b_only(lp, ly):  # a=1: pure level shift
    return {"a": 1.0, "b": float(np.mean(ly - lp))}


def fit_isotonic(lp, ly, bins: int = 50):
    qs = np.quantile(lp, np.linspace(0, 1, bins + 1))
    qs[0] -= 1e-9
    qs[-1] += 1e-9
    idx = np.clip(np.searchsorted(qs, lp, side="right") - 1, 0, bins - 1)
    centers, targets, weights = [], [], []
    for i in range(bins):
        m = idx == i
        n = int(m.sum())
        if n == 0:
            continue
        centers.append(float(lp[m].mean()))
        targets.append(float(ly[m].mean()))
        weights.append(n)
    iso = IsotonicRegression(increasing=True, out_of_bounds="clip")
    iso.fit(np.array(centers), np.array(targets), sample_weight=np.array(weights))
    return {"iso": iso, "centers": centers, "targets": targets}


def apply_isotonic(lp, p):
    return p["iso"].predict(lp)


G_GRID = np.round(np.arange(0.50, 1.51, 0.02), 2)


def fit_power(lp, ly):
    best = None
    for g in G_GRID:
        z = lp ** g
        c = float(np.dot(z, ly) / np.dot(z, z))
        s = lrmse(ly, c * z)
        if best is None or s < best[2]:
            best = (float(g), c, s)
    return {"gamma": best[0], "c": best[1]}


def apply_power(lp, p):
    return p["c"] * lp ** p["gamma"]


FAMILIES = {
    "zero_floor": (fit_zero_floor, apply_zero_floor),
    "affine_log": (fit_affine, apply_affine),
    "affine_a_only": (fit_affine_a_only, apply_affine),
    "affine_b_only": (fit_affine_b_only, apply_affine),
    "isotonic50": (fit_isotonic, apply_isotonic),
    "power": (fit_power, apply_power),
}


def fmt_params(p):
    return json.dumps(
        {k: (round(v, 4) if isinstance(v, float) else v)
         for k, v in p.items() if k in ("t", "a", "b", "gamma", "c")}
    ) if any(k in p for k in ("t", "a", "b", "gamma", "c")) else "(50-bin monotone map)"


# ---------------------------------------------------------------- evaluation

def two_fold_eval(name, fit_fn, apply_fn, lp, ly, half):
    rows = []
    for fold, (tr, ho) in enumerate([(half, ~half), (~half, half)]):
        p = fit_fn(lp[tr], ly[tr])
        base = lrmse(ly[ho], lp[ho])
        post = lrmse(ly[ho], apply_fn(lp[ho], p))
        rows.append({"fold": fold, "base": base, "post": post,
                     "gain": base - post, "params": p})
    gain = float(np.mean([r["gain"] for r in rows]))
    return {"name": name, "gain": gain, "folds": rows}


def combo_eval(f1, f2, lp, ly, half):
    """Compose two families; order chosen by in-sample fit on the train fold."""
    rows = []
    orders = []
    for fold, (tr, ho) in enumerate([(half, ~half), (~half, half)]):
        cand = []
        for first, second in [(f1, f2), (f2, f1)]:
            fit_a, app_a = FAMILIES[first]
            fit_b, app_b = FAMILIES[second]
            pa = fit_a(lp[tr], ly[tr])
            mid_tr = np.clip(app_a(lp[tr], pa), 0, None)
            pb = fit_b(mid_tr, ly[tr])
            insample = lrmse(ly[tr], app_b(mid_tr, pb))
            cand.append((insample, first, second, pa, pb, app_a, app_b))
        cand.sort(key=lambda c: c[0])
        _, first, second, pa, pb, app_a, app_b = cand[0]
        orders.append(f"{first}->{second}")
        base = lrmse(ly[ho], lp[ho])
        mid_ho = np.clip(app_a(lp[ho], pa), 0, None)
        post = lrmse(ly[ho], app_b(mid_ho, pb))
        rows.append({"fold": fold, "base": base, "post": post,
                     "gain": base - post, "params": {"order": orders[-1]}})
    gain = float(np.mean([r["gain"] for r in rows]))
    return {"name": f"combo({f1}+{f2})", "gain": gain, "folds": rows,
            "orders": orders}


def sweep_base(base_name: str, uid, y, ly):
    dv = pl.read_parquet(PREDS_DIR / f"{base_name}_val.parquet").sort("user_id")
    assert (dv["user_id"].to_numpy() == uid).all(), f"user mismatch for {base_name}"
    lp = np.log1p(np.clip(dv["pred"].to_numpy().astype(np.float64), 0, None))
    base_full = lrmse(ly, lp)

    rng = np.random.default_rng(0)
    half = rng.permutation(len(uid)) < len(uid) // 2

    results = []
    for name, (fit_fn, apply_fn) in FAMILIES.items():
        results.append(two_fold_eval(name, fit_fn, apply_fn, lp, ly, half))

    # combo of the top-2 *main* families (probes excluded)
    main = [r for r in results if r["name"] not in ("affine_a_only", "affine_b_only")]
    top2 = sorted(main, key=lambda r: -r["gain"])[:2]
    combo = combo_eval(top2[0]["name"], top2[1]["name"], lp, ly, half)
    results.append(combo)

    return {"base": base_name, "base_full_rmsle": base_full, "lp": lp,
            "half": half, "results": results}


# ---------------------------------------------------------------- main

def main():
    val = load_anchor(VAL_ANCHOR, columns=["user_id", "target"]).sort("user_id")
    uid = val["user_id"].to_numpy()
    y = val["target"].to_numpy().astype(np.float64)
    ly = np.log1p(y)

    out = {}
    for base_name in ["my26", "mlp2_big_cal"]:
        out[base_name] = sweep_base(base_name, uid, y, ly)
        print(f"\n=== {base_name} (val rmsle {out[base_name]['base_full_rmsle']:.6f}) ===")
        for r in out[base_name]["results"]:
            f0, f1 = r["folds"]
            verdict = ("APPLY" if r["gain"] >= GAIN_APPLY else
                       "useful" if r["gain"] >= GAIN_USEFUL else "no")
            print(f"{r['name']:26s} gain {r['gain']:+.6f} "
                  f"[fold0 {f0['gain']:+.6f} | fold1 {f1['gain']:+.6f}] {verdict} "
                  f"p0={fmt_params(f0['params'])} p1={fmt_params(f1['params'])}")

    # decide on application for my26
    my = out["my26"]
    main_res = [r for r in my["results"]]
    best = max(main_res, key=lambda r: r["gain"])
    applied = False
    applied_info = {}
    if best["gain"] >= GAIN_APPLY:
        applied = True
        applied_info = apply_best_to_my26(best, my, uid, ly)
    return out, best, applied, applied_info


def transform_from_result(res, lp_fit, ly_fit):
    """Fit the winning family (or combo) on (lp_fit, ly_fit); return apply fn."""
    if res["name"].startswith("combo("):
        # honor the order the folds chose (they should agree; take fold-0 order)
        order = res["orders"][0].split("->")
        fit_a, app_a = FAMILIES[order[0]]
        fit_b, app_b = FAMILIES[order[1]]
        pa = fit_a(lp_fit, ly_fit)
        mid = np.clip(app_a(lp_fit, pa), 0, None)
        pb = fit_b(mid, ly_fit)
        return lambda lp: app_b(np.clip(app_a(lp, pa), 0, None), pb), {"order": order, "pa": pa, "pb": pb}
    fit_fn, apply_fn = FAMILIES[res["name"]]
    p = fit_fn(lp_fit, ly_fit)
    return lambda lp: apply_fn(lp, p), p


def apply_best_to_my26(best, my, uid, ly):
    lp, half = my["lp"], my["half"]
    # val: cross-fitted (fit on A -> apply to B and vice versa)
    lv = np.empty_like(lp)
    for tr, ho in [(half, ~half), (~half, half)]:
        f, _ = transform_from_result(best, lp[tr], ly[tr])
        lv[ho] = np.clip(f(lp[ho]), 0, None)
    val_rmsle = lrmse(ly, lv)

    # test: fit on the full validation
    f_full, params_full = transform_from_result(best, lp, ly)
    dt = pl.read_parquet(PREDS_DIR / "my26_test.parquet").sort("user_id")
    lt = np.log1p(np.clip(dt["pred"].to_numpy().astype(np.float64), 0, None))
    ltp = np.clip(f_full(lt), 0, None)

    save_preds("my26_pp", "val", uid, np.expm1(lv))
    save_preds("my26_pp", "test", dt["user_id"].to_numpy(), np.expm1(ltp))
    log_score("my26_pp", val_rmsle,
              f"postproc {best['name']} on my26; honest 2-fold gain {best['gain']:+.6f}; "
              f"val=cross-fit, test=fit-on-full-val")
    return {"val_rmsle": val_rmsle, "params_full": params_full,
            "test_changed_frac": float(np.mean(np.abs(ltp - lt) > 1e-12))}


if __name__ == "__main__":
    main()

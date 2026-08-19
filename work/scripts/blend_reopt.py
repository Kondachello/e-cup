"""Strict re-optimisation of blend composition and weights in log1p space.

Motivation: the champion mix (mlpziln_cal .536 / mlpbin_cal .283 / gru_final .145 /
c_xtw_s42 .036, val 1.667432) was found by greedy coordinate descent days ago, when
the library was much smaller. This script re-fits weights properly and measures,
honestly, how much a correct optimiser buys over the greedy one.

Everything is done on log1p predictions (the space RMSLE lives in):
    minimise || sum_i w_i * lp_i  -  ly ||_2   s.t. w >= 0
in four flavours: {free scale, sum(w)=1} x {no ridge, ridge}, ridge alpha picked by
5-fold CV over USERS.  All comparisons are pooled out-of-fold RMSLE (each user scored
with weights fitted without them), so weight-overfit is visible as in-sample vs OOF gap.
Stability: 50 user-bootstrap refits of the winning recipe.

Library hygiene (this matters more than the optimiser):
  * only preds on the standard VAL anchor 2026-01-14 with a matching *_test.parquet;
  * hjit37/hjit44 dropped (h=37/44 day windows, not the standard 30d target);
  * smoke / probe / cand / applied / path artefacts dropped;
  * four library tiers, see LIBS below, separating leak-free models from the ones
    whose val preds were post-hoc fitted on val targets (*_cal, *_chcal, *_dampfit)

Usage: python work/scripts/blend_reopt.py [--save] [--boot 50]
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "POLARS_MAX_THREADS"):
    os.environ.setdefault(_v, "3")

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
from scipy.linalg import cholesky, solve_triangular
from scipy.optimize import lsq_linear, minimize, nnls

sys.path.insert(0, str(Path(__file__).parent))
from common import PREDS_DIR, REPORTS_DIR, VAL_ANCHOR, load_anchor, rmsle  # noqa: E402
from exp_lib import log_score, save_preds  # noqa: E402

# ---------------------------------------------------------------- library spec
EXCL_SUBSTR = ("smoke", "probe", "cand", "applied", "hjit", "path")
# derived files (blends / gates over blends): including them is circular
BLEND_LIKE = {"blend_w1a", "blend_w2", "caruana_v1", "my26", "my27",
              "base_best", "whale_final", "stack_meta"}
# префиксы производных файлов: их нельзя класть в библиотеку как «модель», иначе
# оптимизатор выбирает готовый бленд с весом 1.0 и переоптимизации не происходит
BLEND_PREFIX = ("blend", "caruana", "lbmix", "stack")


def is_blend_like(n: str) -> bool:
    base = n[: -len("_cal")] if n.endswith("_cal") else n
    return n in BLEND_LIKE or base in BLEND_LIKE or n.startswith(BLEND_PREFIX)

OLD_ERA = {"lgblog_final", "xgblog_final", "mlp_final", "gru_final"}
# val preds contain a post-hoc transform fitted on val targets (24 bin shifts /
# damp scalar).  Small leak (~0.0005) but it is a leak; CV over users cannot see it.
CAL_SUFFIX = ("_cal", "_chcal", "_dampfit")

CURRENT = {"mlpziln_cal": 0.536, "mlpbin_cal": 0.283, "gru_final": 0.145,
           "c_xtw_s42": 0.036}

N_FOLDS = 5
SEED = 42
ALPHA_REL_GRID = [0.0, 1e-6, 1e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3, 1.0]
NAME = "blend_opt"


def discover() -> list[str]:
    out = []
    for p in sorted(PREDS_DIR.glob("*_val.parquet")):
        n = p.name[: -len("_val.parquet")]
        if not (PREDS_DIR / f"{n}_test.parquet").exists():
            continue
        if any(s in n for s in EXCL_SUBSTR):
            continue
        cols = pl.read_parquet_schema(PREDS_DIR / f"{n}_val.parquet")
        if "pred" not in cols:            # e.g. hmmsim_aux (feature dump)
            continue
        out.append(n)
    return out


# ------------------------------------------------------------------- solvers
def _chol_system(G: np.ndarray, b: np.ndarray, alpha: float):
    """Return (R, z) with R'R = G+alpha*I so that ||Rw - z||^2 == w'(G+aI)w - 2b'w + c."""
    m = G.shape[0]
    jitter = 1e-11 * float(np.trace(G)) / m
    R = cholesky(G + (alpha + jitter) * np.eye(m), lower=False)
    z = solve_triangular(R, b, trans="mdl_larvik", lower=False)
    return R, z


def fit_nnls(G, b, alpha=0.0):
    """w >= 0, free scale, ridge alpha."""
    R, z = _chol_system(G, b, alpha)
    w, _ = nnls(R, z)
    return w


def fit_ols(G, b, alpha=0.0):
    """Unconstrained least squares (negative weights allowed), ridge alpha."""
    m = G.shape[0]
    jitter = 1e-11 * float(np.trace(G)) / m
    return np.linalg.solve(G + (alpha + jitter) * np.eye(m), b)


def fit_bvls(G, b, alpha=0.0, free_idx=()):
    """w >= 0 except positions in free_idx (unbounded).  Ridge skips free_idx."""
    m = G.shape[0]
    d = np.full(m, alpha)
    d[list(free_idx)] = 0.0
    jitter = 1e-9 * float(np.trace(G)) / m
    R = cholesky(G + np.diag(d + jitter), lower=False)
    z = solve_triangular(R, b, trans="mdl_larvik", lower=False)
    lo = np.zeros(m)
    lo[list(free_idx)] = -np.inf
    r = lsq_linear(R, z, bounds=(lo, np.full(m, np.inf)), method="bvls",
                   max_iter=500, tol=1e-12)
    return r.x


def fit_nnls_sum1(G, b, alpha=0.0, w0=None):
    """w >= 0, sum(w) == 1, ridge alpha.  SLSQP on the (small) Gram problem."""
    m = G.shape[0]
    A = G + alpha * np.eye(m)

    def f(w):
        return float(w @ A @ w - 2.0 * b @ w)

    def g(w):
        return 2.0 * (A @ w - b)

    starts = []
    if w0 is not None and w0.sum() > 0:
        starts.append(np.clip(w0, 0, None) / w0.sum())
    starts.append(np.full(m, 1.0 / m))
    best_w, best_f = None, np.inf
    for s in starts:
        r = minimize(f, s, jac=g, method="SLSQP", bounds=[(0.0, None)] * m,
                     constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0,
                                   "jac": lambda w: np.ones_like(w)}],
                     options={"maxiter": 400, "ftol": 1e-14})
        if r.fun < best_f:
            best_f, best_w = float(r.fun), np.clip(r.x, 0, None)
    best_w = best_w / best_w.sum()
    return best_w


def greedy_gram(G: np.ndarray, b: np.ndarray, yy: float, steps: int = 60,
                stop_no_improve: bool = True) -> np.ndarray:
    """blend.py / caruana.py hill-climb with replacement -> weights (sum 1)."""
    m = G.shape[0]
    counts = np.zeros(m, dtype=np.int64)
    g = np.zeros(m)
    bc, cGc = 0.0, 0.0
    diag = np.diag(G).copy()
    best_prev = np.sqrt(max(yy, 0.0))
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
    return counts / max(counts.sum(), 1)


# ---------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", action="store_true")
    ap.add_argument("--boot", type=int, default=50)
    ap.add_argument("--lib", default="B_plus_cal")
    ap.add_argument("--exclude", default="",
                    help="доп. подстроки (через запятую) для исключения из библиотеки — "
                         "нужно для ЧЕСТНОГО парного замера «до/после» добавления семейства "
                         "моделей: обе прогонки делаются в один момент на одном пуле")
    ap.add_argument("--json", default="blend_reopt.json", help="имя файла отчёта в work/reports")
    args = ap.parse_args()
    global EXCL_SUBSTR
    if args.exclude:
        EXCL_SUBSTR = EXCL_SUBSTR + tuple(s for s in args.exclude.split(",") if s)

    t0 = time.time()
    val = load_anchor(VAL_ANCHOR, columns=["user_id", "target"]).sort("user_id")
    uid = val["user_id"].to_numpy()
    y = val["target"].to_numpy().astype(np.float64)
    ly = np.log1p(np.clip(y, 0, None))
    N = len(uid)

    names = discover()
    P, T, solo = {}, {}, {}
    uid_t = None
    for n in names:
        dv = pl.read_parquet(PREDS_DIR / f"{n}_val.parquet").sort("user_id")
        assert np.array_equal(dv["user_id"].to_numpy(), uid), f"val uid mismatch {n}"
        P[n] = np.log1p(np.clip(dv["pred"].to_numpy().astype(np.float64), 0, None))
        dt = pl.read_parquet(PREDS_DIR / f"{n}_test.parquet").sort("user_id")
        if uid_t is None:
            uid_t = dt["user_id"].to_numpy()
        else:
            assert np.array_equal(dt["user_id"].to_numpy(), uid_t), f"test uid mismatch {n}"
        T[n] = np.log1p(np.clip(dt["pred"].to_numpy().astype(np.float64), 0, None))
        solo[n] = rmsle(y, np.expm1(P[n]))

    cal = [n for n in names if n.endswith(CAL_SUFFIX) and not is_blend_like(n)]
    old = [n for n in names if n in OLD_ERA]
    der = [n for n in names if is_blend_like(n)]
    base = [n for n in names if n not in set(cal) | set(old) | set(der)]

    LIBS = {
        "A_raw_clean": sorted(base),
        "B_plus_cal": sorted(base + cal),
        "C_plus_oldera": sorted(base + cal + old),
        "D_literal_all": sorted(base + cal + old + der),
    }
    print(f"[lib] paired+filtered={len(names)}  base={len(base)} cal={len(cal)} "
          f"old_era={len(old)} derived={len(der)}")
    for k, v in LIBS.items():
        print(f"  {k}: {len(v)}")
    print("\nsolo val RMSLE (library A/B/C members):")
    for n in sorted(base + cal + old, key=lambda x: solo[x]):
        tag = "CAL-LEAK" if n in cal else ("OLD-ERA" if n in old else "")
        print(f"  {n:26s} {solo[n]:.6f}  {tag}")

    rng = np.random.default_rng(SEED)
    fold = rng.permutation(N) % N_FOLDS

    # ---------- reference: current champion mix ----------
    lv_cur = sum(CURRENT[n] * P[n] for n in CURRENT)
    lt_cur = sum(CURRENT[n] * T[n] for n in CURRENT)
    cur_val = float(np.sqrt(np.mean((lv_cur - ly) ** 2)))
    print(f"\n[current] fixed mix val RMSLE = {cur_val:.6f}  (sum w = {sum(CURRENT.values()):.3f})")

    results = {}
    per_lib = {}

    def shape(e):
        """RMSLE reachable after the optimal global log-shift = std of log-error.

        The pipeline re-measures the level on the LB anyway (KNOWLEDGE R9), so the
        level-free number is the fair way to rank blends for submission."""
        return float(np.std(e))

    for lib_name, lib in LIBS.items():
        m = len(lib)
        # last column = intercept (global log-space level); methods that do not want
        # it simply operate on the leading m x m block.
        Xa = np.empty((N, m + 1), dtype=np.float64)
        for j, n in enumerate(lib):
            Xa[:, j] = P[n]
        Xa[:, m] = 1.0
        X = Xa[:, :m]

        Gf, bf, yyf, nf = [], [], [], []
        for f in range(N_FOLDS):
            idx = fold == f
            Xf = Xa[idx]
            lyf = ly[idx]
            Gf.append(Xf.T @ Xf)
            bf.append(Xf.T @ lyf)
            yyf.append(float(lyf @ lyf))
            nf.append(int(idx.sum()))
        Gt, bt, yyt = sum(Gf), sum(bf), sum(yyf)
        Ga_full, ba_full, yy_full = Gt / N, bt / N, yyt / N
        G_full, b_full = Ga_full[:m, :m], ba_full[:m]

        def oof(fitter, icpt=False):
            """Pooled out-of-fold predictions for fitter(G,b,yy)->w."""
            pred = np.empty(N)
            ws = []
            for f in range(N_FOLDS):
                ntr = N - nf[f]
                Ga = (Gt - Gf[f]) / ntr
                ba = (bt - bf[f]) / ntr
                yytr = (yyt - yyf[f]) / ntr
                w = fitter(Ga, ba, yytr) if icpt else \
                    fitter(Ga[:m, :m], ba[:m], yytr)
                ws.append(w)
                idx = fold == f
                pred[idx] = (Xa[idx] if icpt else X[idx]) @ w
            return pred, np.array(ws)

        def insample(fitter, icpt=False):
            w = fitter(Ga_full, ba_full, yy_full) if icpt else \
                fitter(G_full, b_full, yy_full)
            p = (Xa if icpt else X) @ w
            return w, float(np.sqrt(np.mean((p - ly) ** 2)))

        # ridge alpha by CV (free-scale NNLS and sum-1 NNLS)
        scale = float(np.trace(G_full)) / m
        ridge_cv = {"free": {}, "sum1": {}}
        for a_rel in ALPHA_REL_GRID:
            a = a_rel * scale
            pf, _ = oof(lambda G, b, yy, a=a: fit_nnls(G, b, a))
            ridge_cv["free"][a_rel] = float(np.sqrt(np.mean((pf - ly) ** 2)))
            ps, _ = oof(lambda G, b, yy, a=a: fit_nnls_sum1(G, b, a, fit_nnls(G, b, a)))
            ridge_cv["sum1"][a_rel] = float(np.sqrt(np.mean((ps - ly) ** 2)))
        best_a_free = min(ridge_cv["free"], key=ridge_cv["free"].get)
        best_a_sum1 = min(ridge_cv["sum1"], key=ridge_cv["sum1"].get)

        methods = {
            "nnls_free":   (lambda G, b, yy: fit_nnls(G, b, 0.0), False),
            "nnls_sum1":   (lambda G, b, yy: fit_nnls_sum1(G, b, 0.0,
                                                           fit_nnls(G, b, 0.0)), False),
            "greedy":      (lambda G, b, yy: greedy_gram(G, b, yy, 60, True), False),
            "ridge_free":  (lambda G, b, yy, a=best_a_free * scale: fit_nnls(G, b, a), False),
            "ridge_sum1":  (lambda G, b, yy, a=best_a_sum1 * scale:
                            fit_nnls_sum1(G, b, a, fit_nnls(G, b, a)), False),
            # affine variants: nonneg model weights + free global log-level
            "nnls_icpt":   (lambda G, b, yy: fit_bvls(G, b, 0.0, (G.shape[0] - 1,)), True),
            "ridge_icpt":  (lambda G, b, yy, a=best_a_free * scale:
                            fit_bvls(G, b, a, (G.shape[0] - 1,)), True),
            # sign-free (negative weights allowed), as in the LB Gram mixes (R12)
            "ols_free":    (lambda G, b, yy: fit_ols(G, b, 0.0), False),
            "ols_icpt":    (lambda G, b, yy: fit_ols(G, b, 0.0), True),
        }

        tab = {}
        for mn, (fn, icpt) in methods.items():
            p_oof, ws = oof(fn, icpt)
            e = p_oof - ly
            s_oof = float(np.sqrt(np.mean(e ** 2)))
            w_in, s_in = insample(fn, icpt)
            k = m + 1 if icpt else m
            wd = {(lib + ["_intercept"])[i]: float(w_in[i])
                  for i in range(k) if abs(w_in[i]) > 1e-6}
            tab[mn] = dict(oof=s_oof, insample=s_in, gap=s_oof - s_in,
                           oof_shape=shape(e), oof_bias=float(e.mean()),
                           sumw=float(w_in[:m].sum()), nnz=len(wd),
                           weights=dict(sorted(wd.items(), key=lambda kv: -kv[1])),
                           fold_w_std=float(np.mean(ws.std(axis=0))))
            print(f"[{lib_name}] {mn:11s} oof={s_oof:.6f} in={s_in:.6f} "
                  f"gap={s_oof - s_in:+.6f} shape={shape(e):.6f} "
                  f"sumw={w_in[:m].sum():.3f} nnz={len(wd)}")
        tab["_ridge_cv"] = {"scale": scale, "best_alpha_rel_free": best_a_free,
                            "best_alpha_rel_sum1": best_a_sum1,
                            "curve_free": ridge_cv["free"], "curve_sum1": ridge_cv["sum1"]}
        per_lib[lib_name] = tab
        results[lib_name] = {k: v for k, v in tab.items() if k != "_ridge_cv"}
        del Xa, X

    # ---------- honest OOF for the CURRENT recipe (refit its 4 models per fold) ----
    cur_lib = sorted(CURRENT)
    Xc = np.column_stack([P[n] for n in cur_lib])
    Gfc, bfc, yyfc, nfc = [], [], [], []
    for f in range(N_FOLDS):
        idx = fold == f
        Xf, lyf = Xc[idx], ly[idx]
        Gfc.append(Xf.T @ Xf); bfc.append(Xf.T @ lyf)
        yyfc.append(float(lyf @ lyf)); nfc.append(int(idx.sum()))
    Gtc, btc, yytc = sum(Gfc), sum(bfc), sum(yyfc)
    pred_cur_refit = np.empty(N)
    for f in range(N_FOLDS):
        ntr = N - nfc[f]
        w = fit_nnls_sum1((Gtc - Gfc[f]) / ntr, (btc - bfc[f]) / ntr, 0.0,
                          fit_nnls((Gtc - Gfc[f]) / ntr, (btc - bfc[f]) / ntr, 0.0))
        idx = fold == f
        pred_cur_refit[idx] = Xc[idx] @ w
    cur_refit_oof = float(np.sqrt(np.mean((pred_cur_refit - ly) ** 2)))
    w_cur_refit = fit_nnls_sum1(Gtc / N, btc / N, 0.0, fit_nnls(Gtc / N, btc / N, 0.0))
    cur_refit_in = float(np.sqrt(np.mean((Xc @ w_cur_refit - ly) ** 2)))
    print(f"\n[current-4 refit sum1] in={cur_refit_in:.6f} oof={cur_refit_oof:.6f} "
          f"w={dict(zip(cur_lib, np.round(w_cur_refit, 4)))}")

    # ---------- pick winner (library B = task scope, leak-controlled by A) --------
    def best_of(lib_name):
        t = results[lib_name]
        return min(((k, v["oof"]) for k, v in t.items()), key=lambda kv: kv[1])

    print("\n[summary] best OOF per library:", {k: best_of(k) for k in LIBS})

    # shippable = non-negative weights, no val-fitted intercept (the level does NOT
    # transfer val->test, KNOWLEDGE F17/R9: it is re-measured on the LB instead).
    SHIPPABLE = ("nnls_free", "nnls_sum1", "ridge_free", "ridge_sum1", "greedy")
    WIN_LIB = args.lib
    win_method = min(((k, results[WIN_LIB][k]["oof"]) for k in SHIPPABLE),
                     key=lambda kv: kv[1])[0]
    lib = LIBS[WIN_LIB]
    m = len(lib)
    X = np.column_stack([P[n] for n in lib])
    G_full = X.T @ X / N
    b_full = X.T @ ly / N
    scale = float(np.trace(G_full)) / m
    a_rel = per_lib[WIN_LIB]["_ridge_cv"]["best_alpha_rel_free" if "free" in win_method
                                          else "best_alpha_rel_sum1"]
    a = a_rel * scale if win_method.startswith("ridge") else 0.0
    if win_method in ("nnls_free", "ridge_free"):
        fitter = lambda G, b, yy: fit_nnls(G, b, a)          # noqa: E731
    elif win_method in ("nnls_sum1", "ridge_sum1"):
        fitter = lambda G, b, yy: fit_nnls_sum1(G, b, a, fit_nnls(G, b, a))  # noqa: E731
    else:
        fitter = lambda G, b, yy: greedy_gram(G, b, yy, 60, True)  # noqa: E731
    w_win = fitter(G_full, b_full, float(ly @ ly) / N)
    print(f"\n[winner] lib={WIN_LIB} method={win_method} alpha_rel={a_rel} "
          f"oof={results[WIN_LIB][win_method]['oof']:.6f}")

    # ---------- bootstrap stability (user resampling) -----------------------------
    boot_w = np.zeros((args.boot, m))
    brng = np.random.default_rng(SEED + 1)
    for it in range(args.boot):
        c = brng.multinomial(N, np.full(N, 1.0 / N)).astype(np.float64)
        Gb = (X * c[:, None]).T @ X / N
        bb = X.T @ (c * ly) / N
        boot_w[it] = fitter(Gb, bb, float((c * ly * ly).sum()) / N)
        if (it + 1) % 10 == 0:
            print(f"  boot {it + 1}/{args.boot}  ({time.time() - t0:.0f}s)")
    sel_freq = (boot_w > 5e-3).mean(axis=0)
    w_mean, w_std = boot_w.mean(axis=0), boot_w.std(axis=0)
    stab = sorted(
        [{"model": lib[i], "w_full": float(w_win[i]), "w_mean": float(w_mean[i]),
          "w_std": float(w_std[i]), "sel_freq": float(sel_freq[i])} for i in range(m)],
        key=lambda d: -d["w_mean"])
    print("\n[bootstrap] model / w_full / w_mean+-std / sel_freq")
    for d in stab:
        if d["w_mean"] > 1e-4 or d["sel_freq"] > 0.05:
            print(f"  {d['model']:26s} {d['w_full']:.4f}  {d['w_mean']:.4f}"
                  f"+-{d['w_std']:.4f}  {d['sel_freq']:.2f}")
    stable_models = [d["model"] for d in stab if d["sel_freq"] >= 0.9]

    # ---------- apply to test, level diagnostics ---------------------------------
    Xt = np.column_stack([T[n] for n in lib])
    lt_new = Xt @ w_win
    lv_new = X @ w_win
    new_val = float(np.sqrt(np.mean((lv_new - ly) ** 2)))
    lev = {
        "new_val_meanlog": float(lv_new.mean()), "new_test_meanlog": float(lt_new.mean()),
        "cur_val_meanlog": float(lv_cur.mean()), "cur_test_meanlog": float(lt_cur.mean()),
        "delta_test_meanlog_new_minus_cur": float(lt_new.mean() - lt_cur.mean()),
        "corr_test_new_cur": float(np.corrcoef(lt_new, lt_cur)[0, 1]),
        "corr_val_err": float(np.corrcoef(lv_new - ly, lv_cur - ly)[0, 1]),
        "cur_val_shape": shape(lv_cur - ly), "new_val_shape": shape(lv_new - ly),
        "cur_val_bias": float((lv_cur - ly).mean()),
        "new_val_bias": float((lv_new - ly).mean()),
        "test_sd_new": float(lt_new.std()), "test_sd_cur": float(lt_cur.std()),
    }
    print("\n[levels]", json.dumps({k: round(v, 5) for k, v in lev.items()}))

    if args.save:
        save_preds(NAME, "val", uid, np.expm1(np.clip(lv_new, 0, None)))
        save_preds(NAME, "test", uid_t, np.expm1(np.clip(lt_new, 0, None)))
        top = {k: round(v, 4) for k, v in
               sorted({lib[i]: float(w_win[i]) for i in range(m) if w_win[i] > 1e-3}.items(),
                      key=lambda kv: -kv[1])}
        log_score(NAME, new_val,
                  f"re-optimised blend: lib={WIN_LIB}({m}) method={win_method} "
                  f"alpha_rel={a_rel}; OOF(5f by user)={results[WIN_LIB][win_method]['oof']:.6f}; "
                  f"current mix val {cur_val:.6f}; w={top}")

    out = {
        "n_paired_filtered": len(names),
        "libs": {k: v for k, v in LIBS.items()},
        "solo": {n: round(solo[n], 6) for n in sorted(solo, key=lambda x: solo[x])},
        "current_val": round(cur_val, 6),
        "current_refit_in": round(cur_refit_in, 6),
        "current_refit_oof": round(cur_refit_oof, 6),
        "current_refit_w": {k: round(float(v), 4) for k, v in zip(cur_lib, w_cur_refit)},
        "results": results,
        "ridge_cv": {k: per_lib[k]["_ridge_cv"] for k in LIBS},
        "winner": {"lib": WIN_LIB, "method": win_method, "alpha_rel": a_rel,
                   "val": round(new_val, 6),
                   "oof": round(results[WIN_LIB][win_method]["oof"], 6),
                   "weights": {lib[i]: round(float(w_win[i]), 6)
                               for i in np.argsort(-w_win) if w_win[i] > 1e-6}},
        "bootstrap": stab,
        "stable_models": stable_models,
        "levels": lev,
        "runtime_s": round(time.time() - t0, 1),
    }
    (REPORTS_DIR / args.json).write_text(json.dumps(out, indent=1))
    print(f"\nJSON written to work/reports/{args.json} "
          f"({time.time() - t0:.0f}s)")
    return out


if __name__ == "__main__":
    main()

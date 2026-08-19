"""stack_meta3.py — разложение выигрыша ridge_pairwise на компоненты.

ridge_pairwise (единственный вариант, обошедший линейный NNLS) отличается от базы
СРАЗУ ТРЕМЯ вещами: свободный знак весов, свободный уровень (intercept) и попарные
произведения топ-5. Нужно понять, что из этого работает: нелинейность или банальная
аффинная перекалибровка (её пайплайн и так снимает LB-замером, KNOWLEDGE F17/R9 —
уровень val->test НЕ переносится).

Лестница моделей, все на тех же 5 фолдах по юзерам:
  1 NNLS  (w>=0, без свободного уровня)        — база
  2 NNLS + intercept
  3 OLS   (свободный знак, без intercept)
  4 OLS   + intercept
  5 4 + попарные произведения топ-5            (= ridge_pairwise)
  6 4 + квадраты всех 22                       (чистая «кривизна» без взаимодействий)
  7 4 + произведения топ-5 x юзер-признаки     (веса, зависящие от юзера, явно)

Метрики: обычный OOF RMSLE и УРОВНЕ-ИНВАРИАНТНЫЙ (std ошибки) — второй показывает,
что осталось бы после LB-перекалибровки уровня.

Запуск: .venv/bin/python work/scripts/stack_meta3.py
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "POLARS_MAX_THREADS"):
    os.environ.setdefault(_v, "2")

import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from common import FEATURES_DIR, REPORTS_DIR, VAL_ANCHOR  # noqa: E402
from stack_meta import (SEED, USER_FEATS, build_pool, fit_nnls, load_lp,  # noqa: E402
                        paired, rmse)

N_FOLDS = 5
ALPHAS = [0.0, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]


def main() -> int:
    t0 = time.time()
    cols = ["user_id", "target"] + USER_FEATS
    fv = pl.read_parquet(FEATURES_DIR / f"anchor={VAL_ANCHOR.isoformat()}.parquet",
                         columns=cols).sort("user_id")
    uid = fv["user_id"].to_numpy()
    ly = np.log1p(np.clip(fv["target"].to_numpy().astype(np.float64), 0, None))
    N = len(uid)
    pool = build_pool()
    m = len(pool)
    X = np.column_stack([load_lp(n, "val", uid) for n in pool]).astype(np.float64)
    fold = np.random.default_rng(SEED).permutation(N) % N_FOLDS
    nf = np.array([int((fold == f).sum()) for f in range(N_FOLDS)])

    # --------- юзер-признаки: стандартизованы, NaN -> 0 + флаг
    U = fv.select(USER_FEATS).to_numpy().astype(np.float64)
    nan_flag = np.isnan(U).any(axis=1).astype(np.float64)
    U = np.where(np.isnan(U), 0.0, U)
    U = (U - U.mean(0)) / np.where(U.std(0) < 1e-12, 1.0, U.std(0))

    # --------- база NNLS
    Gf = np.zeros((N_FOLDS, m, m)); bf = np.zeros((N_FOLDS, m))
    for f in range(N_FOLDS):
        idx = fold == f
        Gf[f] = X[idx].T @ X[idx]; bf[f] = X[idx].T @ ly[idx]
    lin = np.empty(N)
    for f in range(N_FOLDS):
        tr = [g for g in range(N_FOLDS) if g != f]
        ntr = int(nf[tr].sum())
        w = fit_nnls(Gf[tr].sum(0) / ntr, bf[tr].sum(0) / ntr, 0.0)
        lin[fold == f] = X[fold == f] @ w
    e_lin = lin - ly
    lin_oof, lin_shape = rmse(e_lin), float(np.std(e_lin))
    w_full = fit_nnls(Gf.sum(0) / N, bf.sum(0) / N, 0.0)
    top5 = [int(i) for i in np.argsort(-w_full)[:5]]
    print(f"[NNLS база] OOF={lin_oof:.6f}  уровне-инвариантно={lin_shape:.6f}")
    print(f"[топ-5] {[pool[i] for i in top5]}", flush=True)

    def ridge_ladder(tag: str, Z: np.ndarray, nnls_mode: bool = False):
        """Честная OOF: alpha выбирается вложенной CV внутри трейна."""
        mz = Z.shape[1]
        sd = Z.std(0); sd[sd < 1e-12] = 1.0
        Zs = Z / sd
        Gz = np.zeros((N_FOLDS, mz, mz)); bz = np.zeros((N_FOLDS, mz))
        yyz = np.zeros(N_FOLDS)
        for f in range(N_FOLDS):
            idx = fold == f
            Gz[f] = Zs[idx].T @ Zs[idx]; bz[f] = Zs[idx].T @ ly[idx]
            yyz[f] = float(ly[idx] @ ly[idx])
        pred = np.empty(N)
        picks = []
        for f in range(N_FOLDS):
            tr = [g for g in range(N_FOLDS) if g != f]
            best, best_a = np.inf, None
            for al in ALPHAS:
                tot, cnt = 0.0, 0
                for h in tr:
                    tr2 = [g for g in tr if g != h]
                    n2 = int(nf[tr2].sum())
                    Ga = Gz[tr2].sum(0) / n2; ba = bz[tr2].sum(0) / n2
                    s = float(np.trace(Ga)) / mz
                    wz = (fit_nnls(Ga, ba, al * s) if nnls_mode else
                          np.linalg.solve(Ga + (al + 1e-12) * s * np.eye(mz), ba))
                    tot += float(wz @ Gz[h] @ wz - 2 * bz[h] @ wz + yyz[h])
                    cnt += int(nf[h])
                v = np.sqrt(tot / cnt)
                if v < best:
                    best, best_a = v, al
            ntr = int(nf[tr].sum())
            Ga = Gz[tr].sum(0) / ntr; ba = bz[tr].sum(0) / ntr
            s = float(np.trace(Ga)) / mz
            wz = (fit_nnls(Ga, ba, best_a * s) if nnls_mode else
                  np.linalg.solve(Ga + (best_a + 1e-12) * s * np.eye(mz), ba))
            pred[fold == f] = Zs[fold == f] @ wz
            picks.append(best_a)
        e = pred - ly
        r = dict(oof=rmse(e), gain=lin_oof - rmse(e), shape=float(np.std(e)),
                 shape_gain=lin_shape - float(np.std(e)), bias=float(e.mean()),
                 n_terms=mz, alphas=picks, paired_vs_linear=paired(e, e_lin))
        print(f"[{tag:26s}] OOF={r['oof']:.6f} gain={r['gain']:+.6f} | "
              f"уровне-инв={r['shape']:.6f} gain={r['shape_gain']:+.6f} | "
              f"k={mz} a={picks[0]:g} ({time.time() - t0:.0f}s)", flush=True)
        return r

    one = np.ones((N, 1))
    prod5 = np.column_stack([X[:, i] * X[:, j]
                             for ii, i in enumerate(top5) for j in top5[ii:]])
    sq22 = X ** 2
    inter_u = np.column_stack([X[:, i] * U[:, k] for i in top5
                               for k in range(U.shape[1])])

    out = {"linear_nnls": dict(oof=lin_oof, shape=lin_shape, gain=0.0, shape_gain=0.0),
           "top5": [pool[i] for i in top5]}
    out["nnls_icpt"] = ridge_ladder("2 NNLS + intercept",
                                    np.column_stack([X, one]), nnls_mode=True)
    out["ols"] = ridge_ladder("3 OLS (свободный знак)", X)
    out["ols_icpt"] = ridge_ladder("4 OLS + intercept", np.column_stack([X, one]))
    out["ols_icpt_prod5"] = ridge_ladder("5 + произведения топ-5",
                                         np.column_stack([X, prod5, one]))
    out["ols_icpt_sq22"] = ridge_ladder("6 + квадраты 22",
                                        np.column_stack([X, sq22, one]))
    out["ols_icpt_userint"] = ridge_ladder("7 + топ5 x юзер (12)",
                                           np.column_stack([X, inter_u, one]))
    out["ols_icpt_all"] = ridge_ladder("8 + всё вместе",
                                       np.column_stack([X, prod5, sq22, inter_u,
                                                        nan_flag[:, None], one]))
    print("\n=== разложение выигрыша ===")
    base = out["ols_icpt"]["gain"]
    for k in ("nnls_icpt", "ols", "ols_icpt", "ols_icpt_prod5", "ols_icpt_sq22",
              "ols_icpt_userint", "ols_icpt_all"):
        print(f"  {k:20s} gain={out[k]['gain']:+.6f}  сверх аффинной части="
              f"{out[k]['gain'] - base:+.6f}  уровне-инв={out[k]['shape_gain']:+.6f}")
    (REPORTS_DIR / "stack_meta3.json").write_text(json.dumps(out, indent=1,
                                                            ensure_ascii=False))
    print(f"JSON -> work/reports/stack_meta3.json ({time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

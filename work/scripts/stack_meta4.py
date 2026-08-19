"""stack_meta4.py — контроли для модели с ЮЗЕР-ЗАВИСИМЫМИ ВЕСАМИ.

stack_meta3 показал: попарные произведения моделей не дают ничего (+1e-6), а
взаимодействия «топ-5 моделей x признаки юзера» дают +0.00041 сверх аффинной части
(итог +0.00080 к NNLS-базе). Это ровно проверяемая гипотеза «веса зависят от юзера»,
но в линейной (varying-coefficient) форме, а не древесной.

Прежде чем что-то отправлять, три контроля:
  1. ПЛАЦЕБО: те же 60 членов взаимодействия, но признаки юзера ПЕРЕМЕШАНЫ между
     юзерами (связь юзер<->признак разорвана). Честный протокол обязан дать ~0.
  2. УСТОЙЧИВОСТЬ: 5 разных разбиений на фолды.
  3. ИНТЕРПРЕТАЦИЯ: насколько сильно «плывут» подразумеваемые веса по популяции
     (std эффективного веса модели по юзерам) и какие признаки за это отвечают
     (drop-one по группам признаков).

Запуск: .venv/bin/python work/scripts/stack_meta4.py
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
ALPHAS = [0.0, 1e-6, 1e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 1e-1, 1.0]
SEEDS = [42, 7, 1337, 2024, 555]


def prep_user(fv: pl.DataFrame) -> np.ndarray:
    U = fv.select(USER_FEATS).to_numpy().astype(np.float64)
    U = np.where(np.isnan(U), 0.0, U)
    sd = U.std(0); sd[sd < 1e-12] = 1.0
    return (U - U.mean(0)) / sd


def oof_ridge(Z: np.ndarray, ly: np.ndarray, fold: np.ndarray, nf: np.ndarray,
              n_folds: int) -> tuple[np.ndarray, list]:
    """Честная OOF-предсказание ridge; alpha выбирается вложенной CV внутри трейна."""
    N, mz = Z.shape
    sd = Z.std(0); sd[sd < 1e-12] = 1.0
    Zs = Z / sd
    Gz = np.zeros((n_folds, mz, mz)); bz = np.zeros((n_folds, mz)); yyz = np.zeros(n_folds)
    for f in range(n_folds):
        idx = fold == f
        Gz[f] = Zs[idx].T @ Zs[idx]; bz[f] = Zs[idx].T @ ly[idx]
        yyz[f] = float(ly[idx] @ ly[idx])
    pred = np.empty(N); picks = []
    for f in range(n_folds):
        tr = [g for g in range(n_folds) if g != f]
        best, best_a = np.inf, None
        for al in ALPHAS:
            tot, cnt = 0.0, 0
            for h in tr:
                tr2 = [g for g in tr if g != h]
                n2 = int(nf[tr2].sum())
                Ga = Gz[tr2].sum(0) / n2; ba = bz[tr2].sum(0) / n2
                s = float(np.trace(Ga)) / mz
                wz = np.linalg.solve(Ga + (al + 1e-12) * s * np.eye(mz), ba)
                tot += float(wz @ Gz[h] @ wz - 2 * bz[h] @ wz + yyz[h]); cnt += int(nf[h])
            v = np.sqrt(tot / cnt)
            if v < best:
                best, best_a = v, al
        ntr = int(nf[tr].sum())
        Ga = Gz[tr].sum(0) / ntr; ba = bz[tr].sum(0) / ntr
        s = float(np.trace(Ga)) / mz
        wz = np.linalg.solve(Ga + (best_a + 1e-12) * s * np.eye(mz), ba)
        pred[fold == f] = Zs[fold == f] @ wz
        picks.append(best_a)
    return pred, picks


def main() -> int:
    t0 = time.time()
    fv = pl.read_parquet(FEATURES_DIR / f"anchor={VAL_ANCHOR.isoformat()}.parquet",
                         columns=["user_id", "target"] + USER_FEATS).sort("user_id")
    uid = fv["user_id"].to_numpy()
    ly = np.log1p(np.clip(fv["target"].to_numpy().astype(np.float64), 0, None))
    N = len(uid)
    pool = build_pool(); m = len(pool)
    X = np.column_stack([load_lp(n, "val", uid) for n in pool]).astype(np.float64)
    U = prep_user(fv)
    one = np.ones((N, 1))

    out = {"pool": pool, "user_feats": USER_FEATS, "seeds": {}}

    for sd_i, s in enumerate(SEEDS):
        fold = np.random.default_rng(s).permutation(N) % N_FOLDS
        nf = np.array([int((fold == f).sum()) for f in range(N_FOLDS)])
        Gf = np.zeros((N_FOLDS, m, m)); bf = np.zeros((N_FOLDS, m))
        for f in range(N_FOLDS):
            idx = fold == f
            Gf[f] = X[idx].T @ X[idx]; bf[f] = X[idx].T @ ly[idx]
        lin = np.empty(N)
        for f in range(N_FOLDS):
            tr = [g for g in range(N_FOLDS) if g != f]
            ntr = int(nf[tr].sum())
            lin[fold == f] = X[fold == f] @ fit_nnls(Gf[tr].sum(0) / ntr,
                                                     bf[tr].sum(0) / ntr, 0.0)
        e_lin = lin - ly
        lin_oof = rmse(e_lin)
        w_full = fit_nnls(Gf.sum(0) / N, bf.sum(0) / N, 0.0)
        top5 = [int(i) for i in np.argsort(-w_full)[:5]]

        aff = np.column_stack([X, one])
        p_aff, _ = oof_ridge(aff, ly, fold, nf, N_FOLDS)
        inter = np.column_stack([X[:, i] * U[:, k] for i in top5
                                 for k in range(U.shape[1])])
        Zi = np.column_stack([X, inter, one])
        p_int, picks = oof_ridge(Zi, ly, fold, nf, N_FOLDS)

        rec = dict(linear_nnls=lin_oof, affine=rmse(p_aff - ly),
                   interact=rmse(p_int - ly),
                   gain_affine=lin_oof - rmse(p_aff - ly),
                   gain_interact=lin_oof - rmse(p_int - ly),
                   gain_over_affine=rmse(p_aff - ly) - rmse(p_int - ly),
                   alphas=picks,
                   paired_vs_linear=paired(p_int - ly, e_lin),
                   paired_vs_affine=paired(p_int - ly, p_aff - ly))
        # плацебо только на первом сиде (дорого не нужно, эффект одинаков)
        if sd_i == 0:
            prng = np.random.default_rng(999)
            plac = {}
            for rep in range(3):
                Up = U[prng.permutation(N)]
                Zp = np.column_stack([X] + [X[:, i] * Up[:, k] for i in top5
                                            for k in range(Up.shape[1])] + [one])
                pp, _ = oof_ridge(Zp, ly, fold, nf, N_FOLDS)
                plac[f"rep{rep}"] = dict(oof=rmse(pp - ly),
                                         gain_over_affine=rmse(p_aff - ly) - rmse(pp - ly))
                print(f"  [плацебо {rep}] OOF={rmse(pp - ly):.6f} "
                      f"сверх аффинной={rmse(p_aff - ly) - rmse(pp - ly):+.6f}", flush=True)
            rec["placebo"] = plac
            # drop-one по группам признаков юзера
            groups = {"recency": ["rec_order", "rec_active"],
                      "frequency": ["ord_days_90", "ord_days_365", "ord_rate_90"],
                      "money": ["log_gmv_sum_30", "log_gmv_sum_90", "log_gmv_sum_365",
                                "gmv_per_ordday_365"],
                      "activity": ["act_density", "searches_30", "tenure"]}
            drop = {}
            for gname, feats in groups.items():
                keep = [k for k, f_ in enumerate(USER_FEATS) if f_ not in feats]
                Zd = np.column_stack([X] + [X[:, i] * U[:, k] for i in top5
                                            for k in keep] + [one])
                pd_, _ = oof_ridge(Zd, ly, fold, nf, N_FOLDS)
                drop[gname] = dict(oof=rmse(pd_ - ly),
                                   loss=rmse(pd_ - ly) - rmse(p_int - ly))
                print(f"  [drop {gname:10s}] OOF={rmse(pd_ - ly):.6f} "
                      f"потеря={rmse(pd_ - ly) - rmse(p_int - ly):+.6f}", flush=True)
            rec["drop_one_group"] = drop
            # разброс подразумеваемых весов по популяции (на полной подгонке)
            sdz = Zi.std(0); sdz[sdz < 1e-12] = 1.0
            Zn = Zi / sdz
            Ga = Zn.T @ Zn / N; ba = Zn.T @ ly / N
            sc = float(np.trace(Ga)) / Zi.shape[1]
            wz = np.linalg.solve(Ga + 1e-4 * sc * np.eye(Zi.shape[1]), ba) / sdz
            spread = {}
            for jj, i in enumerate(top5):
                w_i = np.full(N, wz[i])
                for k in range(U.shape[1]):
                    w_i = w_i + wz[m + jj * U.shape[1] + k] * U[:, k]
                spread[pool[i]] = dict(mean=float(w_i.mean()), std=float(w_i.std()),
                                       p05=float(np.quantile(w_i, 0.05)),
                                       p95=float(np.quantile(w_i, 0.95)),
                                       nnls=float(w_full[i]))
            rec["implied_weight_spread"] = spread
            rec["top5"] = [pool[i] for i in top5]
        out["seeds"][str(s)] = rec
        print(f"[seed {s}] NNLS={lin_oof:.6f} affine={rec['affine']:.6f} "
              f"interact={rec['interact']:.6f} gain={rec['gain_interact']:+.6f} "
              f"(сверх аффинной {rec['gain_over_affine']:+.6f}) "
              f"({time.time() - t0:.0f}s)", flush=True)

    g = [out["seeds"][str(s)]["gain_interact"] for s in SEEDS]
    ga = [out["seeds"][str(s)]["gain_over_affine"] for s in SEEDS]
    out["summary"] = dict(gain_interact_mean=float(np.mean(g)), gain_interact_std=float(np.std(g)),
                          gain_over_affine_mean=float(np.mean(ga)),
                          gain_over_affine_std=float(np.std(ga)))
    print(f"\n[устойчивость] выигрыш к NNLS {np.mean(g):+.6f} +- {np.std(g):.6f}; "
          f"сверх аффинной {np.mean(ga):+.6f} +- {np.std(ga):.6f}")
    print("\n[разброс подразумеваемых весов]")
    for k, v in out["seeds"][str(SEEDS[0])]["implied_weight_spread"].items():
        print(f"  {k:18s} nnls={v['nnls']:.3f} -> {v['mean']:+.3f} +- {v['std']:.3f} "
              f"[{v['p05']:+.3f} .. {v['p95']:+.3f}]")
    (REPORTS_DIR / "stack_meta4.json").write_text(json.dumps(out, indent=1,
                                                            ensure_ascii=False))
    print(f"JSON -> work/reports/stack_meta4.json ({time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

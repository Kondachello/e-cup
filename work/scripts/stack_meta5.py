"""stack_meta5.py — комбинация двух работающих мета-механизмов.

Отдельно работают два варианта (честный OOF, 5 фолдов по юзерам, база NNLS 1.666792):
  * varying-coefficient ridge (веса линейно зависят от юзера): 1.665982 (+0.00081)
  * остаточный бустинг в РЕГУЛЯРИЗОВАННОМ режиме (15 листьев, 200 деревьев,
    lr 0.03, min_data 1000): 1.666189 (+0.00060)

Вопрос: это один и тот же сигнал или разные? Считаем честный OOF бустинга поверх
varying-coefficient модели. Если сумма ~= сумме выигрышей — сигналы независимы,
если ~= максимуму — один и тот же.

Запуск: .venv/bin/python work/scripts/stack_meta5.py
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
from stack_meta4 import ALPHAS  # noqa: E402

N_FOLDS = 5
CFG = dict(num_leaves=15, n=200, lr=0.03, mdl=1000)


def main() -> int:
    import lightgbm as lgb
    t0 = time.time()
    fv = pl.read_parquet(FEATURES_DIR / f"anchor={VAL_ANCHOR.isoformat()}.parquet",
                         columns=["user_id", "target"] + USER_FEATS).sort("user_id")
    uid = fv["user_id"].to_numpy()
    ly = np.log1p(np.clip(fv["target"].to_numpy().astype(np.float64), 0, None))
    N = len(uid)
    pool = build_pool(); m = len(pool)
    X = np.column_stack([load_lp(n, "val", uid) for n in pool]).astype(np.float64)
    Uraw = np.nan_to_num(fv.select(USER_FEATS).to_numpy().astype(np.float64), nan=0.0)
    sd = Uraw.std(0); sd[sd < 1e-12] = 1.0
    U = (Uraw - Uraw.mean(0)) / sd
    F = np.column_stack([X.astype(np.float32), U.astype(np.float32)])
    fn = pool + USER_FEATS

    fold = np.random.default_rng(SEED).permutation(N) % N_FOLDS
    nf = np.array([int((fold == f).sum()) for f in range(N_FOLDS)])

    params = dict(objective="regression", metric="l2", num_leaves=CFG["num_leaves"],
                  learning_rate=CFG["lr"], min_data_in_leaf=CFG["mdl"],
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
                  lambda_l2=1.0, num_threads=2, verbosity=-1, seed=SEED,
                  force_row_wise=True)

    # ---- база NNLS + топ-5
    Gf = np.zeros((N_FOLDS, m, m)); bf = np.zeros((N_FOLDS, m))
    for f in range(N_FOLDS):
        idx = fold == f
        Gf[f] = X[idx].T @ X[idx]; bf[f] = X[idx].T @ ly[idx]
    w_full = fit_nnls(Gf.sum(0) / N, bf.sum(0) / N, 0.0)
    top5 = [int(i) for i in np.argsort(-w_full)[:5]]
    Z = np.column_stack([X] + [X[:, i] * U[:, k] for i in top5
                               for k in range(U.shape[1])] + [np.ones(N)])
    zsd = Z.std(0); zsd[zsd < 1e-12] = 1.0
    Zs = Z / zsd
    mz = Z.shape[1]
    Gz = np.zeros((N_FOLDS, mz, mz)); bz = np.zeros((N_FOLDS, mz)); yyz = np.zeros(N_FOLDS)
    for f in range(N_FOLDS):
        idx = fold == f
        Gz[f] = Zs[idx].T @ Zs[idx]; bz[f] = Zs[idx].T @ ly[idx]
        yyz[f] = float(ly[idx] @ ly[idx])

    lin = np.empty(N); vc = np.empty(N); comb = np.empty(N); rb = np.empty(N)
    for f in range(N_FOLDS):
        te = fold == f; tr = ~te
        trf = [g for g in range(N_FOLDS) if g != f]
        ntr = int(nf[trf].sum())
        w = fit_nnls(Gf[trf].sum(0) / ntr, bf[trf].sum(0) / ntr, 0.0)
        lin[te] = X[te] @ w
        lin_tr = X[tr] @ w
        # varying-coefficient (alpha вложенной CV)
        best, best_a = np.inf, None
        for al in ALPHAS:
            tot, cnt = 0.0, 0
            for h in trf:
                tr2 = [g for g in trf if g != h]
                n2 = int(nf[tr2].sum())
                Ga = Gz[tr2].sum(0) / n2; ba = bz[tr2].sum(0) / n2
                sc = float(np.trace(Ga)) / mz
                wz = np.linalg.solve(Ga + (al + 1e-12) * sc * np.eye(mz), ba)
                tot += float(wz @ Gz[h] @ wz - 2 * bz[h] @ wz + yyz[h]); cnt += int(nf[h])
            v = np.sqrt(tot / cnt)
            if v < best:
                best, best_a = v, al
        Ga = Gz[trf].sum(0) / ntr; ba = bz[trf].sum(0) / ntr
        sc = float(np.trace(Ga)) / mz
        wz = np.linalg.solve(Ga + (best_a + 1e-12) * sc * np.eye(mz), ba)
        vc[te] = Zs[te] @ wz
        vc_tr = Zs[tr] @ wz
        # бустинг поверх линейного (эталон) и поверх varying-coefficient
        ds1 = lgb.Dataset(F[tr], label=ly[tr] - lin_tr, feature_name=fn, free_raw_data=False)
        b1 = lgb.train(params, ds1, num_boost_round=CFG["n"])
        rb[te] = lin[te] + b1.predict(F[te], num_iteration=CFG["n"])
        ds2 = lgb.Dataset(F[tr], label=ly[tr] - vc_tr, feature_name=fn, free_raw_data=False)
        b2 = lgb.train(params, ds2, num_boost_round=CFG["n"])
        comb[te] = vc[te] + b2.predict(F[te], num_iteration=CFG["n"])
        print(f"  fold {f} готов ({time.time() - t0:.0f}s)", flush=True)

    e_lin = lin - ly
    r = {}
    for tag, p in (("linear_nnls", lin), ("varying_coef", vc),
                   ("resid_boost_reg", rb), ("vc_plus_boost", comb)):
        e = p - ly
        r[tag] = dict(oof=rmse(e), gain=rmse(e_lin) - rmse(e),
                      shape=float(np.std(e)), bias=float(e.mean()),
                      paired_vs_linear=paired(e, e_lin))
        print(f"[{tag:16s}] OOF={rmse(e):.6f} gain={rmse(e_lin) - rmse(e):+.6f} "
              f"t={r[tag]['paired_vs_linear']['t']:+.1f}")
    r["additivity"] = dict(
        sum_of_parts=r["varying_coef"]["gain"] + r["resid_boost_reg"]["gain"],
        combined=r["vc_plus_boost"]["gain"],
        overlap=r["varying_coef"]["gain"] + r["resid_boost_reg"]["gain"]
        - r["vc_plus_boost"]["gain"])
    print(f"\n[аддитивность] сумма частей {r['additivity']['sum_of_parts']:+.6f}, "
          f"вместе {r['additivity']['combined']:+.6f}, "
          f"пересечение {r['additivity']['overlap']:+.6f}")
    r["boost_cfg"] = CFG
    (REPORTS_DIR / "stack_meta5.json").write_text(json.dumps(r, indent=1, ensure_ascii=False))
    print(f"JSON -> work/reports/stack_meta5.json ({time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

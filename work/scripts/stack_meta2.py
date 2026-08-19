"""stack_meta2.py — ОРАКУЛЬНЫЙ ПОТОЛОК бустинга поверх линейного бленда.

stack_meta.py показал, что при фиксированной конфигурации нелинейный мета-уровень
проигрывает. Остаётся вопрос: это плохая настройка или структуры нет вообще?

Здесь считается ВЕРХНЯЯ ГРАНИЦА: для сетки конфигураций (от очень слабых до
сильных) берётся честный OOF остаточного бустинга, а затем ещё и ОРАКУЛЬНЫЙ
коэффициент усадки c* = argmin ||e_lin - c*r_hat||^2 по всему пулу OOF.
c* подобран НА ОЦЕНОЧНЫХ данных — это заведомо оптимистично; если даже такой
потолок < 0.0005, направление закрыто независимо от настройки.

Дополнительно: корреляция r_hat с остатком линейного бленда (сколько остатка
вообще объяснимо) и та же величина в разбивке по сегментам активности.

Запуск: .venv/bin/python work/scripts/stack_meta2.py
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
                        rmse)

N_FOLDS = 5
GRID = [
    dict(tag="tiny",    num_leaves=7,  n=100, lr=0.03, mdl=2000),
    dict(tag="small",   num_leaves=15, n=200, lr=0.03, mdl=1000),
    dict(tag="spec",    num_leaves=31, n=300, lr=0.05, mdl=100),
    dict(tag="strong",  num_leaves=63, n=600, lr=0.05, mdl=100),
    dict(tag="deepslow", num_leaves=31, n=600, lr=0.02, mdl=500),
]


def main() -> int:
    import lightgbm as lgb
    t0 = time.time()
    cols = ["user_id", "target"] + USER_FEATS
    fv = pl.read_parquet(FEATURES_DIR / f"anchor={VAL_ANCHOR.isoformat()}.parquet",
                         columns=cols).sort("user_id")
    uid = fv["user_id"].to_numpy()
    ly = np.log1p(np.clip(fv["target"].to_numpy().astype(np.float64), 0, None))
    U = fv.select(USER_FEATS).to_numpy().astype(np.float32)
    N = len(uid)
    pool = build_pool()
    m = len(pool)
    X = np.column_stack([load_lp(n, "val", uid) for n in pool]).astype(np.float64)
    F = np.column_stack([X.astype(np.float32), U])
    fn = pool + USER_FEATS

    fold = np.random.default_rng(SEED).permutation(N) % N_FOLDS

    Gf = np.zeros((N_FOLDS, m, m)); bf = np.zeros((N_FOLDS, m))
    nf = np.zeros(N_FOLDS, dtype=np.int64)
    for f in range(N_FOLDS):
        idx = fold == f
        Gf[f] = X[idx].T @ X[idx]; bf[f] = X[idx].T @ ly[idx]; nf[f] = int(idx.sum())
    lin = np.empty(N)
    W = np.zeros((N_FOLDS, m))
    for f in range(N_FOLDS):
        tr = [g for g in range(N_FOLDS) if g != f]
        ntr = int(nf[tr].sum())
        W[f] = fit_nnls(Gf[tr].sum(0) / ntr, bf[tr].sum(0) / ntr, 0.0)
        lin[fold == f] = X[fold == f] @ W[f]
    e_lin = lin - ly
    lin_oof = rmse(e_lin)
    print(f"[linear] OOF = {lin_oof:.6f}", flush=True)

    # сегменты активности для разбивки объяснимости остатка
    rec = fv["rec_order"].to_numpy().astype(np.float64)
    seg = np.where(np.isnan(rec), 4, np.where(rec <= 7, 0, np.where(rec <= 30, 1,
                   np.where(rec <= 90, 2, 3))))
    seg_names = {0: "заказ 0-7д", 1: "8-30д", 2: "31-90д", 3: "91+д", 4: "никогда"}

    out = {"linear_oof": round(lin_oof, 6), "grid": {}}
    for cfg in GRID:
        params = dict(objective="regression", metric="l2", num_leaves=cfg["num_leaves"],
                      learning_rate=cfg["lr"], min_data_in_leaf=cfg["mdl"],
                      feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
                      lambda_l2=1.0, num_threads=2, verbosity=-1, seed=SEED,
                      force_row_wise=True)
        r_hat = np.empty(N)
        for f in range(N_FOLDS):
            te = fold == f; tr = ~te
            lin_tr = X[tr] @ W[f]
            ds = lgb.Dataset(F[tr], label=ly[tr] - lin_tr, feature_name=fn,
                             free_raw_data=False)
            bst = lgb.train(params, ds, num_boost_round=cfg["n"])
            r_hat[te] = bst.predict(F[te], num_iteration=cfg["n"])
        oof_raw = rmse(e_lin + r_hat)
        # оракульная усадка: c* = argmin ||e_lin - c*(-r_hat)||^2  ->  минимум по c
        num = float(np.dot(-e_lin, r_hat)); den = float(np.dot(r_hat, r_hat))
        c_star = num / max(den, 1e-15)
        oof_shrunk = rmse(e_lin + c_star * r_hat)
        corr = float(np.corrcoef(-e_lin, r_hat)[0, 1])
        per_seg = {}
        for s, nm in seg_names.items():
            ms = seg == s
            per_seg[nm] = dict(n=int(ms.sum()),
                               corr=float(np.corrcoef(-e_lin[ms], r_hat[ms])[0, 1]),
                               c_star=float(np.dot(-e_lin[ms], r_hat[ms]) /
                                            max(np.dot(r_hat[ms], r_hat[ms]), 1e-15)))
        out["grid"][cfg["tag"]] = dict(
            cfg=cfg, oof=oof_raw, gain=lin_oof - oof_raw,
            oof_oracle_shrunk=oof_shrunk, gain_oracle=lin_oof - oof_shrunk,
            c_star=c_star, corr_with_residual=corr, per_segment=per_seg)
        print(f"[{cfg['tag']:9s}] OOF={oof_raw:.6f} gain={lin_oof - oof_raw:+.6f} | "
              f"оракул c*={c_star:.3f} -> {oof_shrunk:.6f} "
              f"gain={lin_oof - oof_shrunk:+.6f} | corr(r̂, -e)={corr:+.4f} "
              f"({time.time() - t0:.0f}s)", flush=True)

    best = max(out["grid"], key=lambda k: out["grid"][k]["gain_oracle"])
    out["oracle_ceiling"] = dict(cfg=best, gain=out["grid"][best]["gain_oracle"])
    print(f"\n[ПОТОЛОК] лучшая конфигурация {best}: оракульный выигрыш "
          f"{out['grid'][best]['gain_oracle']:+.6f} (подгонка c* на оценочных данных)")
    (REPORTS_DIR / "stack_meta2.json").write_text(json.dumps(out, indent=1,
                                                            ensure_ascii=False))
    print(f"JSON -> work/reports/stack_meta2.json ({time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

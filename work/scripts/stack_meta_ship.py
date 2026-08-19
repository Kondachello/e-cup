"""stack_meta_ship.py — сборка тестового прогноза победившего мета-варианта.

Победитель (stack_meta3/4): ЛИНЕЙНАЯ модель с юзер-зависимыми весами —
    ly ~ sum_i w_i * lp_i  +  sum_{i in топ-5, k} c_ik * lp_i * u_k  +  const
(ridge, alpha по вложенной CV). Честный OOF 1.665982 против 1.666792 у NNLS
(+0.00081 на сиде 42, +0.000728 +- 0.000057 по 5 разбиениям).
Древесные варианты все проиграли (stack_meta.json).

Что делает скрипт:
  * сохраняет ЧЕСТНЫЕ (кросс-фитнутые) val-прогнозы как stack_meta_val.parquet —
    так файл можно без утечки класть в дальнейшие бленды;
  * обучает модель на ВСЕЙ валидации и применяет к тестовым предсказаниям
    (юзер-признаки теста стандартизуются статистиками ВАЛИДАЦИИ);
  * тем же кодом собирает эталонный линейный NNLS-бленд (для сравнения направлений);
  * пишет два сабмита по штатному рецепту для новых файлов из *_cal-пула
    (наклон до sd=1.628, сдвиг до mean=2.3275, KNOWLEDGE) — сравнимые между собой.

Запуск: .venv/bin/python work/scripts/stack_meta_ship.py [--no-save]
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "POLARS_MAX_THREADS"):
    os.environ.setdefault(_v, "2")

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from common import (FEATURES_DIR, REPORTS_DIR, ROOT, SAMPLE_SUBMIT,  # noqa: E402
                    TEST_ANCHOR, VAL_ANCHOR)
from exp_lib import log_score, save_preds  # noqa: E402
from stack_meta import (SEED, USER_FEATS, build_pool, fit_nnls, load_lp,  # noqa: E402
                        rmse)
from stack_meta4 import oof_ridge  # noqa: E402

N_FOLDS = 5
TARGET_SD = 1.628          # нужная дисперсия log1p на тесте (KNOWLEDGE)
TARGET_MEAN = 2.3275       # mean_P(t), замерено на LB
SUB = ROOT / "submissions"


def write_sub(name: str, uid_t: np.ndarray, lp: np.ndarray) -> Path:
    """Штатный рецепт для новых файлов: наклон до sd=1.628, сдвиг до mean=2.3275."""
    k = TARGET_SD / float(lp.std())
    adj = TARGET_MEAN + k * (lp - lp.mean())
    vals = np.expm1(np.clip(adj, 0, None))
    sample = pl.read_csv(SAMPLE_SUBMIT, schema_overrides={"user_id": pl.Int64})
    out = (sample.select("user_id")
           .join(pl.DataFrame({"user_id": uid_t.astype(np.int64),
                               "predict": vals}), on="user_id", how="left"))
    assert out["predict"].null_count() == 0
    SUB.mkdir(exist_ok=True)
    p = SUB / f"{name}.csv"
    out.write_csv(p)
    print(f"  {p.name}: наклон {k:.4f}, mean(lp) {lp.mean():.4f}->{adj.mean():.4f}, "
          f"sd {lp.std():.4f}->{adj.std():.4f}")
    return p


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-save", action="store_true")
    a = ap.parse_args()

    fv = pl.read_parquet(FEATURES_DIR / f"anchor={VAL_ANCHOR.isoformat()}.parquet",
                         columns=["user_id", "target"] + USER_FEATS).sort("user_id")
    ft = pl.read_parquet(FEATURES_DIR / f"anchor={TEST_ANCHOR.isoformat()}.parquet",
                         columns=["user_id"] + USER_FEATS).sort("user_id")
    uid, uid_t = fv["user_id"].to_numpy(), ft["user_id"].to_numpy()
    ly = np.log1p(np.clip(fv["target"].to_numpy().astype(np.float64), 0, None))
    N = len(uid)
    pool = build_pool(); m = len(pool)
    X = np.column_stack([load_lp(n, "val", uid) for n in pool]).astype(np.float64)
    Xt = np.column_stack([load_lp(n, "test", uid_t) for n in pool]).astype(np.float64)

    # --- юзер-признаки: NaN->0, стандартизация СТАТИСТИКАМИ ВАЛИДАЦИИ (и для теста)
    Uv = np.nan_to_num(fv.select(USER_FEATS).to_numpy().astype(np.float64), nan=0.0)
    Ut = np.nan_to_num(ft.select(USER_FEATS).to_numpy().astype(np.float64), nan=0.0)
    mu, sd = Uv.mean(0), Uv.std(0)
    sd[sd < 1e-12] = 1.0
    Uv = (Uv - mu) / sd
    Ut = (Ut - mu) / sd

    fold = np.random.default_rng(SEED).permutation(N) % N_FOLDS
    nf = np.array([int((fold == f).sum()) for f in range(N_FOLDS)])

    # --- линейная база NNLS (эталон направления)
    Gf = np.zeros((N_FOLDS, m, m)); bf = np.zeros((N_FOLDS, m))
    for f in range(N_FOLDS):
        idx = fold == f
        Gf[f] = X[idx].T @ X[idx]; bf[f] = X[idx].T @ ly[idx]
    lin_oofp = np.empty(N)
    for f in range(N_FOLDS):
        tr = [g for g in range(N_FOLDS) if g != f]
        ntr = int(nf[tr].sum())
        lin_oofp[fold == f] = X[fold == f] @ fit_nnls(Gf[tr].sum(0) / ntr,
                                                      bf[tr].sum(0) / ntr, 0.0)
    lin_oof = rmse(lin_oofp - ly)
    w_nnls = fit_nnls(Gf.sum(0) / N, bf.sum(0) / N, 0.0)
    top5 = [int(i) for i in np.argsort(-w_nnls)[:5]]
    lp_lin_t = Xt @ w_nnls
    print(f"[NNLS] OOF={lin_oof:.6f}; топ-5 {[pool[i] for i in top5]}")

    # --- мета: честный OOF (для сохранения) + полная подгонка (для теста)
    def design(Xa: np.ndarray, Ua: np.ndarray) -> np.ndarray:
        return np.column_stack([Xa] + [Xa[:, i] * Ua[:, k] for i in top5
                                       for k in range(Ua.shape[1])]
                               + [np.ones(len(Xa))])

    Z, Zt = design(X, Uv), design(Xt, Ut)
    meta_oofp, picks = oof_ridge(Z, ly, fold, nf, N_FOLDS)
    meta_oof = rmse(meta_oofp - ly)
    alpha = float(np.median(picks))
    print(f"[meta] честный OOF={meta_oof:.6f}  gain={lin_oof - meta_oof:+.6f}  "
          f"alpha по фолдам={picks} -> {alpha:g}")

    zsd = Z.std(0); zsd[zsd < 1e-12] = 1.0
    Zs = Z / zsd
    Ga = Zs.T @ Zs / N; ba = Zs.T @ ly / N
    sc = float(np.trace(Ga)) / Z.shape[1]
    wz = np.linalg.solve(Ga + (alpha + 1e-12) * sc * np.eye(Z.shape[1]), ba) / zsd
    lp_meta_v = Z @ wz
    lp_meta_t = Zt @ wz
    print(f"[meta] in-sample val={rmse(lp_meta_v - ly):.6f} (оптимистично)")

    # --- вариант B: + регуляризованный бустинг на остатке VC (stack_meta5: +0.000957)
    import lightgbm as lgb
    CFG = dict(num_leaves=15, n=200, lr=0.03, mdl=1000)
    params = dict(objective="regression", metric="l2", num_leaves=CFG["num_leaves"],
                  learning_rate=CFG["lr"], min_data_in_leaf=CFG["mdl"],
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
                  lambda_l2=1.0, num_threads=2, verbosity=-1, seed=SEED,
                  force_row_wise=True)
    Fv = np.column_stack([X.astype(np.float32), Uv.astype(np.float32)])
    Ft = np.column_stack([Xt.astype(np.float32), Ut.astype(np.float32)])
    fn = pool + USER_FEATS
    # бустинг учится на КРОСС-ФИТНУТОМ остатке VC — так же, как при оценке OOF
    ds = lgb.Dataset(Fv, label=ly - meta_oofp, feature_name=fn, free_raw_data=False)
    bst = lgb.train(params, ds, num_boost_round=CFG["n"])
    lp_vcb_t = lp_meta_t + bst.predict(Ft, num_iteration=CFG["n"])
    print(f"[meta+boost] тестовый сдвиг: mean {lp_meta_t.mean():.4f}->"
          f"{lp_vcb_t.mean():.4f}, sd {lp_meta_t.std():.4f}->{lp_vcb_t.std():.4f}")

    lev = dict(val_mean=float(lp_meta_v.mean()), test_mean=float(lp_meta_t.mean()),
               val_sd=float(lp_meta_v.std()), test_sd=float(lp_meta_t.std()),
               lin_test_mean=float(lp_lin_t.mean()), lin_test_sd=float(lp_lin_t.std()),
               corr_test_meta_vs_lin=float(np.corrcoef(lp_meta_t, lp_lin_t)[0, 1]),
               sd_diff=float((lp_meta_t - lp_lin_t).std()),
               test_min=float(lp_meta_t.min()), test_max=float(lp_meta_t.max()),
               frac_below_zero=float((lp_meta_t < 0).mean()))
    print("[уровни]", json.dumps({k: round(v, 5) for k, v in lev.items()},
                                 ensure_ascii=False))

    out = dict(pool=pool, top5=[pool[i] for i in top5], user_feats=USER_FEATS,
               alpha=alpha, linear_oof=round(lin_oof, 6), meta_oof=round(meta_oof, 6),
               gain=round(lin_oof - meta_oof, 6), levels=lev, n_terms=int(Z.shape[1]))

    if not a.no_save:
        # честные (кросс-фитнутые) val-прогнозы — без утечки для дальнейших блендов
        save_preds("stack_meta", "val", uid, np.expm1(np.clip(meta_oofp, 0, None)))
        save_preds("stack_meta", "test", uid_t, np.expm1(np.clip(lp_meta_t, 0, None)))
        log_score("stack_meta", meta_oof,
                  f"varying-coefficient meta-stack: {m} cal models + top5 x "
                  f"{len(USER_FEATS)} user feats ({Z.shape[1]} terms, ridge a={alpha:g}); "
                  f"honest OOF(5f by user) {meta_oof:.6f} vs linear NNLS {lin_oof:.6f}; "
                  f"val preds saved CROSS-FITTED (OOF), test preds from full-val fit")
        save_preds("stack_meta_vcb", "test", uid_t, np.expm1(np.clip(lp_vcb_t, 0, None)))
        print("\n[сабмиты] штатный рецепт (наклон->sd 1.628, сдвиг->mean 2.3275):")
    (REPORTS_DIR / "stack_meta_ship.json").write_text(
        json.dumps(out, indent=1, ensure_ascii=False))
    print("JSON -> work/reports/stack_meta_ship.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

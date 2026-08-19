"""stack_meta.py — НЕЛИНЕЙНЫЙ мета-уровень над пулом калиброванных моделей.

Гипотеза: оптимальные веса смеси зависят от пользователя (активным подходят одни
модели, спящим другие), и мета-модель выучит это сама, без ручной сегментации.
Линейный NNLS в log1p даёт OOF 1.666791 — это база сравнения.

Протокол (полностью честный, ТЕ ЖЕ фолды, что у blend_reopt / blend_segments):
  * 5-фолдовая CV ПО ЮЗЕРАМ, fold = default_rng(42).permutation(N) % 5,
    юзеры отсортированы по user_id;
  * мета-модель обучается на 4 фолдах, оценивается на 5-м; pooled OOF RMSLE;
  * гиперпараметры НЕ подбираются по внешнему фолду (ridge alpha — вложенная CV
    внутри трейна; для LGB число деревьев фиксировано, кривая по итерациям
    печатается ТОЛЬКО как диагностика и помечена oracle);
  * парный тест значимости: разность квадратов ошибок по юзерам -> SE и t.

Варианты:
  linear            NNLS в log1p (база)
  meta_lgb          LGB на 22 log1p-предсказаниях + 12 признаках юзера
  meta_preds_only   LGB только на предсказаниях (даёт ли что-то нелинейность сама)
  meta_lgb_linfeat  LGB на предсказаниях + юзере + линейном бленде как признаке
  residual_boost    линейный NNLS + LGB на ОСТАТКЕ линейного
  ridge_pairwise    ridge: 22 линейных члена + попарные произведения топ-5 моделей

Запуск: .venv/bin/python work/scripts/stack_meta.py [--save] [--folds 5]
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "POLARS_MAX_THREADS"):
    os.environ.setdefault(_v, "2")

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
from scipy.linalg import cholesky, solve_triangular  # noqa: E402
from scipy.optimize import nnls  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from common import FEATURES_DIR, PREDS_DIR, REPORTS_DIR, TEST_ANCHOR, VAL_ANCHOR  # noqa: E402
from exp_lib import log_score, save_preds  # noqa: E402

EXCLUDE = {"blend"}
CONTAMINATED = {"lgblog_final", "xgblog_final", "cblog_final", "mlp_final", "gru_final",
                "hjit37", "hjit44"}

USER_FEATS = ["rec_order", "ord_days_90", "ord_days_365",
              "log_gmv_sum_30", "log_gmv_sum_90", "log_gmv_sum_365",
              "ord_rate_90", "act_density", "gmv_per_ordday_365",
              "searches_30", "rec_active", "tenure"]

N_FOLDS = 5
SEED = 42
NTREES = 300
LGB_PARAMS = dict(objective="regression", metric="l2", num_leaves=31,
                  learning_rate=0.05, min_data_in_leaf=100, feature_fraction=0.8,
                  bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                  num_threads=2, verbosity=-1, seed=SEED, force_row_wise=True)
CURVE_ITERS = [50, 100, 150, 200, 250, 300]
TRANSFER = 0.584          # val->test перенос связи (KNOWLEDGE)
NAME = "stack_meta"


def build_pool() -> list[str]:
    names = []
    for p in sorted(PREDS_DIR.glob("*_cal_test.parquet")):
        stem = p.name[: -len("_cal_test.parquet")]
        if stem in EXCLUDE or stem in CONTAMINATED:
            continue
        if not (PREDS_DIR / f"{stem}_cal_val.parquet").exists():
            continue
        names.append(stem + "_cal")
    if (PREDS_DIR / "channel3_chcal_test.parquet").exists() and \
       (PREDS_DIR / "channel3_chcal_val.parquet").exists():
        names.append("channel3_chcal")
    return names


def load_lp(name: str, split: str, uid_ref: np.ndarray) -> np.ndarray:
    d = pl.read_parquet(PREDS_DIR / f"{name}_{split}.parquet").sort("user_id")
    if not np.array_equal(d["user_id"].to_numpy(), uid_ref):
        raise ValueError(f"{name}_{split}: user_id не совпадает с базисом")
    return np.log1p(np.clip(d["pred"].to_numpy().astype(np.float64), 0, None))


def fit_nnls(G: np.ndarray, b: np.ndarray, alpha: float = 0.0) -> np.ndarray:
    m = G.shape[0]
    jitter = 1e-10 * float(np.trace(G)) / m
    R = cholesky(G + (alpha + jitter) * np.eye(m), lower=False)
    z = solve_triangular(R, b, trans="mdl_larvik", lower=False)
    w, _ = nnls(R, z)
    return w


def rmse(e: np.ndarray) -> float:
    return float(np.sqrt(np.mean(e ** 2)))


def paired(e_new: np.ndarray, e_base: np.ndarray) -> dict:
    """Парный тест: насколько значима разница RMSE двух наборов ошибок."""
    d = e_base ** 2 - e_new ** 2            # >0 => новый лучше
    n = len(d)
    md, sd = float(d.mean()), float(d.std(ddof=1))
    se = sd / np.sqrt(n)
    b2, n2 = float(np.mean(e_base ** 2)), float(np.mean(e_new ** 2))
    # дельта RMSE и её SE (дельта-метод)
    drmse = np.sqrt(b2) - np.sqrt(n2)
    se_rmse = se / (2 * np.sqrt(n2))
    return dict(gain_rmse=float(drmse), se_rmse=float(se_rmse),
                t=float(md / se) if se > 0 else 0.0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", action="store_true")
    ap.add_argument("--folds", type=int, default=N_FOLDS)
    ap.add_argument("--min-gain", type=float, default=0.0005)
    a = ap.parse_args()
    t0 = time.time()
    import lightgbm as lgb

    # ------------------------------------------------------------- данные
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
    print(f"[пул] {m} калиброванных чистых моделей: {', '.join(pool)}", flush=True)
    print(f"[юзер-признаки] {len(USER_FEATS)}: {', '.join(USER_FEATS)}")
    print(f"[N] {N}  доля NaN в юзер-признаках: "
          f"{float(np.isnan(U).mean()):.4f}", flush=True)

    rng = np.random.default_rng(SEED)
    fold = rng.permutation(N) % a.folds

    Xf32 = X.astype(np.float32)
    F_all = np.column_stack([Xf32, U])                    # preds + user
    names_all = pool + USER_FEATS

    # --------------------------------------------------- 1. линейный NNLS
    Gf = np.zeros((a.folds, m, m)); bf = np.zeros((a.folds, m))
    nf = np.zeros(a.folds, dtype=np.int64)
    for f in range(a.folds):
        idx = fold == f
        Xs, lys = X[idx], ly[idx]
        Gf[f] = Xs.T @ Xs; bf[f] = Xs.T @ lys; nf[f] = int(idx.sum())
    pred_lin = np.empty(N)
    W_lin = np.zeros((a.folds, m))
    for f in range(a.folds):
        tr = [g for g in range(a.folds) if g != f]
        ntr = int(nf[tr].sum())
        w = fit_nnls(Gf[tr].sum(0) / ntr, bf[tr].sum(0) / ntr, 0.0)
        W_lin[f] = w
        pred_lin[fold == f] = X[fold == f] @ w
    e_lin = pred_lin - ly
    linear_oof = rmse(e_lin)
    w_full = fit_nnls(Gf.sum(0) / N, bf.sum(0) / N, 0.0)
    print(f"\n[linear NNLS] OOF = {linear_oof:.6f}  (ожидалось ~1.666791)")
    print("  веса:", {pool[i]: round(float(w_full[i]), 4)
                      for i in np.argsort(-w_full) if w_full[i] > 1e-4}, flush=True)

    results = {"linear": dict(oof=linear_oof, gain=0.0)}
    oof_preds = {"linear": pred_lin}

    # ------------------------------------------------------ 2. LGB-варианты
    def run_lgb(tag: str, feat_idx: np.ndarray, feat_names: list[str],
                residual: bool = False, add_lin: bool = False):
        """5-фолдовая честная OOF для LGB на подмножестве признаков."""
        pred = np.empty(N)
        curve = {k: np.empty(N) for k in CURVE_ITERS}
        fn = list(feat_names) + (["_linblend"] if add_lin else [])
        gains = np.zeros(len(fn))
        insample = []
        for f in range(a.folds):
            te = fold == f
            tr = ~te
            Ftr = F_all[:, feat_idx][tr]
            Fte = F_all[:, feat_idx][te]
            if add_lin or residual:
                lin_tr = X[tr] @ W_lin[f]
                lin_te = X[te] @ W_lin[f]
            if add_lin:
                Ftr = np.column_stack([Ftr, lin_tr.astype(np.float32)])
                Fte = np.column_stack([Fte, lin_te.astype(np.float32)])
            ytr = (ly[tr] - lin_tr) if residual else ly[tr]
            ds = lgb.Dataset(Ftr, label=ytr, feature_name=fn, free_raw_data=False)
            bst = lgb.train(LGB_PARAMS, ds, num_boost_round=NTREES)
            base_te = lin_te if residual else 0.0
            base_tr = lin_tr if residual else 0.0
            pred[te] = bst.predict(Fte, num_iteration=NTREES) + base_te
            for k in CURVE_ITERS:
                curve[k][te] = bst.predict(Fte, num_iteration=k) + base_te
            insample.append(rmse(bst.predict(Ftr, num_iteration=NTREES) + base_tr - ly[tr]))
            g = bst.feature_importance("gain")
            gains += g / max(g.sum(), 1e-12)
            print(f"  [{tag}] fold {f} готов ({time.time() - t0:.0f}s)", flush=True)
        e = pred - ly
        oof = rmse(e)
        cur = {k: rmse(curve[k] - ly) for k in CURVE_ITERS}
        gains /= a.folds
        imp = sorted(zip(fn, gains), key=lambda kv: -kv[1])
        user_share = float(sum(v for k, v in zip(fn, gains) if k in USER_FEATS))
        res = dict(oof=oof, gain=linear_oof - oof, insample=float(np.mean(insample)),
                   oof_shape=float(np.std(e)), oof_bias=float(e.mean()),
                   curve={str(k): v for k, v in cur.items()},
                   curve_best_oracle=min(cur.values()),
                   user_feat_gain_share=user_share,
                   importance={k: round(float(v), 5) for k, v in imp[:15]},
                   paired_vs_linear=paired(e, e_lin))
        results[tag] = res
        oof_preds[tag] = pred
        print(f"[{tag}] OOF={oof:.6f} gain={linear_oof - oof:+.6f} "
              f"in={np.mean(insample):.6f} user_gain_share={user_share:.3f}")
        print(f"   кривая по итерациям: " +
              " ".join(f"{k}:{v:.6f}" for k, v in cur.items()), flush=True)
        return res

    idx_preds = np.arange(m)
    idx_all = np.arange(m + len(USER_FEATS))

    run_lgb("meta_lgb", idx_all, names_all)
    run_lgb("meta_preds_only", idx_preds, pool)
    run_lgb("meta_lgb_linfeat", idx_all, names_all, add_lin=True)
    run_lgb("residual_boost", idx_all, names_all, residual=True)

    # ------------------------------------------- 3. ridge с попарными топ-5
    top5 = [int(i) for i in np.argsort(-w_full)[:5]]
    print(f"\n[ridge_pairwise] топ-5 моделей: {[pool[i] for i in top5]}", flush=True)
    prod_pairs = [(i, j) for ii, i in enumerate(top5) for j in top5[ii:]]
    Z = np.column_stack([X] + [X[:, i] * X[:, j] for i, j in prod_pairs])
    Z = np.column_stack([Z, np.ones(N)])
    mz = Z.shape[1]
    zscale = Z.std(axis=0); zscale[zscale < 1e-12] = 1.0
    Z = Z / zscale
    ALPHAS = [1e-8, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]
    Gz = np.zeros((a.folds, mz, mz)); bz = np.zeros((a.folds, mz))
    yyz = np.zeros(a.folds)
    for f in range(a.folds):
        idx = fold == f
        Zs, lys = Z[idx], ly[idx]
        Gz[f] = Zs.T @ Zs; bz[f] = Zs.T @ lys; yyz[f] = float(lys @ lys)
    pred_rp = np.empty(N)
    picks = []
    for f in range(a.folds):
        tr = [g for g in range(a.folds) if g != f]
        # вложенная CV внутри трейна: выбор alpha без внешнего фолда
        best, best_a = np.inf, None
        for al in ALPHAS:
            tot, cnt = 0.0, 0
            for h in tr:
                tr2 = [g for g in tr if g != h]
                n2 = int(nf[tr2].sum())
                Ga = Gz[tr2].sum(0) / n2; ba = bz[tr2].sum(0) / n2
                s = float(np.trace(Ga)) / mz
                wz = np.linalg.solve(Ga + al * s * np.eye(mz), ba)
                tot += float(wz @ Gz[h] @ wz - 2 * bz[h] @ wz + yyz[h]); cnt += int(nf[h])
            v = np.sqrt(tot / cnt)
            if v < best:
                best, best_a = v, al
        ntr = int(nf[tr].sum())
        Ga = Gz[tr].sum(0) / ntr; ba = bz[tr].sum(0) / ntr
        s = float(np.trace(Ga)) / mz
        wz = np.linalg.solve(Ga + best_a * s * np.eye(mz), ba)
        pred_rp[fold == f] = Z[fold == f] @ wz
        picks.append(best_a)
        print(f"  [ridge_pairwise] fold {f}: alpha={best_a:g} ({time.time() - t0:.0f}s)",
              flush=True)
    e_rp = pred_rp - ly
    results["ridge_pairwise"] = dict(oof=rmse(e_rp), gain=linear_oof - rmse(e_rp),
                                     oof_shape=float(np.std(e_rp)),
                                     oof_bias=float(e_rp.mean()), alphas=picks,
                                     n_terms=mz,
                                     paired_vs_linear=paired(e_rp, e_lin))
    oof_preds["ridge_pairwise"] = pred_rp
    print(f"[ridge_pairwise] OOF={rmse(e_rp):.6f} gain={linear_oof - rmse(e_rp):+.6f}",
          flush=True)

    # ----------------------------------------------------------- 4. сводка
    best_tag = min(results, key=lambda k: results[k]["oof"])
    best_gain = linear_oof - results[best_tag]["oof"]
    print("\n=== СВОДКА (OOF RMSLE, 5 фолдов по юзерам) ===")
    for k in sorted(results, key=lambda x: results[x]["oof"]):
        r = results[k]
        p = r.get("paired_vs_linear", {})
        print(f"  {k:18s} {r['oof']:.6f}  gain={r['gain']:+.6f}"
              + (f"  t={p.get('t', 0):+.1f}" if p else ""))
    print(f"\nлучший: {best_tag}  gain={best_gain:+.6f}  "
          f"ожидаемый перенос на тест ~{best_gain * TRANSFER:+.6f}")

    out = dict(pool=pool, user_feats=USER_FEATS, n=N, folds=a.folds,
               linear_oof=round(linear_oof, 6),
               linear_weights={pool[i]: round(float(w_full[i]), 6)
                               for i in np.argsort(-w_full) if w_full[i] > 1e-6},
               results=results, best=best_tag, best_gain=round(best_gain, 6),
               expected_test_gain=round(best_gain * TRANSFER, 6),
               runtime_s=round(time.time() - t0, 1))

    # ------------------------------------------- 5. тестовый прогноз (если есть выигрыш)
    out["shipped"] = False
    if best_gain >= a.min_gain and best_tag != "linear":
        print(f"\n[ship] выигрыш {best_gain:.6f} >= {a.min_gain} — собираю тест", flush=True)
        ft = pl.read_parquet(FEATURES_DIR / f"anchor={TEST_ANCHOR.isoformat()}.parquet",
                             columns=["user_id"] + USER_FEATS).sort("user_id")
        uid_t = ft["user_id"].to_numpy()
        Ut = ft.select(USER_FEATS).to_numpy().astype(np.float32)
        Xt = np.column_stack([load_lp(n, "test", uid_t) for n in pool]).astype(np.float64)
        Ft = np.column_stack([Xt.astype(np.float32), Ut])
        lin_v = X @ w_full
        lin_t = Xt @ w_full
        if best_tag == "ridge_pairwise":
            Zt = np.column_stack([Xt] + [Xt[:, i] * Xt[:, j] for i, j in prod_pairs])
            Zt = np.column_stack([Zt, np.ones(len(uid_t))]) / zscale
            al = float(np.median(picks))
            Ga = Gz.sum(0) / N; ba = bz.sum(0) / N
            s = float(np.trace(Ga)) / mz
            wz = np.linalg.solve(Ga + al * s * np.eye(mz), ba)
            lv, lt = Z @ wz, Zt @ wz
        else:
            residual = best_tag == "residual_boost"
            add_lin = best_tag == "meta_lgb_linfeat"
            fidx = idx_preds if best_tag == "meta_preds_only" else idx_all
            fn = (pool if best_tag == "meta_preds_only" else names_all)
            Ftr, Fte = F_all[:, fidx], Ft[:, fidx]
            if add_lin:
                Ftr = np.column_stack([Ftr, lin_v.astype(np.float32)])
                Fte = np.column_stack([Fte, lin_t.astype(np.float32)])
                fn = fn + ["_linblend"]
            ytr = (ly - lin_v) if residual else ly
            ds = lgb.Dataset(Ftr, label=ytr, feature_name=list(fn), free_raw_data=False)
            bst = lgb.train(LGB_PARAMS, ds, num_boost_round=NTREES)
            lv = bst.predict(Ftr, num_iteration=NTREES) + (lin_v if residual else 0.0)
            lt = bst.predict(Fte, num_iteration=NTREES) + (lin_t if residual else 0.0)
        lv = np.clip(lv, 0, None); lt = np.clip(lt, 0, None)
        out["test_levels"] = dict(val_meanlog=float(lv.mean()), test_meanlog=float(lt.mean()),
                                  lin_val_meanlog=float(lin_v.mean()),
                                  lin_test_meanlog=float(lin_t.mean()),
                                  test_sd=float(lt.std()), lin_test_sd=float(lin_t.std()),
                                  corr_test_vs_lin=float(np.corrcoef(lt, lin_t)[0, 1]))
        print("[ship] уровни:", json.dumps({k: round(v, 5)
                                            for k, v in out["test_levels"].items()}))
        if a.save:
            save_preds(NAME, "val", uid, np.expm1(lv))
            save_preds(NAME, "test", uid_t, np.expm1(lt))
            log_score(NAME, float(np.sqrt(np.mean((lv - ly) ** 2))),
                      f"nonlinear meta-stack {best_tag} over {m} cal models + "
                      f"{len(USER_FEATS)} user feats; honest OOF(5f by user)="
                      f"{results[best_tag]['oof']:.6f} vs linear {linear_oof:.6f}")
            out["shipped"] = True
    else:
        print(f"\n[ship] выигрыш {best_gain:.6f} < {a.min_gain} — тест НЕ собираю")

    (REPORTS_DIR / "stack_meta.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(f"\nJSON -> work/reports/stack_meta.json ({time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

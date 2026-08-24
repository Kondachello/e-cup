"""Перепроверка теоремы оболочки об ТЕКУЩИЙ бленд (могильник, 22.08).

Теорема «остаток бленда не выводится из признаков» доказывалась об старые бленды:
H1 LGB-стек OOF R²=-0.019 (c_cand, 1.6717), TabPFN на 4к строк (-0.0116 при плацебо
-0.0011). Состав бленда с тех пор радикально сменился (kostya46_cal 0.246 + tfm3b 0.20
+ gseq + lagd28, эталон 1.665647) — остаток новой эпохи мог приобрести структуру.

Повтор на ПОЛНЫХ 250к против остатка resid = log1p(y) - blend (колонка пакета,
уже калиброванная):
  lgb_full196        LGB-стек OOF по юзерам, 5 фолдов, лёгкие параметры, 196 фичей
  lgb_safe117        то же на 117 колонках, не задетых обрезкой MAX_BACK
                     (production-safe: на тестовом якоре их вход не искажён)
  lgb_full196_lb     зеркало H1: те же 196 + log1p(бленд) как признак
  lgb_placebo        тот же конвейер, остаток перемешан между юзерами -> калибровка нуля
  ridge_full196      линейная опора на тех же фолдах (протокол exp_resid_ridge)
  ridge_placebo      её ноль

Критерий задачи: OOF R² > +0.002 -> оболочка прохудилась; <= 0 -> теорема подтверждена
для новой эпохи. Честный выигрыш = sb - score(lb + oof) — та же величина, что в
exp_resid_ridge (порог команды 0.0003, шум 0.000022).

Запуск (~6 мин, 2 потока, последовательно):
  USE_V2=1 USE_V3=1 POLARS_MAX_THREADS=2 OMP_NUM_THREADS=2 \
    .venv/bin/python work/reports/grave_hull_probe.py
Артефакт: work/reports/grave_hull_probe.json
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from common import REPORTS_DIR, ROOT, VAL_ANCHOR, feature_cols, load_anchor
from margin import score

FOLDS = 5
ALPHAS = [1e0, 1e1, 1e2, 1e3, 1e4, 1e5]
LGB_PARAMS = dict(n_estimators=350, learning_rate=0.08, num_leaves=63,
                  min_child_samples=500, subsample=0.8, subsample_freq=1,
                  colsample_bytree=0.8, max_bin=127, n_jobs=2, verbose=-1)
STAGES = (100, 200, 350)


def r2(y: np.ndarray, p: np.ndarray) -> float:
    return float(1.0 - np.mean((y - p) ** 2) / np.var(y))


def lgb_oof(X: np.ndarray, r: np.ndarray, folds: np.ndarray, t0: float, tag: str):
    import lightgbm as lgb
    oof = {s: np.zeros_like(r) for s in STAGES}
    fold_r2 = []
    ins_r2 = None
    for f in range(FOLDS):
        tr, te = folds != f, folds == f
        m = lgb.LGBMRegressor(random_state=100 + f, **LGB_PARAMS)
        m.fit(X[tr], r[tr])
        for s in STAGES:
            oof[s][te] = m.predict(X[te], num_iteration=s)
        fold_r2.append(round(r2(r[te], oof[STAGES[-1]][te]), 5))
        if f == 0:
            ins_r2 = round(r2(r[tr], m.predict(X[tr])), 4)
        print(f"  {tag} fold {f}: r2={fold_r2[-1]:+.5f} [{time.time()-t0:.0f}s]", flush=True)
    return oof, fold_r2, ins_r2


def ridge_oof(X: np.ndarray, r: np.ndarray, folds: np.ndarray, seed: int):
    mu, sd = X.mean(0), X.std(0)
    sd[sd == 0] = 1.0
    Z = np.column_stack([(X - mu) / sd, np.ones(len(X))]).astype(np.float64)
    rng = np.random.default_rng(seed)
    oof = np.zeros_like(r)
    alphas = []
    for f in range(FOLDS):
        tr, te = folds != f, folds == f
        Zt, rt = Z[tr], r[tr]
        inner = rng.permutation(len(rt)) < int(0.8 * len(rt))
        G = Zt[inner].T @ Zt[inner]
        b = Zt[inner].T @ rt[inner]
        best = (9e9, ALPHAS[0])
        for al in ALPHAS:
            w = np.linalg.solve(G + al * np.eye(len(G)), b)
            best = min(best, (float(np.mean((Zt[~inner] @ w - rt[~inner]) ** 2)), al))
        al = best[1]
        w = np.linalg.solve(Zt.T @ Zt + al * np.eye(Z.shape[1]), Zt.T @ rt)
        oof[te] = Z[te] @ w
        alphas.append(al)
    return oof, alphas


def main():
    assert os.environ.get("USE_V2") and os.environ.get("USE_V3"), "нужны USE_V2=1 USE_V3=1"
    t0 = time.time()
    pack = pl.read_parquet(ROOT / "work" / "preds_pack" / "val_preds.parquet").sort("user_id")
    uid = pack["user_id"].to_numpy()
    ly = np.log1p(np.clip(pack["target"].to_numpy().astype(np.float64), 0, None))
    lb = pack["blend"].to_numpy().astype(np.float64)
    sb = score(lb, ly)
    resid = ly - lb                     # положительный прогноз ДОБАВЛЯЕТСЯ к бленду

    val = load_anchor(VAL_ANCHOR).sort("user_id")
    assert np.array_equal(val["user_id"].to_numpy(), uid), "user_id пакета и якоря разошлись"
    cols = feature_cols(val)
    bad = {x.split(" ")[0] for x in
           json.loads((REPORTS_DIR / "maxback_affected_cols.json").read_text())}
    safe = [c for c in cols if c not in bad]

    def matrix(cs):
        M = val.select(cs).to_numpy().astype(np.float32)
        return np.nan_to_num(M, nan=0.0, posinf=0.0, neginf=0.0)

    X = matrix(cols)
    Xs = matrix(safe)
    Xlb = np.column_stack([X, lb.astype(np.float32)])
    print(f"эталон {sb:.6f}, признаков {len(cols)} / safe {len(safe)}, "
          f"sd остатка {resid.std():.4f} [{time.time()-t0:.0f}s]", flush=True)

    rng = np.random.default_rng(0)
    folds = rng.permutation(len(ly)) % FOLDS          # одна строка = один юзер
    resid_perm = np.random.default_rng(12345).permutation(resid)

    out = {"reference_blend": round(sb, 6), "n_rows": len(ly),
           "n_features": len(cols), "n_safe": len(safe),
           "folds": FOLDS, "lgb_params": {k: v for k, v in LGB_PARAMS.items()},
           "runs": {}}

    def add(tag, oof, r_true, extra):
        gain = sb - score(lb + oof, ly)
        gain_clip = sb - score(np.clip(lb + oof, 0, None), ly)
        rec = {"oof_r2": round(r2(r_true, oof), 6),
               "gain": round(float(gain), 6), "gain_clip": round(float(gain_clip), 6),
               "oof_sd": round(float(oof.std()), 5), **extra}
        out["runs"][tag] = rec
        print(f"{tag:<18} mdl_flint={rec['oof_r2']:+.6f} выигрыш {gain:+.6f} {extra}", flush=True)

    for tag, M, r_t in (("lgb_full196", X, resid), ("lgb_safe117", Xs, resid),
                        ("lgb_full196_lb", Xlb, resid), ("lgb_placebo", X, resid_perm)):
        oof, fr2, ins = lgb_oof(M, r_t, folds, t0, tag)
        stage_r2 = {str(s): round(r2(r_t, oof[s]), 6) for s in STAGES}
        add(tag, oof[STAGES[-1]], r_t,
            {"fold_r2": fr2, "r2_by_trees": stage_r2, "insample_r2_f0": ins})

    oof_r, alph = ridge_oof(X, resid, folds, seed=0)
    add("ridge_full196", oof_r, resid, {"alphas": alph})
    oof_rp, alph_p = ridge_oof(X, resid_perm, folds, seed=0)
    add("ridge_placebo", oof_rp, resid_perm, {"alphas": alph_p})

    best = max(out["runs"][t]["oof_r2"] for t in
               ("lgb_full196", "lgb_safe117", "lgb_full196_lb"))
    out["best_real_r2"] = best
    out["verdict"] = ("оболочка прохудилась" if best > 0.002 else
                      "теорема подтверждена для новой эпохи" if best <= 0 else
                      "слабый сигнал ниже порога 0.002")
    (REPORTS_DIR / "grave_hull_probe.json").write_text(
        json.dumps(out, indent=1, ensure_ascii=False))
    print(f"\nВЕРДИКТ: {out['verdict']} (best real mdl_flint {best:+.6f})")
    print(f"JSON: work/reports/grave_hull_probe.json [{time.time()-t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()

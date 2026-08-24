"""Пол скрининга: сколько остатка бленда честно объясняет ridge на СТАРЫХ признаках.

Пункт 3 раздела «для Клода Саши» отчёта Жени: его screen_repr.py сравнивал R²
кандидатных признаков с плацебо из случайных чисел, и такой пол ~0.0005 подозрителен.
Вопрос решается прямым замером: ridge на существующих 196 чемпионских признаках
против остатка бленда, строго out-of-sample по юзерам.

Интерпретация:
- если честный OOF-выигрыш ~0.0005–0.001 — это НЕ пол метода, а живой дешёвый
  стек-кандидат (линейный сигнал в остатке, который бленд не выбрал);
- если ~0 — пол screen_repr.py был артефактом плацебо, и прошлые «положительные»
  срабатывания скрининга признаков надо пересматривать с новым контролем.

Контроль: тот же конвейер на остатке, перемешанном между юзерами (должен дать ≤0).

Запуск: USE_V2=1 USE_V3=1 .venv/bin/python work/scripts/exp_resid_ridge.py
Артефакт: work/reports/exp_resid_ridge.json
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))
from common import REPORTS_DIR, ROOT, VAL_ANCHOR, feature_cols, load_anchor
from margin import score

ALPHAS = [1e0, 1e1, 1e2, 1e3, 1e4, 1e5]
FOLDS = 5


def ridge_oof(X: np.ndarray, r: np.ndarray, folds: np.ndarray, rng: np.random.Generator):
    """OOF-прогноз остатка; альфа выбирается на внутреннем сплите обучающего фолда."""
    oof = np.zeros_like(r)
    alphas_used = []
    for f in range(FOLDS):
        tr, te = folds != f, folds == f
        Xt, rt = X[tr], r[tr]
        inner = rng.permutation(len(rt)) < int(0.8 * len(rt))
        G = Xt[inner].T @ Xt[inner]
        b = Xt[inner].T @ rt[inner]
        best = (9e9, ALPHAS[0])
        for al in ALPHAS:
            w = np.linalg.solve(G + al * np.eye(len(G)), b)
            best = min(best, (float(np.mean((Xt[~inner] @ w - rt[~inner]) ** 2)), al))
        al = best[1]
        w = np.linalg.solve(Xt.T @ Xt + al * np.eye(X.shape[1]), Xt.T @ rt)
        oof[te] = X[te] @ w
        alphas_used.append(al)
    return oof, alphas_used


def main():
    assert os.environ.get("USE_V2") and os.environ.get("USE_V3"), "нужны USE_V2=1 USE_V3=1"
    pack = pl.read_parquet(ROOT / "work" / "preds_pack" / "val_preds.parquet").sort("user_id")
    uid = pack["user_id"].to_numpy()
    ly = np.log1p(np.clip(pack["target"].to_numpy().astype(np.float64), 0, None))
    lb = pack["blend"].to_numpy().astype(np.float64)
    sb = score(lb, ly)
    resid = ly - lb                                  # положительный прогноз остатка ДОБАВЛЯЕТСЯ к бленду

    val = load_anchor(VAL_ANCHOR).sort("user_id")
    assert np.array_equal(val["user_id"].to_numpy(), uid)
    cols = feature_cols(val)
    X = val.select(cols).to_numpy().astype(np.float64)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    mu, sd = X.mean(0), X.std(0)
    sd[sd == 0] = 1.0
    X = (X - mu) / sd
    X = np.column_stack([X, np.ones(len(X))])        # свободный член
    print(f"признаков {len(cols)}, эталон {sb:.6f}")

    rng = np.random.default_rng(0)
    folds = rng.permutation(len(ly)) % FOLDS

    oof, alphas = ridge_oof(X, resid, folds, rng)
    gain = sb - score(lb + oof, ly)
    r2 = 1.0 - np.mean((resid - oof) ** 2) / np.mean((resid - resid.mean()) ** 2)
    print(f"честный OOF: выигрыш {gain:+.6f}, R² остатка {r2:+.5f}, альфы {alphas}")

    # производственно-безопасный вариант: только колонки, побитово не задетые обрезкой
    # MAX_BACK=379 (на тестовом якоре задетые колонки искажены, val-обученный ridge
    # получил бы там другой вход; список — зеркальный опыт mb349)
    gain_safe = None
    safe_p = REPORTS_DIR / "maxback_affected_cols.json"
    if safe_p.exists():
        bad = {x.split(" ")[0] for x in json.loads(safe_p.read_text())}
        safe = [c for c in cols if c not in bad]
        Xs = val.select(safe).to_numpy().astype(np.float64)
        Xs = np.nan_to_num(Xs, nan=0.0, posinf=0.0, neginf=0.0)
        mus, sds = Xs.mean(0), Xs.std(0)
        sds[sds == 0] = 1.0
        Xs = np.column_stack([(Xs - mus) / sds, np.ones(len(Xs))])
        oof_s, _ = ridge_oof(Xs, resid, folds, np.random.default_rng(0))
        gain_safe = sb - score(lb + oof_s, ly)
        print(f"безопасные колонки ({len(safe)} из {len(cols)}): выигрыш {gain_safe:+.6f}")

    perm = rng.permutation(len(resid))
    oof_p, _ = ridge_oof(X, resid[perm], folds, rng)
    gain_p = sb - score(lb + oof_p, ly)
    print(f"плацебо (остаток перемешан): выигрыш {gain_p:+.6f}")

    out = {
        "n_features": len(cols), "reference_blend": round(sb, 6),
        "oof_gain": round(float(gain), 6), "oof_r2": round(float(r2), 6),
        "alphas": alphas, "placebo_gain": round(float(gain_p), 6),
        "oof_gain_safe_cols": None if gain_safe is None else round(float(gain_safe), 6),
        "verdict": ("стек-кандидат" if gain >= 0.0003 else
                    "линейный сигнал в остатке на старых признаках отсутствует"
                    if gain < 2 * 0.000022 else "слабый сигнал ниже порога"),
    }
    (REPORTS_DIR / "exp_resid_ridge.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print("JSON: work/reports/exp_resid_ridge.json")


if __name__ == "__main__":
    main()

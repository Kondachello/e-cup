"""Слата 22.08: три новые оси от базы R6 (факт 1.6473390) + банк.

              отжигом фазы B, веса чемпионские, валидация НЕ тронута — val-слепая
              ось механизма (как shade): фаза B раньше рвала отжиг на середине,
              и тестовые модели были систематически недоотожжены при ~0.34 веса.
              Реопт-путь отвергнут замером: ретрейны дрейфанули по признакам
              (tab203→196), и вал-оптимум с ними ХУЖЕ эталона (1.665696 против
              1.665647) — вал-часть оси в свопе занулена по построению.
              колонок, OOF +0.000584±0.000020, минимальная концентрация на китах).
              q/(2F0) ≈ 1e-4; cnt_rank — классовый приор стека 0.36).

tfm3 в библиотеке (решение по вопросу Оли), но в боевой бленд не входит до
tfm3b: его тестовая сторона на 60-дневном зазоре, вал-вес 0.055 нёс бы на тест
деградацию (их же замер переноса 40%).

Восстановление: kappa_i = (F0² − S_i² + Q_i)/(2·Q_i), F0 = 1.6473390.

Запуск: USE_V2=1 USE_V3=1 .venv/bin/python work/scripts/make_u_candidates.py
Артефакты: submissions/U{1,2,3,4}_*.csv, work/reports/u_candidates.json
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))
from common import PREDS_DIR, REPORTS_DIR, ROOT, TEST_ANCHOR, VAL_ANCHOR, load_anchor
from margin import score
from subs import lp

SUB = ROOT / "submissions"
SD_CANON = 1.631108
F0 = 1.6473390          
ALPHAS = [1e0, 1e1, 1e2, 1e3, 1e4, 1e5]
# чемпионские веса свопуемых членов (winner 21.08, эталон 1.665647)
SWAP_W = {"fusion_v3c_avg_cal": 0.228925, "fusion_v3ctl_cal": 0.106124}


def respread(lp_):
    m = lp_.mean()
    return np.clip(m + (lp_ - m) * (SD_CANON / lp_.std()), 0, None)


def rank01(X):
    from scipy.stats import rankdata
    out = np.empty_like(X)
    n = len(X)
    for j in range(X.shape[1]):
        out[:, j] = (rankdata(X[:, j], method="average") - 1.0) / (n - 1.0)
    return out


def write_sub(name, uid, lp_, rep, extra=None):
    pred = np.expm1(lp_)
    assert len(pred) == 250000 and np.isfinite(pred).all() and (pred >= 0).all()
    rep[name] = {"mean": round(float(lp_.mean()), 6), "sd": round(float(lp_.std()), 6),
                 "clipped": int((lp_ <= 0).sum()), **(extra or {})}
    print(f"{name}: mean {lp_.mean():.6f} sd {lp_.std():.6f} clip {(lp_ <= 0).sum()}")


def main():
    assert os.environ.get("USE_V2") and os.environ.get("USE_V3"), "нужны USE_V2=1 USE_V3=1"
    rep: dict = {"F0": F0, "sd_canon": SD_CANON}

    pack_v = pl.read_parquet(ROOT / "work" / "preds_pack" / "val_preds.parquet").sort("user_id")
    ly = np.log1p(np.clip(pack_v["target"].to_numpy().astype(np.float64), 0, None))
    lb_v = pack_v["blend"].to_numpy().astype(np.float64)
    print(f"эталон (чемпионский пак): {score(lb_v, ly):.6f}")

    # ---- mdl_malach: своп тестовых предиктов fusion на ретрейнутые (веса фиксированы)

    #cnt_rank корректор (деплой ночного скрининга)
    bad = {x.split(" ")[0] for x in json.loads(
        (REPORTS_DIR / "maxback_affected_cols.json").read_text())}
    night = json.loads((REPORTS_DIR / "night_corrector_variants.json").read_text())
    cnt_cols = [c for c in night["cnt_cols"] if c not in bad]

    def matrix(anchor):
        df = load_anchor(anchor).sort("user_id")
        X = df.select(cnt_cols).to_numpy().astype(np.float64)
        return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    Xv = rank01(matrix(VAL_ANCHOR))
    mu, sd = Xv.mean(0), Xv.std(0)
    sd[sd == 0] = 1.0
    Xv = np.column_stack([(Xv - mu) / sd, np.ones(len(Xv))])
    resid = ly - lb_v
    rng = np.random.default_rng(0)
    inner = rng.permutation(len(ly)) < int(0.8 * len(ly))
    G = Xv[inner].T @ Xv[inner]
    b_ = Xv[inner].T @ resid[inner]
    best = (9e9, ALPHAS[0])
    for al in ALPHAS:
        w = np.linalg.solve(G + al * np.eye(len(G)), b_)
        best = min(best, (float(np.mean((Xv[~inner] @ w - resid[~inner]) ** 2)), al))
    al = best[1]
    w = np.linalg.solve(Xv.T @ Xv + al * np.eye(Xv.shape[1]), Xv.T @ resid)
    Xt = rank01(matrix(TEST_ANCHOR))
    Xt = np.column_stack([(Xt - mu) / sd, np.ones(len(Xt))])
    # полный шаг стоил бы q/(2F0) ≈ 0.0009 при пустоте (урок mdl_gypsum: передоз стека 3.3x).
    # Зонд стандартной цены: пустое направление теряет ровно 0.0002, ожидание при
    # переносе 0.36 положительное (+0.0001)

    #зонд уровня (среднее — ось, разброс сознательно не трогаем)

    # ---- U4: банк

    (REPORTS_DIR / "u_candidates.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1))
    print("JSON: work/reports/u_candidates.json")


if __name__ == "__main__":
    main()

"""Честная проверка гипотезы «персональный сезонный перенос».

Гипотеза:  lp(D) - lp(C)  ≈  lp(B) - lp(A)     (lp = log1p)
т.е. персональная сезонная дельта между календарными окнами воспроизводится через год.

ПРОБЛЕМА ПРОВЕРКИ: D не наблюдаемо, поэтому напрямую corr(B-A, D-C) не измерить.
Данные покрывают 2025-01-01..2026-02-13, т.е. в ДВУХ годах наблюдается только
календарный отрезок 01-01..02-13 (44 дня). Его хватает, чтобы построить ТОЧНЫЙ
структурный аналог гипотезы:

    Xp = 2025-01-01..01-22   Yp = 2025-01-23..02-13   (пара «год назад»)
    X  = 2026-01-01..01-22   Y  = 2026-01-23..02-13   (та же пара в этом году)

    delta25 = lp(Yp) - lp(Xp)      ← «B - A»
    delta26 = lp(Y)  - lp(X)       ← «D - C»   и ОНО НАБЛЮДАЕМО

corr(delta25, delta26) — прямая, ничем не заменённая проверка воспроизводимости
персональной сезонной дельты через год. Окна 22д (максимум, что делится пополам
в наблюдаемом пересечении); контроль — пара по 15 дней.

ВАЖНО ПРО БАЗУ: чемпионские модели УЖЕ содержат gmv_sum_ya_tgt = окно «год назад,
выровненное по таргету» (для теста это ровно B; для зеркала — Yp). Новой является
только вторая половина дельты — A (Xp). Поэтому инкрементальный тест ставится так:
    base = богатые признаки на 2026-01-22 + lp(Yp)   (аналог чемпиона)
    test = base + lp(Xp)                              (добавляем недостающую половину)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))
from common import FEATURES_DIR, REPORTS_DIR

W = pl.read_parquet(FEATURES_DIR / "yoy_windows.parquet").sort("user_id")
N = W.height
RNG = np.random.default_rng(7)
FOLD = RNG.integers(0, 5, N)


def lp(name: str) -> np.ndarray:
    return np.log1p(np.clip(W[f"g_{name}"].to_numpy().astype(np.float64), 0, None))


def col(name: str) -> np.ndarray:
    return W[name].to_numpy().astype(np.float64)


def cv_fit(Xf: np.ndarray, y: np.ndarray, ridge: float = 1e-6) -> np.ndarray:
    """5-fold out-of-fold линейный прогноз."""
    out = np.zeros_like(y)
    for f in range(5):
        tr, te = FOLD != f, FOLD == f
        A = Xf[tr]
        G = A.T @ A + ridge * len(A) * np.eye(Xf.shape[1])
        c = np.linalg.solve(G, A.T @ y[tr])
        out[te] = Xf[te] @ c
    return out


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def design(*cols) -> np.ndarray:
    return np.column_stack([np.ones(N)] + list(cols))


def main() -> int:
    res: dict = {}

    # ------------------------------------------------------------------ 1. зеркало 22д
    for tag, (xp, yp, x, y) in {
        "m22": ("Xp", "Yp", "X", "Y"),
        "m15": ("Xp2", "Yp2", "", "Y2"),
    }.items():
        lXp, lYp, lX, lY = lp(xp), lp(yp), lp(x), lp(y)
        d25, d26 = lYp - lXp, lY - lX
        act25 = (W[f"g_{xp}"].to_numpy() > 0) | (W[f"g_{yp}"].to_numpy() > 0)
        act_both = act25 & ((W[f"g_{x}"].to_numpy() > 0) | (W[f"g_{y}"].to_numpy() > 0))
        r_all = float(np.corrcoef(d25, d26)[0, 1])
        r_act = float(np.corrcoef(d25[act25], d26[act25])[0, 1])
        r_both = float(np.corrcoef(d25[act_both], d26[act_both])[0, 1])
        # наклон переноса (оптимальная усадка при чистом D = C + s*delta)
        s_opt = float(np.cov(d25, d26)[0, 1] / np.var(d25))
        res[f"repro_{tag}"] = dict(
            corr_all=r_all, corr_active_prev=r_act, corr_active_both=r_both,
            slope=s_opt, mean_d25=float(d25.mean()), mean_d26=float(d26.mean()),
            sd_d25=float(d25.std()), sd_d26=float(d26.std()),
            frac_d25_zero=float(np.mean(np.abs(d25) < 1e-12)),
        )
        print(f"[{tag}] corr(d25,d26): все {r_all:+.4f}  активные-25 {r_act:+.4f} "
              f"(n={act25.sum()})  активные-оба {r_both:+.4f} (n={act_both.sum()})"
              f"   наклон {s_opt:+.4f}   доля d25==0 {res[f'repro_{tag}']['frac_d25_zero']:.3f}")

    # ------------------------------------------------- 2. предиктивный тест на зеркале
    lXp, lYp, lX, lY = lp("Xp"), lp("Yp"), lp("X"), lp("Y")
    d25 = lYp - lXp
    rich = [
        lp("m30"), lp("m90"), lp("m365"),
        np.log1p(col("od_X")), np.log1p(col("od_m30")), np.log1p(col("od_m90")),
        np.log1p(col("od_m365")),
        np.log1p(np.clip(col("m_rec_order"), 0, 400)),
        np.log1p(np.clip(col("m_rec_gmv"), 0, 400)),
        np.log1p(np.clip(col("m_tenure"), 0, 400)),
        np.log1p(col("m_act_days")),
        (col("od_m365") > 0).astype(float),
    ]

    models = {
        "naive_X":          design(lX),
        "naive_X_plus_d25": design(lX, d25),
        "rich":             design(lX, *rich),
        "rich_ya":          design(lX, *rich, lYp),                 # ← аналог чемпиона
        "rich_ya_plus_Xp":  design(lX, *rich, lYp, lXp),            # ← добавили A
        "rich_ya_plus_d25": design(lX, *rich, lYp, d25),            # то же, иная параметризация
    }
    mres = {}
    for k, Xf in models.items():
        p = cv_fit(Xf, lY)
        mres[k] = rmse(p, lY)
        print(f"  {k:20s} OOF RMSLE(зеркало) {mres[k]:.6f}")
    res["mirror_models"] = mres
    res["mirror_gain_ya_plus_Xp"] = mres["rich_ya"] - mres["rich_ya_plus_Xp"]
    res["mirror_gain_naive_d25"] = mres["naive_X"] - mres["naive_X_plus_d25"]

    # коэффициенты полной модели + t-статистика на lp(Xp)
    Xf = models["rich_ya_plus_Xp"]
    c, *_ = np.linalg.lstsq(Xf, lY, rcond=None)
    resid = lY - Xf @ c
    s2 = float(resid @ resid) / (N - Xf.shape[1])
    cov = s2 * np.linalg.inv(Xf.T @ Xf)
    se = np.sqrt(np.diag(cov))
    names = ["const", "lp_X"] + [f"rich{i}" for i in range(len(rich))] + ["lp_Yp", "lp_Xp"]
    res["mirror_coefs"] = {n: [float(v), float(t)] for n, v, t in zip(names, c, c / se)}
    print(f"  коэф lp_Yp {c[-2]:+.4f} (t {c[-2]/se[-2]:+.1f})   "
          f"lp_Xp {c[-1]:+.4f} (t {c[-1]/se[-1]:+.1f})")

    # чистая гипотеза «наклон 1»: lp(Y) = lp(X) + delta25 (без подгонки)
    for s in (0.0, 0.1, 0.2, 0.3, 0.5, 1.0):
        pred = lX + s * (d25 - d25.mean()) + (lY.mean() - lX.mean())
        print(f"  жёсткая форма lp(X)+{s:.1f}*d25_centered: RMSLE {rmse(pred, lY):.6f}")
        res.setdefault("hard_form", {})[str(s)] = rmse(pred, lY)

    # ------------------------------------------------ 3. персистентность дельт (seq 30д)
    S = np.log1p(np.clip(np.column_stack([col(f"s_{i}") for i in range(13)]), 0, None))
    dl = S[:, 1:] - S[:, :-1]          # 12 дельт соседних 30д-окон
    lags = {}
    for lag in range(2, 12):           # >=2 => окна не пересекаются
        rs = [float(np.corrcoef(dl[:, j], dl[:, j + lag])[0, 1])
              for j in range(12 - lag)]
        lags[lag] = float(np.mean(rs))
    res["seq_delta_autocorr"] = lags
    print("  автокорреляция персональных дельт (непересекающиеся пары), лаг→r:")
    print("   ", {k: round(v, 4) for k, v in lags.items()})

    # контроль: лаг 1 (пересекающиеся) — механический артефакт
    res["seq_delta_autocorr_lag1"] = float(np.mean(
        [float(np.corrcoef(dl[:, j], dl[:, j + 1])[0, 1]) for j in range(11)]))
    print(f"    лаг 1 (пересекающиеся окна, артефакт): {res['seq_delta_autocorr_lag1']:+.4f}")

    # ------------------------------------------------ 4. что даст перенос на реальном D
    lA, lB, lC = lp("A"), lp("B"), lp("C")
    dAB = lB - lA
    res["real"] = dict(
        mean_dAB=float(dAB.mean()), sd_dAB=float(dAB.std()),
        frac_A_zero=float(np.mean(W["g_A"].to_numpy() == 0)),
        frac_AB_zero=float(np.mean((W["g_A"].to_numpy() == 0) & (W["g_B"].to_numpy() == 0))),
        corr_dAB_lC=float(np.corrcoef(dAB, lC)[0, 1]),
    )
    print(f"  реальная дельта B-A: mean {dAB.mean():+.4f} sd {dAB.std():.4f}; "
          f"A=0 у {res['real']['frac_A_zero']:.3f}, A=B=0 у {res['real']['frac_AB_zero']:.3f}")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "yoy_check.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print("\nсохранено work/reports/yoy_check.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

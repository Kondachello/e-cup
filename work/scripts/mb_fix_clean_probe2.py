"""Устойчивость честного замера: три сида + проверка, что зеркало бьёт по признакам
не слабее, чем реальный дефект бьёт по тесту.

Первый прогон дал цену обрезки +0.000037 при пороге 0.0003. Число настолько мало,
что обязано быть проверено на разбросе по сидам, иначе это «один замер».
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))
from calibrate import apply_shifts, fit_shifts
from common import FEATURES_DIR, REPORTS_DIR, TEST_ANCHOR, VAL_ANCHOR, rmsle
from mb_fix_clean_probe import anchor_frame, cols_of, mat

SEEDS = (7, 42, 1337)
BINS = 24

AFFECTED = ["tenure", "active_days_full", "gmv_sum_full", "log_gmv_sum_full",
            "act_density", "act_gap_mean", "act_gap_std", "ord_gap_mean",
            "ord_gap_std", "ord_days_full", "rec_over_gap"]


def perturbation(anchor: str, sfx: str) -> None:
    a = pl.read_parquet(FEATURES_DIR / f"anchor={anchor}.parquet").sort("user_id")
    b = pl.read_parquet(FEATURES_DIR / f"anchor={anchor}{sfx}.parquet").sort("user_id")
    print(f"\n  якорь {anchor} против {sfx}:")
    for c in ("tenure", "gmv_sum_full", "active_days_full"):
        x, y = a[c].to_numpy().astype(np.float64), b[c].to_numpy().astype(np.float64)
        m = ~(np.isnan(x) | np.isnan(y))
        chg = float((np.abs(x[m] - y[m]) > 1e-6).mean())
        sp = np.corrcoef(np.argsort(np.argsort(x[m])), np.argsort(np.argsort(y[m])))[0, 1]
        print(f"    {c:18s} изменилось у {chg:.4f}, среднее {x[m].mean():9.3f} -> "
              f"{y[m].mean():9.3f}, ранговая корр {sp:.5f}")


def main() -> None:
    import lightgbm as lgb
    from mb_fix_clean_probe import GAP_DAYS, N_TREES  # noqa: F401
    from datetime import timedelta
    from common import HORIZON, train_anchors

    print("СИЛА ВОЗМУЩЕНИЯ ПРИЗНАКОВ (зеркало обязано бить не слабее реального дефекта)")
    perturbation(TEST_ANCHOR.isoformat(), ".mb409")   # реальный дефект на тесте
    perturbation(VAL_ANCHOR.isoformat(), ".mb349")    # зеркало на валидации

    cols = cols_of()
    last_ok = VAL_ANCHOR + timedelta(days=1) - timedelta(days=GAP_DAYS + HORIZON)
    use = [a for a in train_anchors(14) if a <= last_ok][-6:]
    Xs, ys = [], []
    for a in use:
        d = anchor_frame(a.isoformat())
        Xs.append(mat(d, cols)); ys.append(d["target"].to_numpy().astype(np.float64))
        del d
    X = np.vstack(Xs); y = np.log1p(np.concatenate(ys)); del Xs, ys

    va = VAL_ANCHOR.isoformat()
    full, cut = anchor_frame(va), anchor_frame(va, cut=True)
    yv = full["target"].to_numpy().astype(np.float64)
    ly = np.log1p(yv)
    Xf, Xc = mat(full, cols), mat(cut, cols)
    del full, cut
    n = len(yv)
    half = np.random.default_rng(0).permutation(n) < n // 2
    ev = ~half

    print(f"\nЦЕНА ОБРЕЗКИ ВНЕ ВЫБОРКИ, {len(SEEDS)} сида, обучение на {len(use)} якорях")
    print(f"{'сид':>6s} {'сырой FULL':>11s} {'сырой CUT':>11s} {'калибр FULL':>12s} "
          f"{'калибр CUT':>12s} {'ЦЕНА':>10s} {'sd сдвига':>10s}")
    costs = []
    for s in SEEDS:
        t0 = time.time()
        m = lgb.LGBMRegressor(
            objective="tweedie", tweedie_variance_power=1.45, n_estimators=N_TREES,
            learning_rate=0.05, num_leaves=255, min_child_samples=300,
            subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
            n_jobs=3, verbose=-1, random_state=s).fit(X, y)
        lf_ = np.clip(m.predict(Xf), 0, None)
        lc_ = np.clip(m.predict(Xc), 0, None)
        cf, sf = fit_shifts(lf_[half], ly[half], BINS)
        cc, sc = fit_shifts(lc_[half], ly[half], BINS)
        cal_f = rmsle(yv[ev], np.expm1(apply_shifts(lf_[ev], cf, sf)))
        cal_c = rmsle(yv[ev], np.expm1(apply_shifts(lc_[ev], cc, sc)))
        costs.append(cal_c - cal_f)
        print(f"{s:6d} {rmsle(yv[ev], np.expm1(lf_[ev])):11.6f} "
              f"{rmsle(yv[ev], np.expm1(lc_[ev])):11.6f} {cal_f:12.6f} {cal_c:12.6f} "
              f"{cal_c-cal_f:+10.6f} {(lc_-lf_).std():10.5f}  ({time.time()-t0:.0f}s)")
    c = np.array(costs)
    print(f"\nЦЕНА среднее {c.mean():+.6f}, sd по сидам {c.std(ddof=1):+.6f}, "
          f"диапазон [{c.min():+.6f}, {c.max():+.6f}]")
    print(f"порог проекта 0.0003 -> отношение {c.mean()/0.0003:.2f}; "
          f"шум замера LB 0.000022")
    np.save(REPORTS_DIR / "mb_fix_clean_costs.npy", c)


if __name__ == "__main__":
    main()

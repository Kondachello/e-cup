"""Ещё пять сидов к честному замеру цены обрезки: 0.000355 +- 0.000278 по трём сидам
стоит ровно на пороге 0.0003, а решение на пороге требует меньшей погрешности.
"""
from __future__ import annotations

import sys
import time
from datetime import timedelta
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from calibrate import apply_shifts, fit_shifts
from common import HORIZON, REPORTS_DIR, VAL_ANCHOR, rmsle, train_anchors
from mb_fix_clean_probe import GAP_DAYS, N_TREES, anchor_frame, cols_of, mat

SEEDS = (1, 2, 3, 5, 11)
BINS = 24
PREV = np.array([0.000037, 0.000554, 0.000476])   # сиды 7 / 42 / 1337


def main() -> None:
    import lightgbm as lgb

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
        print(f"сид {s:5d}: FULL {cal_f:.6f} CUT {cal_c:.6f} ЦЕНА {cal_c-cal_f:+.6f} "
              f"({time.time()-t0:.0f}s)", flush=True)

    allc = np.concatenate([PREV, np.array(costs)])
    se = allc.std(ddof=1) / np.sqrt(len(allc))
    print(f"\nВСЕ {len(allc)} сидов: среднее {allc.mean():+.6f}, sd {allc.std(ddof=1):.6f}, "
          f"SE {se:.6f}, t={allc.mean()/se:.2f}")
    print(f"95% интервал примерно [{allc.mean()-2*se:+.6f}, {allc.mean()+2*se:+.6f}]")
    print(f"порог проекта 0.0003 -> отношение {allc.mean()/0.0003:.2f}")
    np.save(REPORTS_DIR / "mb_fix_clean_costs_all.npy", allc)


if __name__ == "__main__":
    main()

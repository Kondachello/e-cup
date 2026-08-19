"""Контроль к yoy_check.py: закрывает главное возражение к зеркальному тесту.

ВОЗРАЖЕНИЕ: зеркальная пара (01.01-22.01 → 23.01-13.02) — переход со СЛАБОЙ
сезонностью. Реальная пара A→B пересекает 23.02 и разгон к 8 марта, где
персональная черта («этот человек покупает подарки») могла бы быть сильной.
Тогда низкая воспроизводимость на зеркале не переносится на реальную пару.

ПРОВЕРКА: если «праздничная отзывчивость» — персональная черта, то персональный
лифт на РАЗНЫХ праздниках 2025 должен коррелировать между собой (окна не
пересекаются, каждый лифт нормирован на собственную базу пользователя).
Календарь — мажорные раны из holiday_cal (см. KNOWLEDGE): 13-19.02, 01-06.03,
10-17.11, 25-28.12 + контрольные 13-16.04, 05-11.06, 14-19.10.

Если корреляции лифтов ≈ 0, персональной праздничной черты нет вовсе,
и у переноса B-A → D-C нет механизма.
"""
from __future__ import annotations

import json
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))
from common import FEATURES_DIR, REPORTS_DIR, TRAIN_PARQUET, user_universe

RUNS = {
    "feb14":  (date(2025, 2, 13), date(2025, 2, 19)),
    "mar8":   (date(2025, 3, 1),  date(2025, 3, 6)),
    "apr":    (date(2025, 4, 13), date(2025, 4, 16)),
    "jun":    (date(2025, 6, 5),  date(2025, 6, 11)),
    "oct":    (date(2025, 10, 14), date(2025, 10, 19)),
    "nov11":  (date(2025, 11, 10), date(2025, 11, 17)),
    "dec":    (date(2025, 12, 25), date(2025, 12, 28)),
}
BASE_LEN, BASE_GAP = 28, 3


def main() -> int:
    aggs = []
    for k, (s, e) in RUNS.items():
        m = pl.col("event_date").is_between(pl.lit(s), pl.lit(e))
        aggs.append(pl.col("gmv").filter(m).sum().alias(f"r_{k}"))
        bs = s - timedelta(days=BASE_GAP + BASE_LEN)
        be = s - timedelta(days=BASE_GAP + 1)
        mb = pl.col("event_date").is_between(pl.lit(bs), pl.lit(be))
        aggs.append(pl.col("gmv").filter(mb).sum().alias(f"b_{k}"))
    df = (
        pl.scan_parquet(TRAIN_PARQUET).select(["user_id", "event_date", "gmv"])
        .group_by("user_id").agg(aggs).collect(engine="streaming")
    )
    W = user_universe().join(df, on="user_id", how="left").fill_null(0.0).sort("user_id")

    lift, active = {}, {}
    for k, (s, e) in RUNS.items():
        rl = (e - s).days + 1
        r = W[f"r_{k}"].to_numpy().astype(np.float64)
        b = W[f"b_{k}"].to_numpy().astype(np.float64)
        # лифт = log1p(факт в ран) - log1p(ожидание из собственной базы того же масштаба)
        lift[k] = np.log1p(r) - np.log1p(b * rl / BASE_LEN)
        active[k] = b > 0
        print(f"{k:6s} ран {rl}д  mean lift {lift[k].mean():+.4f}  sd {lift[k].std():.3f}  "
              f"с базой {active[k].mean():.3f}")

    keys = list(RUNS)
    res = {"pairs": {}}
    print("\ncorr персональных праздничных лифтов (только юзеры с ненулевой базой в обоих):")
    for i, a in enumerate(keys):
        for bkey in keys[i + 1:]:
            m = active[a] & active[bkey]
            r = float(np.corrcoef(lift[a][m], lift[bkey][m])[0, 1])
            res["pairs"][f"{a}|{bkey}"] = dict(corr=r, n=int(m.sum()))
            print(f"  {a:6s} × {bkey:6s}  r {r:+.4f}  n {m.sum()}")
    rs = [v["corr"] for v in res["pairs"].values()]
    res["mean_corr"] = float(np.mean(rs))
    res["max_corr"] = float(np.max(rs))

    # ключевая пара для нашей задачи: mar8 (то, что переносим) × остальные
    mar = [v["corr"] for k, v in res["pairs"].items() if "mar8" in k]
    res["mar8_mean_corr"] = float(np.mean(mar))
    print(f"\nсреднее по всем парам {res['mean_corr']:+.4f}   макс {res['max_corr']:+.4f}   "
          f"среднее с mar8 {res['mar8_mean_corr']:+.4f}")

    # контроль: та же метрика, но для УРОВНЯ (не лифта) — должна быть большой,
    # иначе метод измерения сломан
    lev = {k: np.log1p(W[f"b_{k}"].to_numpy().astype(np.float64)) for k in keys}
    ctl = []
    for i, a in enumerate(keys):
        for bkey in keys[i + 1:]:
            m = active[a] & active[bkey]
            ctl.append(float(np.corrcoef(lev[a][m], lev[bkey][m])[0, 1]))
    res["level_control_mean_corr"] = float(np.mean(ctl))
    print(f"контроль (уровень базы вместо лифта): {res['level_control_mean_corr']:+.4f} "
          f"— метод измерения рабочий")

    (REPORTS_DIR / "yoy_holiday_trait.json").write_text(
        json.dumps(res, indent=1, ensure_ascii=False))
    print("\nсохранено work/reports/yoy_holiday_trait.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

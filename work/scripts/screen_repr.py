"""Скрининг представлений: что объясняет остаток бленда, не обучая моделей.

Логика. Доказано тождество: вклад модели в бленд равен примерно 7.1*delta^2, где
delta — доля модели вне линейной оболочки бленда, и остаток бленда не предсказуем из
наших 203 агрегатных признаков. Значит помочь может только представление данных,
которое НЕ является функцией этого набора.

Проверить это можно без обучения: построить представление, регрессировать на него
остаток бленда (подбор на половине пользователей, замер на другой) и сравнить с
плацебо той же размерности. Положительный R^2 вне выборки при отрицательном плацебо
означает сигнал вне оболочки.

Перевод в метрику: выигрыш = sb - sb*sqrt(1-R^2).

Запуск: POLARS_MAX_THREADS=3 .venv/bin/python work/scripts/screen_repr.py
"""
from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))
from common import PREDS_DIR, VAL_ANCHOR, load_anchor
from err_corr import BLEND
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import Ridge

TRAIN = Path("/Users/alexanderkondakov/ozon-cup/train.parquet")
DATA_START = date(2025, 1, 1)


def blend_residual():
    val = load_anchor(VAL_ANCHOR, columns=["user_id", "target"]).sort("user_id")
    uid = val["user_id"].to_numpy()
    y = np.log1p(val["target"].to_numpy().astype(np.float64))

    def lp(n):
        d = pl.read_parquet(PREDS_DIR / f"{n}_val.parquet").sort("user_id")
        return np.log1p(np.clip(d["pred"].to_numpy().astype(np.float64), 0, None))

    b = sum(w * lp(n) for n, w in BLEND.items())
    return uid, y - b


def dense(rows, cols, vals, n_rows, n_cols, idx):
    M = np.zeros((n_rows, n_cols), dtype=np.float32)
    keep = np.fromiter((u in idx for u in rows), bool, len(rows))
    M[[idx[u] for u in rows[keep]], cols[keep]] = vals[keep]
    return M


def main():
    uid, e = blend_residual()
    idx = {u: i for i, u in enumerate(uid)}
    n = len(uid)
    sb = float(np.sqrt((e ** 2).mean()))
    rng = np.random.default_rng(0)
    m = rng.random(n) < 0.5
    print(f"остаток бленда: sb={sb:.6f}, юзеров {n}\n")

    lf = pl.scan_parquet(TRAIN).filter(pl.col("event_date") <= VAL_ANCHOR)

    def r2(F: np.ndarray) -> float:
        F = np.nan_to_num(F, nan=0.0, posinf=0.0, neginf=0.0)
        F = (F - F.mean(0)) / (F.std(0) + 1e-9)
        p = Ridge(alpha=10.0).fit(F[m], e[m]).predict(F[~m])
        return float(1 - ((e[~m] - p) ** 2).sum() / ((e[~m] - e[~m].mean()) ** 2).sum())

    reps: dict[str, np.ndarray] = {}

    # A. сырые дневные значения за последние 28 дней (не агрегат, а сам профиль)
    d28 = (lf.filter(pl.col("event_date") > VAL_ANCHOR - timedelta(days=28))
             .with_columns(((pl.lit(VAL_ANCHOR) - pl.col("event_date")).dt.total_days() - 1)
                           .alias("dd"))
             .group_by(["user_id", "dd"]).agg(pl.col("gmv").sum().alias("g"))
             .collect(engine="streaming"))
    reps["сырые 28 дней GMV"] = dense(d28["user_id"].to_numpy(), d28["dd"].to_numpy(),
                                      np.log1p(d28["g"].to_numpy().astype(float)), n, 28, idx)

    # B. профиль по дням недели: доля активности и денег по каждому дню
    dow = (lf.with_columns(pl.col("event_date").dt.weekday().alias("wd"))
             .group_by(["user_id", "wd"])
             .agg([pl.col("gmv").sum().alias("g"), pl.len().alias("c")])
             .collect(engine="streaming"))
    wd = dow["wd"].to_numpy() - 1
    G = dense(dow["user_id"].to_numpy(), wd, np.log1p(dow["g"].to_numpy().astype(float)), n, 7, idx)
    C = dense(dow["user_id"].to_numpy(), wd, dow["c"].to_numpy().astype(float), n, 7, idx)
    reps["профиль по дням недели"] = np.hstack([G / (G.sum(1, keepdims=True) + 1e-6),
                                                C / (C.sum(1, keepdims=True) + 1e-6)])

    # C и D. разложения недельных матриц по разным величинам
    wk = (lf.with_columns(((pl.col("event_date") - pl.lit(DATA_START)).dt.total_days() // 7)
                          .alias("wk"))
            .group_by(["user_id", "wk"])
            .agg([pl.col("gmv").sum().alias("g"), pl.col("to_ord").sum().alias("o"),
                  pl.col("searches").sum().alias("s")])
            .collect(engine="streaming"))
    nw = int(wk["wk"].max()) + 1
    ui, wi = wk["user_id"].to_numpy(), wk["wk"].to_numpy()
    for tag, col, fn in [("GMV", "g", np.log1p), ("заказы", "o", np.log1p),
                         ("поиски", "s", np.log1p)]:
        M = dense(ui, wi, fn(wk[col].to_numpy().astype(float)), n, nw, idx)
        reps[f"разложение недель: {tag}"] = TruncatedSVD(32, random_state=0).fit_transform(M)

    print(f"{'представление':30s} {'разм':>5s} {'mdl_flint вне выборки':>15s} {'плацебо':>10s} {'выигрыш':>10s}")
    for name, F in reps.items():
        R = r2(F)
        P = r2(rng.normal(size=(n, F.shape[1])))
        gain = sb - sb * np.sqrt(max(1 - max(R, 0.0), 0.0))
        flag = "  <-- вне оболочки" if R > 0 and R > P + 0.0005 else ""
        print(f"{name:30s} {F.shape[1]:5d} {R:15.6f} {P:10.6f} {gain:10.6f}{flag}")

    # всё вместе: сколько дают представления совместно
    allF = np.hstack(list(reps.values()))
    R = r2(allF)
    print(f"\n{'ВСЕ ВМЕСТЕ':30s} {allF.shape[1]:5d} {R:15.6f} "
          f"{'':10s} {sb - sb*np.sqrt(max(1-max(R,0.0),0.0)):10.6f}")


if __name__ == "__main__":
    main()

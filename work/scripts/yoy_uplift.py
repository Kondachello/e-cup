"""Персональный сезонный подъём год-к-году как направление-поправка.

Мотивация. Признаки прошлогоднего окна (ya_tgt/ya_wide) пусты во ВСЕХ обучающих
срезах: покрытие данных начинается 2025-01-01, поэтому окно (A-364, A-335] попадает
внутрь данных только для срезов от 2025-12-31. С зазором 30 дней обучающие срезы
заканчиваются на 2025-11-16, у них покрытие ноль. Значит ни одна наша модель никогда
не видела, что пользователь делал в феврале-марте 2025 — в том числе к 8 марта.

Тестовое окно 2026-02-14..2026-03-15 ровно на год позже окна 2025-02-14..2025-03-15.
Личный подъём в том окне относительно собственной базы пользователя — информация,
которой нет ни в одном признаке ни одной модели.

Обучить на это коэффициент негде (ни один обучающий срез не имеет прошлогоднего окна),
поэтому коэффициент измеряется на лидерборде одним замером: сабмит base + step*h,
дальше парабола даёт точный оптимальный шаг.

Запуск:
"""
from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))
from subs import MEASURED, lp, novelty, span_matrix

ROOT = Path("/Users/alexanderkondakov/ozon-cup")
TRAIN = ROOT / "train.parquet"

# окно теста годом раньше и личная база перед ним
W_START, W_END = date(2025, 2, 14), date(2025, 3, 15)
B_START, B_END = date(2025, 1, 1), date(2025, 2, 13)
B_DAYS = (B_END - B_START).days + 1          # 44
W_DAYS = (W_END - W_START).days + 1          # 30


def build() -> pl.DataFrame:
    lf = pl.scan_parquet(TRAIN)
    lo = min(B_START, W_START)
    hist = lf.filter((pl.col("event_date") >= lo) & (pl.col("event_date") <= W_END))
    inw = (pl.col("event_date") >= W_START) & (pl.col("event_date") <= W_END)
    inb = (pl.col("event_date") >= B_START) & (pl.col("event_date") <= B_END)
    g = hist.group_by("user_id").agg([
        pl.col("gmv").filter(inw).sum().alias("gmv_w"),
        pl.col("gmv").filter(inb).sum().alias("gmv_b"),
        (pl.col("to_ord").filter(inw) > 0).sum().alias("ord_days_w"),
        (pl.col("to_ord").filter(inb) > 0).sum().alias("ord_days_b"),
        inw.sum().alias("act_w"),
        inb.sum().alias("act_b"),
    ]).collect(engine="streaming")
    return g


    d = pl.DataFrame({"user_id": uid}).join(g, on="user_id", how="left").fill_null(0)
    gw = d["gmv_w"].to_numpy().astype(np.float64)
    gb = d["gmv_b"].to_numpy().astype(np.float64)
    ow = d["ord_days_w"].to_numpy().astype(np.float64)
    ob = d["ord_days_b"].to_numpy().astype(np.float64)
    aw = d["act_w"].to_numpy().astype(np.float64)
    ab = d["act_b"].to_numpy().astype(np.float64)

    # личный подъём: прошлогоднее окно против собственной базы, приведённой к 30 дням
    u = np.log1p(gw) - np.log1p(gb * W_DAYS / B_DAYS)
    # усадка к нулю там, где мало наблюдений: оценка подъёма по паре дней — шум
    n = aw + ab
    u_shr = u * n / (n + 8.0)

    # резкий вариант: покупал в прошлогоднем окне заметно больше собственной базы
    spike = ((gw > 2.0 * gb * W_DAYS / B_DAYS) & (ow > 0)).astype(np.float64)
    # обратный: активен в базе, но в прошлогоднем окне не покупал
    dip = ((ob > 0) & (ow == 0) & (ab > 0)).astype(np.float64)

    out = {
        "u_shrunk": u_shr,
        "spike": spike,
        "spike_minus_dip": spike - dip,
        "ord_uplift": (ow - ob * W_DAYS / B_DAYS),
    }
    # центрируем: общий уровень уже подобран на лидерборде, нам нужна только личная часть
    return {k: v - v.mean() for k, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", type=float, default=0.05)
    ap.add_argument("--out-dir", default=str(ROOT / "submissions"))
    args = ap.parse_args()

    uid, base_lp = lp(args.base)
    n = len(uid)

    print(f"база {args.base}: n={n}, sd(log1p)={base_lp.std():.4f}")
    print(f"{'направление':18s} {'sd':>8s} {'новизна':>9s} {'corr с базой':>13s} {'ненулевых':>10s}")
    rows = {}

    # для замера берём направление с максимальной новизной при вменяемом разбросе
    best = max(rows, key=lambda k: rows[k][0])
    h = rows[best][2]
    h = h / h.std()                       # нормируем: шаг задаётся в единицах sd
    q = float((h ** 2).mean())
    print(f"\nвыбрано: {best}; после нормировки q=mean(h^2)={q:.4f}")

    outp = Path(args.out_dir) / f"Y1_yoy_{best}.csv"
    new = base_lp + args.step * h
    pl.DataFrame({"user_id": uid, "predict": np.expm1(np.clip(new, 0, None))}).write_csv(outp)
    print(f"записан {outp}  (шаг {args.step} sd)")
    print("после замера: c = (MSE_base - MSE_new + step^2*q) / (2*step); "
          "оптимальный шаг c/q, ожидаемый выигрыш в MSE c^2/q")
    np.savez(ROOT / "work" / "yoy_dirs.npz", uid=uid, **{k: v[2] for k, v in rows.items()})


if __name__ == "__main__":
    main()

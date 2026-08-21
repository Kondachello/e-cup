"""R1. Точная календарная линза для викенд-оси.

Прошлая оценка (1.5 сигмы) считалась грубо: доля gmv по дням недели, умноженная
на разность счётчиков. Здесь честная линза: для каждого юзера строим его дневной
профиль по дням недели, затем ОЖИДАЕМУЮ сумму окна как сумму по фактическим датам
окна, и берём разность ожидаемых сумм test-окна и val-окна. Это ровно тот эффект,
который даёт замена состава дней, без примеси уровня.
"""
import os
import numpy as np, polars as pl
from datetime import date, timedelta

df = pl.read_parquet("train.parquet", columns=["user_id", "event_date", "gmv"])
uid = np.sort(df["user_id"].unique().to_numpy())
F0, NPUB, NOISE = 1.666395, 50_000, 0.000022


def days(s, e):
    return [s + timedelta(days=i) for i in range((e - s).days + 1)]


def profile(anchor: date, look: int = 364):
    """средний дневной gmv юзера по дню недели, по истории до якоря"""
    h = df.filter(pl.col("event_date").is_between(anchor - timedelta(days=look - 1), anchor))
    h = h.with_columns(pl.col("event_date").dt.weekday().alias("w"))
    g = h.group_by(["user_id", "w"]).agg(pl.col("gmv").sum().alias("s"))
    n = {}
    for d in days(anchor - timedelta(days=look - 1), anchor):
        n[d.isoweekday()] = n.get(d.isoweekday(), 0) + 1
    g = g.with_columns(pl.col("w").replace_strict(n, default=1).alias("cnt"))
    g = g.with_columns((pl.col("s") / pl.col("cnt")).alias("m"))
    piv = g.pivot(on="w", index="user_id", values="m").fill_null(0.0)
    piv = piv.with_columns(pl.col("user_id").cast(pl.Int64))
    piv = pl.DataFrame({"user_id": uid}).join(piv, on="user_id", how="left").fill_null(0.0)
    cols = [c for c in piv.columns if c != "user_id"]
    M = np.zeros((len(uid), 7))
    for c in cols:
        M[:, int(c) - 1] = piv[c].to_numpy()
    return M                                   # [юзер, день недели] средний дневной gmv


def wcount(s, e):
    c = np.zeros(7)
    for d in days(s, e):
        c[d.isoweekday() - 1] += 1
    return c


def expected(M, s, e):
    return M @ wcount(s, e)                    # ожидаемая сумма окна по профилю


def lp(s, e):
    w = df.filter(pl.col("event_date").is_between(s, e)).group_by("user_id").agg(
        pl.col("gmv").sum().alias("y"))
    y = pl.DataFrame({"user_id": uid}).join(w, on="user_id", how="left")["y"].to_numpy().astype(float)
    return np.log1p(np.nan_to_num(y))


# --- проверка на 2025: якорь 14.01.2025, но истории там всего 14 дней ---
# поэтому берём ВТОРУЮ проверку на якоре с полной историей: 14.01.2026 не годится
# (там таргет известен только для val). Используем пары 2025 года с полной историей.
print("Линза проверяется на парах окон с ПОЛНОЙ историей профиля (>=180 дней).\n")
PAIRS = [(date(2025, 9, 14), date(2025, 9, 15), date(2025, 10, 14), date(2025, 10, 15), date(2025, 11, 13)),
         (date(2025, 10, 14), date(2025, 10, 15), date(2025, 11, 13), date(2025, 11, 14), date(2025, 12, 13)),
         (date(2025, 11, 14), date(2025, 11, 15), date(2025, 12, 14), date(2025, 12, 15), date(2026, 1, 13))]
print(f"{'якорь':12s} {'выходных val->test':>20} {'выигрыш':>11} {'сигм':>7}")
for A, s1, e1, s2, e2 in PAIRS:
    M = profile(A, look=min(364, (A - date(2025, 1, 1)).days + 1))
    h = expected(M, s2, e2) - expected(M, s1, e1)
    h = np.sign(h) * np.log1p(np.abs(h))
    h = h - h.mean()
    a = lp(s1, e1); b = lp(s2, e2)
    e = (a + float((b - a).mean())) - b                    # остаток после глобального сдвига
    hh = float(np.mean(h * h))
    c = float(np.mean(e * h))
    gain = c * c / (hh * 2 * F0)
    z = abs(c) / max(float(np.std(e * h)) / np.sqrt(NPUB), 1e-12)
    wv, wt = wcount(s1, e1)[5:].sum(), wcount(s2, e2)[5:].sum()
    print(f"{A}   {int(wv):>2} -> {int(wt):<2}          {gain:>11.6f} {z:>7.1f}")

print(f"\n=== ЦЕЛЕВАЯ ПАРА: val 15.01-13.02.2026 -> test 14.02-15.03.2026 ===")
A = date(2026, 1, 14)
M = profile(A)
h = expected(M, date(2026, 2, 14), date(2026, 3, 15)) - expected(M, date(2026, 1, 15), date(2026, 2, 13))
print(f"  выходных: {int(wcount(date(2026,1,15),date(2026,2,13))[5:].sum())} -> "
      f"{int(wcount(date(2026,2,14),date(2026,3,15))[5:].sum())}")
print(f"  средний ожидаемый сдвиг gmv: {h.mean():+.4f}; юзеров с |сдвиг|>1: {int((np.abs(h)>1).sum()):,}")
hs = np.sign(h) * np.log1p(np.abs(h)); hs = hs - hs.mean()
print(f"  направление построено, sd = {hs.std():.5f}")
out = pl.DataFrame({"user_id": uid, "step": hs})
p = os.environ.get("ZH_OUT", "work/zhenya_eda/out") + "/dir_weekend.parquet"
out.write_parquet(p)
print(f"  записано {p}")

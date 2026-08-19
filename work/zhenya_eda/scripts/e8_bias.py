"""E8. Насколько отбор юниверса искажает валидационное окно относительно теста.

В валидации команды исчезнувших НЕТ (отбор). В тесте они БУДУТ.
Меряем на исторических окнах: mean log1p(y) и долю нулей
  (a) на всей отобранной популяции  = как будет в ТЕСТЕ
  (b) только на присутствующих      = как устроена ВАЛИДАЦИЯ команды
"""
import polars as pl, numpy as np
from datetime import date, timedelta
df = pl.read_parquet("train.parquet", columns=["user_id","event_date","gmv"])

print("якорь        | N отобр | mean lp (ТЕСТ-режим) | mean lp (ВАЛ-режим) | разница | нулей тест | нулей вал")
res=[]
for A in [date(2025,6,7), date(2025,7,19), date(2025,8,9), date(2025,9,20), date(2025,10,11), date(2025,11,1)]:
    sel = df.filter(pl.col("event_date").is_between(A-timedelta(days=29), A))["user_id"].unique()
    fut = df.filter(pl.col("event_date").is_between(A+timedelta(days=1), A+timedelta(days=30)))
    tgt = (pl.DataFrame({"user_id": sel})
             .join(fut.group_by("user_id").agg(pl.col("gmv").sum().alias("y")), on="user_id", how="left"))
    present = tgt["y"].is_not_null().to_numpy()
    y = np.nan_to_num(tgt["y"].to_numpy().astype(np.float64))
    lp = np.log1p(y)
    m_all, m_pres = lp.mean(), lp[present].mean()
    z_all, z_pres = (y == 0).mean(), (y[present] == 0).mean()
    res.append((A, len(sel), m_all, m_pres, z_all, z_pres, 1-present.mean()))
    print(f"{A} | {len(sel):>7,} |        {m_all:.4f}        |       {m_pres:.4f}      | {m_pres-m_all:+.4f} "
          f"|   {100*z_all:5.2f}%   |  {100*z_pres:5.2f}%")

d = np.array([r[3]-r[2] for r in res]); f = np.array([r[6] for r in res])
print(f"\nСМЕЩЕНИЕ ВАЛ-РЕЖИМА: mean={d.mean():+.4f}  диапазон [{d.min():+.4f}, {d.max():+.4f}]")
print(f"доля исчезнувших:    mean={100*f.mean():.2f}%")
print(f"проверка тождества mean_all = (1-f)*mean_pres:  {[f'{r[2]:.4f} vs {(1-r[6])*r[3]:.4f}' for r in res[:3]]}")

print("\n=== ПЕРЕСЧЁТ СЕЗОННОГО СДВИГА КОМАНДЫ ===")
val_mean = 2.2421      # KNOWLEDGE.md Ф5, валидационное окно (БЕЗ исчезнувших)
lb_mean  = 2.3275      # Ф18, замерено на лидерборде для ТЕСТА (С исчезнувшими)
print(f"  val mean log1p (вал-режим, исчезнувших нет): {val_mean}")
print(f"  test mean log1p (замер LB, исчезнувшие есть): {lb_mean}")
print(f"  наблюдаемый разрыв, отнесённый командой к сезону: {lb_mean-val_mean:+.4f}")
for fv in (0.03, 0.05, 0.06, 0.08):
    drag = fv * val_mean
    print(f"  при доле исчезнувших {100*fv:4.1f}%: просадка теста {-drag:+.4f} -> "
          f"ЧИСТАЯ сезонность {lb_mean-val_mean+drag:+.4f}")
print(f"\n  для сравнения: прошлогодний аналог окон дал сезонный подъём +0.1759 (KNOWLEDGE.md)")

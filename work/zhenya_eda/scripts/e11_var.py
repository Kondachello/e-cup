"""E11. Объясняет ли режим исчезновения «недодисперсность бленда»?
Команда: sd прогнозов 1.510 против нужных тесту 1.628, лечится множителем 1.077.
Гипотеза: часть дисперсии таргета создаёт режим исчезновения, которого в валидации нет.
"""
import polars as pl, numpy as np
from datetime import date, timedelta
df = pl.read_parquet("train.parquet", columns=["user_id","event_date","gmv"])
print("якорь        |  sd(lp) ТЕСТ-режим | sd(lp) ВАЛ-режим | отношение | доля исчезн.")
R=[]
for A in [date(2025,6,7),date(2025,7,19),date(2025,8,9),date(2025,9,20),date(2025,10,11),date(2025,11,1)]:
    sel = df.filter(pl.col("event_date").is_between(A-timedelta(days=29),A))["user_id"].unique()
    fut = df.filter(pl.col("event_date").is_between(A+timedelta(days=1),A+timedelta(days=30)))
    t = pl.DataFrame({"user_id":sel}).join(fut.group_by("user_id").agg(pl.col("gmv").sum().alias("y")),
                                            on="user_id", how="left")
    y = t["y"].to_numpy().astype(np.float64); pres = ~np.isnan(y)
    lp = np.log1p(np.nan_to_num(y))
    sa, sp, f = lp.std(), lp[pres].std(), 1-pres.mean()
    R.append((sa,sp,f)); print(f"{A} |      {sa:.4f}        |     {sp:.4f}      |   {sa/sp:.4f}  |   {100*f:5.2f}%")

sa=np.array([r[0] for r in R]); sp=np.array([r[1] for r in R]); f=np.array([r[2] for r in R])
print(f"\nотношение sd(тест-режим)/sd(вал-режим): mean={np.mean(sa/sp):.4f}")

print("\n=== СВЕРКА С ЧИСЛАМИ КОМАНДЫ ===")
sd_blend, sd_need = 1.510, 1.628
print(f"  их бленд sd={sd_blend}, нужный тесту sd={sd_need}, их множитель {sd_need/sd_blend:.4f}")
print(f"  недостающая ДИСПЕРСИЯ: {sd_need**2 - sd_blend**2:.4f}")
mu = 2.3275/(1-0.05)   # среднее среди присутствующих при доле исчезнувших 5%
for fv in (0.03,0.04,0.05,0.06):
    extra = fv*(1-fv)*mu**2
    print(f"  при доле исчезнувших {100*fv:4.1f}%: режим добавляет дисперсии {extra:.4f} "
          f"= {100*extra/(sd_need**2-sd_blend**2):5.1f}% недостающей")

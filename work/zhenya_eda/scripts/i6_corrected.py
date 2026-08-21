"""I6. ИСПРАВЛЕНИЕ. Ревью право по трём пунктам:
  (а) i4_kalman склеивал параметры одного окна с sd другого;
  (б) среди баз i5_robust не было ВАЛИДАЦИОННОГО окна, а бленд меряется именно на нём;
  (в) «потолок задачи» — переобобщение: это бенчмарк одномерного фильтра
      собственных 30-дневных сумм, и бленд его превышает.
Пересчитываем всё на валидационном окне, параметры и sd из него же.
"""
import polars as pl, numpy as np
from datetime import date, timedelta
from scipy.optimize import least_squares
df = pl.read_parquet("train.parquet", columns=["user_id","event_date","gmv"])
uid = np.sort(pl.read_parquet("train.parquet", columns=["user_id"])["user_id"].unique().to_numpy())
def lp(s,e):
    w = df.filter(pl.col("event_date").is_between(s,e)).group_by("user_id").agg(pl.col("gmv").sum().alias("y"))
    y = pl.DataFrame({"user_id":uid}).join(w,on="user_id",how="left")["y"].to_numpy().astype(np.float64)
    return np.log1p(np.nan_to_num(y))

BASE_END = date(2026,2,13)     # ВАЛИДАЦИОННОЕ окно 2026-01-15..2026-02-13
b = lp(BASE_END-timedelta(days=29), BASE_END)
sd = float(b.std())
lags = [1,2,3,4,5,6,8]
r = {}
for k in lags:
    e = BASE_END - timedelta(days=30*k)
    r[k] = float(np.corrcoef(b, lp(e-timedelta(days=29), e))[0,1])
print(f"база: 2026-01-15..2026-02-13 (ВАЛИДАЦИОННОЕ окно), sd={sd:.4f}")
print("  " + "  ".join(f"r({k})={r[k]:.4f}" for k in lags))

K = np.array(lags,float); R = np.array([r[k] for k in lags])
s = least_squares(lambda t: t[0]+t[1]*t[2]**K - R, [0.4,0.18,0.8], bounds=([0,0,0],[1,1,0.999]))
p,q,lam = s.x
print(f"\nразложение: var(mu)={p:.4f}  var(s)={q:.4f}  lam={lam:.4f}  var(e)={1-p-q:.4f}  "
      f"невязка={np.abs(s.fun).max():.5f}")
print(f"  полупериод медленной компоненты: {30*np.log(2)/(-np.log(lam)):.1f} дней")

Rn=1-p-q; Q=q*(1-lam**2); P=Q
for _ in range(20000): P = lam**2*(P*Rn/(P+Rn))+Q
floor = sd*np.sqrt(P+Rn)
print(f"\n=== БЕНЧМАРК одномерного фильтра (НЕ потолок задачи) ===")
print(f"  RMSLE такого предиктора: {floor:.4f}")
sb = 1.666395
print(f"  действующий бленд:       {sb:.4f}")
print(f"  бленд ЛУЧШЕ бенчмарка на {floor-sb:+.4f}")
print(f"\n  mdl_flint бенчмарка {1-(floor/sd)**2:.4f}  против mdl_flint бленда {1-(sb/sd)**2:.4f}")
print(f"\nВЫВОД (исправленный): бленд превышает одномерный бенчмарк, значит бенчмарк")
print(f"НЕ ограничивает дальнейшие выигрыши. Прежняя формулировка «бленд на потолке,")
print(f"крупных выигрышей нет» ОТОЗВАНА. Величина превышения {floor-sb:.4f} — это цена")
print(f"всей информации сверх собственных 30-дневных сумм юзера (дневное разрешение,")
print(f"кросс-секция, сезонность, воронка).")

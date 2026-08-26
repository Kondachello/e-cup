"""E12. Артефакты выгрузки: пропуски дат, сбои, аномальные дни, странные юзеры."""
import polars as pl, numpy as np
from datetime import date, timedelta
df = pl.read_parquet("train.parquet")

d = df.group_by("event_date").agg([
    pl.len().alias("rows"), pl.col("user_id").n_unique().alias("dau"),
    pl.col("gmv").sum().alias("gmv"), pl.col("to_ord").sum().alias("ord"),
    pl.col("searches").sum().alias("srch")]).sort("event_date")
all_days = pl.date_range(date(2025,1,1), date(2026,2,13), "1d", eager=True)
missing = set(all_days.to_list()) - set(d["event_date"].to_list())
print(f"пропущенных дат в данных: {len(missing)}  {sorted(missing)[:10] if missing else ''}")

x = d["dau"].to_numpy().astype(float)
med = np.array([np.median(x[max(0,i-3):i+4]) for i in range(len(x))])
r = x/med
print("\n=== дни с аномальным DAU (отношение к локальной медиане) ===")
for i in np.argsort(r)[:6]:  print(f"  ПРОВАЛ  {d['event_date'][int(i)]}  dau={int(x[i]):>6,}  x{r[i]:.3f}")
for i in np.argsort(r)[-6:]: print(f"  ПИК     {d['event_date'][int(i)]}  dau={int(x[i]):>6,}  x{r[i]:.3f}")

print("\n=== средний чек и конверсия по кварталам ===")
q = df.with_columns(pl.col("event_date").dt.strftime("%Y-Q%q").alias("q")).group_by("q").agg([
    (pl.col("gmv").sum()/pl.col("to_ord").sum()).alias("aov"),
    (pl.col("to_ord").sum()/pl.col("to_cart").sum()).alias("cart2ord"),
    (pl.col("to_cart").sum()/pl.col("searches").sum()).alias("srch2cart"),
    pl.col("gmv").sum().alias("gmv")]).sort("q")
print(q)

print("\n=== юзеры-аномалии ===")
u = df.group_by("user_id").agg([
    pl.col("gmv").sum().alias("gmv"), pl.col("to_ord").sum().alias("ord"),
    pl.col("searches").sum().alias("srch"), pl.len().alias("days"),
    (pl.col("to_ord")>0).sum().alias("odays")])
print(f"  0 заказов за 409 дней:        {int((u['ord']==0).sum()):>7,} ({100*(u['ord']==0).mean():.2f}%)")
print(f"  0 поисков вообще:             {int((u['srch']==0).sum()):>7,}")
print(f"  заказы каждый 2-й активный день: {int((u['odays']/u['days']>0.5).sum()):>7,}")
print(f"  gmv > 100k за период:         {int((u['gmv']>100000).sum()):>7,}  максимум {u['gmv'].max():,.0f}")
top = u.sort('gmv', descending=True).head(3)
print(f"  топ-3 по gmv: {[f'{v:,.0f}' for v in top['gmv']]}  их доля в общем gmv: "
      f"{100*top['gmv'].sum()/u['gmv'].sum():.3f}%")
print(f"  доля gmv у топ-1% юзеров:     {100*u.sort('gmv',descending=True).head(2500)['gmv'].sum()/u['gmv'].sum():.1f}%")

print("\n=== «пустые» дни (визит без единого действия) — есть ли у них сигнал ===")
emp = ((pl.col("searches")==0)&(pl.col("cat")==0)&(pl.col("to_cart")==0)&(pl.col("to_ord")==0))
A = date(2026,1,14)
h = df.filter(pl.col("event_date").is_between(A-timedelta(days=29), A)).group_by("user_id").agg([
    emp.sum().alias("empty30"), pl.len().alias("act30")])
fut = df.filter(pl.col("event_date").is_between(A+timedelta(days=1), A+timedelta(days=30)))
g = fut.group_by("user_id").agg(pl.col("gmv").sum().alias("y"))
j = h.join(g, on="user_id", how="left").with_columns(pl.col("y").fill_null(0))
j = j.with_columns((pl.col("empty30")/pl.col("act30")).alias("share"))
print("доля пустых дней в последние 30д -> средний log1p(таргета):")
for lo,hi in [(0,0.001),(0.001,0.1),(0.1,0.2),(0.2,0.35),(0.35,0.5),(0.5,1.01)]:
    s = j.filter((pl.col("share")>=lo)&(pl.col("share")<hi))
    if s.height: print(f"  [{lo:.2f},{hi:.2f}) n={s.height:>7,}  mean lp={np.log1p(s['y'].to_numpy()).mean():.4f}  "
                       f"доля нулей={100*(s['y']==0).mean():5.2f}%")

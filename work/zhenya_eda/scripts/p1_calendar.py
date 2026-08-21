"""mdl_amber. Разложение разности окон val(15.01-13.02) -> test(14.02-15.03).
Единственный наблюдаемый экземпляр того же сдвига — 2025 год. Меряем по нему."""
import polars as pl, numpy as np
from datetime import date, timedelta
df = pl.read_parquet("train.parquet", columns=["user_id","event_date","gmv","to_ord","searches","to_cart"])
V26=(date(2026,1,15),date(2026,2,13)); T26=(date(2026,2,14),date(2026,3,15))
V25=(date(2025,1,15),date(2025,2,13)); T25=(date(2025,2,14),date(2025,3,15))

print("=== 1. СОСТАВ ДНЕЙ НЕДЕЛИ (чистая арифметика календаря) ===")
def wd(s,e):
    d=[s+timedelta(days=i) for i in range((e-s).days+1)]
    c=np.zeros(7,int)
    for x in d: c[x.weekday()]+=1
    return c,len(d)
for nm,(s,e) in (("val 2026",V26),("test 2026",T26),("val 2025",V25),("test 2025",T25)):
    c,n=wd(s,e); print(f"  {nm:10s} дней {n}  пн-вс {c}")
cv,_=wd(*V26); ct,_=wd(*T26)
print(f"  разница test-val 2026: {ct-cv}  (выходных: {cv[5:].sum()} -> {ct[5:].sum()})")

print("\n=== 2. ДНЕВНОЙ УРОВЕНЬ GMV по календарю 2025 (что даёт праздник) ===")
d25 = df.filter(pl.col("event_date").is_between(date(2025,1,15),date(2025,3,15))) \
        .group_by("event_date").agg([pl.col("gmv").sum().alias("g"),
                                     pl.col("user_id").n_unique().alias("dau")]).sort("event_date")
g=d25["g"].to_numpy(); dts=d25["event_date"].to_list()
med=np.array([np.median(g[max(0,i-7):i+8]) for i in range(len(g))])
r=g/med
print("  день          gmv/локальная медиана")
for i,dt in enumerate(dts):
    if r[i]>1.10 or r[i]<0.90: print(f"   {dt}  x{r[i]:.3f}")

print("\n=== 3. ПЕРСОНАЛЬНЫЙ СДВИГ val->test в 2025 (это и есть ось сезонности) ===")
uid=np.sort(df["user_id"].unique().to_numpy())
def lp(s,e):
    w=df.filter(pl.col("event_date").is_between(s,e)).group_by("user_id").agg(pl.col("gmv").sum().alias("y"))
    y=pl.DataFrame({"user_id":uid}).join(w,on="user_id",how="left")["y"].to_numpy().astype(float)
    return np.log1p(np.nan_to_num(y))
a25,b25=lp(*V25),lp(*T25); a26=lp(*V26)
h=b25-a25
print(f"  средний сдвиг 2025: {h.mean():+.4f}, sd {h.std():.4f}")
print(f"  q = mean(h²) = {float(np.mean(h*h)):.6f}  (для сравнения: mdl_amber..mdl_realgr имели q=0.00066)")
print(f"  ЦЕНТРИРОВАННЫЙ (уровень уже снят пробой ): q_c = {float(np.mean((h-h.mean())**2)):.6f}")

print("\n=== 4. РАЗЛОЖЕНИЕ сдвига по сегментам активности ===")
A=date(2026,1,14)
act=df.filter(pl.col("event_date").is_between(A-timedelta(days=29),A)).group_by("user_id").agg([
    pl.col("to_ord").sum().alias("o"), pl.len().alias("n")])
act=pl.DataFrame({"user_id":uid}).join(act,on="user_id",how="left").fill_null(0)
o=act["o"].to_numpy()
hc=h-h.mean()
print(f"  {'сегмент':22s} {'n':>8} {'сдвиг 2025':>12} {'вклад в q_c':>12}")
segs=[("заказов 0",o==0),("1-2",(o>=1)&(o<=2)),("3-5",(o>=3)&(o<=5)),
      ("6-10",(o>=6)&(o<=10)),("11+",o>=11)]
for nm,m in segs:
    print(f"  {nm:22s} {int(m.sum()):>8,} {h[m].mean():>12.4f} {float(np.mean(hc[m]**2)*m.mean()):>12.6f}")

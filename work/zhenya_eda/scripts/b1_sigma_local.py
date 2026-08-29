# -*- coding: utf-8 -*-
"""B1. Локальная sigma_u вместо константы F0*sqrt(q) — вопрос, переданный треку теории.

ВЫВОД. Для среднесохраняющей индикаторной оси сегмента S с долей p:
    h_i = 1{i in S} - p   (с точностью до масштаба)
    q      = mean(h^2) = p(1-p)
    sigma_u^2 = Var(h*r) ~ E[(1_S - p)^2 r^2] = p(1-p)[(1-p)A + pB]
где A = E[r^2 | S] (MSE ВНУТРИ сегмента), B = E[r^2 | не S] (снаружи).
Поскольку F0^2 = pA + (1-p)B, множитель гетероскедастичности:

    g^2 = sigma_u^2 / (F0^2 * q) = ((1-p)A + pB) / (pA + (1-p)B)

То есть g ОПРЕДЕЛЯЕТСЯ ДОЛЕЙ MSE, которую несёт сегмент, и считается ЛОКАЛЬНО
одной строкой — без единой посылки. Для малого сегмента g^2 -> A/B.
"""
import math, sys, json
import numpy as np, polars as pl
from datetime import date, timedelta
sys.stdout.reconfigure(encoding="utf-8")
F0N,NP_=1.6470,50_000
FPC=math.sqrt(0.8)

z=np.load("out/v1_erafix.npz")
uid=z["uid"]; y=z["y"]; r=z["pred_ctl"]-z["y"]
r2=r*r; F0=math.sqrt(float(r2.mean()))
print(f"валидационное окно: n={len(uid)}, F0={F0:.4f}")

ANCH=date(2026,1,14)
df=pl.read_parquet("../repo2/train.parquet",
                   columns=["user_id","event_date","gmv","to_ord","searches","cat","to_cart"])
h=df.filter(pl.col("event_date")<=ANCH)
def days_since(expr,name):
    return (pl.lit(ANCH)-expr.max()).dt.total_days().alias(name)
agg=h.group_by("user_id").agg([
    days_since(pl.col("event_date").filter(pl.col("to_ord")>0),"rec_ord"),
    days_since(pl.col("event_date"),"rec_act"),
    (pl.col("to_ord")>0).sum().alias("n_ord"),
    pl.col("searches").sum().alias("srch"),
    pl.col("to_cart").sum().alias("cart"),
    pl.len().alias("ndays")]).sort("user_id")
agg=pl.DataFrame({"user_id":uid}).join(agg,on="user_id",how="left")
ro=agg["rec_ord"].fill_null(9999).to_numpy(); ra=agg["rec_act"].fill_null(9999).to_numpy()
no=agg["n_ord"].fill_null(0).to_numpy(); nd=agg["ndays"].fill_null(0).to_numpy()

SEG={" ядро recency 15-90":(ro>=15)&(ro<=90),
     " спящие 91-365":(ro>=91)&(ro<=365),
     " никогда не покупавшие":no==0,
     " спящие-браузеры":(no==0)&(nd>=10),
     "узкий: топ-5% активных":(nd>=np.quantile(nd,0.95)),
     "узкий: свежие 0-14":(ro<=14)}
print(f"\n{'сегмент':28s}{'p':>8}{'доля MSE':>10}{'A/F0^2':>9}{'B/F0^2':>9}"
      f"{'g (формула)':>13}{'g (прямо)':>11}")
OUT={}
for nm,m in SEG.items():
    p=float(m.mean()); A=float(r2[m].mean()); B=float(r2[~m].mean())
    share=p*A/(F0*F0)
    g_form=math.sqrt(((1-p)*A+p*B)/(p*A+(1-p)*B))
    hh=(m.astype(float)-p); u=hh*r
    g_dir=float(np.std(u))/(F0*math.sqrt(float((hh*hh).mean())))
    OUT[nm]=dict(p=p,share=share,g=g_form)
    print(f"{nm:28s}{p:8.3f}{share:10.3f}{A/F0**2:9.3f}{B/F0**2:9.3f}{g_form:13.3f}{g_dir:11.3f}")
print("\n  колонки «формула» и «прямо» должны совпадать — это проверка вывода")
json.dump(OUT,open("out/b1_sigma_local.json","w",encoding="utf-8"),ensure_ascii=False,indent=1)

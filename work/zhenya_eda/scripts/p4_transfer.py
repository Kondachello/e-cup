"""mdl_marble. Переносится ли посегментная структура сдвига между годами?
Считаем посегментный сдвиг на 2025 (янв->фев-мар) и на наблюдаемых парах 2026,
сравниваем векторы. Если корреляция отрицательна — сегментные оси мертвы."""
import polars as pl, numpy as np
from datetime import date, timedelta
df=pl.read_parquet("train.parquet",columns=["user_id","event_date","gmv","to_ord"])
uid=np.sort(df["user_id"].unique().to_numpy())
def lp(s,e):
    w=df.filter(pl.col("event_date").is_between(s,e)).group_by("user_id").agg(pl.col("gmv").sum().alias("y"))
    y=pl.DataFrame({"user_id":uid}).join(w,on="user_id",how="left")["y"].to_numpy().astype(float)
    return np.log1p(np.nan_to_num(y))
def segvec(A,s1,e1,s2,e2,nb=10):
    """посегментный (по активности на якоре A) сдвиг между окнами, центрированный"""
    w=df.filter(pl.col("event_date").is_between(A-timedelta(days=29),A)).group_by("user_id").agg(pl.len().alias("n"))
    n_=pl.DataFrame({"user_id":uid}).join(w,on="user_id",how="left").fill_null(0)["n"].to_numpy().astype(float)
    q=np.quantile(n_,np.linspace(0,1,nb+1)); q[0],q[-1]=-np.inf,np.inf
    b=np.digitize(n_,q[1:-1]); d=lp(s2,e2)-lp(s1,e1)
    v=np.array([d[b==k].mean() if (b==k).sum()>50 else np.nan for k in range(nb)])
    return v-np.nanmean(v)

pairs={
 "2025 янв->фев-мар (сезонная)": (date(2025,1,14),date(2025,1,15),date(2025,2,13),date(2025,2,14),date(2025,3,15)),
 "2025 сент->окт-ноя (контроль)":(date(2025,9,14),date(2025,9,15),date(2025,10,14),date(2025,10,15),date(2025,11,13)),
 "2025 ноя->дек-янв":            (date(2025,11,14),date(2025,11,15),date(2025,12,14),date(2025,12,15),date(2026,1,13)),
 "2026 дек->янв-фев (ближайшая)":(date(2025,12,15),date(2025,12,16),date(2026,1,14),date(2026,1,15),date(2026,2,13)),
}
V={k:segvec(*v) for k,v in pairs.items()}
print(f"{'пара окон':32s} " + " ".join(f"с{i}" for i in range(10)))
for k,v in V.items(): print(f"{k:32s} " + " ".join(f"{x:+.2f}" for x in v))
ks=list(V); base=ks[0]
print(f"\nкорреляция посегментного вектора с сезонной парой 2025:")
for k in ks[1:]:
    a,b=V[base],V[k]; m=np.isfinite(a)&np.isfinite(b)
    print(f"  {k:32s} {np.corrcoef(a[m],b[m])[0,1]:+.3f}")
print(f"\nвывод: если корреляция сезонной пары 2025 с ближайшей парой 2026 отрицательна,")
print(f"то поправка из 2025 направлена ПРОТИВ структуры 2026 — сегментные оси мертвы.")

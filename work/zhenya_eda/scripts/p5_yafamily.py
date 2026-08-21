"""mdl_realgr. Семейство «год назад», открытое пробой mdl_gabbro (κ=0.454).
Проверяемо НА НАБЛЮДАЕМОЙ годовой паре: окно 15.01-13.02.2025 против того же
окна 2026 года. Вопрос: даёт ли прошлогоднее окно сигнал СВЕРХ свежей активности?
Если да — семейство живо, и подокна (трети, отдельные праздники) специфицируемы."""
import polars as pl, numpy as np
from datetime import date, timedelta
from sklearn.linear_model import Ridge
df=pl.read_parquet("train.parquet",columns=["user_id","event_date","gmv","to_ord","searches","to_cart","cat"])
uid=np.sort(df["user_id"].unique().to_numpy()); F0=1.666395; NPUB=50_000; NOISE=0.000022
def agg(s,e,col="gmv"):
    w=df.filter(pl.col("event_date").is_between(s,e)).group_by("user_id").agg(pl.col(col).sum().alias("y"))
    y=pl.DataFrame({"user_id":uid}).join(w,on="user_id",how="left")["y"].to_numpy().astype(float)
    return np.nan_to_num(y)

# цель: валидационное окно 2026 (известно). База: бленд команды.
v=pl.read_parquet("work/preds_pack/val_preds.parquet").sort("user_id")
assert np.array_equal(v["user_id"].to_numpy(),uid)
ly=np.log1p(np.clip(v["target"].to_numpy().astype(float),0,None))
e=v["blend"].to_numpy().astype(float)-ly
print(f"остаток бленда на валидации: RMSE {np.sqrt(np.mean(e**2)):.6f}\n")

# ПРОШЛОГОДНЕЕ окно, выровненное на валидационное: 15.01-13.02.2025
YA=(date(2025,1,15),date(2025,2,13))
subw={
 "ya полное окно 15.01-13.02":      YA,
 "ya треть 1 (15.01-24.01)":        (date(2025,1,15),date(2025,1,24)),
 "ya треть 2 (25.01-03.02)":        (date(2025,1,25),date(2025,2,3)),
 "ya треть 3 (04.02-13.02)":        (date(2025,2,4),date(2025,2,13)),
 "ya широкое (01.01-28.02)":        (date(2025,1,1),date(2025,2,28)),
}
print(f"{'ось':34s} {'выигрыш':>11} {'сигм':>7} {'вердикт':>10}")
for nm,(s,en) in subw.items():
    h=np.log1p(agg(s,en)); h=h-h.mean()
    hh=float(np.mean(h*h)); c=float(np.mean(e*h))
    gain=c*c/(hh*2*F0); z=abs(c)/max(float(np.std(e*h))/np.sqrt(NPUB),1e-12)
    print(f"{nm:34s} {gain:>11.6f} {z:>7.1f} {'ЖИВА' if gain>NOISE and z>2 else 'ниже шума':>10}")

print(f"\n=== КОНТРОЛЬ: не дублирует ли ya свежую активность ===")
A=date(2026,1,14)
w30=df.filter(pl.col("event_date").is_between(A-timedelta(days=364),A)).group_by("user_id").agg([
    pl.col("gmv").sum().alias("g365"),pl.col("to_ord").sum().alias("o365")])
w30=pl.DataFrame({"user_id":uid}).join(w30,on="user_id",how="left").fill_null(0)
R=np.column_stack([np.log1p(w30["g365"].to_numpy().astype(float)),
                   np.log1p(w30["o365"].to_numpy().astype(float))])
ya=np.log1p(agg(*YA))
mu,sd=R.mean(0),R.std(0)+1e-9
resid_ya=ya-Ridge(alpha=10.).fit((R-mu)/sd,ya).predict((R-mu)/sd)
for nm,h in (("ya сырое",ya-ya.mean()),("ya ЗА ВЫЧЕТОМ 365д-активности",resid_ya-resid_ya.mean())):
    hh=float(np.mean(h*h)); c=float(np.mean(e*h))
    gain=c*c/(hh*2*F0); z=abs(c)/max(float(np.std(e*h))/np.sqrt(NPUB),1e-12)
    print(f"  {nm:34s} выигрыш {gain:.6f}  {z:.1f} сигм")
print(f"\nЕсли остаточная ось сохраняет выигрыш — сигнал НЕ дублирует активность,")
print(f"и семейство «год назад» действительно вне того, что видели модели.")

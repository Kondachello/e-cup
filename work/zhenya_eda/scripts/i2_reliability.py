"""I2. ПОЛ МЕТРИКИ через устойчивость таргета между окнами.

Модель: lp_t = mu_u + eps_t, mu_u — устойчивый уровень юзера, eps_t — транзиент.
   corr(lp_1, lp_2) = rho = var(mu)/(var(mu)+var(eps))
Идеальная модель знает mu_u и оставляет var(eps), то есть
   ПОЛ RMSLE = sd(lp) * sqrt(1 - rho)
Считаем rho на непересекающихся 30-дневных окнах при разных разносах.
"""
import polars as pl, numpy as np
from datetime import date, timedelta
df = pl.read_parquet("train.parquet", columns=["user_id","event_date","gmv"])
ALL = pl.DataFrame({"user_id": np.arange(0,0)})

def win_lp(s, e, uid):
    w = df.filter(pl.col("event_date").is_between(s,e)).group_by("user_id").agg(pl.col("gmv").sum().alias("y"))
    y = pl.DataFrame({"user_id":uid}).join(w,on="user_id",how="left")["y"].to_numpy().astype(np.float64)
    return np.log1p(np.nan_to_num(y))

uid = np.sort(pl.read_parquet("train.parquet", columns=["user_id"])["user_id"].unique().to_numpy())
print(f"юзеров {len(uid):,}\n")
print(f"{'окно 1':>24} {'окно 2':>24} {'зазор':>7} {'rho':>8} {'sd':>7} {'ПОЛ RMSLE':>11}")
res=[]
pairs = [
    (date(2025,11,16), date(2025,12,15), date(2025,12,16), date(2026,1,14), 0),
    (date(2025,10,17), date(2025,11,15), date(2025,12,16), date(2026,1,14), 30),
    (date(2025,9,17),  date(2025,10,16), date(2025,12,16), date(2026,1,14), 60),
    (date(2025,7,19),  date(2025,8,17),  date(2025,12,16), date(2026,1,14), 120),
    (date(2025,4,20),  date(2025,5,19),  date(2025,12,16), date(2026,1,14), 210),
    (date(2025,12,16), date(2026,1,14),  date(2026,1,15), date(2026,2,13), 0),
]
for s1,e1,s2,e2,gap in pairs:
    a, b = win_lp(s1,e1,uid), win_lp(s2,e2,uid)
    rho = float(np.corrcoef(a,b)[0,1]); sd = float(b.std())
    floor = sd*np.sqrt(max(1-rho,0))
    res.append((gap,rho,floor))
    print(f"{str(s1)+'..'+str(e1):>24} {str(s2)+'..'+str(e2):>24} {gap:>7} {rho:>8.4f} {sd:>7.4f} {floor:>11.4f}")

print(f"\n=== СВЕРКА С РЕАЛЬНОСТЬЮ ===")
v = pl.read_parquet("work/preds_pack/val_preds.parquet").sort("user_id")
ly = np.log1p(np.clip(v["target"].to_numpy().astype(np.float64),0,None))
sb = float(np.sqrt(np.mean((v["blend"].to_numpy().astype(np.float64)-ly)**2)))
sd = ly.std()
print(f"  бленд val RMSLE      {sb:.4f}")
print(f"  sd таргета           {sd:.4f}")
rho_need = 1-(sb/sd)**2
print(f"  бленд эквивалентен знанию mu_u с rho = {rho_need:.4f}")
r0 = [r for g,r,f in res if g==0][0]
print(f"  измеренная устойчивость соседних окон rho = {r0:.4f}")
if r0 > rho_need:
    fl = sd*np.sqrt(1-r0)
    print(f"  => ПОЛ {fl:.4f}, запас до него {sb-fl:.4f} (разрыв до топ-1 всего 0.0031)")
else:
    print(f"  => бленд УЖЕ извлекает больше, чем даёт одно соседнее окно")
    print(f"     (одно окно — шумная оценка mu_u; модель усредняет много окон истории)")

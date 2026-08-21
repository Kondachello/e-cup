"""J1. Разложение на ДНЕВНОМ разрешении.

Ревью справедливо: сетка 30 дней схлопывает в «белый шум» всё, что живёт короче
месяца. Меряем автокорреляцию дневных рядов и раскладываем на сумму экспонент —
это и есть постоянные времени, которые должно кодировать представление.
"""
import os
import polars as pl, numpy as np
from datetime import date, timedelta
from scipy.optimize import least_squares

A = date(2026,1,14)          # валидационный якорь: история ДО него
L = 364
df = pl.read_parquet("train.parquet", columns=["user_id","event_date","gmv","to_ord","to_cart","searches"])
df = df.filter(pl.col("event_date").is_between(A-timedelta(days=L-1), A))
uids = np.sort(df["user_id"].unique().to_numpy())
uidx = {u:i for i,u in enumerate(uids)}
print(f"юзеров {len(uids):,}, дней {L}")

d0 = A - timedelta(days=L-1)
ui = df["user_id"].to_numpy(); di = (df["event_date"].to_numpy() - np.datetime64(d0)).astype("timedelta64[D]").astype(int)
row = np.array([uidx[u] for u in ui], dtype=np.int32)

def build(col, transform):
    M = np.zeros((len(uids), L), dtype=np.float32)
    M[row, di] = transform(df[col].to_numpy())
    return M

SER = {
    "log1p(gmv) дневной": build("gmv", lambda v: np.log1p(v)),
    "заказ (0/1)":        build("to_ord", lambda v: (v>0).astype(np.float32)),
    "корзина (0/1)":      build("to_cart", lambda v: (v>0).astype(np.float32)),
    "log1p(поиски)":      build("searches", lambda v: np.log1p(v)),
}

def acf(M, lags):
    """корреляция между столбцами t и t+k, усреднённая по t (по всем юзерам)"""
    out=[]
    for k in lags:
        a = M[:, :L-k].ravel(); b = M[:, k:].ravel()
        out.append(float(np.corrcoef(a,b)[0,1]))
    return np.array(out)

lags = np.array([1,2,3,5,7,10,14,21,30,45,60,90,120,180,240])
print(f"\n{'ряд':22s} " + " ".join(f"r{k:<4}" for k in lags))
ACF={}
for nm, M in SER.items():
    r = acf(M, lags); ACF[nm]=r
    print(f"{nm:22s} " + " ".join(f"{x:5.3f}" for x in r))

print("\n=== подгонка r(k) = p + q1*e^(-k/t1) + q2*e^(-k/t2) ===")
print(f"{'ряд':22s} {'p (пост.)':>10} {'q1':>8} {'t1, дн':>8} {'q2':>8} {'t2, дн':>9} {'невязка':>9}")
FIT={}
for nm, r in ACF.items():
    f = lambda th: th[0] + th[1]*np.exp(-lags/th[2]) + th[3]*np.exp(-lags/th[4]) - r
    best=None
    for t1g,t2g in [(3,60),(5,90),(2,30),(10,150)]:
        try:
            s = least_squares(f, [r[-1]*0.8, 0.05, t1g, 0.05, t2g],
                              bounds=([0,0,0.5,0,10],[1,1,30,1,400]))
            if best is None or s.cost < best.cost: best = s
        except Exception: pass
    p,q1,t1,q2,t2 = best.x; FIT[nm]=best.x
    print(f"{nm:22s} {p:>10.4f} {q1:>8.4f} {t1:>8.2f} {q2:>8.4f} {t2:>9.2f} {np.abs(best.fun).max():>9.5f}")

print("\n=== ЧТО ЭТО ЗНАЧИТ ===")
p,q1,t1,q2,t2 = FIT["log1p(gmv) дневной"]
print(f"  У дневного log1p(gmv) три масштаба:")
print(f"    постоянный уровень юзера        {p:.4f}")
print(f"    БЫСТРАЯ компонента, tau={t1:>6.2f} дн  {q1:.4f}  <- 30-дневная сетка её СХЛОПЫВАЕТ")
print(f"    медленная компонента, tau={t2:>6.2f} дн {q2:.4f}")
print(f"    остаток (истинно дневной шум)   {1-p-q1-q2:.4f}")
print(f"\n  полупериоды: быстрая {t1*np.log(2):.1f} дн, медленная {t2*np.log(2):.1f} дн")
print(f"  У команды экспоненциальные затухания с полупериодами 7 / 30 / 120 дней (тир v2).")
np.save(os.environ.get("ZH_OUT", "work/zhenya_eda/out") + "/j1_acf.npy", {k:v for k,v in ACF.items()}, allow_pickle=True)

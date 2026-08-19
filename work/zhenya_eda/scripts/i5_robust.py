"""I5. Устойчивость оценки потолка: другая база окон + бутстрап."""
import polars as pl, numpy as np
from datetime import date, timedelta
from scipy.optimize import least_squares
df = pl.read_parquet("train.parquet", columns=["user_id","event_date","gmv"])
uid = np.sort(pl.read_parquet("train.parquet", columns=["user_id"])["user_id"].unique().to_numpy())

def lp(s,e):
    w = df.filter(pl.col("event_date").is_between(s,e)).group_by("user_id").agg(pl.col("gmv").sum().alias("y"))
    y = pl.DataFrame({"user_id":uid}).join(w,on="user_id",how="left")["y"].to_numpy().astype(np.float64)
    return np.log1p(np.nan_to_num(y))

def acf_from(base_end):
    """r(k) между окном [base_end-29, base_end] и окнами на k*30 дней раньше"""
    b = lp(base_end-timedelta(days=29), base_end)
    out={}
    for k in (1,2,3,4,5,6,8):
        e = base_end - timedelta(days=30*k)
        out[k] = float(np.corrcoef(b, lp(e-timedelta(days=29), e))[0,1])
    return b, out

def fit(out):
    k = np.array(sorted(out)); r = np.array([out[i] for i in k], float)
    s = least_squares(lambda t: t[0]+t[1]*t[2]**k - r, [0.4,0.18,0.8],
                      bounds=([0,0,0],[1,1,0.999]))
    return s.x, float(np.abs(s.fun).max())

def floor_from(p,q,lam,sd):
    R=1-p-q; Q=q*(1-lam**2); P=Q
    for _ in range(5000): P = lam**2*(P*R/(P+R))+Q
    return sd*np.sqrt(P+R), sd*np.sqrt(R)

print(f"{'база окна':>26} {'r(1)':>7} {'p':>7} {'q':>7} {'lam':>7} {'невязка':>9} {'ПОЛ':>8} {'белый шум':>10}")
for be in (date(2026,1,14), date(2025,12,15), date(2025,11,15), date(2025,10,16)):
    b, out = acf_from(be); (p,q,lam), res = fit(out); sd = float(b.std())
    fl, wn = floor_from(p,q,lam,sd)
    print(f"{str(be-timedelta(days=29))+'..'+str(be):>26} {out[1]:>7.4f} {p:>7.4f} {q:>7.4f} "
          f"{lam:>7.4f} {res:>9.5f} {fl:>8.4f} {wn:>10.4f}")

print("\n=== бутстрап по юзерам для базового окна 2025-12-16..2026-01-14 ===")
b, out = acf_from(date(2026,1,14))
cols = {k: lp(date(2026,1,14)-timedelta(days=30*k+29), date(2026,1,14)-timedelta(days=30*k)) for k in sorted(out)}
rng = np.random.default_rng(0); fl=[]
for _ in range(40):
    i = rng.integers(0, len(b), len(b))
    o = {k: float(np.corrcoef(b[i], v[i])[0,1]) for k,v in cols.items()}
    try:
        (p,q,lam), _ = fit(o); f_,_w = floor_from(p,q,lam, float(b[i].std())); fl.append(f_)
    except Exception: pass
fl = np.array(fl)
print(f"  ПОЛ: среднее {fl.mean():.4f}  sd {fl.std():.4f}  [p5={np.percentile(fl,5):.4f}, p95={np.percentile(fl,95):.4f}]")
print(f"  бленд 1.6664 -> запас до пола {1.666395-fl.mean():+.4f} +- {fl.std():.4f}")

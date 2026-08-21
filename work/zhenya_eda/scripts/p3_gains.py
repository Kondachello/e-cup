"""mdl_halite. Достижимый выигрыш каждой структурной оси, замеренный на 2025.

Ставим себя в 2025: якорь 14.01.2025, «валидация» 15.01-13.02, «тест» 14.02-15.03.
это ровно то, что структурная ось могла бы починить.
   e = (lp_val + сдвиг) - lp_test
   выигрыш оси h = mean(e·h)² / (mean(h²) · 2·F0)    — инвариантен к нормировке h
"""
import polars as pl, numpy as np
from datetime import date, timedelta
df = pl.read_parquet("train.parquet", columns=["user_id","event_date","gmv","to_ord","searches","cat"])
uid=np.sort(df["user_id"].unique().to_numpy()); F0=1.666395; NPUB=50_000
def lp(s,e):
    w=df.filter(pl.col("event_date").is_between(s,e)).group_by("user_id").agg(pl.col("gmv").sum().alias("y"))
    y=pl.DataFrame({"user_id":uid}).join(w,on="user_id",how="left")["y"].to_numpy().astype(float)
    return np.log1p(np.nan_to_num(y))

A25=date(2025,1,14)
v25,t25 = lp(date(2025,1,15),date(2025,2,13)), lp(date(2025,2,14),date(2025,3,15))
shift = float((t25-v25).mean())
e = (v25+shift) - t25                      # остаток ПОСЛЕ глобального сдвига
print(f"глобальный сдвиг 2025 = {shift:+.4f}; остаток: sd={e.std():.4f}, RMSE={np.sqrt(np.mean(e**2)):.4f}\n")

# признаки, доступные НА ЯКОРЕ 14.01.2025 (ничего из будущего)
w30=df.filter(pl.col("event_date").is_between(A25-timedelta(days=29),A25))
g=w30.group_by("user_id").agg([pl.col("to_ord").sum().alias("o"),pl.len().alias("n"),
    pl.col("gmv").sum().alias("gm"),(pl.col("cat")>0).sum().alias("cd"),
    pl.col("searches").sum().alias("sr")])
g=pl.DataFrame({"user_id":uid}).join(g,on="user_id",how="left").fill_null(0)
o,n_,gm,cd,sr=(g[c].to_numpy().astype(float) for c in ("o","n","gm","cd","sr"))

hist=df.filter(pl.col("event_date")<=A25)
wd=hist.with_columns(pl.col("event_date").dt.weekday().alias("w")).group_by(["user_id","w"]).agg(
    pl.col("gmv").sum().alias("g")).pivot(on="w",index="user_id",values="g").fill_null(0)
wd=pl.DataFrame({"user_id":uid}).join(wd,on="user_id",how="left").fill_null(0)
W=wd.drop("user_id").to_numpy().astype(float); W=W/np.maximum(W.sum(1,keepdims=True),1e-9)
def wdc(s,en):
    c=np.zeros(7)
    for i in range((en-s).days+1): c[(s+timedelta(days=i)).weekday()]+=1
    return c
dw=(wdc(date(2025,2,14),date(2025,3,15))-wdc(date(2025,1,15),date(2025,2,13)))/30.0

dom=hist.with_columns((pl.col("event_date").dt.day()<=15).alias("h1")).group_by(["user_id","h1"]).agg(
    pl.col("gmv").sum().alias("g")).pivot(on="h1",index="user_id",values="g").fill_null(0)
dom=pl.DataFrame({"user_id":uid}).join(dom,on="user_id",how="left").fill_null(0)
D=dom.drop("user_id").to_numpy().astype(float); dshare=D[:,0]/np.maximum(D.sum(1),1e-9)

def seg(key,nb=10):
    q=np.quantile(key,np.linspace(0,1,nb+1)); q[0],q[-1]=-np.inf,np.inf
    b=np.digitize(key,q[1:-1])
    return np.array([e[b==k].mean() if (b==k).sum()>50 else 0.0 for k in range(nb)])[b]

AXES={
 "сегменты по активности":   seg(n_),
 "сегменты по заказам":      seg(o),
 "сегменты по gmv":          seg(gm),
 "сегменты по каталогу":     seg(cd),
 "сегменты по поискам":      seg(sr),
 "состав дней недели":       W@dw,
 "зарплатная фаза":          dshare-dshare.mean(),
 "молчание (пустые визиты)": (n_==0).astype(float),
 "уровень (контроль)":       np.ones(len(e)),
}
print(f"{'структурная ось':28s} {'выигрыш':>11} {'сигм на 50k':>12} {'вердикт':>10}")
NOISE=0.000022
res=[]
for nm,h in AXES.items():
    h=h-h.mean() if nm!="уровень (контроль)" else h
    hh=float(np.mean(h*h))
    if hh<1e-14: print(f"{nm:28s}  вырождена"); continue
    c=float(np.mean(e*h)); gain=c*c/(hh*2*F0)
    se_c=float(np.std(e*h))/np.sqrt(NPUB); z=abs(c)/max(se_c,1e-12)
    res.append((nm,gain,z))
    print(f"{nm:28s} {gain:>11.6f} {z:>12.1f} {'ЖИВА' if gain>NOISE and z>2 else 'ниже шума':>10}")
print(f"\nпорог: выигрыш > {NOISE:.6f} И значимость > 2 сигм")
print("ОГОВОРКА: это выигрыш на 2025 году. Перенос на 2026 — отдельный вопрос,")
print("сегментная сезонность уже опровергнута (season_segments.md, корр -0.93).")

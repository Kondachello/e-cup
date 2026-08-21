"""mdl_gabbro. Каталог осей типа S: для каждой структурной компоненты считаем
СИСТЕМАТИЧЕСКУЮ (посегментную) величину q и ожидаемую k."""
import polars as pl, numpy as np
from datetime import date, timedelta
df = pl.read_parquet("train.parquet", columns=["user_id","event_date","gmv","to_ord","searches","to_cart","cat"])
uid = np.sort(df["user_id"].unique().to_numpy())
F0, NPUB = 1.666395, 50_000
sig = lambda q: F0/np.sqrt(NPUB*max(q,1e-12))
dose = lambda q: 0.0416/(0.0416+sig(q)**2)

def lp(s,e):
    w=df.filter(pl.col("event_date").is_between(s,e)).group_by("user_id").agg(pl.col("gmv").sum().alias("y"))
    y=pl.DataFrame({"user_id":uid}).join(w,on="user_id",how="left")["y"].to_numpy().astype(float)
    return np.log1p(np.nan_to_num(y))

V25,T25=(date(2025,1,15),date(2025,2,13)),(date(2025,2,14),date(2025,3,15))
h25 = lp(*T25)-lp(*V25)                       # наблюдаемый сдвиг окна в 2025
A=date(2026,1,14)

def q_of(vals):
    """q систематической оси = дисперсия ПОСЕГМЕНТНЫХ средних сдвига"""
    v=np.asarray(vals,float); return float(np.mean((v-v.mean())**2))

# --- сегментации, по которым раскладываем сдвиг ---
w30=df.filter(pl.col("event_date").is_between(A-timedelta(days=29),A))
g=w30.group_by("user_id").agg([pl.col("to_ord").sum().alias("o"), pl.len().alias("n"),
                               pl.col("gmv").sum().alias("gm"),
                               (pl.col("cat")>0).sum().alias("cd"),
                               pl.col("searches").sum().alias("sr")])
g=pl.DataFrame({"user_id":uid}).join(g,on="user_id",how="left").fill_null(0)
o,n_,gm,cd,sr = (g[c].to_numpy().astype(float) for c in ("o","n","gm","cd","sr"))

def seg_axis(key, nb=10):
    """строим ось: каждому юзеру приписан средний сдвиг 2025 его сегмента"""
    q_=np.quantile(key,np.linspace(0,1,nb+1)); q_[0],q_[-1]=-np.inf,np.inf
    b=np.digitize(key,q_[1:-1])
    m=np.array([h25[b==k].mean() if (b==k).sum()>50 else h25.mean() for k in range(nb)])
    return m[b]

CAT=[]
for nm,key in [("сезонный сдвиг по активности", n_), ("по числу заказов", o),
               ("по gmv за 30д", gm), ("по каталогу", cd), ("по поискам", sr)]:
    ax=seg_axis(key); qq=q_of(ax)
    CAT.append((nm,qq))

# --- календарные компоненты ---
def wd_counts(s,e):
    c=np.zeros(7); 
    for i in range((e-s).days+1): c[(s+timedelta(days=i)).weekday()]+=1
    return c
dv,dt_=wd_counts(date(2026,1,15),date(2026,2,13)),wd_counts(date(2026,2,14),date(2026,3,15))
# персональный недельный профиль -> сколько юзер теряет/получает от смены состава дней
prof=df.filter(pl.col("event_date")>=A-timedelta(days=180)).with_columns(
    pl.col("event_date").dt.weekday().alias("wd")).group_by(["user_id","wd"]).agg(
    pl.col("gmv").sum().alias("g")).pivot(on="wd",index="user_id",values="g").fill_null(0)
prof=pl.DataFrame({"user_id":uid}).join(prof,on="user_id",how="left").fill_null(0)
P=prof.drop("user_id").to_numpy().astype(float)
P=P/np.maximum(P.sum(1,keepdims=True),1e-9)                  # доля gmv по дням недели
delta_wd=(dt_-dv)/30.0
wd_ax=np.log1p(np.abs(P@delta_wd))*np.sign(P@delta_wd)*7      # эффект смены состава
CAT.append(("состав дней недели", q_of(wd_ax)))

# фаза месяца: доля gmv в первой/второй половине месяца
dom=df.filter(pl.col("event_date")>=A-timedelta(days=180)).with_columns(
    (pl.col("event_date").dt.day()<=15).alias("h1")).group_by(["user_id","h1"]).agg(
    pl.col("gmv").sum().alias("g")).pivot(on="h1",index="user_id",values="g").fill_null(0)
dom=pl.DataFrame({"user_id":uid}).join(dom,on="user_id",how="left").fill_null(0)
D=dom.drop("user_id").to_numpy().astype(float)
share=D[:,0]/np.maximum(D.sum(1),1e-9)
CAT.append(("зарплатная фаза (половина месяца)", q_of(np.log1p(share))))

print(f"{'структурная компонента':38s} {'q':>10} {'σ_κ':>7} {'доза':>6}")
for nm,qq in sorted(CAT,key=lambda x:-x[1]):
    print(f"{nm:38s} {qq:>10.6f} {sig(qq):>7.3f} {dose(qq):>6.3f}")
print(f"\nдля сравнения: пробы mdl_amber..mdl_realgr имели q=0.00066 (σ=0.290, доза 0.33)")
print(f"перелом дозы w=0.5 при q=0.00134")

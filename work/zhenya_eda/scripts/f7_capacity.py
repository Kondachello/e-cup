"""F7. КОНТРОЛЬ РАВНОЙ ЁМКОСТЬЮ на настоящем валидационном якоре 2026-01-14.
Правило команды: новый набор признаков сравнивать не с базой, а с базой + СТОЛЬКО ЖЕ
СТАРЫХ признаков. Так были отклонены тиры v6, v8, v10.
Сравнение только ПОСЛЕ калибровки (правило №1)."""
import polars as pl, numpy as np, lightgbm as lgb
from datetime import date, timedelta
df = pl.read_parquet("train.parquet")
VAL = date(2026,1,14)
TRAIN = [VAL - timedelta(days=30+14*i) for i in range(0,10)]   # зазор 30
W=(7,14,30,60,90,180,365)

def base_feats(A):
    hist = df.filter(pl.col("event_date")<=A); Ad=pl.lit(A); a=[]
    for w in W:
        m = pl.col("event_date")>=A-timedelta(days=w-1)
        a += [m.sum().alias(f"act_{w}"), pl.col("gmv").filter(m).sum().alias(f"gmv_{w}"),
              pl.col("to_ord").filter(m).sum().alias(f"ord_{w}"),
              pl.col("to_cart").filter(m).sum().alias(f"cart_{w}"),
              pl.col("searches").filter(m).sum().alias(f"srch_{w}"),
              (pl.col("to_ord")>0).filter(m).sum().alias(f"od_{w}")]
    a += [(Ad-pl.col("event_date").max()).dt.total_days().alias("rec_act"),
          (Ad-pl.col("event_date").filter(pl.col("to_ord")>0).max()).dt.total_days().alias("rec_ord"),
          (Ad-pl.col("event_date").filter(pl.col("to_cart")>0).max()).dt.total_days().alias("rec_cart"),
          (Ad-pl.col("event_date").min()).dt.total_days().alias("tenure"), pl.len().alias("nd"),
          pl.col("event_date").diff().dt.total_days().mean().alias("gm"),
          pl.col("event_date").diff().dt.total_days().max().alias("gx")]
    X = hist.sort(["user_id","event_date"]).group_by("user_id").agg(a)
    sel = df.filter(pl.col("event_date").is_between(A-timedelta(days=29),A))["user_id"].unique()
    return X.filter(pl.col("user_id").is_in(sel.implode()))

def r3_feats(A, uid):
    """12 признаков: визиты без единого действия + дни «поиск И каталог»"""
    e = ((pl.col("searches")==0)&(pl.col("cat")==0)&(pl.col("to_cart")==0)&(pl.col("to_ord")==0))
    a=[]
    for w in (14,30,90,365):
        m = pl.col("event_date")>=A-timedelta(days=w-1)
        a += [e.filter(m).sum().alias(f"emp_{w}"),
              (e.filter(m).sum()/pl.max_horizontal(m.sum(),pl.lit(1))).alias(f"empsh_{w}"),
              ((pl.col("searches")>0)&(pl.col("cat")>0)).filter(m).sum().alias(f"both_{w}")]
    g = df.filter(pl.col("event_date")<=A).group_by("user_id").agg(a)
    return pl.DataFrame({"user_id":uid}).join(g,on="user_id",how="left").drop("user_id")

def ctl_feats(A, uid):
    """КОНТРОЛЬ: 12 СТАРЫХ по типу признаков (те же окна, другие величины) — ноль новой информации"""
    a=[]
    for w in (14,30,90,365):
        m = pl.col("event_date")>=A-timedelta(days=w-1)
        a += [pl.col("gmv_search").filter(m).sum().alias(f"gs_{w}"),
              pl.col("search_to_cart").filter(m).sum().alias(f"s2c_{w}"),
              pl.col("cat_to_cart").filter(m).sum().alias(f"c2c_{w}")]
    g = df.filter(pl.col("event_date")<=A).group_by("user_id").agg(a)
    return pl.DataFrame({"user_id":uid}).join(g,on="user_id",how="left").drop("user_id")

def tgt(A, uid):
    fut = df.filter(pl.col("event_date").is_between(A+timedelta(days=1),A+timedelta(days=30)))
    g = fut.group_by("user_id").agg(pl.col("gmv").sum().alias("y"))
    return np.nan_to_num(pl.DataFrame({"user_id":uid}).join(g,on="user_id",how="left")["y"].to_numpy().astype(float))

print("сборка признаков...")
DATA={}
for A in TRAIN+[VAL]:
    X = base_feats(A); u = X["user_id"].to_numpy()
    DATA[A] = (X.drop("user_id"), r3_feats(A,u), ctl_feats(A,u), tgt(A,u), u)
    print(f"  {A} n={len(u):,}")

def build(arm):
    tr = [DATA[a] for a in TRAIN]
    def stack(d):
        b,r,c,_,_ = d
        parts = [b.to_numpy().astype(float)]
        if arm in ("r3",): parts.append(r.to_numpy().astype(float))
        if arm in ("ctl",): parts.append(c.to_numpy().astype(float))
        return np.nan_to_num(np.hstack(parts), nan=-1.0, posinf=1e9, neginf=-1e9)
    Xtr = np.vstack([stack(d) for d in tr]); ytr = np.concatenate([d[3] for d in tr])
    return Xtr, np.log1p(ytr), stack(DATA[VAL]), np.log1p(DATA[VAL][3])

def calibrated(lp, ly, nb=24):
    rng = np.random.default_rng(0); i = rng.permutation(len(ly)); h=len(ly)//2
    out = lp.copy()
    for tr, te in ((i[:h],i[h:]),(i[h:],i[:h])):
        q = np.quantile(lp[tr], np.linspace(0,1,nb+1)); q[0],q[-1]=-np.inf,np.inf
        b1,b2 = np.digitize(lp[tr],q[1:-1]), np.digitize(lp[te],q[1:-1])
        sh = np.array([np.mean(ly[tr][b1==k]-lp[tr][b1==k]) if (b1==k).sum()>50 else 0. for k in range(nb)])
        out[te] = lp[te]+sh[b2]
    return float(np.sqrt(np.mean((out-ly)**2)))

print(f"\n{'вариант':28s} {'признаков':>10} {'сырой':>11} {'калиброванный':>15}")
res={}
for arm, lab in (("base","база"), ("r3","база + mdl_gypsum (12 новых)"), ("ctl","база + 12 СТАРЫХ (контроль)")):
    Xtr, ytr, Xte, yte = build(arm)
    m = lgb.LGBMRegressor(objective="tweedie", tweedie_variance_power=1.45, learning_rate=.05,
        num_leaves=63, min_child_samples=100, subsample=.8, colsample_bytree=.8,
        n_estimators=700, verbose=-1, n_jobs=4, random_state=42).fit(Xtr, ytr)
    lp = np.clip(m.predict(Xte),0,None)
    raw = float(np.sqrt(np.mean((lp-yte)**2))); cal = calibrated(lp, yte)
    res[arm]=(raw,cal); print(f"{lab:28s} {Xtr.shape[1]:>10} {raw:>11.6f} {cal:>15.6f}")

print(f"\nR3 против БАЗЫ:               {res['r3'][1]-res['base'][1]:+.6f} калиброванно")
print(f"КОНТРОЛЬ против БАЗЫ:         {res['ctl'][1]-res['base'][1]:+.6f} калиброванно")
print(f"mdl_gypsum против КОНТРОЛЯ РАВНОЙ ЁМКОСТИ: {res['r3'][1]-res['ctl'][1]:+.6f}  <- РЕШАЮЩЕЕ ЧИСЛО")
print(f"(отрицательное = mdl_gypsum лучше; порог приёмки 0.0003, шум 0.000022)")

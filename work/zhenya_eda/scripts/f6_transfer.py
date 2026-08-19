"""F6. ПРОВЕРКА ПЕРЕНОСА для mdl_gypsum (визиты без действий).
Команда обожглась на матричных факторах: R^2 +0.0012 внутри окна и -0.031 при
разносе окон на 30+ дней. Тот же тест здесь.

Обучаем ОДНУ модель на срезах <= 2025-08-21, берём её остатки на двух якорях,
целевые окна которых НЕ пересекаются. Подбираем регрессию остатка на первом,
меряем на втором.
"""
import polars as pl, numpy as np, lightgbm as lgb
from datetime import date, timedelta
from sklearn.linear_model import Ridge
df = pl.read_parquet("train.parquet")
FEATW=(7,14,30,60,90,180,365)

def agg_base(A):
    hist = df.filter(pl.col("event_date")<=A); Ad=pl.lit(A); a=[]
    for w in FEATW:
        m = pl.col("event_date") >= A-timedelta(days=w-1)
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

def mdl_gypsum(A, uid):
    e = ((pl.col("searches")==0)&(pl.col("cat")==0)&(pl.col("to_cart")==0)&(pl.col("to_ord")==0))
    a=[]
    for w in (14,30,90,365):
        m = pl.col("event_date") >= A-timedelta(days=w-1)
        a += [e.filter(m).sum().alias(f"emp_{w}"),
              (e.filter(m).sum()/pl.max_horizontal(m.sum(),pl.lit(1))).alias(f"empsh_{w}"),
              ((pl.col("searches")>0)&(pl.col("cat")>0)).filter(m).sum().alias(f"both_{w}")]
    g = df.filter(pl.col("event_date")<=A).group_by("user_id").agg(a)
    M = pl.DataFrame({"user_id":uid}).join(g,on="user_id",how="left").drop("user_id").to_numpy().astype(float)
    M = np.nan_to_num(M); M = np.sign(M)*np.log1p(np.abs(M))
    return M

def tgt(A, uid):
    fut = df.filter(pl.col("event_date").is_between(A+timedelta(days=1),A+timedelta(days=30)))
    g = fut.group_by("user_id").agg(pl.col("gmv").sum().alias("y"))
    y = pl.DataFrame({"user_id":uid}).join(g,on="user_id",how="left")["y"].to_numpy().astype(float)
    return np.log1p(np.nan_to_num(y))

A1, A2 = date(2025,9,20), date(2025,11,1)     # целевые окна 09-21..10-20 и 11-02..12-01, НЕ пересекаются
TR_A = [date(2025,8,21)-timedelta(days=14*i) for i in range(0,7)]
print(f"обучение на {len(TR_A)} срезах {TR_A[-1]}..{TR_A[0]}; остатки на {A1} и {A2}")
Xs, ys = [], []
for a in TR_A:
    X = agg_base(a); Xs.append(X.drop("user_id")); ys.append(np.expm1(tgt(a, X["user_id"].to_numpy())))
m = lgb.LGBMRegressor(objective="tweedie", tweedie_variance_power=1.45, learning_rate=.05,
      num_leaves=63, min_child_samples=100, subsample=.8, colsample_bytree=.8,
      n_estimators=700, verbose=-1, n_jobs=4).fit(
      np.vstack([x.to_numpy().astype(float) for x in Xs]), np.log1p(np.concatenate(ys)))

D = {}
for tag, A in (("A1",A1), ("A2",A2)):
    X = agg_base(A); u = X["user_id"].to_numpy()
    pred = np.clip(m.predict(X.drop("user_id").to_numpy().astype(float)),0,None)
    y = tgt(A,u); D[tag] = (u, y-pred, mdl_gypsum(A,u))
    print(f"  {tag} {A}: n={len(u):,} RMSLE={np.sqrt(np.mean((y-pred)**2)):.6f}")

u1,r1,M1 = D["A1"]; u2,r2_,M2 = D["A2"]
common = np.intersect1d(u1,u2); i1 = np.searchsorted(u1,common); i2 = np.searchsorted(u2,common)
r1,M1,r2_,M2 = r1[i1],M1[i1],r2_[i2],M2[i2]
print(f"  общих юзеров: {len(common):,}")

def r2score(fitX, fity, evX, evy):
    mu, sd = fitX.mean(0), fitX.std(0)+1e-9
    md = Ridge(alpha=10.0).fit((fitX-mu)/sd, fity)
    p = md.predict((evX-mu)/sd)
    return 1 - np.sum((evy-p)**2)/np.sum((evy-evy.mean())**2)

rng = np.random.default_rng(0); n=len(common); idx=rng.permutation(n); h=n//2
print(f"\n{'тест':52s} {'mdl_flint':>11}")
print(f"{'ТО ЖЕ ОКНО, другая половина юзеров (режим ловушки)':52s} "
      f"{r2score(M1[idx[:h]], r1[idx[:h]], M1[idx[h:]], r1[idx[h:]]):>11.6f}")
print(f"{'ПЕРЕНОС на окно через 42 дня (честный тест)':52s} {r2score(M1, r1, M2, r2_):>11.6f}")
Pl = rng.normal(size=M1.shape); Pl2 = rng.normal(size=M2.shape)
print(f"{'ПЛАЦЕБО того же размера, перенос':52s} {r2score(Pl, r1, Pl2, r2_):>11.6f}")
print(f"{'обратный перенос A2 -> A1':52s} {r2score(M2, r2_, M1, r1):>11.6f}")

print("\n=== ТО ЖЕ, НО НА ЦЕНТРИРОВАННЫХ ОСТАТКАХ (уровень окна снят — его чинит LB-сдвиг) ===")
c1, c2 = r1 - r1.mean(), r2_ - r2_.mean()
print(f"  среднее остатка A1={r1.mean():+.4f}  A2={r2_.mean():+.4f}  (разница уровня {r2_.mean()-r1.mean():+.4f})")
print(f"{'то же окно, другая половина':52s} "
      f"{r2score(M1[idx[:h]], c1[idx[:h]], M1[idx[h:]], c1[idx[h:]]):>11.6f}")
t = r2score(M1, c1, M2, c2); p = r2score(Pl, c1, Pl2, c2)
print(f"{'ПЕРЕНОС A1 -> A2':52s} {t:>11.6f}")
print(f"{'ПЛАЦЕБО, перенос':52s} {p:>11.6f}")
print(f"{'обратный перенос A2 -> A1':52s} {r2score(M2, c2, M1, c1):>11.6f}")
print(f"\n  превышение над плацебо: {t-p:+.6f}")
sb = 1.6664
if t > 0:
    print(f"  перевод в метрику: {sb - sb*np.sqrt(1-t):+.6f} RMSLE")
    print(f"  порог приёмки команды 0.0003, уровень шума 0.000022")
else:
    print("  mdl_flint переноса отрицателен -> представление НЕ переносится между окнами, направление закрыто")

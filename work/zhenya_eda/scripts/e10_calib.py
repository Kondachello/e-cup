"""E10. ЦЕНА ПРОТОКОЛА. Команда калибрует на валидации, где исчезнувших НЕТ
(отбор юниверса), и применяет к тесту, где они ЕСТЬ. Сколько это стоит?

Зеркало: калибруем на якоре A1 двумя способами (только присутствующие = режим
валидации; вся популяция = честно) и применяем к якорю A2. Затем — глобальный
сдвиг, который команда получает замером на лидерборде, и смотрим, что осталось.
"""
import polars as pl, numpy as np, lightgbm as lgb
from datetime import date, timedelta
df = pl.read_parquet("train.parquet")
FEATW = (7,14,30,60,90,180,365)

def feats(A):
    hist = df.filter(pl.col("event_date") <= A); Ad = pl.lit(A); aggs=[]
    for w in FEATW:
        m = pl.col("event_date") >= A - timedelta(days=w-1)
        aggs += [m.sum().alias(f"act_{w}"), pl.col("gmv").filter(m).sum().alias(f"gmv_{w}"),
                 pl.col("to_ord").filter(m).sum().alias(f"ord_{w}"),
                 pl.col("to_cart").filter(m).sum().alias(f"cart_{w}"),
                 pl.col("searches").filter(m).sum().alias(f"srch_{w}"),
                 (pl.col("to_ord")>0).filter(m).sum().alias(f"odays_{w}")]
    aggs += [(Ad-pl.col("event_date").max()).dt.total_days().alias("rec_act"),
             (Ad-pl.col("event_date").filter(pl.col("to_ord")>0).max()).dt.total_days().alias("rec_ord"),
             (Ad-pl.col("event_date").filter(pl.col("to_cart")>0).max()).dt.total_days().alias("rec_cart"),
             (Ad-pl.col("event_date").min()).dt.total_days().alias("tenure"), pl.len().alias("n_days"),
             pl.col("event_date").diff().dt.total_days().mean().alias("gap_mean"),
             pl.col("event_date").diff().dt.total_days().max().alias("gap_max"),
             pl.col("event_date").diff().dt.total_days().std().alias("gap_std")]
    X = hist.sort(["user_id","event_date"]).group_by("user_id").agg(aggs)
    sel = df.filter(pl.col("event_date").is_between(A-timedelta(days=29), A))["user_id"].unique()
    return X.filter(pl.col("user_id").is_in(sel.implode()))

def tgt(A, uid):
    fut = df.filter(pl.col("event_date").is_between(A+timedelta(days=1), A+timedelta(days=30)))
    g = fut.group_by("user_id").agg(pl.col("gmv").sum().alias("y"))
    y = pl.DataFrame({"user_id": uid}).join(g, on="user_id", how="left")["y"].to_numpy().astype(np.float64)
    return np.nan_to_num(y), np.isnan(y)

A1, A2 = date(2025,8,9), date(2025,10,11)          # A1 = «валидация», A2 = «тест»
TRAIN = [A1 - timedelta(days=30+14*i) for i in range(1,8)]
print(f"обучение на {len(TRAIN)} якорях {TRAIN[-1]}..{TRAIN[0]}; калибровка на {A1}; замер на {A2}")
Xs, ys = [], []
for a in TRAIN:
    X = feats(a); y,_ = tgt(a, X["user_id"].to_numpy()); Xs.append(X.drop("user_id")); ys.append(y)
FE = list(Xs[0].columns)
m = lgb.LGBMRegressor(objective="tweedie", tweedie_variance_power=1.45, learning_rate=0.05,
        num_leaves=63, min_child_samples=100, subsample=.8, colsample_bytree=.8,
        n_estimators=700, verbose=-1, n_jobs=4).fit(
        np.vstack([x.to_numpy().astype(np.float64) for x in Xs]), np.log1p(np.concatenate(ys)))

out = {}
for tag, A in (("cal", A1), ("test", A2)):
    X = feats(A); u = X["user_id"].to_numpy(); y, v = tgt(A, u)
    out[tag] = (np.clip(m.predict(X.drop("user_id").to_numpy().astype(np.float64)),0,None), np.log1p(y), v)
(lc, lyc, vc), (lt, lyt, vt) = out["cal"], out["test"]
print(f"исчезнувших: на калибровочном {100*vc.mean():.2f}%, на замерном {100*vt.mean():.2f}%")

def fit_binned(l, ly, msk, nb=24):
    q = np.quantile(l[msk], np.linspace(0,1,nb+1)); q[0],q[-1] = -np.inf,np.inf
    b = np.digitize(l[msk], q[1:-1])
    sh = np.array([np.mean(ly[msk][b==k]-l[msk][b==k]) if (b==k).sum()>50 else 0.0 for k in range(nb)])
    return q, sh
def apply_binned(l, qs):
    q, sh = qs; return l + sh[np.digitize(l, q[1:-1])]

rm = lambda l: float(np.sqrt(np.mean((l-lyt)**2)))
present = ~vc
cal_valmode = fit_binned(lc, lyc, present)      # режим команды: только присутствующие
cal_honest  = fit_binned(lc, lyc, np.ones_like(vc, bool))
pv_, ph_ = apply_binned(lt, cal_valmode), apply_binned(lt, cal_honest)
print(f"\nбез калибровки                       {rm(lt):.6f}")
print(f"калибровка ВАЛ-РЕЖИМА (как у команды) {rm(pv_):.6f}")
print(f"калибровка ЧЕСТНАЯ (с исчезнувшими)   {rm(ph_):.6f}   разница {rm(pv_)-rm(ph_):+.6f}")

# теперь глобальный сдвиг, который даёт замер на лидерборде (оба файла к истинному среднему)
def shift_to_mean(l): return l + (lyt.mean() - l.mean())
sv, sh_ = shift_to_mean(pv_), shift_to_mean(ph_)
print(f"\nпосле глобального сдвига по замеру среднего (то, что делает LB):")
print(f"  вал-режим {rm(sv):.6f}   честная {rm(sh_):.6f}   ОСТАТОЧНАЯ разница {rm(sv)-rm(sh_):+.6f}")
print(f"\nсдвиг, который потребовался вал-режимной калибровке: {lyt.mean()-pv_.mean():+.4f}")
print(f"сдвиг, который потребовался честной:                 {lyt.mean()-ph_.mean():+.4f}")

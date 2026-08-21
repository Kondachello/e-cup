"""E9. ЧЕСТНЫЙ ЗАМЕР ВЫИГРЫША. Даёт ли поправка, обусловленная вероятностью
ИСЧЕЗНОВЕНИЯ, что-то СВЕРХ обычной калибровки по уровню прогноза?

Протокол: зазор 30 дней; обучение на якорях <= A-30; замер на якоре A;
все поправки подбираются на половине юзеров и меряются на другой.
Плацебо = та же процедура, но обусловленная самим прогнозом (то, что команда
уже делает и считает исчерпанным).
"""
import os
import polars as pl, numpy as np, lightgbm as lgb
from datetime import date, timedelta
from sklearn.metrics import roc_auc_score

df = pl.read_parquet("train.parquet")
FEATW = (7, 14, 30, 60, 90, 180, 365)

def feats(A: date) -> pl.DataFrame:
    hist = df.filter(pl.col("event_date") <= A)
    Ad = pl.lit(A); aggs = []
    for w in FEATW:
        m = pl.col("event_date") >= A - timedelta(days=w-1)
        aggs += [m.sum().alias(f"act_{w}"), pl.col("gmv").filter(m).sum().alias(f"gmv_{w}"),
                 pl.col("to_ord").filter(m).sum().alias(f"ord_{w}"),
                 pl.col("to_cart").filter(m).sum().alias(f"cart_{w}"),
                 pl.col("searches").filter(m).sum().alias(f"srch_{w}"),
                 (pl.col("to_ord") > 0).filter(m).sum().alias(f"odays_{w}")]
    aggs += [(Ad - pl.col("event_date").max()).dt.total_days().alias("rec_act"),
             (Ad - pl.col("event_date").filter(pl.col("to_ord") > 0).max()).dt.total_days().alias("rec_ord"),
             (Ad - pl.col("event_date").filter(pl.col("to_cart") > 0).max()).dt.total_days().alias("rec_cart"),
             (Ad - pl.col("event_date").min()).dt.total_days().alias("tenure"), pl.len().alias("n_days"),
             pl.col("event_date").diff().dt.total_days().mean().alias("gap_mean"),
             pl.col("event_date").diff().dt.total_days().max().alias("gap_max"),
             pl.col("event_date").diff().dt.total_days().std().alias("gap_std")]
    X = hist.sort(["user_id","event_date"]).group_by("user_id").agg(aggs)
    sel = df.filter(pl.col("event_date").is_between(A-timedelta(days=29), A))["user_id"].unique()
    return X.filter(pl.col("user_id").is_in(sel))

def targets(A: date, uid: np.ndarray):
    fut = df.filter(pl.col("event_date").is_between(A+timedelta(days=1), A+timedelta(days=30)))
    g = fut.group_by("user_id").agg(pl.col("gmv").sum().alias("y"))
    t = pl.DataFrame({"user_id": uid}).join(g, on="user_id", how="left")
    y = t["y"].to_numpy().astype(np.float64)
    return np.nan_to_num(y), np.isnan(y)   # y, vanished

A = date(2025, 10, 11)
TRAIN = [A - timedelta(days=30 + 14*i) for i in range(1, 9)]
print(f"замерный якорь {A}; обучающих якорей {len(TRAIN)} ({TRAIN[-1]} .. {TRAIN[0]}), зазор 30")

Xs, ys, vs = [], [], []
for a in TRAIN:
    X = feats(a); u = X["user_id"].to_numpy(); y, v = targets(a, u)
    Xs.append(X.drop("user_id")); ys.append(y); vs.append(v)
FE = Xs[0].columns
Xtr = np.vstack([x.to_numpy().astype(np.float64) for x in Xs])
ytr = np.concatenate(ys); vtr = np.concatenate(vs)
Xte_df = feats(A); uid = Xte_df["user_id"].to_numpy()
Xte = Xte_df.drop("user_id").to_numpy().astype(np.float64)
yte, vte = targets(A, uid)
print(f"train {Xtr.shape}  test {Xte.shape}  исчезли в тесте {vte.mean()*100:.2f}%")

P = dict(objective="tweedie", tweedie_variance_power=1.45, learning_rate=0.05,
         num_leaves=63, min_child_samples=100, subsample=0.8, colsample_bytree=0.8,
         n_estimators=700, verbose=-1, n_jobs=4)
m = lgb.LGBMRegressor(**P).fit(Xtr, np.log1p(ytr), feature_name=list(FE))
lp = np.clip(m.predict(Xte), 0, None)
mv = lgb.LGBMClassifier(objective="binary", learning_rate=0.05, num_leaves=63,
                        min_child_samples=100, n_estimators=500, verbose=-1, n_jobs=4).fit(Xtr, vtr)
pv = mv.predict_proba(Xte)[:, 1]
np.savez(os.environ.get("ZH_OUT", "work/zhenya_eda/out") + "/e9_cache.npz", lp=lp, pv=pv, yte=yte, vte=vte, uid=uid)
ly = np.log1p(yte)
print(f"AUC исчезновения на замерном якоре: {roc_auc_score(vte, pv):.4f}")

rm = lambda l: float(np.sqrt(np.mean((l - ly) ** 2)))
rng = np.random.default_rng(0); idx = rng.permutation(len(ly)); h = len(ly)//2; F, G = idx[:h], idx[h:]

def affine(l, tr, te):
    b = np.polyfit(l[tr], ly[tr], 1)
    return b[0]*l[te] + b[1]
base = np.empty_like(lp)
for tr, te in ((F, G), (G, F)): base[te] = affine(lp, tr, te)
print(f"\nсырой RMSLE            {rm(lp):.6f}")
print(f"после глобального аффина {rm(base):.6f}   <- база (то, что даёт LB-сдвиг)")

def binned(cond, nb=20):
    """поправка среднего остатка по децилям cond, подбор на одной половине, замер на другой"""
    out = base.copy()
    for tr, te in ((F, G), (G, F)):
        q = np.quantile(cond[tr], np.linspace(0, 1, nb+1)); q[0], q[-1] = -np.inf, np.inf
        btr = np.digitize(cond[tr], q[1:-1]); bte = np.digitize(cond[te], q[1:-1])
        sh = np.zeros(nb)
        for b in range(nb):
            msk = btr == b
            if msk.sum() > 50: sh[b] = np.mean(ly[tr][msk] - base[tr][msk])
        out[te] = base[te] + sh[bte]
    return rm(out)

r_pred = binned(base)
r_van  = binned(pv)
print(f"\n+ поправка по УРОВНЮ ПРОГНОЗА (плацебо, команда это делает)  {r_pred:.6f}  ({rm(base)-r_pred:+.6f})")
print(f"+ поправка по P(ИСЧЕЗНЕТ)                                      {r_van:.6f}  ({rm(base)-r_van:+.6f})")

# совместная: сначала уровень, потом остаток по p_vanish
out = base.copy()
for tr, te in ((F, G), (G, F)):
    q = np.quantile(base[tr], np.linspace(0,1,21)); q[0],q[-1] = -np.inf, np.inf
    b1, b2 = np.digitize(base[tr], q[1:-1]), np.digitize(base[te], q[1:-1])
    sh = np.array([np.mean(ly[tr][b1==b]-base[tr][b1==b]) if (b1==b).sum()>50 else 0 for b in range(20)])
    out[te] = base[te] + sh[b2]
step1 = out.copy()
for tr, te in ((F, G), (G, F)):
    q = np.quantile(pv[tr], np.linspace(0,1,21)); q[0],q[-1] = -np.inf, np.inf
    b1, b2 = np.digitize(pv[tr], q[1:-1]), np.digitize(pv[te], q[1:-1])
    sh = np.array([np.mean(ly[tr][b1==b]-step1[tr][b1==b]) if (b1==b).sum()>50 else 0 for b in range(20)])
    out[te] = step1[te] + sh[b2]
print(f"уровень, ЗАТЕМ P(исчезнет) СВЕРХУ                             {rm(out):.6f}  "
      f"(сверх уровня {r_pred-rm(out):+.6f})")

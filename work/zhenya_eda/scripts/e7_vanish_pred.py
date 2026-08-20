"""E7. РЕШАЮЩИЙ ТЕСТ. Нулевая масса теста состоит из ДВУХ режимов:
   (1) юзер присутствует, но не покупает   <- ЕДИНСТВЕННЫЙ режим в валидации команды
   (2) юзер исчезает целиком               <- в валидации ОТСУТСТВУЕТ по построению отбора
Вопрос: различаются ли они по предсказуемости? Команда измерила потолок AUC 0.6286
в «зоне нулей» — но измеряла его на популяции БЕЗ режима (2).
"""
import polars as pl, numpy as np
from datetime import date, timedelta
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

df = pl.read_parquet("train.parquet")

def build(A: date):
    hist = df.filter(pl.col("event_date") <= A)
    Ad = pl.lit(A)
    aggs = []
    for w in (7, 14, 30, 60, 90, 180, 365):
        m = pl.col("event_date") >= A - timedelta(days=w-1)
        aggs += [m.sum().alias(f"act_{w}"),
                 pl.col("gmv").filter(m).sum().alias(f"gmv_{w}"),
                 pl.col("to_ord").filter(m).sum().alias(f"ord_{w}"),
                 pl.col("to_cart").filter(m).sum().alias(f"cart_{w}"),
                 pl.col("searches").filter(m).sum().alias(f"srch_{w}")]
    aggs += [
        (Ad - pl.col("event_date").max()).dt.total_days().alias("rec_act"),
        (Ad - pl.col("event_date").filter(pl.col("to_ord") > 0).max()).dt.total_days().alias("rec_ord"),
        (Ad - pl.col("event_date").filter(pl.col("to_cart") > 0).max()).dt.total_days().alias("rec_cart"),
        (Ad - pl.col("event_date").min()).dt.total_days().alias("tenure"),
        pl.len().alias("n_days"),
        pl.col("event_date").diff().dt.total_days().mean().alias("gap_mean"),
        pl.col("event_date").diff().dt.total_days().max().alias("gap_max"),
        pl.col("event_date").diff().dt.total_days().std().alias("gap_std"),
    ]
    X = hist.sort(["user_id","event_date"]).group_by("user_id").agg(aggs)
    sel = set(df.filter(pl.col("event_date").is_between(A-timedelta(days=29), A))["user_id"].unique().to_list())
    X = X.filter(pl.col("user_id").is_in(sel))
    fut = df.filter(pl.col("event_date").is_between(A+timedelta(days=1), A+timedelta(days=30)))
    pres = set(fut["user_id"].unique().to_list())
    buy = set(fut.group_by("user_id").agg(pl.col("gmv").sum()).filter(pl.col("gmv")>0)["user_id"].to_list())
    uid = X["user_id"].to_numpy()
    return X, uid, np.array([u not in pres for u in uid]), np.array([u not in buy for u in uid])

def auc_of(X, y, cols):
    M = X.select(cols).to_numpy().astype(np.float64)
    M = np.nan_to_num(M, nan=-1.0, posinf=1e6, neginf=-1e6)
    M = np.sign(M)*np.log1p(np.abs(M))
    rng = np.random.default_rng(0); idx = rng.permutation(len(y)); h = len(y)//2
    tr, te = idx[:h], idx[h:]
    sc = StandardScaler().fit(M[tr])
    lr = LogisticRegression(max_iter=1000, C=1.0).fit(sc.transform(M[tr]), y[tr])
    return roc_auc_score(y[te], lr.predict_proba(sc.transform(M[te]))[:,1])

for A in [date(2025,6,7), date(2025,8,9), date(2025,10,11)]:
    X, uid, y_van, y_zero = build(A)
    cols = [c for c in X.columns if c != "user_id"]
    n = len(uid)
    present = ~y_van
    y_zero_p = y_zero[present]           # нулевые СРЕДИ присутствующих = режим валидации
    print(f"\n=== якорь {A}  популяция {n:,} ===")
    print(f"  исчезли:            {y_van.sum():>7,} ({100*y_van.mean():5.2f}%)")
    print(f"  нулевой gmv всего:  {y_zero.sum():>7,} ({100*y_zero.mean():5.2f}%)")
    print(f"  из них исчезнувшие: {(y_zero & y_van).sum():>7,} = {100*(y_zero&y_van).sum()/y_zero.sum():5.2f}% всей нулевой массы")
    print(f"  AUC «ИСЧЕЗНЕТ»                       : {auc_of(X, y_van, cols):.4f}")
    print(f"  AUC «нулевой gmv» (вся популяция)    : {auc_of(X, y_zero, cols):.4f}")
    Xp = X.filter(pl.Series(present))
    print(f"  AUC «не купит» СРЕДИ ПРИСУТСТВУЮЩИХ  : {auc_of(Xp, y_zero_p, cols):.4f}   <- режим валидации")

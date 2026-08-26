"""V1. Поправка на слом логирования каталога 01.04.2025.

Замерено (t1_verify): cat -18.5%, cat_to_ord -16.4%, cat_to_cart -16.2% ровно
на границе 01.04.2025, при том что search -2.1% и to_ord +6.0%. Слом специфичен
для каталожного канала.

Доля 365-дневного окна в дословной эре падает монотонно:
   обучающие якоря 40.4% -> 25.8%,  валидация 20.8%,  ТЕСТ 12.6%
То есть модель учится на завышенной каталожной истории и применяется к менее
завышенной. Это систематический дрейф, а не поведение.

Поправка: каталожные счётчики до 01.04.2025 умножаются на 0.82.
Две руки, идентичные во всём кроме поправки; сравнение ПОСЛЕ калибровки.
"""
import os
import numpy as np, polars as pl, lightgbm as lgb
from datetime import date, timedelta
from pathlib import Path

OUT = Path(os.environ.get("ZH_OUT", "work/zhenya_eda/out")); OUT.mkdir(parents=True, exist_ok=True)
BREAK = date(2025, 4, 1)
SCALE = 0.82
W = (7, 14, 30, 60, 90, 180, 365)
VAL = date(2026, 1, 14)
TRAIN = [VAL - timedelta(days=30 + 14 * i) for i in range(0, 10)]
SEEDS = (42, 555)

df = pl.read_parquet("train.parquet")
CATCOLS = ["cat", "cat_to_ord", "cat_to_cart", "gmv_cat"]


def build(A: date, fix: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    h = df.filter(pl.col("event_date") <= A)
    if fix:                       # масштабируем каталожные счётчики дословной эры
        pre = pl.col("event_date") < BREAK
        h = h.with_columns([
            pl.when(pre).then(pl.col(c) * SCALE).otherwise(pl.col(c)).alias(c) for c in CATCOLS])
    Ad = pl.lit(A); a = []
    for w in W:
        m = pl.col("event_date") >= A - timedelta(days=w - 1)
        a += [m.sum().alias(f"act_{w}"), pl.col("gmv").filter(m).sum().alias(f"gmv_{w}"),
              pl.col("to_ord").filter(m).sum().alias(f"ord_{w}"),
              pl.col("to_cart").filter(m).sum().alias(f"cart_{w}"),
              pl.col("searches").filter(m).sum().alias(f"srch_{w}"),
              (pl.col("to_ord") > 0).filter(m).sum().alias(f"od_{w}"),
              pl.col("gmv_cat").filter(m).sum().alias(f"gc_{w}"),
              (pl.col("cat") > 0).filter(m).sum().alias(f"cd_{w}"),
              pl.col("cat_to_ord").filter(m).sum().alias(f"c2o_{w}"),
              pl.col("cat_to_cart").filter(m).sum().alias(f"c2c_{w}")]
    a += [(Ad - pl.col("event_date").max()).dt.total_days().alias("rec_act"),
          (Ad - pl.col("event_date").filter(pl.col("to_ord") > 0).max()).dt.total_days().alias("rec_ord"),
          (Ad - pl.col("event_date").filter(pl.col("cat") > 0).max()).dt.total_days().alias("rec_cat"),
          (Ad - pl.col("event_date").min()).dt.total_days().alias("tenure"), pl.len().alias("nd"),
          pl.col("event_date").diff().dt.total_days().mean().alias("gm"),
          pl.col("event_date").diff().dt.total_days().max().alias("gx")]
    X = h.sort(["user_id", "event_date"]).group_by("user_id").agg(a)
    sel = df.filter(pl.col("event_date").is_between(A - timedelta(days=29), A))["user_id"].unique()
    X = X.filter(pl.col("user_id").is_in(sel.implode())).sort("user_id")
    fut = df.filter(pl.col("event_date").is_between(A + timedelta(days=1), A + timedelta(days=30)))
    g = fut.group_by("user_id").agg(pl.col("gmv").sum().alias("y"))
    y = X.select("user_id").join(g, on="user_id", how="left")["y"].to_numpy().astype(float)
    M = np.nan_to_num(X.drop("user_id").to_numpy().astype(np.float32), nan=-1., posinf=1e9, neginf=-1e9)
    return M, np.log1p(np.nan_to_num(y)), X["user_id"].to_numpy()


def calibrated(lp, ly, nb=24):
    rng = np.random.default_rng(0); i = rng.permutation(len(ly)); h = len(ly) // 2
    out = lp.copy()
    for tr, te in ((i[:h], i[h:]), (i[h:], i[:h])):
        q = np.quantile(lp[tr], np.linspace(0, 1, nb + 1)); q[0], q[-1] = -np.inf, np.inf
        b1, b2 = np.digitize(lp[tr], q[1:-1]), np.digitize(lp[te], q[1:-1])
        sh = np.array([np.mean(ly[tr][b1 == k] - lp[tr][b1 == k]) if (b1 == k).sum() > 50 else 0.
                       for k in range(nb)])
        out[te] = lp[te] + sh[b2]
    return float(np.sqrt(np.mean((out - ly) ** 2))), out


res = {}
for fix in (False, True):
    tag = "С ПОПРАВКОЙ" if fix else "без поправки"
    Xs, ys = [], []
    for a in TRAIN:
        M, y, _ = build(a, fix); Xs.append(M); ys.append(y)
    Xtr = np.vstack(Xs); ytr = np.concatenate(ys)
    Xv, yv, uidv = build(VAL, fix)
    raws, cals, preds = [], [], []
    for s in SEEDS:
        m = lgb.LGBMRegressor(objective="tweedie", tweedie_variance_power=1.45, learning_rate=.05,
                              num_leaves=63, min_child_samples=100, subsample=.8,
                              colsample_bytree=.8, n_estimators=700, verbose=-1, n_jobs=4,
                              random_state=s).fit(Xtr, ytr)
        lp = np.clip(m.predict(Xv), 0, None)
        c, out = calibrated(lp, yv)
        raws.append(float(np.sqrt(np.mean((lp - yv) ** 2)))); cals.append(c); preds.append(out)
    res[fix] = (np.mean(raws), np.mean(cals), np.mean(preds, axis=0), uidv, yv)
    print(f"{tag:14s} признаков {Xtr.shape[1]}  сырой {np.mean(raws):.6f}  "
          f"калиброванный {np.mean(cals):.6f}  разброс сидов {np.std(cals):.6f}", flush=True)

d = res[True][1] - res[False][1]
print(f"\nПОПРАВКА против КОНТРОЛЯ: {d:+.6f} калиброванно")
print(f"  (отрицательное = поправка помогает; порог шума 0.000022)")
np.savez(OUT / "v1_erafix.npz", uid=res[True][3], y=res[True][4],
         pred_fix=res[True][2], pred_ctl=res[False][2])

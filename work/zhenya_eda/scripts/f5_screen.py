"""F5. Скрининг представлений по методике команды (screen_repr.py):
регрессия ОСТАТКА ДЕЙСТВУЮЩЕГО БЛЕНДА на представление, честный разрез по юзерам,
плацебо той же размерности. Положительный R^2 вне выборки при отрицательном плацебо
= сигнал вне линейной оболочки бленда.

ВНИМАНИЕ: у метода известен ложноположительный режим (разрез по юзерам не контролирует
окно). Всё, что сработает, обязано пройти проверку переноса между окнами.
"""
import polars as pl, numpy as np
from datetime import date, timedelta
from sklearn.linear_model import Ridge

A = date(2026,1,14)
v = pl.read_parquet("work/preds_pack/val_preds.parquet").sort("user_id")
uid = v["user_id"].to_numpy()
ly = np.log1p(np.clip(v["target"].to_numpy().astype(np.float64),0,None))
blend = v["blend"].to_numpy().astype(np.float64)
resid = ly - blend
sb = float(np.sqrt(np.mean(resid**2)))
print(f"бленд val RMSLE={sb:.6f}  n={len(uid):,}")

df = pl.read_parquet("train.parquet").filter(pl.col("event_date")<=A)
base = pl.DataFrame({"user_id": uid})

def R_latency():
    """R1. Задержка корзина->заказ. Временная микроструктура, в агрегатах её нет."""
    d = df.filter((pl.col("to_cart")>0)|(pl.col("to_ord")>0)).sort(["user_id","event_date"])
    d = d.with_columns([
        pl.when(pl.col("to_cart")>0).then(pl.col("event_date")).otherwise(None)
          .forward_fill().over("user_id").alias("last_cart")])
    d = d.filter(pl.col("to_ord")>0).with_columns(
        (pl.col("event_date")-pl.col("last_cart")).dt.total_days().alias("lag"))
    g = d.group_by("user_id").agg([
        pl.col("lag").mean().alias("lag_mean"), pl.col("lag").median().alias("lag_med"),
        pl.col("lag").std().alias("lag_std"), pl.col("lag").max().alias("lag_max"),
        (pl.col("lag")==0).mean().alias("lag_same"), (pl.col("lag")<=1).mean().alias("lag_fast"),
        pl.len().alias("lag_n")])
    return base.join(g, on="user_id", how="left")

def R_runs():
    """mdl_flint. Длины серий подряд идущих активных дней и пауз."""
    d = df.select(["user_id","event_date"]).sort(["user_id","event_date"])
    d = d.with_columns(pl.col("event_date").diff().dt.total_days().over("user_id").alias("g"))
    d = d.with_columns(((pl.col("g")>1)|pl.col("g").is_null()).cum_sum().over("user_id").alias("run"))
    r = d.group_by(["user_id","run"]).len().rename({"len":"L"})
    g = r.group_by("user_id").agg([
        pl.col("L").mean().alias("run_mean"), pl.col("L").max().alias("run_max"),
        pl.col("L").std().alias("run_std"), pl.len().alias("run_n"),
        (pl.col("L")==1).mean().alias("run_solo")])
    gaps = d.filter(pl.col("g")>1).group_by("user_id").agg([
        pl.col("g").mean().alias("gap_mean2"), pl.col("g").max().alias("gap_max2"),
        pl.col("g").std().alias("gap_std2"), pl.col("g").quantile(.9).alias("gap_p90")])
    return base.join(g, on="user_id", how="left").join(gaps, on="user_id", how="left")

def R_empty():
    """mdl_gypsum. Визиты без единого действия (14.85% строк), по окнам."""
    e = ((pl.col("searches")==0)&(pl.col("cat")==0)&(pl.col("to_cart")==0)&(pl.col("to_ord")==0))
    aggs=[]
    for w in (14,30,90,365):
        m = pl.col("event_date") >= A - timedelta(days=w-1)
        aggs += [e.filter(m).sum().alias(f"emp_{w}"),
                 (e.filter(m).sum()/pl.max_horizontal(m.sum(),pl.lit(1))).alias(f"empsh_{w}"),
                 ((pl.col("searches")>0)&(pl.col("cat")>0)).filter(m).sum().alias(f"both_{w}")]
    g = df.group_by("user_id").agg(aggs)
    return base.join(g, on="user_id", how="left")

def R_pricedist():
    """R4. ФОРМА распределения цен юзера (у команды только суммы и средние)."""
    d = df.filter(pl.col("to_ord")>0).with_columns((pl.col("gmv")/pl.col("to_ord")).alias("ap"))
    g = d.group_by("user_id").agg([
        pl.col("ap").log1p().mean().alias("lap_mean"), pl.col("ap").log1p().std().alias("lap_std"),
        pl.col("ap").quantile(.1).log1p().alias("lap_p10"), pl.col("ap").quantile(.5).log1p().alias("lap_p50"),
        pl.col("ap").quantile(.9).log1p().alias("lap_p90"),
        (pl.col("ap").max()/(pl.col("ap").min()+1e-9)).log1p().alias("lap_rng"),
        pl.col("to_ord").mean().alias("items_mean"), pl.col("to_ord").max().alias("items_max")])
    return base.join(g, on="user_id", how="left")

def R_weekday():
    """mdl_gneis2. Персональная недельная фаза активности и покупок."""
    d = df.with_columns(pl.col("event_date").dt.weekday().alias("wd"))
    piv = d.group_by(["user_id","wd"]).agg(pl.len().alias("n")).pivot(
        on="wd", index="user_id", values="n").fill_null(0)
    piv = piv.rename({c: f"wd_{c}" for c in piv.columns if c != "user_id"})
    return base.join(piv, on="user_id", how="left")

REPS = {"R1 задержка корзина->заказ": R_latency, "mdl_flint серии активных дней": R_runs,
        "mdl_gypsum визиты без действий": R_empty, "R4 форма распределения цен": R_pricedist,
        "mdl_gneis2 недельная фаза": R_weekday}

rng = np.random.default_rng(0); idx = rng.permutation(len(uid)); h = len(uid)//2
TR, TE = idx[:h], idx[h:]
print(f"\n{'представление':32s} {'k':>3} {'mdl_flint вне выборки':>15} {'плацебо':>10} {'выигрыш':>10}  вердикт")
for name, fn in REPS.items():
    M = fn().drop("user_id").to_numpy().astype(np.float64)
    M = np.nan_to_num(M, nan=0.0, posinf=0.0, neginf=0.0)
    M = np.sign(M)*np.log1p(np.abs(M))
    M = (M - M[TR].mean(0)) / (M[TR].std(0) + 1e-9)
    def r2(X):
        m = Ridge(alpha=10.0).fit(X[TR], resid[TR])
        pr = m.predict(X[TE])
        return 1 - np.sum((resid[TE]-pr)**2)/np.sum((resid[TE]-resid[TR].mean())**2)
    a = r2(M)
    P = rng.normal(size=M.shape)
    b = r2(P)
    gain = sb - sb*np.sqrt(max(1-a, 0)) if a > 0 else 0.0
    verd = "СИГНАЛ" if (a > 0 and a - b > 3e-4) else "ноль"
    print(f"{name:32s} {M.shape[1]:>3} {a:>15.6f} {b:>10.6f} {gain:>10.6f}  {verd}")

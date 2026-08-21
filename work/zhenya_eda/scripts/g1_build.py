"""G1. Сборка кэша признаков на все якоря. Один раз — дальше опыты мгновенные.

БАЗА  — аналог 203-признакового набора команды (суммы/счётчики/давности/тренды).
DAYTYPE — таксономия дня, найденная в разведке:
   A пустой визит (ни поиска, ни каталога, ни корзины, ни заказа) — 14.85% строк
   B только поиск/каталог без корзины                              — 44.86%
   C корзина без заказа                                            — 21.47%
   D заказ                                                         — 15.46%
   у команды есть D (ord_days) и C+D (cart_days) только для 30/90/365; A нет вовсе.
TRANS — переходы между типами дней подряд (марковская структура).
CTL   — контроль равной ёмкости: столько же СТАРЫХ величин на тех же окнах.
"""
import os
import polars as pl, numpy as np, sys
from datetime import date, timedelta
from pathlib import Path

df = pl.read_parquet("train.parquet")
CACHE = Path(os.environ.get("ZH_CACHE", "work/zhenya_eda/cache")); CACHE.mkdir(exist_ok=True, parents=True)
VAL, TEST = date(2026, 1, 14), date(2026, 2, 13)
W = (7, 14, 30, 60, 90, 180, 365)
DW = (7, 14, 30, 60, 90, 180, 365)

A_EXPR = (pl.col("searches") == 0) & (pl.col("cat") == 0) & (pl.col("to_cart") == 0) & (pl.col("to_ord") == 0)
C_EXPR = (pl.col("to_cart") > 0) & (pl.col("to_ord") == 0)
D_EXPR = pl.col("to_ord") > 0
B_EXPR = ~A_EXPR & ~C_EXPR & ~D_EXPR


def build(A: date) -> pl.DataFrame:
    hist = df.filter(pl.col("event_date") <= A)
    Ad = pl.lit(A)
    a = []
    # ---- БАЗА ----
    for w in W:
        m = pl.col("event_date") >= A - timedelta(days=w - 1)
        a += [m.sum().alias(f"b_act_{w}"),
              pl.col("gmv").filter(m).sum().alias(f"b_gmv_{w}"),
              pl.col("to_ord").filter(m).sum().alias(f"b_ord_{w}"),
              pl.col("to_cart").filter(m).sum().alias(f"b_cart_{w}"),
              pl.col("searches").filter(m).sum().alias(f"b_srch_{w}"),
              (pl.col("to_ord") > 0).filter(m).sum().alias(f"b_od_{w}"),
              pl.col("gmv_search").filter(m).sum().alias(f"b_gs_{w}"),
              pl.col("gmv_cat").filter(m).sum().alias(f"b_gc_{w}")]
    for s, e in [(59, 30), (89, 60), (179, 90), (364, 180)]:
        m = pl.col("event_date").is_between(A - timedelta(days=s), A - timedelta(days=e))
        a += [pl.col("gmv").filter(m).sum().alias(f"b_gmv_b{s}"),
              m.sum().alias(f"b_act_b{s}"),
              pl.col("to_ord").filter(m).sum().alias(f"b_ord_b{s}")]
    a += [(Ad - pl.col("event_date").max()).dt.total_days().alias("b_rec_act"),
          (Ad - pl.col("event_date").filter(D_EXPR).max()).dt.total_days().alias("b_rec_ord"),
          (Ad - pl.col("event_date").filter(pl.col("to_cart") > 0).max()).dt.total_days().alias("b_rec_cart"),
          (Ad - pl.col("event_date").filter(pl.col("searches") > 0).max()).dt.total_days().alias("b_rec_srch"),
          (Ad - pl.col("event_date").filter(pl.col("cat") > 0).max()).dt.total_days().alias("b_rec_cat"),
          (Ad - pl.col("event_date").min()).dt.total_days().alias("b_tenure"),
          pl.len().alias("b_nd"),
          pl.col("event_date").diff().dt.total_days().mean().alias("b_gm"),
          pl.col("event_date").diff().dt.total_days().std().alias("b_gsd"),
          pl.col("event_date").diff().dt.total_days().max().alias("b_gx"),
          pl.col("event_date").filter(D_EXPR).diff().dt.total_days().mean().alias("b_ogm"),
          pl.col("event_date").filter(D_EXPR).diff().dt.total_days().std().alias("b_ogs")]
    # ---- DAYTYPE ----
    for w in DW:
        m = pl.col("event_date") >= A - timedelta(days=w - 1)
        n = pl.max_horizontal(m.sum(), pl.lit(1))
        for tag, ex in (("A", A_EXPR), ("B", B_EXPR), ("C", C_EXPR), ("D", D_EXPR)):
            a += [ex.filter(m).sum().alias(f"dt_{tag}_{w}"),
                  (ex.filter(m).sum() / n).alias(f"dt_{tag}sh_{w}")]
    a += [(Ad - pl.col("event_date").filter(A_EXPR).max()).dt.total_days().alias("dt_rec_A"),
          (Ad - pl.col("event_date").filter(C_EXPR).max()).dt.total_days().alias("dt_rec_C")]
    # ---- CTL: равная ёмкость, старые величины ----
    for w in DW:
        m = pl.col("event_date") >= A - timedelta(days=w - 1)
        n = pl.max_horizontal(m.sum(), pl.lit(1))
        a += [pl.col("search_to_cart").filter(m).sum().alias(f"ct_s2c_{w}"),
              (pl.col("search_to_cart").filter(m).sum() / n).alias(f"ct_s2csh_{w}"),
              pl.col("cat_to_cart").filter(m).sum().alias(f"ct_c2c_{w}"),
              (pl.col("cat_to_cart").filter(m).sum() / n).alias(f"ct_c2csh_{w}"),
              pl.col("search_to_ord").filter(m).sum().alias(f"ct_s2o_{w}"),
              (pl.col("search_to_ord").filter(m).sum() / n).alias(f"ct_s2osh_{w}"),
              pl.col("cat_to_ord").filter(m).sum().alias(f"ct_c2o_{w}"),
              (pl.col("cat_to_ord").filter(m).sum() / n).alias(f"ct_c2osh_{w}")]
    a += [pl.col("gmv").filter(D_EXPR).mean().alias("ct_aov"),
          pl.col("gmv").filter(D_EXPR).max().alias("ct_amax")]
    X = hist.sort(["user_id", "event_date"]).group_by("user_id").agg(a)

    # ---- TRANS: переходы типов дней в подряд идущие дни, окно 90 ----
    t = hist.filter(pl.col("event_date") >= A - timedelta(days=89)).select([
        "user_id", "event_date",
        pl.when(A_EXPR).then(0).when(C_EXPR).then(2).when(D_EXPR).then(3).otherwise(1).alias("ty")])
    t = t.sort(["user_id", "event_date"]).with_columns([
        pl.col("ty").shift(1).over("user_id").alias("pt"),
        pl.col("event_date").diff().dt.total_days().over("user_id").alias("dd")])
    t = t.filter(pl.col("dd") == 1)
    tr = t.group_by(["user_id", "pt", "ty"]).len()
    tr = tr.with_columns((pl.lit("tr_") + pl.col("pt").cast(pl.Utf8) + pl.col("ty").cast(pl.Utf8)).alias("k"))
    piv = tr.pivot(on="k", index="user_id", values="len").fill_null(0)
    X = X.join(piv, on="user_id", how="left")

    sel = df.filter(pl.col("event_date").is_between(A - timedelta(days=29), A))["user_id"].unique()
    X = X.filter(pl.col("user_id").is_in(sel.implode()))
    fut = df.filter(pl.col("event_date").is_between(A + timedelta(days=1), A + timedelta(days=30)))
    g = fut.group_by("user_id").agg(pl.col("gmv").sum().alias("target"))
    X = X.join(g, on="user_id", how="left")
    return X.with_columns(pl.col("target").fill_null(0.0)).sort("user_id")


anchors = [VAL - timedelta(days=30 + 14 * i) for i in range(0, 10)] + [VAL]
if len(sys.argv) > 1 and sys.argv[1] == "test":
    anchors = [TEST]
for A in anchors:
    p = CACHE / f"a{A}.parquet"
    if p.exists():
        print(f"  {A} уже есть", flush=True)
        continue
    X = build(A)
    X.write_parquet(p)
    print(f"  {A}: {X.height:,} x {X.width}", flush=True)
print("готово")

"""E6. ЗЕРКАЛЬНЫЙ ТЕСТ. Воспроизводим правило отбора на исторических якорях
и меряем, какая доля отобранной популяции ПОЛНОСТЬЮ исчезает в следующие 30 дней.

Правило отбора организаторов: юзер активен в [A-29, A] (для A = 2026-02-13 это весь юниверс).
Вопрос: какая доля такой популяции отсутствует в [A+1, A+30]?
В валидации команды ответ 0% ПО ПОСТРОЕНИЮ. В тесте — нет.
"""
import os
import polars as pl, numpy as np
from datetime import date, timedelta
df = pl.read_parquet("train.parquet", columns=["user_id","event_date","gmv"])

def window_users(s, e):
    return set(df.filter(pl.col("event_date").is_between(s,e))["user_id"].unique().to_list())

def window_buyers(s, e):
    w = df.filter(pl.col("event_date").is_between(s,e)).group_by("user_id").agg(pl.col("gmv").sum())
    return set(w.filter(pl.col("gmv")>0)["user_id"].to_list())

print("якорь A     |отобрано (акт.30д)| исчезли в +30д |  доля  | нулевых gmv | доля")
out=[]
A = date(2025,3,15)
while A + timedelta(days=30) <= date(2026,2,13):
    sel = window_users(A-timedelta(days=29), A)
    nxt = window_users(A+timedelta(days=1), A+timedelta(days=30))
    buy = window_buyers(A+timedelta(days=1), A+timedelta(days=30))
    vanish = len(sel - nxt); zero = len(sel - buy)
    out.append((A, len(sel), vanish, zero))
    print(f"{A} | {len(sel):>14,} | {vanish:>13,} | {100*vanish/len(sel):5.2f}% | {zero:>10,} | {100*zero/len(sel):5.2f}%")
    A += timedelta(days=21)

import json, pathlib
pathlib.Path(os.environ.get("ZH_OUT", "work/zhenya_eda/out") + "/vanish.json").write_text(json.dumps(
    [{"anchor":str(a),"sel":s,"vanish":v,"zero":z} for a,s,v,z in out], indent=1))

v = np.array([100*v/s for _,s,v,_ in out])
print(f"\nдоля исчезающих: mean={v.mean():.2f}%  последние 5 якорей={v[-5:].mean():.2f}%  тренд={np.polyfit(range(len(v)),v,1)[0]:+.3f}%/якорь")

print("\n=== МАКСИМАЛЬНЫЙ РАЗРЫВ между активными днями (проверка утверждения «29 дней») ===")
g = (df.select(["user_id","event_date"]).sort(["user_id","event_date"])
       .with_columns(pl.col("event_date").diff().over("user_id").dt.total_days().alias("gap"))
       .group_by("user_id").agg(pl.col("gap").max().alias("mx")))
m = g["mx"].drop_nulls()
print(f"  max по всем юзерам: {m.max()}  p50={m.median():.0f} p95={m.quantile(.95):.0f} p99={m.quantile(.99):.0f}")
for t in (30, 60, 90, 120):
    print(f"  юзеров с разрывом >= {t:>3}д: {int((m>=t).sum()):>7,} ({100*(m>=t).mean():5.2f}%)")

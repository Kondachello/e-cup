"""E5. Доля ПОЛНОСТЬЮ отсутствующих юзеров в 30-дневном окне.

Юниверс отобран по «активен в 2026-01-15..2026-02-13» = валидационное окно команды.
Значит в валидации доля отсутствующих РАВНА НУЛЮ ПО ПОСТРОЕНИЮ, а в тесте — нет.
Меряем естественную долю по всем историческим окнам.
"""
import os
import polars as pl, numpy as np
from datetime import date, timedelta
df = pl.read_parquet("train.parquet", columns=["user_id","event_date","to_ord","gmv"])
N = 250000
D1 = date(2026,2,13)

print("окно(30д)            отсутств.  доля   |  нулевой gmv  доля")
rows=[]
a = date(2025,1,14)
while a + timedelta(days=30) <= D1:
    s, e = a+timedelta(days=1), a+timedelta(days=30)
    w = df.filter(pl.col("event_date").is_between(s,e))
    present = w["user_id"].n_unique()
    absent = N - present
    buyers = w.group_by("user_id").agg(pl.col("gmv").sum()).filter(pl.col("gmv")>0).height
    zero = N - buyers
    rows.append((a, absent, zero))
    print(f"{s}..{e}  {absent:>8,} {100*absent/N:5.2f}%  |  {zero:>8,} {100*zero/N:5.2f}%")
    a += timedelta(days=14)

import json, pathlib
pathlib.Path(os.environ.get("ZH_OUT", "work/zhenya_eda/out") + "/absence.json").write_text(json.dumps(
    [{"anchor":str(r[0]),"absent":r[1],"zero":r[2]} for r in rows], indent=1))

print("\n=== КОНТРОЛЬ: активны ли все 250k в 30 дней ПЕРЕД валидационным якорем? ===")
w = df.filter(pl.col("event_date").is_between(date(2025,12,16), date(2026,1,14)))
print(f"  присутствуют {w['user_id'].n_unique():,} из 250,000  -> отсутствуют {N-w['user_id'].n_unique():,}")
print("=== и перед тестовым якорём (2026-01-15..2026-02-13) — окно отбора ===")
w = df.filter(pl.col("event_date").is_between(date(2026,1,15), D1))
print(f"  присутствуют {w['user_id'].n_unique():,} из 250,000  -> отсутствуют {N-w['user_id'].n_unique():,}")

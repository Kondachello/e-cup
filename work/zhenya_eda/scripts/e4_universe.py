"""E4. Как отбирали юниверс: первое/последнее событие, tenure, правило отбора."""
import polars as pl, numpy as np
from datetime import date
df = pl.read_parquet("train.parquet", columns=["user_id","event_date","to_ord","gmv","searches"])
D0, D1 = date(2025,1,1), date(2026,2,13)

u = df.group_by("user_id").agg([
    pl.col("event_date").min().alias("first"), pl.col("event_date").max().alias("last"),
    pl.len().alias("days"), pl.col("to_ord").sum().alias("ord"), pl.col("gmv").sum().alias("gmv"),
]).sort("user_id")
print(f"юзеров {u.height:,}")

f = u["first"]; l = u["last"]
print(f"\nПЕРВОЕ событие: min={f.min()} max={f.max()}")
print(f"  ровно 2025-01-01: {int((f==D0).sum()):,} ({100*(f==D0).mean():.1f}%)")
print(f"  в январе 2025:    {int((f<date(2025,2,1)).sum()):,} ({100*(f<date(2025,2,1)).mean():.1f}%)")
print(f"ПОСЛЕДНЕЕ событие: min={l.min()} max={l.max()}")
print(f"  ровно 2026-02-13: {int((l==D1).sum()):,} ({100*(l==D1).mean():.1f}%)")

print("\n--- распределение ПОСЛЕДНЕГО события (хвост) ---")
lc = u.group_by("last").len().sort("last")
print(lc.tail(20))
print("\n--- распределение ПЕРВОГО события (голова) ---")
fc = u.group_by("first").len().sort("first")
print(fc.head(15))
print("...\nпервое событие ПОЗЖЕ 2025-06-01:", int((f>date(2025,6,1)).sum()))
print("первое событие ПОЗЖЕ 2025-12-01:", int((f>date(2025,12,1)).sum()))

# ПРАВИЛО ОТБОРА: сколько дней назад последнее событие относительно конца данных
rec = (pl.lit(D1) - pl.col("last")).dt.total_days()
r = u.with_columns(rec.alias("rec_days"))["rec_days"]
print(f"\nдавность последнего события от 2026-02-13: max={r.max()} p99={r.quantile(.99):.0f} p50={r.median():.0f}")
print("гистограмма давности (дней):")
for lo, hi in [(0,0),(1,6),(7,13),(14,29),(30,59),(60,89),(90,10000)]:
    k = int(((r>=lo)&(r<=hi)).sum()); print(f"  {lo:>3}-{hi if hi<1000 else '∞':>4}: {k:>8,} ({100*k/250000:5.2f}%)")

# заказы за весь период
print(f"\nюзеров с 0 заказов за 409 дней: {int((u['ord']==0).sum()):,} ({100*(u['ord']==0).mean():.2f}%)")
print(f"юзеров с 0 gmv:                 {int((u['gmv']==0).sum()):,}")
print(f"дней активности: p1={u['days'].quantile(.01):.0f} p50={u['days'].median():.0f} p99={u['days'].quantile(.99):.0f}")

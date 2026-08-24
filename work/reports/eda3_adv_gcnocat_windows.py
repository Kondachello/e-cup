# Кросс-оконная проверка: разрыв сегмента gc_nocat в ЧЕСТНЫХ OOF-остатках seq-модели
# на 9 размеченных якорях (2025-11-19 .. 2026-01-14). Механизм "слепое пятно всего
# семейства" обязан жить в каждом окне; окно-специфичный сигнал = мёртв (закон переноса 2%).
import datetime as dt
import numpy as np
import polars as pl

anchors = [dt.date(2025, 11, 19), dt.date(2025, 11, 26), dt.date(2025, 12, 3),
           dt.date(2025, 12, 10), dt.date(2025, 12, 17), dt.date(2025, 12, 24),
           dt.date(2025, 12, 31), dt.date(2026, 1, 7), dt.date(2026, 1, 14)]

lf = pl.scan_parquet("train.parquet")
rows = []
for A in anchors:
    Aend = A + dt.timedelta(days=30)
    gc_day = (pl.col("gmv_cat") > 0) & (pl.col("cat") == 0)
    agg = (
        lf.group_by("user_id").agg(
            gc_day.filter(pl.col("event_date") <= A).sum().alias("n_gc"),
            (pl.col("cat") > 0).filter(pl.col("event_date") <= A).sum().alias("n_cat_days"),
            (pl.col("gmv_cat") > 0).filter(pl.col("event_date") <= A).sum().alias("n_gcd"),
            pl.col("gmv").filter((pl.col("event_date") > A) & (pl.col("event_date") <= Aend)).sum().alias("tgt"),
        ).collect(engine="streaming")
    )
    oof = pl.read_parquet(f"work/features/anchor={A.isoformat()}.seqoof.parquet")
    j = oof.join(agg, on="user_id", how="left").sort("user_id")
    pred = j["seqoof_pred"].to_numpy().astype(float)
    tgt = np.log1p(j["tgt"].fill_null(0.0).to_numpy().astype(float))
    r = tgt - pred
    f = j["n_gc"].fill_null(0).to_numpy() > 0
    gap = r[f].mean() - r[~f].mean()
    se = np.sqrt(r[f].var() / f.sum() + r[~f].var() / (~f).sum())
    rmse = np.sqrt((r ** 2).mean())
    ovl = max(0, (min(Aend, dt.date(2026, 2, 13)) - max(A + dt.timedelta(days=1), dt.date(2026, 1, 15))).days + 1)
    rows.append((A.isoformat(), int(f.sum()), rmse, gap, se, ovl))
    print(f"{A}  seg {f.sum():6d}  seq_rmse {rmse:.4f}  gap {gap:+.4f} ± {se:.4f}  overlap_val_days {ovl}")

g = np.array([x[3] for x in rows])
s = np.array([x[4] for x in rows])
w = 1 / s ** 2
print(f"\nвзвешенное среднее gap по 9 якорям: {np.sum(g*w)/np.sum(w):+.4f}; min {g.min():+.4f} max {g.max():+.4f}")
nonovl = [x for x in rows if x[5] == 0]
gn = np.array([x[3] for x in nonovl]); sn = np.array([x[4] for x in nonovl])
wn = 1 / sn ** 2
print(f"только якоря БЕЗ пересечения с вал-окном ({len(nonovl)}): weighted gap {np.sum(gn*wn)/np.sum(wn):+.4f}")
import json
json.dump({"rows": rows}, open("work/reports/eda3_adv_gcnocat_windows.json", "w"), default=str)

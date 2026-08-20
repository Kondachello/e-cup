"""№3: правило отбора когорты и явка (стена отбора). Печатает числа §4 отчёта."""
import polars as pl, numpy as np, lightgbm as lgb
from datetime import date
from features import build_features

DAY0 = date(2025, 1, 1)
act = pl.read_parquet("act.parquet").with_columns(
    (pl.col("event_date") - pl.lit(DAY0)).dt.total_days().alias("day"))
users = pl.read_parquet("users_order.parquet")["user_id"].to_numpy()

def n_active(s, e):
    return act.filter((pl.col("day") >= s) & (pl.col("day") < e))["user_id"].n_unique()

print("== перебор окон: где все 250к активны ==")
for s in [289, 300, 310, 316, 319, 322, 330, 340, 349, 379]:
    print(f"  [{s},{s+30}): {n_active(s, s+30)}")
print("  => все активны ровно в блоках [319,349), [349,379), [379,409)")

def active_arr(s, e):
    ids = act.filter((pl.col("day") >= s) & (pl.col("day") < e))["user_id"].unique().to_numpy()
    m = np.zeros(len(users), bool); m[np.searchsorted(users, ids)] = True
    return m

print("\n== явка при согласованном отборе (юзеры активны в 3 блоках подряд) ==")
model = lgb.Booster(model_file="clf_buy.txt")
cube = np.load("cube_val.npy", mmap_mode="r")
for T0 in [253, 260, 267, 274, 281, 288]:
    sel = active_arr(T0-90, T0-60) & active_arr(T0-60, T0-30) & active_arr(T0-30, T0)
    app = active_arr(T0, T0+30)
    X, _ = build_features(T0, cube, 379)
    p = model.predict(X[:, :90])
    q3, q7 = np.quantile(p[sel], [0.30, 0.70])
    pool = sel & (p <= q3); mid = sel & (p > q3) & (p <= q7); top = sel & (p > q7)
    print(f"  T0={T0}: явка пула {app[pool].mean():.4f}  середины {app[mid].mean():.4f}  топа {app[top].mean():.4f}")

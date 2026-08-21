"""eda3 фаза цикла, шаг 2: остаток бленда по бинам фазы у регулярных юзеров.
Фаза: progress u = rec / P (P = медианный межзаказный интервал), и wrap-фаза (u mod 1).
Срезы: регулярность cv, диапазон периода P. Всё против e = log1p(target) - blend.
"""
import numpy as np
import polars as pl

SCRATCH = "/private/tmp/claude-501/-Users-alexanderkondakov-ozon-cup/b994bee8-2354-4524-9a2b-530595cd5feb/scratchpad"

st = pl.read_parquet(f"{SCRATCH}/eda3_user_cycle.parquet")
vp = pl.read_parquet("/Users/alexanderkondakov/ozon-cup/work/preds_pack/val_preds.parquet",
                     columns=["user_id", "target", "blend"])
d = vp.join(st, on="user_id", how="left")
d = d.with_columns(
    e=(pl.col("target").log1p() - pl.col("blend")),
    P=pl.col("gap_median"),
)
d = d.with_columns(u=pl.col("rec") / pl.col("P"))
N_ALL = d.height
e_all = d["e"].to_numpy()
mse0 = float(np.mean(e_all**2))
print(f"N={N_ALL} blend RMSE={np.sqrt(mse0):.6f}")

print("\ncv quantiles (n_ord_days>=5):")
q = d.filter(pl.col("n_ord_days") >= 5)["cv"]
print({p: round(float(q.quantile(p)), 3) for p in [0.01, 0.05, 0.1, 0.25, 0.5, 0.75]})
print("P (gap_median) quantiles (n>=5):",
      {p: float(d.filter(pl.col("n_ord_days") >= 5)["P"].quantile(p)) for p in [0.1, 0.25, 0.5, 0.75, 0.9, 0.99]})

def bin_table(sub: pl.DataFrame, col: str, edges, label: str):
    x = sub[col].to_numpy()
    e = sub["e"].to_numpy()
    idx = np.digitize(x, edges)
    rows = []
    for b in range(len(edges) + 1):
        m = idx == b
        n = int(m.sum())
        if n == 0:
            continue
        me = float(e[m].mean())
        se = float(e[m].std(ddof=1) / max(np.sqrt(n), 1))
        lo = edges[b - 1] if b > 0 else -np.inf
        hi = edges[b] if b < len(edges) else np.inf
        rows.append((f"[{lo:.2f},{hi:.2f})", n, me, se, me / se if se > 0 else 0))
    # потенциал: оракульная поправка биновым средним, вклад в общий MSE
    gain_mse = sum(n * me * me for (_, n, me, se, t) in rows) / N_ALL
    d_rmse = np.sqrt(mse0) - np.sqrt(mse0 - gain_mse)
    print(f"\n== {label}: n_sub={sub.height}  oracle dMSE={gain_mse:.6f} dRMSE={d_rmse:.6f}")
    for r in rows:
        print(f"  {col} {r[0]:>14} n={r[1]:>6} mean_e={r[2]:+.4f} se={r[3]:.4f} t={r[4]:+.2f}")
    return rows

# срезы регулярности
base = d.filter(pl.col("n_ord_days") >= 5, pl.col("P") >= 2)
for cvmax in [0.5, 0.8]:
    sub = base.filter(pl.col("cv") <= cvmax)
    bin_table(sub, "u", [0.25, 0.5, 0.75, 1.0, 1.5, 2.0], f"n>=5 cv<={cvmax}, progress u=rec/P")

# длиннопериодные (фаза решает 0 vs 1 заказов в окне 30д)
for pmin, pmax, cvmax in [(14, 60, 0.6), (14, 60, 1.0), (20, 90, 0.8)]:
    sub = base.filter(pl.col("P") >= pmin, pl.col("P") <= pmax, pl.col("cv") <= cvmax)
    bin_table(sub, "u", [0.25, 0.5, 0.75, 1.0, 1.5, 2.0], f"P in [{pmin},{pmax}] cv<={cvmax}")

# wrap-фаза (u mod 1) у высокорегулярных — резкая периодическая структура?
for cvmax, nmin in [(0.5, 5), (0.8, 5), (0.5, 8)]:
    sub = base.filter(pl.col("cv") <= cvmax, pl.col("n_ord_days") >= nmin, pl.col("u") <= 3)
    sub = sub.with_columns(ph=pl.col("u") % 1.0)
    bin_table(sub, "ph", [0.2, 0.4, 0.6, 0.8], f"wrap-фаза cv<={cvmax} n>={nmin}")

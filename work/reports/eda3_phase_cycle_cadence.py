"""eda3 фаза цикла, шаг 4: тренд каденции r = gap_last / gap_median.
gap_last отсутствует в 203 признаках. Механизм: ускорение/замедление личного ритма,
которое агрегаты gap_mean/std не видят. Замер: e по бинам r, с контролем по rec.
"""
import numpy as np
import polars as pl

SCRATCH = "/private/tmp/claude-501/-Users-alexanderkondakov-ozon-cup/b994bee8-2354-4524-9a2b-530595cd5feb/scratchpad"
st = pl.read_parquet(f"{SCRATCH}/eda3_user_cycle.parquet")
vp = pl.read_parquet("/Users/alexanderkondakov/ozon-cup/work/preds_pack/val_preds.parquet",
                     columns=["user_id", "target", "blend"])
d = vp.join(st, on="user_id", how="left")
d = d.with_columns(e=(pl.col("target").log1p() - pl.col("blend")), P=pl.col("gap_median"))
N_ALL = d.height
mse0 = float((d["e"].to_numpy() ** 2).mean())

base = d.filter(pl.col("n_ord_days") >= 5, pl.col("P") >= 2)
base = base.with_columns(r=pl.col("gap_last") / pl.col("P"))
print("n base:", base.height, " r quantiles:",
      {p: round(float(base["r"].quantile(p)), 2) for p in [0.1, 0.25, 0.5, 0.75, 0.9]})

EDGES = [0.5, 0.8, 1.25, 2.0, 4.0]
x = base["r"].to_numpy(); e = base["e"].to_numpy()
idx = np.digitize(x, EDGES)
print("\nбины r=gap_last/P (все n>=5):")
tot = 0.0
for b in range(len(EDGES) + 1):
    m = idx == b
    n = int(m.sum())
    if n == 0: continue
    me, se = e[m].mean(), e[m].std(ddof=1) / np.sqrt(n)
    tot += n * me * me
    print(f"  bin{b} n={n:>6} mean_e={me:+.4f} se={se:.4f} t={me/se:+.2f}")
print(f"oracle dMSE={tot/N_ALL:.6f} dRMSE={np.sqrt(mse0)-np.sqrt(mse0-tot/N_ALL):.6f}")

# контроль rec: внутри бинов rec остаётся ли структура r?
rec = base["rec"].to_numpy()
rec_edges = [7, 14, 30, 60]
ridx = np.digitize(rec, rec_edges)
print("\nдвойной контроль: mean_e по (rec-бин x r<=1 vs r>1):")
for rb in range(len(rec_edges) + 1):
    for lab, m2 in [("r<=1", x <= 1.0), ("r>1", x > 1.0)]:
        m = (ridx == rb) & m2
        n = int(m.sum())
        if n < 50: continue
        me, se = e[m].mean(), e[m].std(ddof=1) / np.sqrt(n)
        print(f"  rec_bin{rb} {lab:>5} n={n:>6} mean_e={me:+.4f} se={se:.4f} t={me/se:+.2f}")

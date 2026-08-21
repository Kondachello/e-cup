"""eda3 GMV sources: confound tests for gc_nocat_share — recency partialling,
prediction-edge check, replication across pack models."""
import numpy as np
import polars as pl

CUT = pl.date(2026, 1, 14)
lf = pl.scan_parquet("/Users/alexanderkondakov/ozon-cup/train.parquet").filter(pl.col("event_date") <= CUT)
A = pl.date(2026, 1, 14)
rec = lf.group_by("user_id").agg([
    (A - pl.col("event_date").filter(pl.col("cat") > 0).max()).dt.total_days().alias("rec_cat"),
    (A - pl.col("event_date").filter(pl.col("gmv_cat") > 0).max()).dt.total_days().alias("rec_gc"),
    (A - pl.col("event_date").filter(pl.col("gmv") > 0).max()).dt.total_days().alias("rec_ord"),
    (A - pl.col("event_date").filter((pl.col("gmv_cat") > 0) & (pl.col("cat") == 0)).max()).dt.total_days().alias("rec_gcnocat"),
]).collect(engine="streaming")

pp = pl.read_parquet("/Users/alexanderkondakov/ozon-cup/work/preds_pack/val_preds.parquet",
                     columns=["user_id", "target", "blend", "kostya46_cal", "fusion_v3_avg_cal", "wklin", "gseq_big_s42_cal"])
uv = pl.read_parquet("/Users/alexanderkondakov/ozon-cup/work/reports/eda3_gmv_sources_uservec.parquet")
df = pp.join(uv, on="user_id", how="left").join(rec, on="user_id", how="left")
y = np.log1p(df["target"].to_numpy())
b = df["blend"].to_numpy().astype(np.float64)
e = y - b
n = len(e)

def g(c): return np.nan_to_num(df[c].to_numpy().astype(np.float64))
L = np.log1p
x = g("gc_nocat_share")
FILL = 999.0
def gr(c):
    v = df[c].to_numpy().astype(np.float64)
    return np.where(np.isnan(v), FILL, v)

# 1) partial out recency (in-shell) + counts + prediction itself
P = np.column_stack([np.ones(n), L(gr("rec_cat")), L(gr("rec_gc")), L(gr("rec_ord")),
                     (~np.isnan(df["rec_cat"].to_numpy().astype(np.float64))).astype(float),
                     (~np.isnan(df["rec_gc"].to_numpy().astype(np.float64))).astype(float),
                     L(g("n_gc_days")), L(g("n_gmv_days")), g("cat_rub_share"), L(g("sum_gc")), b, b ** 2])
beta, *_ = np.linalg.lstsq(P, x, rcond=None)
rx = x - P @ beta
print(f"corr(e, x) raw {np.corrcoef(x, e)[0,1]:.5f} -> after recency+counts+blend partial: {np.corrcoef(rx, e)[0,1]:.5f} (proxy mdl_flint {1 - rx.var()/x.var():.3f})")

# honest OOF gain of the PARTIALED residual vector
uid = df["user_id"].to_numpy()
fold = uid % 2 == 0
rmse0 = float(np.sqrt(np.mean(e ** 2)))
pred = np.zeros(n)
for m in (fold, ~fold):
    tr, te = ~m, m
    xc = rx[tr] - rx[tr].mean()
    bta = float((xc @ (e[tr] - e[tr].mean())) / np.maximum(xc @ xc, 1e-12))
    pred[te] = bta * (rx[te] - rx[tr].mean()) + e[tr].mean()
pred -= pred.mean()
print(f"OOF gain of partialed vector: {rmse0 - float(np.sqrt(np.mean((e - pred) ** 2))):.7f}")

# 2) prediction-edge check: distribution of blend among share>0 vs 0 (cat buyers)
m = g("n_gc_days") > 0
z = x[m] > 0
bm = b[m]
print(f"mean blend | share>0: {bm[z].mean():.4f}  | share==0: {bm[~z].mean():.4f}  (corr(x,b) overall {np.corrcoef(x, b)[0,1]:.4f})")

# 3) replication across models: mean_e gap (share>0 minus share==0) per model, cat buyers
for c in ["blend", "kostya46_cal", "fusion_v3_avg_cal", "wklin", "gseq_big_s42_cal"]:
    p = df[c].to_numpy().astype(np.float64)
    ee = (y - p)[m]
    gap = ee[z].mean() - ee[~z].mean()
    se = np.sqrt(ee[z].var() / z.sum() + ee[~z].var() / (~z).sum())
    print(f"  {c:22s} gap = {gap:+.4f} +- {se:.4f}")

# 4) matched-on-blend gap: within blend deciles (cat buyers), avg weighted gap
qs = np.quantile(bm, np.linspace(0, 1, 11)); qs[-1] += 1e-9
gaps, ws = [], []
for i in range(10):
    mm = (bm >= qs[i]) & (bm < qs[i + 1])
    if (z & mm).sum() < 50 or (~z & mm).sum() < 50: continue
    ee = (y - b)[m]
    gp = ee[z & mm].mean() - ee[~z & mm].mean()
    w = 1.0 / (ee[z & mm].var() / (z & mm).sum() + ee[~z & mm].var() / (~z & mm).sum())
    gaps.append(gp); ws.append(w)
gaps = np.array(gaps); ws = np.array(ws)
print(f"blend-decile-matched gap: {np.average(gaps, weights=ws):+.4f} +- {1/np.sqrt(ws.sum()):.4f} (deciles used {len(gaps)})")

# 5) freshness split: effect for users whose last gc_nocat day is recent vs old
rgn = gr("rec_gcnocat")
for lab, mm in [("gcnocat<=90d", (x > 0) & (rgn <= 90) & m), ("gcnocat>90d", (x > 0) & (rgn > 90) & (rgn < FILL) & m)]:
    ee = e[mm]
    print(f"  {lab}: n={mm.sum()} mean_e={ee.mean():+.4f} se={ee.std()/np.sqrt(mm.sum()):.4f}")

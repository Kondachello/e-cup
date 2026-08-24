"""eda3 GMV sources: deployable binary corrector OOF + test-anchor segment size/overlap."""
import numpy as np
import polars as pl

pp = pl.read_parquet("/Users/alexanderkondakov/ozon-cup/work/preds_pack/val_preds.parquet",
                     columns=["user_id", "target", "blend"])
uv = pl.read_parquet("/Users/alexanderkondakov/ozon-cup/work/reports/eda3_gmv_sources_uservec.parquet")
df = pp.join(uv, on="user_id", how="left")
y = np.log1p(df["target"].to_numpy())
b = df["blend"].to_numpy().astype(np.float64)
e = y - b
n = len(e)
rmse0 = float(np.sqrt(np.mean(e ** 2)))
x = np.nan_to_num(df["gc_nocat_share"].to_numpy().astype(np.float64))
nn = np.nan_to_num(df["n_gc_nocat"].to_numpy().astype(np.float64))
seg = (nn > 0).astype(np.float64)
uid = df["user_id"].to_numpy()
fold = uid % 2 == 0

def oof_gain(X):
    X = np.atleast_2d(X.T).T
    pred = np.zeros(n)
    for m in (fold, ~fold):
        tr, te = ~m, m
        A = np.column_stack([np.ones(tr.sum()), X[tr]])
        beta, *_ = np.linalg.lstsq(A, e[tr], rcond=None)
        pred[te] = np.column_stack([np.ones(te.sum()), X[te]]) @ beta
    pred -= pred.mean()
    e2 = e - pred
    d = e ** 2 - e2 ** 2
    return rmse0 - float(np.sqrt(np.mean(e2 ** 2))), float((d > 0).mean())

g1, imp1 = oof_gain(seg)
print(f"binary seg (n_gc_nocat>0): OOF gain {g1:.7f}, users improved {imp1:.3f}, seg size {int(seg.sum())}")
g2, imp2 = oof_gain(np.column_stack([seg, x]))
print(f"binary + share: OOF gain {g2:.7f}")
g3, imp3 = oof_gain(np.column_stack([seg, x, np.log1p(nn)]))
print(f"binary + share + logcount: OOF gain {g3:.7f}")
# per-fold delta of binary corrector
for k in (0, 1):
    m = fold if k == 0 else ~fold
    print(f"  fold{k}: delta_seg = {e[m][seg[m] > 0].mean() - e[m][seg[m] == 0].mean():+.4f}")

# test-anchor segment (history <= 2026-02-13) and overlap with val-anchor segment
lf = pl.scan_parquet("/Users/alexanderkondakov/ozon-cup/train.parquet")
tst = lf.group_by("user_id").agg([
    ((pl.col("gmv_cat") > 0) & (pl.col("cat") == 0)).sum().alias("n_gc_nocat_test"),
    ((pl.col("gmv_cat") > 0) & (pl.col("cat") == 0) & (pl.col("event_date") > pl.date(2026, 1, 14))).sum().alias("n_new"),
]).collect(engine="streaming")
j = df.select(["user_id"]).with_columns(pl.Series("seg_val", seg)).join(tst, on="user_id", how="left")
sv = j["seg_val"].to_numpy() > 0
st = j["n_gc_nocat_test"].to_numpy() > 0
print(f"test-anchor seg size: {st.sum()} ({st.mean():.4f}); val-anchor: {sv.sum()}; jaccard {np.logical_and(sv, st).sum()/np.logical_or(sv, st).sum():.3f}")
print(f"users entering seg only in Jan15-Feb13: {(st & ~sv).sum()}")

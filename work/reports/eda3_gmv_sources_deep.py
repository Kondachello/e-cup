"""eda3 lens GMV sources: deep-dive gc_nocat_share. Partial out in-shell proxies,
subgroup analysis, fold-sign consistency, whale-trimmed gain."""
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

def g(c): return np.nan_to_num(df[c].to_numpy().astype(np.float64))
L = np.log1p
x = g("gc_nocat_share")
xl = L(g("n_gc_nocat"))

# 1) partial out in-shell proxies
P = np.column_stack([np.ones(n), L(g("n_gc_days")), L(g("n_gs_days")), L(g("n_gmv_days")),
                     g("cat_rub_share"), L(g("sum_gmv")), L(g("sum_gc")),
                     (g("n_gc_days") > 0).astype(float)])
for name, xv in [("gc_nocat_share", x), ("log_n_gc_nocat", xl)]:
    beta, *_ = np.linalg.lstsq(P, xv, rcond=None)
    r = xv - P @ beta
    c_raw = float(np.corrcoef(xv, e)[0, 1])
    c_res = float(np.corrcoef(r, e)[0, 1])
    print(f"{name}: corr raw {c_raw:.5f} -> after partialling in-shell proxies {c_res:.5f}  (mdl_flint of proxies on x: {1 - r.var()/xv.var():.3f})")

# also: does e correlate with the SHELL proxies themselves here? (should be ~0 by shell theorem)
for i, nm in enumerate(["log_n_gc_days", "log_n_gs_days", "log_n_gmv_days", "cat_rub_share", "log_sum_gmv", "log_sum_gc", "has_gc"]):
    print(f"  shell proxy corr(e, {nm}) = {np.corrcoef(P[:, i+1], e)[0,1]:.5f}")

# 2) subgroup: cat buyers only
m = g("n_gc_days") > 0
print(f"\ncat buyers: {m.sum()} users ({m.mean():.3f}); users with n_gc_nocat>0: {(g('n_gc_nocat')>0).sum()}")
em = e[m]; xm = x[m]
print(f"within cat-buyers: corr(gc_nocat_share, e) = {np.corrcoef(xm, em)[0,1]:.5f}, n={m.sum()}, 1/sqrt(n)={1/np.sqrt(m.sum()):.5f}")

# bin means with SE, within cat buyers
qs = [0, 0.25, 0.5, 0.75, 1.0000001]
edges = np.quantile(xm, [0, .25, .5, .75, 1.0]); edges[-1] += 1e-9
print("bins of gc_nocat_share (cat buyers): [lo,hi) n mean_e se")
for i in range(4):
    mm = (xm >= edges[i]) & (xm < edges[i + 1])
    if mm.sum() == 0: continue
    print(f"  [{edges[i]:.3f},{edges[i+1]:.3f}) n={mm.sum():6d} mean_e={em[mm].mean():+.4f} se={em[mm].std()/np.sqrt(mm.sum()):.4f}")
# exact-zero vs positive share
z = xm == 0
print(f"  share==0: n={z.sum()} mean_e={em[z].mean():+.4f} se={em[z].std()/np.sqrt(z.sum()):.4f}")
print(f"  share>0 : n={(~z).sum()} mean_e={em[~z].mean():+.4f} se={em[~z].std()/np.sqrt((~z).sum()):.4f}")

# 3) fold-sign consistency of beta (4 folds by user_id % 4)
uid = df["user_id"].to_numpy()
for name, xv in [("gc_nocat_share", x)]:
    betas = []
    for k in range(4):
        mk = uid % 4 == k
        xc = xv[mk] - xv[mk].mean()
        betas.append(float((xc @ (e[mk] - e[mk].mean())) / (xc @ xc)))
    print(f"\n{name} per-fold betas: {[round(v,4) for v in betas]}")

# 4) OOF gain with whale trim: exclude top-0.1% |e| users from evaluation, and winsorized-beta variant
fold = uid % 2 == 0
pred = np.zeros(n)
for mm in (fold, ~fold):
    tr, te = ~mm, mm
    xc = x[tr] - x[tr].mean()
    beta = float((xc @ (e[tr] - e[tr].mean())) / np.maximum(xc @ xc, 1e-12))
    a0 = float(e[tr].mean() - beta * x[tr].mean())
    pred[te] = a0 + beta * x[te]
pred -= pred.mean()
e2 = e - pred
d = e ** 2 - e2 ** 2
gain = rmse0 - float(np.sqrt(np.mean(e2 ** 2)))
# trimmed evaluation: drop top 0.1% by |d|
ord_d = np.argsort(-np.abs(d))
keep = np.ones(n, bool); keep[ord_d[: n // 1000]] = False
r0t = float(np.sqrt(np.mean(e[keep] ** 2))); r1t = float(np.sqrt(np.mean(e2[keep] ** 2)))
print(f"\nOOF gain full: {gain:.7f}; after dropping top-0.1% |d| users: {r0t - r1t:.7f}")
# median-user view: share of users improved
print(f"share of users with d>0 (improved): {(d > 0).mean():.4f}; among cat buyers: {(d[m] > 0).mean():.4f}")
# 5) combined two-feature corrector (share + log count), honest OOF
X2 = np.column_stack([x, xl])
pred = np.zeros(n)
for mm in (fold, ~fold):
    tr, te = ~mm, mm
    A = np.column_stack([np.ones(tr.sum()), X2[tr]])
    beta, *_ = np.linalg.lstsq(A, e[tr], rcond=None)
    pred[te] = np.column_stack([np.ones(te.sum()), X2[te]]) @ beta
pred -= pred.mean()
gain2 = rmse0 - float(np.sqrt(np.mean((e - pred) ** 2)))
print(f"two-feature (share+logcount) OOF gain: {gain2:.7f}")

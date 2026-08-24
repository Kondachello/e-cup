"""Final checks: leak-free val-residual link, delta structure vs size, substitution mechanism, quintile table."""
import numpy as np
import polars as pl

SC = "/private/tmp/claude-501/-Users-alexanderkondakov-ozon-cup/b994bee8-2354-4524-9a2b-530595cd5feb/scratchpad"
U = pl.read_parquet(f"{SC}/user_offday.parquet").sort("user_id")
mdl_silica = pl.read_parquet(f"{SC}/user_offday2.parquet").sort("user_id")
V = pl.read_parquet("/Users/alexanderkondakov/ozon-cup/work/preds_pack/val_preds.parquet").sort("user_id")
assert (U["user_id"].to_numpy() == V["user_id"].to_numpy()).all()
assert (mdl_silica["user_id"].to_numpy() == V["user_id"].to_numpy()).all()
N = U.height
g = lambda c: U[c].to_numpy().astype(np.float64)
g2 = lambda c: mdl_silica[c].to_numpy().astype(np.float64)
mdl_flint = 2 * 1.665647

def mkdelta(o_off, o_wk, n_off, n_wk, lam=30.0, comp_from=(22, 8), comp_to=(18, 12)):
    mu_off, mu_wk = o_off.sum() / n_off / N, o_wk.sum() / n_wk / N
    r_off = (o_off + lam * mu_off) / (n_off + lam)
    r_wk = (o_wk + lam * mu_wk) / (n_wk + lam)
    a, b = comp_from; a2, b2 = comp_to
    d = np.log(a2 * r_wk + b2 * r_off) - np.log(a * r_wk + b * r_off)
    return d - d.mean()

# ---- 1. leak-free S-axis check: trait strictly BEFORE val window ----
d_pre = mkdelta(g2("pre_ord_off"), g2("pre_ord_wk"), 104, 241)
y = np.log1p(V["target"].to_numpy().astype(np.float64))
b = V["blend"].to_numpy().astype(np.float64)
e = y - b
print("=== leak-free val-residual link (trait 2025-01-12..2026-01-14) ===")
print(f"corr(d_pre, e_val) = {np.corrcoef(d_pre, e)[0,1]:+.4f}   corr(d_pre, blend) = {np.corrcoef(d_pre, b)[0,1]:+.4f}")
beta_e = (d_pre * (e - e.mean())).mean() / (d_pre**2).mean()
print(f"OLS e ~ d_pre: beta={beta_e:+.4f}; d_pre sd={d_pre.std():.4f}; naive in-sample dMSE={-(np.corrcoef(d_pre,e)[0,1]**2)*e.var():+.2e}")
# vs previous leaky number
d_full = mkdelta(g("full_ord_off"), g("full_ord_wk"), 112, 263)
print(f"(leaky full-trait corr was {np.corrcoef(d_full, e)[0,1]:+.4f} -> leakage share {np.corrcoef(d_full,e)[0,1]-np.corrcoef(d_pre,e)[0,1]:+.4f})")

# ---- 2. is d essentially a size proxy? mean d by blend decile ----
dec = np.clip((np.argsort(np.argsort(b)) / N * 10).astype(int), 0, 9)
print("\nmean d_pre by blend decile (0=smallest):", np.round([d_pre[dec == k].mean() for k in range(10)], 4))
ords = g2("pre_ord_off") + g2("pre_ord_wk")
print("mean pre-orders by decile:", np.round([ords[dec == k].mean() for k in range(10)], 1))
oshare = np.where(ords > 0, g2("pre_ord_off") / np.maximum(ords, 1e-9), np.nan)
print("mean off-share (ord>0) by decile:", np.round([np.nanmean(oshare[dec == k]) for k in range(10)], 4))

# after removing decile-mean (size axis is closed): residual delta
d_res = d_pre.copy()
for k in range(10):
    d_res[dec == k] -= d_pre[dec == k].mean()
print(f"corr(d_res, e) after removing blend-decile means = {np.corrcoef(d_res, e)[0,1]:+.4f}")

# ---- 3. substitution mechanism: trait moves TIMING inside window, not the SUM ----
d_trait = mkdelta(g("p1_ord_off") + g("p2_ord_off"), g("p1_ord_wk") + g("p2_ord_wk"), 23 + 34, 56 + 89)
q5 = np.clip((np.argsort(np.argsort(d_trait)) / N * 5).astype(int), 0, 4)
print("\n=== substitution check: off-day GMV share inside Apr (8/22) vs May (12/18) by trait quintile ===")
print("(if trait is real, off-share rises with quintile in BOTH windows; if sums responded, dlog would too)")
for k in range(5):
    m = q5 == k
    apr_off, apr_wk = g2("m_apr_gmv_off")[m].sum(), g2("m_apr_gmv_wk")[m].sum()
    may_off, may_wk = g2("m_may_gmv_off")[m].sum(), g2("m_may_gmv_wk")[m].sum()
    dlog = np.log1p(g("m_may_gmv")[m]) - np.log1p(g("m_apr_gmv")[m])
    print(f"Q{k}: apr off-share {apr_off/(apr_off+apr_wk):.4f}  may off-share {may_off/(may_off+may_wk):.4f}  "
          f"apr gmv {apr_off+apr_wk:,.0f}  may gmv {may_off+may_wk:,.0f}  ratio {(may_off+may_wk)/(apr_off+apr_wk):.4f}  "
          f"mean dlog1p {dlog.mean():+.4f}")
# per-day rate view for extreme quintiles
for k in [0, 4]:
    m = q5 == k
    a_o, a_w = g2("m_apr_gmv_off")[m].sum() / 8, g2("m_apr_gmv_wk")[m].sum() / 22
    m_o, m_w = g2("m_may_gmv_off")[m].sum() / 12, g2("m_may_gmv_wk")[m].sum() / 18
    print(f"Q{k}: apr per-day gmv off/wk {a_o:,.0f}/{a_w:,.0f} (rho {a_o/a_w:.3f});  may {m_o:,.0f}/{m_w:,.0f} (rho {m_o/m_w:.3f})")

# ---- 4. naive arithmetic vector: what deployment WOULD have added (for the report) ----
print("\n=== naive deployment vector d_pre (22,8)->(18,12): what it would do on test ===")
print(f"sd={d_pre.std():.4f}; if response slope were 1.0: dRMSLE = -sd^2/(2R) = {-(d_pre.std()**2)/mdl_flint:.2e}")
print(f"measured response slope (Apr->May clean trait): -0.084 +- 0.052 -> realizable gain = beta^2*var/(2R) = {-(0.084**2)*(d_pre.std()**2)/mdl_flint:.2e}")
# quintile dlog table (nonparametric slope check), clean trait
print("\nmean dlog1p(May-Apr) by clean-trait quintile:", np.round([ (np.log1p(g('m_may_gmv'))-np.log1p(g('m_apr_gmv')))[q5==k].mean() for k in range(5)],4))

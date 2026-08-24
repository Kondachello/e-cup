"""Weekend-composition lens: reliability, natural experiments Apr->May / Oct->Nov / placebo Sep->Oct,
deployment vector delta, whale concentration, val-residual link, gain ceilings."""
import numpy as np
import polars as pl

SC = "/private/tmp/claude-501/-Users-alexanderkondakov-ozon-cup/b994bee8-2354-4524-9a2b-530595cd5feb/scratchpad"
U = pl.read_parquet(f"{SC}/user_offday.parquet").sort("user_id")
V = pl.read_parquet("/Users/alexanderkondakov/ozon-cup/work/preds_pack/val_preds.parquet").sort("user_id")
assert (U["user_id"].to_numpy() == V["user_id"].to_numpy()).all()
N = U.height
g = lambda c: U[c].to_numpy().astype(np.float64)
mdl_flint = 2 * 1.665647  # 2R for gain conversion

# ---------- day-count denominators ----------
ND = {"p1": (23, 56), "p2": (34, 89), "p3": (73, 159), "full": (112, 263)}  # (off, wk)

def rates(pref, extra=None):
    o_off, o_wk = g(f"{pref}_ord_off"), g(f"{pref}_ord_wk")
    n_off, n_wk = ND[pref]
    if extra:  # pool two periods
        o_off = o_off + g(f"{extra}_ord_off"); o_wk = o_wk + g(f"{extra}_ord_wk")
        n_off += ND[extra][0]; n_wk += ND[extra][1]
    return o_off, o_wk, n_off, n_wk

def delta(pref, comp_from=(22, 8), comp_to=(18, 12), lam=30.0, extra=None, kind="ord"):
    """delta_i = log(nwk'*rwk + noff'*roff) - log(nwk*rwk + noff*roff), shrunk rates."""
    if kind == "ord":
        o_off, o_wk, n_off, n_wk = rates(pref, extra)
    else:
        o_off, o_wk = g(f"{pref}_gmv_off"), g(f"{pref}_gmv_wk")
        n_off, n_wk = ND[pref]
        if extra:
            o_off = o_off + g(f"{extra}_gmv_off"); o_wk = o_wk + g(f"{extra}_gmv_wk")
            n_off += ND[extra][0]; n_wk += ND[extra][1]
    mu_off, mu_wk = o_off.sum() / n_off / N, o_wk.sum() / n_wk / N
    r_off = (o_off + lam * mu_off) / (n_off + lam)
    r_wk = (o_wk + lam * mu_wk) / (n_wk + lam)
    a, b = comp_from; a2, b2 = comp_to
    d = np.log(a2 * r_wk + b2 * r_off) - np.log(a * r_wk + b * r_off)
    return d - d.mean()

print("=== 1. global off/wk per-day rates (full 375d, NY excluded) ===")
for kind, cols in [("orders", ("full_ord_off", "full_ord_wk")), ("gmv", ("full_gmv_off", "full_gmv_wk")),
                   ("active-days", ("full_ad_off", "full_ad_wk"))]:
    off, wk = g(cols[0]).sum(), g(cols[1]).sum()
    print(f"  {kind}: off/day {off/112:.1f}  wk/day {wk/263:.1f}  rho={off/112/(wk/263):.4f}")

print("\n=== 2. split-half reliability of off-share (alternating weeks, full period) ===")
NP = {"off": (56, 56), "wk": (127, 136)}  # par0, par1
for kind in ["ord", "gmv", "ad"]:
    sh = {}
    for p in [0, 1]:
        o = g(f"fp{p}_{kind}_off"); w = g(f"fp{p}_{kind}_wk")
        n_o = NP["off"][p]; n_w = NP["wk"][p]
        tot = o + w
        share = np.where(tot > 0, o / np.maximum(tot, 1e-9), np.nan)
        exp_share = n_o / (n_o + n_w)
        sh[p] = (share - exp_share, tot)
    m = ~np.isnan(sh[0][0]) & ~np.isnan(sh[1][0])
    r = np.corrcoef(sh[0][0][m], sh[1][0][m])[0, 1]
    m8 = m & (sh[0][1] >= 8) & (sh[1][1] >= 8)
    r8 = np.corrcoef(sh[0][0][m8], sh[1][0][m8])[0, 1] if m8.sum() > 100 else np.nan
    print(f"  {kind}: r_all={r:+.4f} (n={m.sum()})  r_cnt>=8/half={r8:+.4f} (n={m8.sum()})")

# delta-form reliability (their style): per-parity delta with light shrinkage
d0 = None
for p in [0, 1]:
    o_off = g(f"fp{p}_ord_off"); o_wk = g(f"fp{p}_ord_wk")
    n_off, n_wk = NP["off"][p], NP["wk"][p]
    mu_off, mu_wk = o_off.sum() / n_off / N, o_wk.sum() / n_wk / N
    lam = 5.0
    r_off = (o_off + lam * mu_off) / (n_off + lam); r_wk = (o_wk + lam * mu_wk) / (n_wk + lam)
    d = np.log(18 * r_wk + 12 * r_off) - np.log(22 * r_wk + 8 * r_off)
    d = d - d.mean()
    if p == 0: d0 = d
    else: d1 = d
cov01 = np.mean(d0 * d1)
print(f"  delta-form (lam=5): rms0={np.sqrt((d0**2).mean()):.4f} rms1={np.sqrt((d1**2).mean()):.4f} "
      f"cov={cov01:+.3e} r={cov01/np.sqrt((d0**2).mean()*(d1**2).mean()):+.4f} "
      f"-> rms(true delta)={np.sqrt(max(cov01,0)):.4f}  oracle gain={max(cov01,0)/mdl_flint:.2e}")

print("\n=== 3. natural experiments: does off-share trait predict window-pair dGMV? ===")
def experiment(name, m_from, m_to, dl, subset=None):
    y = np.log1p(g(f"{m_to}_gmv")) - np.log1p(g(f"{m_from}_gmv"))
    x = dl.copy()
    msk = np.ones(N, bool) if subset is None else subset
    y = y[msk] - y[msk].mean(); x = x[msk] - x[msk].mean()
    vx = (x**2).mean()
    if vx < 1e-12:
        print(f"  {name}: var(x)=0"); return
    beta = (x * y).mean() / vx
    resid = y - beta * x
    se = np.sqrt((resid**2).mean() / vx / len(x))
    # 5-fold by-user OOF gain in predicting y
    idx = np.arange(len(x)); rng = np.random.default_rng(0); rng.shuffle(idx)
    gains = []
    for f in range(5):
        te = idx[f::5]; tr = np.setdiff1d(idx, te)
        b_tr = (x[tr] * y[tr]).mean() / (x[tr]**2).mean()
        gains.append(((y[te]**2).mean() - ((y[te] - b_tr * x[te])**2).mean()))
    print(f"  {name}: n={len(x)} sd(x)={np.sqrt(vx):.4f} beta={beta:+.3f} (se {se:.3f}, t={beta/se:+.2f}) "
          f"corr={beta*np.sqrt(vx)/y.std():+.4f}  OOF dMSE={np.mean(gains):+.2e}  "
          f"implied dRMSLE if transferred={np.mean(gains)/mdl_flint:+.2e}")
    return beta, se

# Apr->May: composition (22,8)->(18,12) exactly = val->test
d_p1p2 = delta("p1", extra="p2", lam=30)          # no-leak max-power trait (202 d)
d_p1 = delta("p1", lam=30)                        # deployment-realistic pre-window trait
experiment("Apr->May | trait mdl_amber+mdl_gabbro (no leak)", "m_apr", "m_may", d_p1p2)
experiment("Apr->May | trait mdl_amber only (79d)", "m_apr", "m_may", d_p1)
# active users only
act = g("p1_ord_off") + g("p1_ord_wk") + g("p2_ord_off") + g("p2_ord_wk") >= 10
experiment("Apr->May | mdl_amber+mdl_gabbro, users>=10 ord", "m_apr", "m_may", d_p1p2, subset=act)
# gmv-based trait
d_gmv = delta("p1", extra="p2", lam=30, kind="gmv")
experiment("Apr->May | gmv-trait mdl_amber+mdl_gabbro", "m_apr", "m_may", d_gmv)
# Oct->Nov: (22,8)->(19,11)
d_p3b = delta("p3", comp_to=(19, 11), lam=30)
experiment("Oct->Nov | trait mdl_halite (Jan-Aug)", "m_oct", "m_nov", d_p3b)
# placebo Sep->Oct with the SAME (22,8)->(18,12) formula: real composition unchanged
d_p3 = delta("p3", lam=30)
experiment("PLACEBO Sep->Oct | trait mdl_halite", "m_sep", "m_oct", d_p3)

print("\n=== 4. deployment vector for test26 (full trait, (22,8)->(18,12)) ===")
d_full = delta("full", lam=30)
q = np.percentile(d_full, [0.1, 1, 5, 25, 50, 75, 95, 99, 99.9])
print(f"  sd={d_full.std():.4f}  pct[0.1,1,5,25,50,75,95,99,99.9]={np.round(q,4)}")
y_val = np.log1p(V["target"].to_numpy().astype(np.float64))
blend = V["blend"].to_numpy().astype(np.float64)
e = y_val - blend
print(f"  corr(delta_full, e_val)={np.corrcoef(d_full, e)[0,1]:+.4f}  (S-axis: must be ~0)")
print(f"  corr(delta_full, blend)={np.corrcoef(d_full, blend)[0,1]:+.4f}")
# concentration: share of sum delta^2 in top users by predicted GMV
pg = np.expm1(blend)
ordr = np.argsort(-pg)
d2 = d_full**2
for top in [0.01, 0.001]:
    k = int(N * top)
    print(f"  share of sum(delta^2) in top-{top:.1%} by blend GMV: {d2[ordr[:k]].sum()/d2.sum():.3f}")
act_full = g("full_ord_off") + g("full_ord_wk")
ordr2 = np.argsort(-act_full)
for top in [0.01]:
    k = int(N * top)
    print(f"  share of sum(delta^2) in top-{top:.1%} by order count: {d2[ordr2[:k]].sum()/d2.sum():.3f}")

print("\n=== 5. holiday-weekday behavior: is there a separate 'holiday off-day' trait? ===")
# hol8: 8 weekday-holidays 2025. Expected orders under user's own off-day rate:
o_off, o_wk, n_off, n_wk = rates("full")
exp_hol = 8.0 * o_off / 112.0
obs_hol = g("hol8_ord")
m = o_off >= 8
dev = obs_hol - exp_hol
print(f"  users with >=8 off-day orders: {m.sum()}; mean obs {obs_hol[m].mean():.3f} vs exp {exp_hol[m].mean():.3f} "
      f"ratio {obs_hol[m].sum()/exp_hol[m].sum():.4f}")
# overdispersion vs binomial: var of (obs-exp) compared to poisson floor
var_obs = dev[m].var(); poiss = exp_hol[m].mean()
print(f"  var(obs-exp)={var_obs:.3f} vs poisson floor~{poiss:.3f} -> excess var {max(var_obs-poiss,0):.3f} "
      f"(trait var upper bound, counts)")

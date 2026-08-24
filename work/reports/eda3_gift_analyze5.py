#!/usr/bin/env python
"""eda3 gift pass 5:
   (C') honest link: pre-anchor-week check/category structure vs CHAMPION val residual;
   (D)  is there a distinct 'gift profile'? compare standardized structure coefficients across anchors;
   (E)  descriptive: does the 2026 test run-up week look like the 2025 one (population + per-user)?
"""
import numpy as np, polars as pl, json

R = "/Users/alexanderkondakov/ozon-cup/work/reports"
A = pl.read_parquet(f"{R}/eda3_gift_anchors.parquet").sort("user_id")
Vw = pl.read_parquet(f"{R}/eda3_gift_valanchor.parquet").sort("user_id")
W1 = pl.read_parquet(f"{R}/eda3_gift_user_windows.parquet").sort("user_id")
V = pl.read_parquet("/Users/alexanderkondakov/ozon-cup/work/preds_pack/val_preds.parquet").sort("user_id")
Z = np.load("/Users/alexanderkondakov/ozon-cup/final_submission/models/chain_test.npz")
N = A.height
lp = np.log1p

def g(c):
    for src in (A, Vw, W1):
        if c in src.columns:
            return src[c].to_numpy().astype(np.float64)
    raise KeyError(c)

def sd_(a, b):
    o = np.full(N, np.nan); m = b > 0; o[m] = a[m] / b[m]; return o

rng = np.random.default_rng(11); fold = rng.integers(0, 5, N)
out = {}

STRUCT_NAMES = ["aov_l", "aov_l_minus_h", "csh_l", "csh_l_minus_h", "ssh_l", "catr_l", "catr_l_minus_h", "cartr_l"]

def base_struct(h, l):
    base = [lp(g(f"{h}_gmv")), lp(g(f"{h}_to_ord")), lp(g(f"{h}_to_cart")), lp(g(f"{h}_search")),
            lp(g(f"{h}_cat")), g(f"{h}_dord").astype(float), g(f"{h}_dact").astype(float),
            lp(g(f"{l}_gmv")), lp(g(f"{l}_to_ord")), lp(g(f"{l}_to_cart")), lp(g(f"{l}_search")),
            lp(g(f"{l}_cat")), g(f"{l}_dord").astype(float), g(f"{l}_dact").astype(float)]
    aov_l = lp(sd_(g(f"{l}_gmv"), g(f"{l}_to_ord"))); aov_h = lp(sd_(g(f"{h}_gmv"), g(f"{h}_to_ord")))
    csh_l = sd_(g(f"{l}_gmv_cat"), g(f"{l}_gmv")); csh_h = sd_(g(f"{h}_gmv_cat"), g(f"{h}_gmv"))
    ssh_l = sd_(g(f"{l}_gmv_search"), g(f"{l}_gmv"))
    catr_l = sd_(g(f"{l}_cat"), g(f"{l}_cat") + g(f"{l}_search"))
    catr_h = sd_(g(f"{h}_cat"), g(f"{h}_cat") + g(f"{h}_search"))
    cartr_l = sd_(g(f"{l}_to_cart"), g(f"{l}_to_ord"))
    struct = [aov_l, aov_l - aov_h, csh_l, csh_l - csh_h, ssh_l, catr_l, catr_l - catr_h, cartr_l]
    return base, struct

def imp(cols):
    """impute NaN with mean, append one missing-flag per NaN-bearing column (fixed order)."""
    X = np.column_stack(cols).astype(float); flags = []
    for j in range(X.shape[1]):
        c = X[:, j]
        if not np.isfinite(c).all():
            miss = ~np.isfinite(c); flags.append(miss.astype(float))
            c = c.copy(); c[miss] = np.nanmean(c); X[:, j] = c
    return np.column_stack([X] + flags) if flags else X

def oof(cols, y):
    X = imp(cols); pred = np.zeros(N)
    for k in range(5):
        tr = fold != k; te = fold == k
        M = np.column_stack([np.ones(tr.sum()), X[tr]])
        cf, *_ = np.linalg.lstsq(M, y[tr], rcond=None)
        pred[te] = cf[0] + X[te] @ cf[1:]
    return float(np.sqrt(((y - pred) ** 2).mean())), pred

# ---------- (C') champion residual at the val anchor ----------
y = lp(V["target"].to_numpy().astype(np.float64))
b = V["blend"].to_numpy().astype(np.float64)
e = y - b
rmse_b = float(np.sqrt((e ** 2).mean()))
p2step = lp(mdl_gabbro["predict"].to_numpy()) - Z["ref_lp"]

base_v, struct_v = base_struct("vAh", "vAl")
sing = {}
for nm, f in zip(STRUCT_NAMES, struct_v):
    m = np.isfinite(f)
    # partial: residualize against blend + p2 step
    B = np.column_stack([np.ones(m.sum()), b[m], p2step[m]])
    cf1, *_ = np.linalg.lstsq(B, f[m], rcond=None)
    cf2, *_ = np.linalg.lstsq(B, e[m], rcond=None)
    sing[nm] = {"cov": round(float(m.mean()), 3),
                "corr_e": round(float(np.corrcoef(f[m], e[m])[0, 1]), 5),
                "corr_e_partial": round(float(np.corrcoef(f[m] - B @ cf1, e[m] - B @ cf2)[0, 1]), 5)}
out["Cp_valanchor_struct_single"] = sing

ctrl = [b, p2step] + base_v
r_ctrl, p_ctrl = oof(ctrl, e)
r_full, p_full = oof(ctrl + struct_v, e)
gi = (e - p_ctrl) ** 2 - (e - p_full) ** 2
o = np.argsort(-np.abs(gi))
out["Cp_valanchor_corrector"] = {
    "rmse_blend": round(rmse_b, 6),
    "oof_rmse_ctrl(levels+blend+mdl_gabbro)": round(r_ctrl, 6),
    "oof_rmse_ctrl+struct": round(r_full, 6),
    "delta_levels_vs_blend": round(r_ctrl - rmse_b, 7),
    "delta_struct_extra": round(r_full - r_ctrl, 7),
    "struct_gain_top1pct_share": round(float(gi[o[: N // 100]].sum() / gi.sum()), 3),
    "struct_gain_top01pct_share": round(float(gi[o[: N // 1000]].sum() / gi.sum()), 3),
}

# ---------- (D) is the gift profile distinct? standardized struct coefficients per anchor ----------
coefs = {}
for a in ["gA", "gB", "c1", "c2", "c3"]:
    base_a, struct_a = base_struct(f"{a}h", f"{a}l")
    ya = lp(g(f"{a}t_gmv"))
    X = imp(base_a + struct_a)
    mu, sdv = X.mean(0), X.std(0) + 1e-9
    Xs = (X - mu) / sdv
    M = np.column_stack([np.ones(N), Xs])
    cf, *_ = np.linalg.lstsq(M, ya, rcond=None)
    coefs[a] = cf[1 + len(base_a): 1 + len(base_a) + len(STRUCT_NAMES)]
out["D_struct_coefs"] = {a: {n: round(float(v), 4) for n, v in zip(STRUCT_NAMES, c)} for a, c in coefs.items()}
pairs = {}
keys = list(coefs)
for i in range(len(keys)):
    for j in range(i + 1, len(keys)):
        a, bb = keys[i], keys[j]
        pairs[f"{a}|{bb}"] = round(float(np.corrcoef(coefs[a], coefs[bb])[0, 1]), 4)
out["D_coef_profile_corr"] = pairs

# ---------- (E) 2026 test run-up week vs 2025 analog: population & per-user ----------
def popstat(p, ndays):
    gmv, ordn = g(f"{p}_gmv").sum(), g(f"{p}_to_ord").sum()
    return {"aov": round(float(gmv / ordn), 2), "ord_per_day": round(float(ordn / ndays), 1),
            "cat_gmv_share": round(float(g(f"{p}_gmv_cat").sum() / gmv), 4),
            "buyers": int((g(f"{p}_to_ord") > 0).sum()),
            "cart_per_ord": round(float(g(f"{p}_to_cart").sum() / ordn), 3)}
out["E_runup_population"] = {
    "2025 vrun 07-13.02": popstat("vrun", 7),
    "2026 vrun 07-13.02": popstat("tAl", 7),
    "2026 pre-val 08-14.01": popstat("vAl", 7),
    "2025 pre-val analog (vrunb 10.01-06.02)": popstat("vrunb", 28),
}
# per-user persistence of gift-run structure 2025 -> 2026 (same calendar week, one year apart)
aov25 = lp(sd_(g("vrun_gmv"), g("vrun_to_ord"))); aov26 = lp(sd_(g("tAl_gmv"), g("tAl_to_ord")))
csh25 = sd_(g("vrun_gmv_cat"), g("vrun_gmv")); csh26 = sd_(g("tAl_gmv_cat"), g("tAl_gmv"))
buy25 = (g("vrun_to_ord") > 0).astype(float); buy26 = (g("tAl_to_ord") > 0).astype(float)
def cc(x, z):
    m = np.isfinite(x) & np.isfinite(z)
    return (round(float(np.corrcoef(x[m], z[m])[0, 1]), 4), int(m.sum()))
B = np.column_stack([np.ones(N), lp(g("year_gmv")), lp(g("year_to_ord")), g("year_dord").astype(float)])
c1_, *_ = np.linalg.lstsq(B, buy25, rcond=None); c2_, *_ = np.linalg.lstsq(B, buy26, rcond=None)
out["E_runup_yoy_peruser"] = {
    "aov_2025_vs_2026": cc(aov25, aov26),
    "catshare_2025_vs_2026": cc(csh25, csh26),
    "buyflag_raw": cc(buy25, buy26),
    "buyflag_partial_on_year_level": round(float(np.corrcoef(buy25 - B @ c1_, buy26 - B @ c2_)[0, 1]), 4),
}

print(json.dumps(out, indent=1, ensure_ascii=False))
with open(f"{R}/eda3_gift_analyze5.json", "w") as fh:
    json.dump(out, fh, indent=1, ensure_ascii=False)

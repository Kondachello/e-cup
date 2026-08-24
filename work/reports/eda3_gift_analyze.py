#!/usr/bin/env python
"""eda3 gift carriers: candidate features vs val residual + cross-holiday structural trait."""
import numpy as np, polars as pl, json

R = "/Users/alexanderkondakov/ozon-cup/work/reports"
W = pl.read_parquet(f"{R}/eda3_gift_user_windows.parquet").sort("user_id")
V = pl.read_parquet("/Users/alexanderkondakov/ozon-cup/work/preds_pack/val_preds.parquet").sort("user_id")
Z = np.load("/Users/alexanderkondakov/ozon-cup/final_submission/models/chain_test.npz")

uid = W["user_id"].to_numpy()
assert (uid == V["user_id"].to_numpy()).all() and (uid == Z["user_id"]).all() and (uid == mdl_gabbro["user_id"].to_numpy()).all()
N = len(uid)

y = np.log1p(V["target"].to_numpy().astype(np.float64))
b = V["blend"].to_numpy().astype(np.float64)
e = y - b
rmse_b = float(np.sqrt((e**2).mean()))

g = lambda c: W[c].to_numpy().astype(np.float64)
lp = np.log1p
p2step = lp(mdl_gabbro["predict"].to_numpy()) - Z["ref_lp"]

# ---- year-ex (full history minus p2 window) ----
yx = {f: g(f"year_{f}") - g(f"p2win_{f}") for f in ["gmv", "to_ord", "cat", "search", "gmv_cat", "gmv_search"]}
yx_dord = g("year_dord") - g("p2win_dord")
lp_year = lp(g("year_gmv"))
lp_ya = lp(g("p2win_gmv"))            # mdl_gabbro underlying variable

def safe_div(a, bb):
    out = np.full(N, np.nan)
    m = bb > 0
    out[m] = a[m] / bb[m]
    return out

# ---- candidate features (2025 gift window, categories & checks) ----
feats = {}
aov_gift = safe_div(g("gift_gmv"), g("gift_to_ord"))
aov_yx = safe_div(yx["gmv"], yx["to_ord"])
feats["A_aov_gift_lift"] = lp(aov_gift) - lp(aov_yx)                      # check-size lift vs own year
feats["B_aov_gift_abs"] = lp(aov_gift)                                    # absolute check in gift window
cs_gift = safe_div(g("gift_gmv_cat"), g("gift_gmv"))
cs_yx = safe_div(yx["gmv_cat"], yx["gmv"])
feats["C_catshare_lift"] = cs_gift - cs_yx                                # category-channel GMV share lift
ss_gift = safe_div(g("gift_gmv_search"), g("gift_gmv"))
ss_yx = safe_div(yx["gmv_search"], yx["gmv"])
feats["C2_searchshare_lift"] = ss_gift - ss_yx
days_yx = 349.0  # 379 days year minus 30 p2win
feats["D_catcnt_lift"] = lp(g("gift_cat") / 23.0) - lp(yx["cat"] / days_yx)          # cat-visit rate lift
feats["E_srchcnt_lift"] = lp(g("gift_search") / 23.0) - lp(yx["search"] / days_yx)
feats["F_ordrate_lift"] = lp(g("gift_to_ord") / 23.0) - lp(yx["to_ord"] / days_yx)   # order-count lift (control)
feats["G_gift_only"] = ((g("gift_to_ord") > 0) & (yx["to_ord"] <= 1)).astype(float)  # gift-only buyer flag
feats["H_dord_lift"] = lp(g("gift_dord") / 23.0) - lp(yx_dord / days_yx)             # order-days lift
# run-level (7d) versions: valentine+mar8 runs vs their own 28d bases
for nm in ["vrun", "mrun"]:
    a_r = safe_div(g(f"{nm}_gmv"), g(f"{nm}_to_ord"))
    a_b = safe_div(g(f"{nm}b_gmv"), g(f"{nm}b_to_ord"))
    feats[f"R_aov_{nm}"] = lp(a_r) - lp(a_b)
feats["S_lp_ya"] = lp_ya.copy()  # reference: the mdl_gabbro variable itself, for calibration of the method

basis = np.column_stack([np.ones(N), b, lp_year, lp_ya, p2step])

def resid_on(x, B, m):
    coef, *_ = np.linalg.lstsq(B[m], x[m], rcond=None)
    return x - B @ coef

def whale_share(gain_i, k):
    idx = np.argsort(-np.abs(gain_i))[: max(1, int(N * k))]
    tot = gain_i.sum()
    return float(gain_i[idx].sum() / tot) if tot != 0 else np.nan

rng = np.random.default_rng(42)
fold = rng.integers(0, 5, N)

res = {}
for name, f in feats.items():
    m = np.isfinite(f)
    cov = float(m.mean())
    if m.sum() < 1000:
        res[name] = {"coverage": cov, "note": "too few"}
        continue
    r_raw = float(np.corrcoef(f[m], e[m])[0, 1])
    # partial: residualize feature and e on basis, fitted on defined subset
    fr = resid_on(f, basis, m)
    er = resid_on(e, basis, m)
    r_part = float(np.corrcoef(fr[m], er[m])[0, 1])
    # OOF 1D corrector on residualized feature (0 outside coverage)
    corr_vec = np.zeros(N)
    for k in range(5):
        tr = m & (fold != k); te = m & (fold == k)
        if tr.sum() < 100 or te.sum() == 0: continue
        X = np.column_stack([np.ones(tr.sum()), fr[tr]])
        cf, *_ = np.linalg.lstsq(X, e[tr], rcond=None)
        corr_vec[te] = cf[0] + cf[1] * fr[te]
    e_new = e - corr_vec
    rmse_new = float(np.sqrt((e_new**2).mean()))
    gain_i = e**2 - e_new**2
    res[name] = {
        "coverage": round(cov, 4), "n_def": int(m.sum()),
        "corr_raw": round(r_raw, 5), "corr_partial": round(r_part, 5),
        "oof_rmse_delta": round(rmse_new - rmse_b, 7),
        "gain_top1pct_share": round(whale_share(gain_i, 0.01), 3),
        "gain_top01pct_share": round(whale_share(gain_i, 0.001), 3),
    }

# ---- cross-holiday structural trait (2025, within-train) ----
runs = ["vrun", "f23", "mrun", "jun", "oct", "nov11", "dec7"]
def run_lifts(nm):
    a_r = safe_div(g(f"{nm}_gmv"), g(f"{nm}_to_ord"))
    a_b = safe_div(g(f"{nm}b_gmv"), g(f"{nm}b_to_ord"))
    aov_l = lp(a_r) - lp(a_b)
    cs_r = safe_div(g(f"{nm}_gmv_cat"), g(f"{nm}_gmv"))
    cs_b = safe_div(g(f"{nm}b_gmv_cat"), g(f"{nm}b_gmv"))
    cs_l = cs_r - cs_b
    cat_l = lp(g(f"{nm}_cat")) - lp(g(f"{nm}b_cat") * 7.0 / 28.0)
    act = (g(f"{nm}_dact") > 0) & (g(f"{nm}b_dact") > 0)
    cat_l[~act] = np.nan
    gmv_l = lp(g(f"{nm}_gmv")) - lp(g(f"{nm}b_gmv") / 4.0)  # GMV lift control (prior art)
    gmv_l[~act] = np.nan
    return {"aov": aov_l, "csh": cs_l, "cat": cat_l, "gmv": gmv_l,
            "lvl_aov": lp(a_r), "lvl_cat": lp(g(f"{nm}_cat"))}

L = {nm: run_lifts(nm) for nm in runs}
pairs = [("vrun", "mrun"), ("vrun", "f23"), ("f23", "mrun"), ("vrun", "dec7"),
         ("mrun", "dec7"), ("mrun", "nov11"), ("jun", "oct"), ("vrun", "jun"),
         ("nov11", "dec7"), ("vrun", "nov11")]
cross = {}
for a, bb in pairs:
    row = {}
    for kind in ["aov", "csh", "cat", "gmv", "lvl_aov", "lvl_cat"]:
        x, z2 = L[a][kind], L[bb][kind]
        mm = np.isfinite(x) & np.isfinite(z2)
        row[kind] = {"corr": round(float(np.corrcoef(x[mm], z2[mm])[0, 1]), 4), "n": int(mm.sum())} if mm.sum() > 500 else None
    cross[f"{a}|{bb}"] = row

out = {"rmse_blend": round(rmse_b, 6), "features": res, "cross_holiday": cross}
print(json.dumps(out, indent=1, ensure_ascii=False))
with open(f"{R}/eda3_gift_analyze.json", "w") as fh:
    json.dump(out, fh, indent=1, ensure_ascii=False)

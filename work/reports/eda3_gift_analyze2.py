#!/usr/bin/env python
"""eda3 gift carriers pass 2: clean vrun->mrun trait, mirror YoY structural increment, profile AUC."""
import numpy as np, polars as pl, json

R = "/Users/alexanderkondakov/ozon-cup/work/reports"
W1 = pl.read_parquet(f"{R}/eda3_gift_user_windows.parquet").sort("user_id")
mdl_onyx = pl.read_parquet(f"{R}/eda3_gift_user_windows2.parquet").sort("user_id")
assert (W1["user_id"].to_numpy() == mdl_onyx["user_id"].to_numpy()).all()
N = W1.height
lp = np.log1p

def g(c):
    src = W1 if c in W1.columns else mdl_onyx
    return src[c].to_numpy().astype(np.float64)

def safe_div(a, b):
    out = np.full(N, np.nan); m = b > 0; out[m] = a[m] / b[m]; return out

def corr(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    return (round(float(np.corrcoef(x[m], y[m])[0, 1]), 4), int(m.sum())) if m.sum() > 500 else (None, int(m.sum()))

out = {}

#vrun -> mrun with DISJOINT bases =================
def lifts(run, base):
    a_r = safe_div(g(f"{run}_gmv"), g(f"{run}_to_ord"))
    a_b = safe_div(g(f"{base}_gmv"), g(f"{base}_to_ord"))
    len_r = {"vrun": 7, "mrun": 7, "jun": 7, "oct": 7, "dec7": 7}[run]
    len_b = 28.0
    cs_r = safe_div(g(f"{run}_gmv_cat"), g(f"{run}_gmv"))
    cs_b = safe_div(g(f"{base}_gmv_cat"), g(f"{base}_gmv"))
    act = (g(f"{run}_dact") > 0) & (g(f"{base}_dact") > 0)
    cat_l = lp(g(f"{run}_cat") / len_r) - lp(g(f"{base}_cat") / len_b); cat_l[~act] = np.nan
    gmv_l = lp(g(f"{run}_gmv") / len_r) - lp(g(f"{base}_gmv") / len_b); gmv_l[~act] = np.nan
    ord_l = lp(g(f"{run}_to_ord") / len_r) - lp(g(f"{base}_to_ord") / len_b); ord_l[~act] = np.nan
    return {"aov": lp(a_r) - lp(a_b), "csh": cs_r - cs_b, "cat": cat_l, "gmv": gmv_l, "ord": ord_l}

Lv = lifts("vrun", "vrunb")     # valentine run vs 28d before it
Lm = lifts("mrun", "postb")     # march-8 run vs 28d AFTER season (disjoint from vrun)
Lj = lifts("jun", "junb")
Lo = lifts("oct", "octb")
Ld = lifts("dec7", "dec7b")
m1 = {}
for kind in ["aov", "csh", "cat", "gmv", "ord"]:
    m1[kind] = {
        "vrun|mrun_clean": corr(Lv[kind], Lm[kind]),
        "jun|oct": corr(Lj[kind], Lo[kind]),
        "vrun|dec7": corr(Lv[kind], Ld[kind]),
    }
# buy-flag persistence given base frequency
def part_flag(run1, run2, b1, b2):
    f1 = (g(f"{run1}_to_ord") > 0).astype(float); f2 = (g(f"{run2}_to_ord") > 0).astype(float)
    B = np.column_stack([np.ones(N), lp(g(f"{b1}_to_ord")), lp(g(f"{b2}_to_ord")), lp(g("year_to_ord")), lp(g("year_gmv"))])
    c1, *_ = np.linalg.lstsq(B, f1, rcond=None); c2, *_ = np.linalg.lstsq(B, f2, rcond=None)
    return round(float(np.corrcoef(f1 - B @ c1, f2 - B @ c2)[0, 1]), 4)
m1["buyflag_partial"] = {
    "vrun|mrun": part_flag("vrun", "mrun", "vrunb", "postb"),
    "jun|oct": part_flag("jun", "oct", "junb", "octb"),
    "vrun|dec7": part_flag("vrun", "dec7", "vrunb", "dec7b"),
}
out["M1_within_season"] = m1

# ================= M2: mirror YoY structural increment =================
yY = lp(g("Y_gmv"))
def feat_block(names):
    return np.column_stack(names)
X_lvl = [lp(g("X_gmv")), lp(g("X_to_ord")), lp(g("X_cat")), lp(g("X_search")), g("X_dord"),
         lp(g("y2X_gmv")), lp(g("y2X_to_ord")), g("y2X_dord"), lp(g("y2X_cat"))]
L1 = [lp(g("Yp_gmv")), lp(g("Xp_gmv"))]
aov_Yp = lp(safe_div(g("Yp_gmv"), g("Yp_to_ord")))
aov_Xp = lp(safe_div(g("Xp_gmv"), g("Xp_to_ord")))
csh_Yp = safe_div(g("Yp_gmv_cat"), g("Yp_gmv"))
csh_Xp = safe_div(g("Xp_gmv_cat"), g("Xp_gmv"))

def prep(cols):
    Xm = np.column_stack(cols)
    add = []
    for j in range(Xm.shape[1]):
        col = Xm[:, j]
        if not np.isfinite(col).all():
            miss = ~np.isfinite(col)
            add.append(miss.astype(float))
            col = col.copy(); col[miss] = np.nanmean(col); Xm[:, j] = col
    if add: Xm = np.column_stack([Xm] + add)
    return Xm

rng = np.random.default_rng(7)
fold = rng.integers(0, 5, N)
def oof_rmse(cols):
    Xm = prep(cols); pred = np.zeros(N)
    for k in range(5):
        tr = fold != k; te = fold == k
        A = np.column_stack([np.ones(tr.sum()), Xm[tr]])
        cf, *_ = np.linalg.lstsq(A, yY[tr], rcond=None)
        pred[te] = cf[0] + Xm[te] @ cf[1:]
    return float(np.sqrt(((yY - pred) ** 2).mean()))

r_b0 = oof_rmse(X_lvl)
r_b1 = oof_rmse(X_lvl + L1)
out["M2_mirror"] = {"oof_rmse_base": round(r_b0, 6), "plus_ya_level": round(r_b1, 6),
                    "plus_ya_structure": round(r_b2, 6),
                    "delta_level": round(r_b1 - r_b0, 6), "delta_structure": round(r_b2 - r_b1, 6)}

# ================= M3: is gift-window buying more predictable? =================
def auc(score, ybin):
    m = np.isfinite(score)
    s, yb = score[m], ybin[m]
    r = np.argsort(np.argsort(s)) + 1
    n1 = yb.sum(); n0 = len(yb) - n1
    return float((r[yb == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))

def run_auc(run, base):
    ybin = (g(f"{run}_to_ord") > 0).astype(int)
    lvl = [lp(g(f"{base}_gmv")), lp(g(f"{base}_to_ord")), g(f"{base}_dord").astype(float), lp(g(f"{base}_to_cart"))]
    strv = [lp(g(f"{base}_cat")), lp(g(f"{base}_search")),
            lp(safe_div(g(f"{base}_gmv"), g(f"{base}_to_ord"))),
            safe_div(g(f"{base}_gmv_cat"), g(f"{base}_gmv"))]
    def ridge_score(cols):
        Xm = prep(cols)
        mu, sd = Xm.mean(0), Xm.std(0) + 1e-9
        Xm = (Xm - mu) / sd
        pred = np.zeros(N)
        for k in range(5):
            tr = fold != k; te = fold == k
            A = Xm[tr]; yb = ybin[tr].astype(float)
            cf = np.linalg.solve(A.T @ A + 1e-3 * len(yb) * np.eye(A.shape[1]), A.T @ (yb - yb.mean()))
            pred[te] = Xm[te] @ cf
        return pred
    a_lvl = auc(ridge_score(lvl), ybin)
    a_all = auc(ridge_score(lvl + strv), ybin)
    return {"rate": round(float(ybin.mean()), 4), "auc_level": round(a_lvl, 4),
            "auc_level+struct": round(a_all, 4), "delta_struct": round(a_all - a_lvl, 4)}

out["M3_profile_auc"] = {
    "mrun(gift)": run_auc("mrun", "mrunb"),
    "vrun(gift)": run_auc("vrun", "vrunb"),
    "dec7(gift)": run_auc("dec7", "dec7b"),
    "jun(ctrl)": run_auc("jun", "junb"),
    "oct(ctrl)": run_auc("oct", "octb"),
    "nov11(sale)": run_auc("nov11", "nov11b"),
}

print(json.dumps(out, indent=1, ensure_ascii=False))
with open(f"{R}/eda3_gift_analyze2.json", "w") as fh:
    json.dump(out, fh, indent=1, ensure_ascii=False)

#!/usr/bin/env python
"""eda3 gift pass 4:
   (a) population-level check/category anomaly of gift windows;
   (b) does run-up-week STRUCTURE (checks+categories) earn more at gift anchors than neutral ones;
   (c) same structure built on 07-13.02.2026 -> link to champion val residual (ortho to mdl_gabbro + blend).
"""
import numpy as np, polars as pl, json

R = "/Users/alexanderkondakov/ozon-cup/work/reports"
W1 = pl.read_parquet(f"{R}/eda3_gift_user_windows.parquet").sort("user_id")
mdl_onyx = pl.read_parquet(f"{R}/eda3_gift_user_windows2.parquet").sort("user_id")
A = pl.read_parquet(f"{R}/eda3_gift_anchors.parquet").sort("user_id")
V = pl.read_parquet("/Users/alexanderkondakov/ozon-cup/work/preds_pack/val_preds.parquet").sort("user_id")
Z = np.load("/Users/alexanderkondakov/ozon-cup/final_submission/models/chain_test.npz")
N = A.height
lp = np.log1p

def g(c):
    for src in (A, W1, mdl_onyx):
        if c in src.columns:
            return src[c].to_numpy().astype(np.float64)
    raise KeyError(c)

def sd_(a, b):
    o = np.full(N, np.nan); m = b > 0; o[m] = a[m] / b[m]; return o

out = {}

# ---------- (a) population-level ----------
pop = {}
wins = {"gift_feb14_mar8": ("gift", 23), "p2win_feb14_mar15": ("p2win", 30), "bjan_pre30": ("bjan", 30),
        "bspr_post30": ("bspr", 30), "vrun7": ("vrun", 7), "f23run7": ("f23", 7), "mrun7": ("mrun", 7),
        "jun7": ("jun", 7), "oct7": ("oct", 7), "nov11run7": ("nov11", 7), "dec7run7": ("dec7", 7)}
for nm, (p, ndays) in wins.items():
    gmv, ordn = g(f"{p}_gmv").sum(), g(f"{p}_to_ord").sum()
    pop[nm] = {"aov": round(float(gmv / ordn), 2),
               "ord_per_day": round(float(ordn / ndays), 1),
               "gmv_per_day_M": round(float(gmv / ndays / 1e6), 2),
               "cat_gmv_share": round(float(g(f"{p}_gmv_cat").sum() / gmv), 4),
               "search_gmv_share": round(float(g(f"{p}_gmv_search").sum() / gmv), 4)}
out["A_population"] = pop

# ---------- (b) anchor experiment ----------
rng = np.random.default_rng(11); fold = rng.integers(0, 5, N)

def prep(cols):
    X = np.column_stack(cols); add = []
    for j in range(X.shape[1]):
        c = X[:, j]
        if not np.isfinite(c).all():
            miss = ~np.isfinite(c); add.append(miss.astype(float))
            c = c.copy(); c[miss] = np.nanmean(c); X[:, j] = c
    return np.column_stack([X] + add) if add else X

def oof_rmse(cols, y):
    X = prep(cols); pred = np.zeros(N)
    for k in range(5):
        tr = fold != k; te = fold == k
        M = np.column_stack([np.ones(tr.sum()), X[tr]])
        cf, *_ = np.linalg.lstsq(M, y[tr], rcond=None)
        pred[te] = cf[0] + X[te] @ cf[1:]
    return float(np.sqrt(((y - pred) ** 2).mean())), pred

def blocks(a):
    h, l = f"{a}h", f"{a}l"
    base = [lp(g(f"{h}_gmv")), lp(g(f"{h}_to_ord")), lp(g(f"{h}_to_cart")), lp(g(f"{h}_search")),
            lp(g(f"{h}_cat")), g(f"{h}_dord").astype(float), g(f"{h}_dact").astype(float),
            lp(g(f"{l}_gmv")), lp(g(f"{l}_to_ord")), lp(g(f"{l}_to_cart")), lp(g(f"{l}_search")),
            lp(g(f"{l}_cat")), g(f"{l}_dord").astype(float), g(f"{l}_dact").astype(float)]
    aov_l = lp(sd_(g(f"{l}_gmv"), g(f"{l}_to_ord")))
    aov_h = lp(sd_(g(f"{h}_gmv"), g(f"{h}_to_ord")))
    csh_l = sd_(g(f"{l}_gmv_cat"), g(f"{l}_gmv"))
    csh_h = sd_(g(f"{h}_gmv_cat"), g(f"{h}_gmv"))
    ssh_l = sd_(g(f"{l}_gmv_search"), g(f"{l}_gmv"))
    catr_l = sd_(g(f"{l}_cat"), g(f"{l}_cat") + g(f"{l}_search"))
    catr_h = sd_(g(f"{h}_cat"), g(f"{h}_cat") + g(f"{h}_search"))
    cartr_l = sd_(g(f"{l}_to_cart"), g(f"{l}_to_ord"))
    struct = [aov_l, aov_l - aov_h, csh_l, csh_l - csh_h, ssh_l, catr_l, catr_l - catr_h, cartr_l]
    y = lp(g(f"{a}t_gmv"))
    return base, struct, y

anc = {}
for a, label in [("gA", "GIFT feb14-mar15"), ("gB", "GIFT dec15-jan13"),
                 ("c1", "neutral jun"), ("c2", "neutral sep"), ("c3", "promo nov11")]:
    base, struct, y = blocks(a)
    r0, _ = oof_rmse(base, y)
    r1, p1 = oof_rmse(base + struct, y)
    # whale concentration of the structure increment
    _, p0 = oof_rmse(base, y)
    gi = (y - p0) ** 2 - (y - p1) ** 2
    o = np.argsort(-np.abs(gi))
    anc[a] = {"label": label, "oof_base": round(r0, 6), "oof_base+struct": round(r1, 6),
              "delta_struct": round(r1 - r0, 6),
              "top1pct_share": round(float(gi[o[: N // 100]].sum() / gi.sum()), 3),
              "top01pct_share": round(float(gi[o[: N // 1000]].sum() / gi.sum()), 3),
              "buyers_in_target": int((g(f"{a}t_to_ord") > 0).sum())}
out["B_anchor_experiment"] = anc

# ---------- (c) 2026 run-up structure vs champion residual ----------
uid = A["user_id"].to_numpy()
assert (uid == V["user_id"].to_numpy()).all() and (uid == Z["user_id"]).all()
y = lp(V["target"].to_numpy().astype(np.float64))
b = V["blend"].to_numpy().astype(np.float64)
e = y - b
rmse_b = float(np.sqrt((e ** 2).mean()))
p2step = lp(mdl_gabbro["predict"].to_numpy()) - Z["ref_lp"]

aov_26 = lp(sd_(g("vrun26_gmv"), g("vrun26_to_ord")))
csh_26 = sd_(g("vrun26_gmv_cat"), g("vrun26_gmv"))
catr_26 = sd_(g("vrun26_cat"), g("vrun26_cat") + g("vrun26_search"))
cartr_26 = sd_(g("vrun26_to_cart"), g("vrun26_to_ord"))
val_aov = lp(sd_(g("valw_gmv"), g("valw_to_ord")))
val_csh = sd_(g("valw_gmv_cat"), g("valw_gmv"))
val_catr = sd_(g("valw_cat"), g("valw_cat") + g("valw_search"))
S26 = {"aov_lastweek": aov_26, "aov_lastweek_lift": aov_26 - val_aov,
       "catshare_lastweek": csh_26, "catshare_lift": csh_26 - val_csh,
       "catratio_lastweek": catr_26, "catratio_lift": catr_26 - val_catr,
       "cartratio_lastweek": cartr_26,
       "lp_gmv_lastweek": lp(g("vrun26_gmv")), "lp_cat_lastweek": lp(g("vrun26_cat")),
       "lp_cart_lastweek": lp(g("vrun26_to_cart")), "dact_lastweek": g("vrun26_dact").astype(float)}
sing = {}
for nm, f in S26.items():
    m = np.isfinite(f)
    sing[nm] = {"cov": round(float(m.mean()), 3), "corr_e": round(float(np.corrcoef(f[m], e[m])[0, 1]), 5)}
out["C_2026_runup_single"] = sing

ctrl = [np.ones(N), b, p2step, lp(g("valw_gmv")), lp(g("valw_to_ord")), g("valw_dord").astype(float)]
X = prep(list(S26.values()) + ctrl[1:])
pred = np.zeros(N)
for k in range(5):
    tr = fold != k; te = fold == k
    M = np.column_stack([np.ones(tr.sum()), X[tr]])
    cf, *_ = np.linalg.lstsq(M, e[tr], rcond=None)
    pred[te] = cf[0] + X[te] @ cf[1:]
e2 = e - pred
gi = e ** 2 - e2 ** 2
o = np.argsort(-np.abs(gi))
out["C_2026_runup_corrector"] = {
    "rmse_blend": round(rmse_b, 6),
    "oof_rmse": round(float(np.sqrt((e2 ** 2).mean())), 6),
    "oof_delta": round(float(np.sqrt((e2 ** 2).mean())) - rmse_b, 7),
    "top1pct_share": round(float(gi[o[: N // 100]].sum() / gi.sum()), 3),
    "top01pct_share": round(float(gi[o[: N // 1000]].sum() / gi.sum()), 3),
}
# control: same corrector but WITHOUT the structure columns (only level controls)
Xc = prep(ctrl[1:])
predc = np.zeros(N)
for k in range(5):
    tr = fold != k; te = fold == k
    M = np.column_stack([np.ones(tr.sum()), Xc[tr]])
    cf, *_ = np.linalg.lstsq(M, e[tr], rcond=None)
    predc[te] = cf[0] + Xc[te] @ cf[1:]
out["C_2026_runup_corrector"]["oof_delta_levels_only"] = round(
    float(np.sqrt(((e - predc) ** 2).mean())) - rmse_b, 7)

print(json.dumps(out, indent=1, ensure_ascii=False))
with open(f"{R}/eda3_gift_analyze4.json", "w") as fh:
    json.dump(out, fh, indent=1, ensure_ascii=False)

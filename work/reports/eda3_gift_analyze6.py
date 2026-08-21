#!/usr/bin/env python
"""eda3 gift pass 6: matched same-calendar-week YoY persistence, gift week vs neutral week.
   Gift pair    : 07-13.02.2025 -> 07-13.02.2026   (Valentine run-up; both observable)
   Neutral pair : 08-14.01.2025 -> 08-14.01.2026
   Level control: 16.03-31.12.2025, disjoint from all four weeks.
"""
import numpy as np, polars as pl, json

R = "/Users/alexanderkondakov/ozon-cup/work/reports"
srcs = [pl.read_parquet(f"{R}/{f}").sort("user_id") for f in
        ["eda3_gift_yoyweeks.parquet", "eda3_gift_valanchor.parquet", "eda3_gift_user_windows.parquet"]]
N = srcs[0].height
lp = np.log1p

def g(c):
    for s in srcs:
        if c in s.columns:
            return s[c].to_numpy().astype(np.float64)
    raise KeyError(c)

def sd_(a, b):
    o = np.full(N, np.nan); m = b > 0; o[m] = a[m] / b[m]; return o

CTRL = np.column_stack([np.ones(N), lp(g("mid_gmv")), lp(g("mid_to_ord")), g("mid_dord").astype(float),
                        lp(g("mid_cat")), lp(g("mid_search")), lp(g("mid_to_cart"))])

def partial_corr(x, z, extra=None):
    m = np.isfinite(x) & np.isfinite(z)
    B = CTRL[m] if extra is None else np.column_stack([CTRL[m]] + [c[m] for c in extra])
    cx, *_ = np.linalg.lstsq(B, x[m], rcond=None)
    cz, *_ = np.linalg.lstsq(B, z[m], rcond=None)
    return round(float(np.corrcoef(x[m] - B @ cx, z[m] - B @ cz)[0, 1]), 4), int(m.sum())

out = {}
pairs = {
    "GIFT 07-13.02 (25->26)": ("vrun", "tAl", "f25b", "f26b"),
    "NEUTRAL 08-14.01 (25->26)": ("j25", "vAl", None, None),
}
res = {}
for label, (a, b, pa, pb) in pairs.items():
    buy_a = (g(f"{a}_to_ord") > 0).astype(float)
    buy_b = (g(f"{b}_to_ord") > 0).astype(float)
    aov_a = lp(sd_(g(f"{a}_gmv"), g(f"{a}_to_ord")))
    aov_b = lp(sd_(g(f"{b}_gmv"), g(f"{b}_to_ord")))
    csh_a = sd_(g(f"{a}_gmv_cat"), g(f"{a}_gmv"))
    csh_b = sd_(g(f"{b}_gmv_cat"), g(f"{b}_gmv"))
    cat_a = lp(g(f"{a}_cat")); cat_b = lp(g(f"{b}_cat"))
    row = {
        "buyers_y1": int(buy_a.sum()), "buyers_y2": int(buy_b.sum()),
        "buyflag_raw": round(float(np.corrcoef(buy_a, buy_b)[0, 1]), 4),
        "buyflag_partial_mid": partial_corr(buy_a, buy_b),
        "aov_partial_mid": partial_corr(aov_a, aov_b),
        "catshare_partial_mid": partial_corr(csh_a, csh_b),
        "catcnt_partial_mid": partial_corr(cat_a, cat_b),
    }
    if pa is not None:  # extra control: activity in the 14 days right before each week
        extra = [lp(g(f"{pa}_gmv")), lp(g(f"{pa}_to_ord")), g(f"{pa}_dord").astype(float),
                 lp(g(f"{pb}_gmv")), lp(g(f"{pb}_to_ord")), g(f"{pb}_dord").astype(float)]
        row["buyflag_partial_mid+prev14"] = partial_corr(buy_a, buy_b, extra)
        row["aov_partial_mid+prev14"] = partial_corr(aov_a, aov_b, extra)
    res[label] = row
out["YoY_matched_weeks"] = res

# how much of the gift-week 2026 buy flag is explainable at all (OOF AUC, level vs +gift-history)
rng = np.random.default_rng(3); fold = rng.integers(0, 5, N)
def auc(s, yb):
    r = np.argsort(np.argsort(s)) + 1
    n1 = yb.sum(); n0 = len(yb) - n1
    return float((r[yb == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
def ridge_auc(cols, yb):
    X = np.column_stack(cols).astype(float)
    for j in range(X.shape[1]):
        c = X[:, j]
        if not np.isfinite(c).all():
            c = c.copy(); c[~np.isfinite(c)] = np.nanmean(c); X[:, j] = c
    X = (X - X.mean(0)) / (X.std(0) + 1e-9)
    pred = np.zeros(N)
    for k in range(5):
        tr = fold != k; te = fold == k
        A = X[tr]; t = yb[tr].astype(float)
        cf = np.linalg.solve(A.T @ A + 1e-3 * len(t) * np.eye(A.shape[1]), A.T @ (t - t.mean()))
        pred[te] = X[te] @ cf
    return round(auc(pred, yb), 4)

yb26 = (g("tAl_to_ord") > 0).astype(int)
lvl = [lp(g("f26b_gmv")), lp(g("f26b_to_ord")), g("f26b_dord").astype(float), lp(g("f26b_to_cart")),
       lp(g("mid_gmv")), lp(g("mid_to_ord")), g("mid_dord").astype(float)]
gifthist = [(g("vrun_to_ord") > 0).astype(float), lp(g("vrun_gmv")),
            (g("mrun_to_ord") > 0).astype(float), lp(g("mrun_gmv")),
            lp(sd_(g("vrun_gmv"), g("vrun_to_ord"))), sd_(g("vrun_gmv_cat"), g("vrun_gmv"))]
out["gift_week_2026_auc"] = {
    "rate": round(float(yb26.mean()), 4),
    "auc_level_only": ridge_auc(lvl, yb26),
    "auc_level+gift_history_2025": ridge_auc(lvl + gifthist, yb26),
}
out["gift_week_2026_auc"]["delta"] = round(
    out["gift_week_2026_auc"]["auc_level+gift_history_2025"] - out["gift_week_2026_auc"]["auc_level_only"], 4)
# neutral-week control of the same increment
yb26n = (g("vAl_to_ord") > 0).astype(int)
lvln = [lp(g("mid_gmv")), lp(g("mid_to_ord")), g("mid_dord").astype(float), lp(g("mid_to_cart"))]
neuhist = [(g("j25_to_ord") > 0).astype(float), lp(g("j25_gmv")),
           (g("jun_to_ord") > 0).astype(float), lp(g("jun_gmv")),
           lp(sd_(g("j25_gmv"), g("j25_to_ord"))), sd_(g("j25_gmv_cat"), g("j25_gmv"))]
out["neutral_week_2026_auc"] = {
    "rate": round(float(yb26n.mean()), 4),
    "auc_level_only": ridge_auc(lvln, yb26n),
    "auc_level+same_week_history_2025": ridge_auc(lvln + neuhist, yb26n),
}
out["neutral_week_2026_auc"]["delta"] = round(
    out["neutral_week_2026_auc"]["auc_level+same_week_history_2025"] - out["neutral_week_2026_auc"]["auc_level_only"], 4)

print(json.dumps(out, indent=1, ensure_ascii=False))
with open(f"{R}/eda3_gift_analyze6.json", "w") as fh:
    json.dump(out, fh, indent=1, ensure_ascii=False)

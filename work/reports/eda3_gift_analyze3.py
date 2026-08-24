#!/usr/bin/env python
"""eda3 gift pass 3: geometric buy-flag control, Yp-structure vs champion residual, M2 decomposition."""
import numpy as np, polars as pl, json

R = "/Users/alexanderkondakov/ozon-cup/work/reports"
W1 = pl.read_parquet(f"{R}/eda3_gift_user_windows.parquet").sort("user_id")
mdl_onyx = pl.read_parquet(f"{R}/eda3_gift_user_windows2.parquet").sort("user_id")
W3 = pl.read_parquet(f"{R}/eda3_gift_user_windows3.parquet").sort("user_id")
V = pl.read_parquet("/Users/alexanderkondakov/ozon-cup/work/preds_pack/val_preds.parquet").sort("user_id")
N = W1.height
lp = np.log1p

def g(c):
    for src in (W1, mdl_onyx, W3):
        if c in src.columns:
            return src[c].to_numpy().astype(np.float64)
    raise KeyError(c)

def safe_div(a, b):
    out = np.full(N, np.nan); m = b > 0; out[m] = a[m] / b[m]; return out

y = lp(V["target"].to_numpy().astype(np.float64))
b = V["blend"].to_numpy().astype(np.float64)
e = y - b
rmse_b = float(np.sqrt((e**2).mean()))
out = {"rmse_blend": round(rmse_b, 6)}

# ---------- 1. buy-flag partial corr: gift pair vs 2 geometric replicas ----------
def part_flag(r1, r2, b1, b2):
    f1 = (g(f"{r1}_to_ord") > 0).astype(float); f2 = (g(f"{r2}_to_ord") > 0).astype(float)
    B = np.column_stack([np.ones(N), lp(g(f"{b1}_to_ord")), lp(g(f"{b2}_to_ord")),
                         lp(g("year_to_ord")), lp(g("year_gmv")), g("year_dord").astype(float)])
    c1, *_ = np.linalg.lstsq(B, f1, rcond=None); c2, *_ = np.linalg.lstsq(B, f2, rcond=None)
    rr = float(np.corrcoef(f1 - B @ c1, f2 - B @ c2)[0, 1])
    return round(rr, 4), int(f1.sum()), int(f2.sum())

out["buyflag_geometric"] = {
    "vrun|mrun (gift)": part_flag("vrun", "mrun", "vrunb", "postb"),
    "june replica": part_flag("jr1", "jr2", "jr1b", "jrpb"),
    "sept replica": part_flag("sr1", "sr2", "sr1b", "srpb"),
}
# same for cat-count lift trait
def part_cnt(r1, r2, b1, b2, field="cat"):
    l1 = lp(g(f"{r1}_{field}") / 7.0) - lp(g(f"{b1}_{field}") / 28.0)
    l2 = lp(g(f"{r2}_{field}") / 7.0) - lp(g(f"{b2}_{field}") / 28.0)
    m = (g(f"{r1}_dact") + g(f"{b1}_dact") > 0) & (g(f"{r2}_dact") + g(f"{b2}_dact") > 0)
    return round(float(np.corrcoef(l1[m], l2[m])[0, 1]), 4), int(m.sum())

out["catlift_geometric"] = {
    "vrun|mrun (gift)": part_cnt("vrun", "mrun", "vrunb", "postb"),
    "june replica": part_cnt("jr1", "jr2", "jr1b", "jrpb"),
    "sept replica": part_cnt("sr1", "sr2", "sr1b", "srpb"),
}

# ---------- 2. Yp-structure vs CHAMPION val residual ----------
# Yp = 23.01-13.02.2025 (year-ago analog of the val window's core), Xp = 01-22.01.2025
aov_Yp = lp(safe_div(g("Yp_gmv"), g("Yp_to_ord")))
aov_Xp = lp(safe_div(g("Xp_gmv"), g("Xp_to_ord")))
csh_Yp = safe_div(g("Yp_gmv_cat"), g("Yp_gmv"))
csh_Xp = safe_div(g("Xp_gmv_cat"), g("Xp_gmv"))
STR = {
    "lp_Yp_gmv": lp(g("Yp_gmv")), "lp_Xp_gmv": lp(g("Xp_gmv")),
    "lp_Yp_ord": lp(g("Yp_to_ord")), "Yp_dord": g("Yp_dord").astype(float),
    "lp_Yp_cat": lp(g("Yp_cat")), "lp_Yp_search": lp(g("Yp_search")),
    "aov_Yp": aov_Yp, "aov_Yp_lift": aov_Yp - aov_Xp,
    "csh_Yp": csh_Yp, "csh_Yp_lift": csh_Yp - csh_Xp,
}
def prep(cols):
    Xm = np.column_stack(cols); add = []
    for j in range(Xm.shape[1]):
        col = Xm[:, j]
        if not np.isfinite(col).all():
            miss = ~np.isfinite(col); add.append(miss.astype(float))
            col = col.copy(); col[miss] = np.nanmean(col); Xm[:, j] = col
    return np.column_stack([Xm] + add) if add else Xm

rng = np.random.default_rng(7); fold = rng.integers(0, 5, N)
def oof_corr_gain(cols, target, base_cols=None):
    """OOF linear model for target; returns rmse delta vs zero-pred and per-user corrections."""
    Xm = prep(cols if base_cols is None else base_cols + cols)
    pred = np.zeros(N)
    for k in range(5):
        tr = fold != k; te = fold == k
        A = np.column_stack([np.ones(tr.sum()), Xm[tr]])
        cf, *_ = np.linalg.lstsq(A, target[tr], rcond=None)
        pred[te] = cf[0] + Xm[te] @ cf[1:]
    return pred

single = {}
for nm, f in STR.items():
    m = np.isfinite(f)
    single[nm] = {"cov": round(float(m.mean()), 3),
                  "corr_e": round(float(np.corrcoef(f[m], e[m])[0, 1]), 5)}
out["Yp_struct_single_corr_e"] = single

# joint OOF corrector of e from full Yp-structure block (with blend as control feature)
pred_e = oof_corr_gain(list(STR.values()) + [b], e)
e_new = e - pred_e
rmse_new = float(np.sqrt((e_new**2).mean()))
gain_i = e**2 - e_new**2
idx1 = np.argsort(-np.abs(gain_i))[: N // 100]
idx01 = np.argsort(-np.abs(gain_i))[: N // 1000]
out["Yp_struct_oof_corrector"] = {
    "oof_rmse_delta": round(rmse_new - rmse_b, 7),
    "total_gain_mse": round(float(gain_i.mean()), 7),
    "top1pct_share": round(float(gain_i[idx1].sum() / gain_i.sum()), 3) if gain_i.sum() != 0 else None,
    "top01pct_share": round(float(gain_i[idx01].sum() / gain_i.sum()), 3) if gain_i.sum() != 0 else None,
}

# ---------- 3. M2 decomposition with stronger base ----------
yY = lp(g("Y_gmv"))
aov_X = lp(safe_div(g("X_gmv"), g("X_to_ord")))
csh_X = safe_div(g("X_gmv_cat"), g("X_gmv"))
BASE = [lp(g("X_gmv")), lp(g("X_to_ord")), lp(g("X_cat")), lp(g("X_search")), g("X_dord").astype(float),
        lp(g("X_to_cart")), aov_X, csh_X,
        lp(g("y2X_gmv")), lp(g("y2X_to_ord")), g("y2X_dord").astype(float), lp(g("y2X_cat")),
        lp(g("y2X_search")), lp(g("y2X_to_cart"))]
L_lvl = [lp(g("Yp_gmv")), lp(g("Xp_gmv"))]
L_ord = [lp(g("Yp_to_ord")), g("Yp_dord").astype(float)]
L_cnt = [lp(g("Yp_cat")), lp(g("Yp_search"))]
L_str = [aov_Yp, aov_Yp - aov_Xp, csh_Yp, csh_Yp - csh_Xp]

def oof_rmse(cols):
    Xm = prep(cols); pred = np.zeros(N)
    for k in range(5):
        tr = fold != k; te = fold == k
        A = np.column_stack([np.ones(tr.sum()), Xm[tr]])
        cf, *_ = np.linalg.lstsq(A, yY[tr], rcond=None)
        pred[te] = cf[0] + Xm[te] @ cf[1:]
    return float(np.sqrt(((yY - pred) ** 2).mean()))

r0 = oof_rmse(BASE)
steps = {
    "base_rich": r0,
    "+ya_gmv_level": oof_rmse(BASE + L_lvl),
    "+ya_ord": oof_rmse(BASE + L_lvl + L_ord),
    "+ya_counts": oof_rmse(BASE + L_lvl + L_ord + L_cnt),
    "+ya_aov_csh": oof_rmse(BASE + L_lvl + L_ord + L_cnt + L_str),
    "aov_csh_only": oof_rmse(BASE + L_str),
    "counts_only": oof_rmse(BASE + L_cnt),
    "ord_only": oof_rmse(BASE + L_ord),
}
out["M2_decomp"] = {k: round(v, 6) for k, v in steps.items()}
out["M2_decomp_deltas"] = {
    "ya_gmv_level": round(steps["+ya_gmv_level"] - r0, 6),
    "ord_extra": round(steps["+ya_ord"] - steps["+ya_gmv_level"], 6),
    "counts_extra": round(steps["+ya_counts"] - steps["+ya_ord"], 6),
    "aov_csh_extra": round(steps["+ya_aov_csh"] - steps["+ya_counts"], 6),
    "aov_csh_alone": round(steps["aov_csh_only"] - r0, 6),
    "counts_alone": round(steps["counts_only"] - r0, 6),
    "ord_alone": round(steps["ord_only"] - r0, 6),
}

print(json.dumps(out, indent=1, ensure_ascii=False))
with open(f"{R}/eda3_gift_analyze3.json", "w") as fh:
    json.dump(out, fh, indent=1, ensure_ascii=False)

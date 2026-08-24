#!/usr/bin/env python
"""eda3 gift pass 7: final checks - channel decomposition sanity, standalone check-size corrector,
   whale concentration of the level-corrector (closed class reference), noise floor for the deltas."""
import numpy as np, polars as pl, json

R = "/Users/alexanderkondakov/ozon-cup/work/reports"
srcs = [pl.read_parquet(f"{R}/{f}").sort("user_id") for f in
        ["eda3_gift_valanchor.parquet", "eda3_gift_user_windows.parquet", "eda3_gift_yoyweeks.parquet"]]
V = pl.read_parquet("/Users/alexanderkondakov/ozon-cup/work/preds_pack/val_preds.parquet").sort("user_id")
Z = np.load("/Users/alexanderkondakov/ozon-cup/final_submission/models/chain_test.npz")
N = srcs[0].height
lp = np.log1p
def g(c):
    for s in srcs:
        if c in s.columns: return s[c].to_numpy().astype(np.float64)
    raise KeyError(c)
def sd_(a, b):
    o = np.full(N, np.nan); m = b > 0; o[m] = a[m] / b[m]; return o

out = {}
# 0. channel sanity: does gmv split exactly into search + cat?
lf = pl.scan_parquet("/Users/alexanderkondakov/ozon-cup/train.parquet")
chk = lf.select([(pl.col("gmv") - pl.col("gmv_search") - pl.col("gmv_cat")).abs().max().alias("max_abs_resid"),
                 pl.col("gmv").sum().alias("gmv"), pl.col("gmv_cat").sum().alias("gmv_cat")]).collect(engine="streaming")
out["channel_sanity"] = {"max|gmv-gmv_search-gmv_cat|": float(chk["max_abs_resid"][0]),
                         "cat_share_all_data": round(float(chk["gmv_cat"][0] / chk["gmv"][0]), 4)}

y = lp(V["target"].to_numpy().astype(np.float64)); b = V["blend"].to_numpy().astype(np.float64)
e = y - b; rmse_b = float(np.sqrt((e**2).mean()))
p2step = lp(mdl_gabbro["predict"].to_numpy()) - Z["ref_lp"]
rng = np.random.default_rng(11); fold = rng.integers(0, 5, N)

def oof_corr(cols):
    X = np.column_stack(cols).astype(float); flags = []
    for j in range(X.shape[1]):
        c = X[:, j]
        if not np.isfinite(c).all():
            m = ~np.isfinite(c); flags.append(m.astype(float)); c = c.copy(); c[m] = np.nanmean(c); X[:, j] = c
    if flags: X = np.column_stack([X] + flags)
    pred = np.zeros(N)
    for k in range(5):
        tr = fold != k; te = fold == k
        M = np.column_stack([np.ones(tr.sum()), X[tr]])
        cf, *_ = np.linalg.lstsq(M, e[tr], rcond=None); pred[te] = cf[0] + X[te] @ cf[1:]
    en = e - pred; gi = e**2 - en**2; o = np.argsort(-np.abs(gi))
    return {"oof_delta": round(float(np.sqrt((en**2).mean())) - rmse_b, 7),
            "top1pct_share": round(float(gi[o[: N // 100]].sum() / gi.sum()), 3),
            "top01pct_share": round(float(gi[o[: N // 1000]].sum() / gi.sum()), 3)}

aov_l = lp(sd_(g("vAl_gmv"), g("vAl_to_ord")))
csh_l = sd_(g("vAl_gmv_cat"), g("vAl_gmv"))
lvl = [lp(g("vAh_gmv")), lp(g("vAh_to_ord")), g("vAh_dord").astype(float), lp(g("vAl_gmv")),
       lp(g("vAl_to_ord")), g("vAl_dord").astype(float)]
out["standalone_correctors"] = {
    "aov_lastweek_only(+blend)": oof_corr([b, aov_l]),
    "catshare_lastweek_only(+blend)": oof_corr([b, csh_l]),
    "levels_only(+blend) [CLOSED class ref]": oof_corr([b] + lvl),
    "levels+P2step": oof_corr([b, p2step] + lvl),
}
# noise floor: same OOF procedure on pure noise features
noise = [rng.normal(size=N) for _ in range(8)]
out["noise_floor_8_random_features"] = oof_corr([b] + noise)
print(json.dumps(out, indent=1, ensure_ascii=False))
with open(f"{R}/eda3_gift_analyze7.json", "w") as fh:
    json.dump(out, fh, indent=1, ensure_ascii=False)

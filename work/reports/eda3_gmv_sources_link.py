"""eda3 lens 'soglasovannost istochnikov GMV': link beyond-shell channel vectors to blend residual.
corr, honest 2-fold OOF (user parity), whale concentration."""
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
print(f"n={n} rmse0={rmse0:.6f} var(e)={np.var(e):.4f}")

def g(c): return np.nan_to_num(df[c].to_numpy().astype(np.float64))

L = np.log1p
has_hist = (g("n_gmv_days") > 0).astype(float)
cands = {
    # cross-day attribution (conversion day without same-day session flag) - joint day-level, beyond shell
    "gc_nocat_share": g("gc_nocat_share"),
    "log_n_gc_nocat": L(g("n_gc_nocat")),
    "gs_nosearch_any": (g("n_gs_nosearch") > 0).astype(float),
    # day-level mix volatility / purity - beyond shell
    "std_daily_catshare": g("std_daily_catshare"),
    "mixed_share": g("mixed_share"),
    "log_n_mixed": L(g("n_mixed")),
    "switch_rate": g("switch_rate"),
    # fine temporal drift of mix
    "catshare_slope": g("catshare_slope"),
    "drift_h2_h1": np.where((g("gmv_h1") > 0) & (g("gmv_h2") > 0), g("catshare_h2") - g("catshare_h1"), 0.0),
    "drift_60_all": np.where(g("gmv_60") > 0, g("catshare_60") - g("cat_rub_share"), 0.0),
    "abs_drift_h2_h1": np.abs(np.where((g("gmv_h1") > 0) & (g("gmv_h2") > 0), g("catshare_h2") - g("catshare_h1"), 0.0)),
    # controls (inside shell): expect ~0 honest gain
    "CTRL_cat_rub_share": g("cat_rub_share"),
    "CTRL_log_n_gmv_days": L(g("n_gmv_days")),
}

fold = (df["user_id"].to_numpy() % 2 == 0)
rows = []
for name, xv in cands.items():
    xv = xv.astype(np.float64)
    c = float(np.corrcoef(xv, e)[0, 1])
    pred = np.zeros(n)
    for m in (fold, ~fold):
        tr, te = ~m, m
        xc = xv[tr] - xv[tr].mean()
        beta = float((xc @ (e[tr] - e[tr].mean())) / np.maximum(xc @ xc, 1e-12))
        a0 = float(e[tr].mean() - beta * xv[tr].mean())
        pred[te] = a0 + beta * xv[te]
    pred_shape = pred - pred.mean()  # level-free: global shift is a closed class
    e2 = e - pred_shape
    rmse_oof = float(np.sqrt(np.mean(e2 ** 2)))
    gain = rmse0 - rmse_oof
    d = e ** 2 - e2 ** 2
    order = np.argsort(-d)
    tot = d.sum()
    top01 = float(d[order[: max(1, n // 1000)]].sum() / tot) if tot != 0 else np.nan
    top1 = float(d[order[: max(1, n // 100)]].sum() / tot) if tot != 0 else np.nan
    rows.append((name, round(c, 5), round(gain, 7), round(top1, 3), round(top01, 3)))

out = pl.DataFrame(rows, schema=["cand", "corr_e", "oof_gain", "top1pct_share", "top01pct_share"], orient="row")
pl.Config.set_tbl_rows(30); pl.Config.set_tbl_width_chars(120)
print(out.sort("oof_gain", descending=True))
print("noise floor |corr| ~", round(1 / np.sqrt(n), 5),
      "; gain 1e-4 needs |corr| ~", round(float(np.sqrt(1e-4 * 2 * rmse0 / np.var(e))), 4))

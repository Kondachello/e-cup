"""eda3 step 5: link candidate exposure vectors to blend residual on val. corr, honest 2-fold OOF, whale concentration."""
import numpy as np
import polars as pl

pp = pl.read_parquet("/Users/alexanderkondakov/ozon-cup/work/preds_pack/val_preds.parquet",
                     columns=["user_id", "target", "blend"])
uv = pl.read_parquet("/Users/alexanderkondakov/ozon-cup/work/reports/eda3_user_vectors.parquet")
df = pp.join(uv, on="user_id", how="left").fill_null(0)

y = np.log1p(df["target"].to_numpy())
b = df["blend"].to_numpy().astype(np.float64)
e = y - b
n = len(e)
rmse0 = float(np.sqrt(np.mean(e ** 2)))
print(f"n={n} rmse0={rmse0:.6f} var(e)={np.var(e):.4f}")

def g(c): return df[c].to_numpy().astype(np.float64)

L = np.log1p
cands = {
    # catalog-step exposure (era split not present in the 203 aggregates)
    "cat_prestep_share": np.where(g("cat_pre_v") + g("cat_post_v") > 0,
                                  g("cat_pre_v") / np.maximum(g("cat_pre_v") + g("cat_post_v"), 1e-9), 0.0),
    "log_cat_pre": L(g("cat_pre_v")),
    "log_cat_post": L(g("cat_post_v")),
    "c2o_prestep_share": np.where(g("c2o_pre_v") + g("c2o_post_v") > 0,
                                  g("c2o_pre_v") / np.maximum(g("c2o_pre_v") + g("c2o_post_v"), 1e-9), 0.0),
    "log_gmvcat_pre": L(g("gmvcat_pre_v")),
    "gmvcat_prestep_share": np.where(g("gmvcat_pre_v") + g("gmvcat_post_v") > 0,
                                     g("gmvcat_pre_v") / np.maximum(g("gmvcat_pre_v") + g("gmvcat_post_v"), 1e-9), 0.0),
    # ghost-row (all-zero counters) exposure
    "ghost_share": g("ghost_v") / np.maximum(g("act_v"), 1),
    "log_ghost": L(g("ghost_v")),
    # generic pre-era activity share (placebo control: pure recency/trend proxy, should be absorbed)
    "act_prestep_share": g("act_pre_v") / np.maximum(g("act_v"), 1),
}

rng = np.random.default_rng(0)
fold = (df["user_id"].to_numpy() % 2 == 0)

rows = []
for name, x in cands.items():
    x = x.astype(np.float64)
    c = float(np.corrcoef(x, e)[0, 1])
    # honest 2-fold projection
    pred = np.zeros(n)
    for m in (fold, ~fold):
        tr, te = ~m, m
        xc = x[tr] - x[tr].mean()
        beta = float((xc @ (e[tr] - e[tr].mean())) / np.maximum(xc @ xc, 1e-12))
        a0 = float(e[tr].mean() - beta * x[tr].mean())
        pred[te] = a0 + beta * x[te]
    # level-free version: subtract global mean of pred so only shape counts
    pred_shape = pred - pred.mean() + 0.0
    e2 = e - pred_shape
    rmse_oof = float(np.sqrt(np.mean(e2 ** 2)))
    gain = rmse0 - rmse_oof
    # whale concentration of the gain
    d = e ** 2 - e2 ** 2
    order = np.argsort(-d)
    tot = d.sum()
    top01 = float(d[order[: max(1, n // 1000)]].sum() / tot) if tot != 0 else np.nan
    top1 = float(d[order[: max(1, n // 100)]].sum() / tot) if tot != 0 else np.nan
    rows.append((name, round(c, 5), round(gain, 7), round(top1, 3), round(top01, 3)))

out = pl.DataFrame(rows, schema=["cand", "corr_e", "oof_gain", "top1pct_share", "top01pct_share"], orient="row")
pl.Config.set_tbl_rows(30)
print(out.sort("oof_gain", descending=True))
print("noise floor |corr| ~", round(1 / np.sqrt(n), 5),
      "; gain needed 1e-4 needs |corr| ~", round(np.sqrt(1e-4 * 2 * rmse0 / np.var(e)), 4))

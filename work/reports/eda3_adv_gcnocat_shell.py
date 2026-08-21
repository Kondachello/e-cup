# Проверка теоремы оболочки для флага gc_nocat:
# (а) насколько флаг/счётчик восстановим из полного набора табличных признаков (LGB, честный 2-fold);
# (б) разложение гейна корректора: часть флага, выразимая оболочкой (flag_hat) vs невязка (flag - flag_hat).
# Если гейн сидит в flag_hat -> переоткрытие закрытого класса; если в невязке -> сигнал вне оболочки.
import os, sys
os.environ.setdefault("USE_V2", "1")
os.environ.setdefault("USE_V3", "1")
os.environ.setdefault("USE_V4", "1")
sys.path.insert(0, "work/scripts")
import numpy as np
import polars as pl
import lightgbm as lgb
from common import VAL_ANCHOR, load_anchor, feature_cols

u = pl.read_parquet("work/reports/eda3_adv_gcnocat_user.parquet").sort("user_id")
uid = u["user_id"].to_numpy()
e = u["e"].to_numpy()
flag = u["flag_v"].to_numpy().astype(float)
n_gc = u["n_gc_nocat"].to_numpy().astype(float)

val = load_anchor(VAL_ANCHOR).sort("user_id")
assert (val["user_id"].to_numpy() == uid).all()
cols = feature_cols(val)
print(f"shell features: {len(cols)}")
X = val.select(cols).with_columns(pl.all().cast(pl.Float64)).fill_null(np.nan).to_numpy()

folds = (uid % 2).astype(int)
params = dict(objective="binary", learning_rate=0.08, num_leaves=63,
              min_data_in_leaf=200, feature_fraction=0.8, bagging_fraction=0.8,
              bagging_freq=1, verbosity=-1, num_threads=8, seed=13)

flag_hat = np.zeros(len(uid))
for f in [0, 1]:
    tr = folds != f
    te = folds == f
    ds = lgb.Dataset(X[tr], label=flag[tr])
    m = lgb.train(params, ds, num_boost_round=250)
    flag_hat[te] = m.predict(X[te])

from sklearn.metrics import roc_auc_score
auc = roc_auc_score(flag, flag_hat)
brier = ((flag_hat - flag) ** 2).mean()
r2_flag = 1 - brier / flag.var()
print(f"flag ~ shell(203): OOF AUC {auc:.4f}  Brier {brier:.4f}  mdl_flint {r2_flag:.3f}")

# счётчик дней (log1p) — насколько сам вектор восстановим
p2 = dict(params); p2["objective"] = "regression"
ngl = np.log1p(n_gc)
ng_hat = np.zeros(len(uid))
for f in [0, 1]:
    tr = folds != f; te = folds == f
    m = lgb.train(p2, lgb.Dataset(X[tr], label=ngl[tr]), num_boost_round=250)
    ng_hat[te] = m.predict(X[te])
r2_cnt = 1 - ((ng_hat - ngl) ** 2).mean() / ngl.var()
print(f"log1p(n_gc_nocat) ~ shell: OOF mdl_flint {r2_cnt:.3f}")

# --- разложение гейна корректора ---
def oof_corr_gain(z, e, folds):
    """centered one-feature OLS corrector, honest by folds"""
    corr = np.zeros_like(e)
    for f in np.unique(folds):
        tr = folds != f; te = folds == f
        zc = z[tr] - z[tr].mean()
        beta = (zc * (e[tr] - e[tr].mean())).sum() / (zc ** 2).sum()
        corr[te] = beta * (z[te] - z[tr].mean())
    return np.sqrt((e**2).mean()) - np.sqrt(((e - corr)**2).mean()), corr

flag_resid = flag - flag_hat
g_flag, _ = oof_corr_gain(flag, e, folds)
g_hat, _ = oof_corr_gain(flag_hat, e, folds)
g_res, _ = oof_corr_gain(flag_resid, e, folds)
# двухфакторный: e ~ flag_hat + flag_resid
def oof_two(z1, z2, e, folds):
    corr = np.zeros_like(e)
    for f in np.unique(folds):
        tr = folds != f; te = folds == f
        A = np.column_stack([np.ones(tr.sum()), z1[tr], z2[tr]])
        beta = np.linalg.lstsq(A, e[tr], rcond=None)[0]
        At = np.column_stack([np.ones(te.sum()), z1[te], z2[te]])
        corr[te] = At @ beta - beta[0]  # без свободного члена в применении? нет — центрируем
        corr[te] = (At @ beta) - (A @ beta).mean()
    return np.sqrt((e**2).mean()) - np.sqrt(((e - corr)**2).mean()), corr
g_two, _ = oof_two(flag_hat, flag_resid, e, folds)
print(f"corrector OOF gains (centered): flag {g_flag*1e4:.2f}e-4 | flag_hat(shell part) {g_hat*1e4:.2f}e-4 | flag-flag_hat(novel part) {g_res*1e4:.2f}e-4 | both {g_two*1e4:.2f}e-4")

# corr остатка с частями
def nc(a, b): return np.corrcoef(a, b)[0, 1]
print(f"corr(e, flag) {nc(e, flag):+.4f} | corr(e, flag_hat) {nc(e, flag_hat):+.4f} | corr(e, flag_resid) {nc(e, flag_resid):+.4f}")

# --- партиализация малым прокси-набором (реплика охотника своим кодом) ---
prox = np.column_stack([
    np.log1p(u["n_cat_days"].to_numpy()),
    np.log1p(u["n_gmvcat_days"].to_numpy()),
    np.log1p(u["n_c2o_days"].to_numpy()),
    np.log1p(u["n_ord_days"].to_numpy()),
    np.log1p(u["gmv_sum"].to_numpy()),
    np.log1p(np.minimum(u["rec_cat"].to_numpy(), 400)),
    np.log1p(np.minimum(u["rec_gmvcat"].to_numpy(), 400)),
    np.log1p(np.minimum(u["rec_ord"].to_numpy(), 400)),
    u["blend"].to_numpy(),
    u["blend"].to_numpy() ** 2,
])
A = np.column_stack([np.ones(len(uid)), prox])
bt = np.linalg.lstsq(A, flag, rcond=None)[0]
flag_p = flag - A @ bt
r2p = 1 - flag_p.var() / flag.var()
bt2 = np.linalg.lstsq(A, e, rcond=None)[0]
e_p = e - A @ bt2
print(f"proxy mdl_flint(flag~10 prox) {r2p:.3f};  partial corr(e_p, flag_p) {nc(e_p, flag_p):+.4f}")
g_part, _ = oof_corr_gain(flag_p, e_p, folds)
print(f"partial corrector gain (e_p ~ flag_p, OOF) {g_part*1e4:.2f}e-4")

np.save("work/reports/eda3_adv_gcnocat_flaghat.npy", flag_hat)
print("saved flag_hat")

# Дожим: сколько стоит именно ВНЕоболочечная часть флага gc_nocat.
# (а) насыщение LGB (600 деревьев) — не растёт ли оболочечная доля дальше;
# (б) бутстреп CI гейна новой части; (в) разрыв e внутри децилей flag_hat;
# (г) дозозависимость по n_gc_nocat.
import os, sys
os.environ.setdefault("USE_V2", "1"); os.environ.setdefault("USE_V3", "1"); os.environ.setdefault("USE_V4", "1")
sys.path.insert(0, "work/scripts")
import numpy as np, polars as pl, lightgbm as lgb
from common import VAL_ANCHOR, load_anchor, feature_cols
from sklearn.metrics import roc_auc_score

RNG = np.random.default_rng(11)
u = pl.read_parquet("work/reports/eda3_adv_gcnocat_user.parquet").sort("user_id")
uid = u["user_id"].to_numpy(); e = u["e"].to_numpy()
flag = u["flag_v"].to_numpy().astype(float)
n_gc = u["n_gc_nocat"].to_numpy().astype(float)
folds = (uid % 2).astype(int)

val = load_anchor(VAL_ANCHOR).sort("user_id")
X = val.select(feature_cols(val)).with_columns(pl.all().cast(pl.Float64)).fill_null(np.nan).to_numpy()

params = dict(objective="binary", learning_rate=0.06, num_leaves=127,
              min_data_in_leaf=100, feature_fraction=0.9, bagging_fraction=0.9,
              bagging_freq=1, verbosity=-1, num_threads=8, seed=13)
fh = np.zeros(len(uid))
for f in [0, 1]:
    tr = folds != f; te = folds == f
    m = lgb.train(params, lgb.Dataset(X[tr], label=flag[tr]), num_boost_round=600)
    fh[te] = m.predict(X[te])
print(f"LGB600: AUC {roc_auc_score(flag, fh):.4f}  mdl_flint {1-((fh-flag)**2).mean()/flag.var():.3f}")

def oof_corr_gain(z, e, folds):
    corr = np.zeros_like(e)
    for f in np.unique(folds):
        tr = folds != f; te = folds == f
        zc = z[tr] - z[tr].mean()
        beta = (zc * (e[tr] - e[tr].mean())).sum() / (zc ** 2).sum()
        corr[te] = beta * (z[te] - z[tr].mean())
    return np.sqrt((e**2).mean()) - np.sqrt(((e - corr)**2).mean()), corr

fr = flag - fh
g_flag, c_flag = oof_corr_gain(flag, e, folds)
g_hat, c_hat = oof_corr_gain(fh, e, folds)
g_res, c_res = oof_corr_gain(fr, e, folds)
print(f"gains: flag {g_flag*1e4:.2f}e-4 | shell part {g_hat*1e4:.2f}e-4 | novel part {g_res*1e4:.2f}e-4")

# бутстреп CI новой части
d = e**2 - (e - c_res)**2
mse0 = (e**2).mean()
gains = np.empty(500)
n = len(e)
for i in range(500):
    s = RNG.integers(0, n, n)
    m0 = (e[s]**2).mean()
    gains[i] = np.sqrt(m0) - np.sqrt(m0 - d[s].mean())
ci = np.percentile(gains, [2.5, 50, 97.5])
print(f"novel part bootstrap CI: [{ci[0]*1e4:.2f}, {ci[1]*1e4:.2f}, {ci[2]*1e4:.2f}]e-4")

# разрыв e по флагу ВНУТРИ децилей flag_hat
qs = np.quantile(fh, np.linspace(0, 1, 11)); qs[0] -= 1; qs[-1] += 1
dec = np.digitize(fh, qs[1:-1])
gaps, ws = [], []
for k in range(10):
    m = dec == k
    n1 = (flag[m] > 0).sum(); n0 = (flag[m] == 0).sum()
    if n1 > 100 and n0 > 100:
        g = e[m][flag[m] > 0].mean() - e[m][flag[m] == 0].mean()
        se = np.sqrt(e[m][flag[m] > 0].var()/n1 + e[m][flag[m] == 0].var()/n0)
        gaps.append(g); ws.append(1/se**2)
        print(f"  fh-decile {k}: n1={n1:6d} n0={n0:6d} gap {g:+.4f} ± {se:.4f}")
gaps = np.array(gaps); ws = np.array(ws)
wg = (gaps*ws).sum()/ws.sum()
print(f"within-flag_hat-decile gap (weighted): {wg:+.4f} ± {np.sqrt(1/ws.sum()):.4f}")

# дозозависимость
for lo, hi, lab in [(1, 1, "n=1"), (2, 2, "n=2"), (3, 4, "n=3-4"), (5, 99, "n>=5")]:
    m = (n_gc >= lo) & (n_gc <= hi)
    print(f"  {lab:6s} users {m.sum():6d}  mean_e {e[m].mean():+.4f}")
print(f"  n=0    users {(n_gc==0).sum():6d}  mean_e {e[n_gc==0].mean():+.4f}")

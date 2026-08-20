"""Train P(buy in [T,T+30)) classifier on val-grid anchors <= 344, predict at val anchor 379.
Reproduce the team's hard-pool phenomenon (predicted-no-buy pool, buyer share, in-pool AUC).
"""
import numpy as np, lightgbm as lgb, json
from sklearn.metrics import roc_auc_score
from features import build_features

cube = np.load("/root/work/cube_val.npy", mmap_mode="r")
buy_mat = np.load("/root/work/buy_mat.npy")
anchor_days = np.load("/root/work/anchor_days.npy")  # 36..379 step 7
day_to_col = {int(d): i for i, d in enumerate(anchor_days)}

TRAIN_DAYS = [295, 302, 309, 316, 323, 330, 337, 344]
VAL_DAY = 379

Xs, ys = [], []
for d in TRAIN_DAYS:
    X, names = build_features(d, cube, 379)
    y = buy_mat[:, day_to_col[d]]
    Xs.append(X); ys.append(y)
    print("built", d, X.shape)
Xtr = np.concatenate(Xs); ytr = np.concatenate(ys)
del Xs
Xv, names = build_features(VAL_DAY, cube, 379)
yv = buy_mat[:, day_to_col[VAL_DAY]]
np.save("/root/work/Xval_anchor379.npy", Xv)
json.dump(names, open("/root/work/feat_names.json", "w"))

params = dict(objective="binary", learning_rate=0.05, num_leaves=127,
              min_data_in_leaf=300, feature_fraction=0.8, bagging_fraction=0.8,
              bagging_freq=1, verbose=-1, num_threads=2, seed=1)
dtr = lgb.Dataset(Xtr, ytr, feature_name=names)
model = lgb.train(params, dtr, num_boost_round=400)
model.save_model("/root/work/clf_buy.txt")
p = model.predict(Xv)
np.save("/root/work/p_buy_val.npy", p)
print("overall AUC (val anchor):", round(roc_auc_score(yv, p), 4))
print("overall buyer share:", round(yv.mean(), 4))

# hard pool scan: pool = p < t
for frac in [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.46]:
    t = np.quantile(p, frac)
    pool = p <= t
    bs = yv[pool].mean()
    auc_in = roc_auc_score(yv[pool], p[pool]) if 0 < yv[pool].mean() < 1 else float("nan")
    print(f"pool=bottom {frac:.0%}  size={pool.sum()}  buyer_share={bs:.4f}  AUC_in_pool={auc_in:.4f}")

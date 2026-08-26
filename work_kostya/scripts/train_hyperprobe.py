"""Full-scale hyperparameter probe (seed 1, val grid): lr 0.035, more rounds, deeper.
Their rule: smokes rank inversely — probe only at full scale, seed-matched vs baseline."""
import numpy as np, lightgbm as lgb, gc
from features import build_features

cube = np.load("/root/work/cube_val.npy", mmap_mode="r")
buy_mat = np.load("/root/work/buy_mat.npy")
gmv_mat = np.load("/root/work/gmv_mat.npy")
anchor_days = np.load("/root/work/anchor_days.npy")
day_to_col = {int(d): i for i, d in enumerate(anchor_days)}
TRAIN_DAYS = [281, 288, 295, 302, 309, 316, 323, 330, 337, 344]
VAL_DAY = 379

Xs, ys, zs, ws = [], [], [], []
for d in TRAIN_DAYS:
    X, names = build_features(d, cube, 379)
    c = day_to_col[d]
    Xs.append(X); ys.append(buy_mat[:, c].copy())
    zs.append(np.log1p(gmv_mat[:, c]).astype(np.float32))
    ws.append(np.full(X.shape[0], 0.5 ** (((VAL_DAY - d) / 7.0) / 26.0), dtype=np.float32))
    print("built", d, flush=True)
Xtr = np.concatenate(Xs); del Xs; gc.collect()
ytr = np.concatenate(ys); ztr = np.concatenate(zs); wtr = np.concatenate(ws)
Xv = np.load("/root/work/Xval3_379.npy")

prm = dict(learning_rate=0.035, num_leaves=200, min_data_in_leaf=300,
           feature_fraction=0.75, bagging_fraction=0.8, bagging_freq=1,
           max_bin=127, verbose=-1, num_threads=2, seed=1)
reg = lgb.train(dict(objective="l2", **prm), lgb.Dataset(Xtr, ztr, weight=wtr, feature_name=names), num_boost_round=1300)
np.save("/root/work/hp_val_pz_s1.npy", reg.predict(Xv).astype(np.float32))
reg.save_model("/root/work/hp_reg_s1.txt"); del reg; gc.collect()
print("hp reg done", flush=True)
clf = lgb.train(dict(objective="binary", **prm), lgb.Dataset(Xtr, ytr, weight=wtr, feature_name=names), num_boost_round=900)
np.save("/root/work/hp_val_p_s1.npy", clf.predict(Xv).astype(np.float32))
clf.save_model("/root/work/hp_clf_s1.txt"); del clf; gc.collect()
print("hp clf done", flush=True)
mb = ytr
size = lgb.train(dict(objective="l2", **prm), lgb.Dataset(Xtr[mb], ztr[mb], weight=wtr[mb], feature_name=names), num_boost_round=1050)
np.save("/root/work/hp_val_s_s1.npy", size.predict(Xv).astype(np.float32))
size.save_model("/root/work/hp_size_s1.txt"); del size; gc.collect()
print("hp size done", flush=True)

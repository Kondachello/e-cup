"""Seeds 3,4 for m2 heads (val grid) — same recipe as train_model2.py."""
import numpy as np, lightgbm as lgb, json, gc
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

base = dict(learning_rate=0.05, num_leaves=160, min_data_in_leaf=400,
            feature_fraction=0.75, bagging_fraction=0.8, bagging_freq=1,
            max_bin=127, verbose=-1, num_threads=2)
for seed in [3, 4]:
    prm = dict(base, seed=seed)
    reg = lgb.train(dict(objective="l2", **prm), lgb.Dataset(Xtr, ztr, weight=wtr, feature_name=names), num_boost_round=800)
    np.save(f"/root/work/m2_val_pz_s{seed}.npy", reg.predict(Xv).astype(np.float32))
    reg.save_model(f"/root/work/m2_reg_s{seed}.txt"); del reg; gc.collect()
    print("seed", seed, "reg done", flush=True)
    clf = lgb.train(dict(objective="binary", **prm), lgb.Dataset(Xtr, ytr, weight=wtr, feature_name=names), num_boost_round=550)
    np.save(f"/root/work/m2_val_p_s{seed}.npy", clf.predict(Xv).astype(np.float32))
    clf.save_model(f"/root/work/m2_clf_s{seed}.txt"); del clf; gc.collect()
    print("seed", seed, "clf done", flush=True)
    mb = ytr
    size = lgb.train(dict(objective="l2", **prm), lgb.Dataset(Xtr[mb], ztr[mb], weight=wtr[mb], feature_name=names), num_boost_round=650)
    np.save(f"/root/work/m2_val_s_s{seed}.npy", size.predict(Xv).astype(np.float32))
    size.save_model(f"/root/work/m2_size_s{seed}.txt"); del size; gc.collect()
    print("seed", seed, "size done", flush=True)

# tweedie on LOG1P target (not raw!) — third loss family, 2 seeds
for seed in [1, 2]:
    prm = dict(base, seed=seed)
    tw = lgb.train(dict(objective="tweedie", tweedie_variance_power=1.3, **prm),
                   lgb.Dataset(Xtr, ztr, weight=wtr, feature_name=names), num_boost_round=800)
    np.save(f"/root/work/m2_val_twlog_s{seed}.npy", tw.predict(Xv).astype(np.float32))
    tw.save_model(f"/root/work/m2_twlog_s{seed}.txt"); del tw; gc.collect()
    print("twlog seed", seed, "done", flush=True)
print("all done", flush=True)

"""Test-grid: seeds 3-4 heads + twlog x2 (mirror of val v2 composition)."""
import numpy as np, lightgbm as lgb, json, gc, os
from features import build_features

cube = np.load("/root/work/cube_test.npy", mmap_mode="r")
G = np.load("/root/work/gmv_mat_testgrid.npy")
TRAIN = json.load(open("/root/work/testgrid_days.json"))
TEST_DAY = 409

Xs, ys, zs, ws = [], [], [], []
for j, d in enumerate(TRAIN):
    X, names = build_features(d, cube, TEST_DAY)
    Xs.append(X); zs.append(np.log1p(G[:, j]).astype(np.float32)); ys.append(G[:, j] > 0)
    ws.append(np.full(X.shape[0], 0.5 ** (((TEST_DAY - d) / 7.0) / 26.0), dtype=np.float32))
    print("built", d, flush=True)
Xtr = np.concatenate(Xs); del Xs; gc.collect()
ytr = np.concatenate(ys); ztr = np.concatenate(zs); wtr = np.concatenate(ws)
Xte = np.load("/root/work/Xtest_409.npy")

base = dict(learning_rate=0.05, num_leaves=160, min_data_in_leaf=400,
            feature_fraction=0.75, bagging_fraction=0.8, bagging_freq=1,
            max_bin=127, verbose=-1, num_threads=2)
for seed in [3, 4]:
    prm = dict(base, seed=seed)
    if not os.path.exists(f"/root/work/mt_test_pz_s{seed}.npy"):
        reg = lgb.train(dict(objective="l2", **prm), lgb.Dataset(Xtr, ztr, weight=wtr, feature_name=names), num_boost_round=800)
        np.save(f"/root/work/mt_test_pz_s{seed}.npy", reg.predict(Xte).astype(np.float32)); del reg; gc.collect()
    print("t-seed", seed, "reg done", flush=True)
    if not os.path.exists(f"/root/work/mt_test_p_s{seed}.npy"):
        clf = lgb.train(dict(objective="binary", **prm), lgb.Dataset(Xtr, ytr, weight=wtr, feature_name=names), num_boost_round=550)
        np.save(f"/root/work/mt_test_p_s{seed}.npy", clf.predict(Xte).astype(np.float32)); del clf; gc.collect()
    print("t-seed", seed, "clf done", flush=True)
    if not os.path.exists(f"/root/work/mt_test_s_s{seed}.npy"):
        mb = ytr
        size = lgb.train(dict(objective="l2", **prm), lgb.Dataset(Xtr[mb], ztr[mb], weight=wtr[mb], feature_name=names), num_boost_round=650)
        np.save(f"/root/work/mt_test_s_s{seed}.npy", size.predict(Xte).astype(np.float32)); del size; gc.collect()
    print("t-seed", seed, "size done", flush=True)
for seed in [1, 2]:
    prm = dict(base, seed=seed)
    if not os.path.exists(f"/root/work/mt_test_twlog_s{seed}.npy"):
        tw = lgb.train(dict(objective="tweedie", tweedie_variance_power=1.3, **prm),
                       lgb.Dataset(Xtr, ztr, weight=wtr, feature_name=names), num_boost_round=800)
        np.save(f"/root/work/mt_test_twlog_s{seed}.npy", tw.predict(Xte).astype(np.float32)); del tw; gc.collect()
    print("t-twlog", seed, "done", flush=True)

# m1-slot on test: existing val-trained m1 models applied at test anchor (121-feature subset)
m1r = lgb.Booster(model_file="/root/work/reg_z.txt")
m1c = lgb.Booster(model_file="/root/work/clf2.txt")
m1s = lgb.Booster(model_file="/root/work/size_z.txt")
X121 = Xte[:, :121]
np.save("/root/work/mt_test_m1pz.npy", m1r.predict(X121).astype(np.float32))
np.save("/root/work/mt_test_m1two.npy", (m1c.predict(X121) * m1s.predict(X121)).astype(np.float32))
print("m1-slot test preds done", flush=True)
print("all done", flush=True)

"""m1-config on the test grid: 7 slices, first 121 features, no weights, seed 1.

LOCAL ADAPTATION (delta vs ../scripts/train_test_m1.py):
1) /root/work -> <repo>/work_kostya/work;
2) seed parameterized via env KSEED (unset -> original 1).
No other changes.
"""
import numpy as np, lightgbm as lgb, json, gc, os
from pathlib import Path
from features import build_features

_W = str(Path(__file__).resolve().parents[1] / "work")  # was /root/work
_KSEED = os.environ.get("KSEED", "").strip()
_SEED1 = int(_KSEED) if _KSEED else 1  # original: 1

cube = np.load(f"{_W}/cube_test.npy", mmap_mode="r")
G = np.load(f"{_W}/gmv_mat_testgrid.npy")
ALL = json.load(open(f"{_W}/testgrid_days.json"))
TRAIN = ALL[-7:]  # 332..374
TEST_DAY = 409

Xs, ys, zs = [], [], []
for d in TRAIN:
    X, names = build_features(d, cube, TEST_DAY)
    j = ALL.index(d)
    Xs.append(X[:, :121])
    zs.append(np.log1p(G[:, j]).astype(np.float32)); ys.append(G[:, j] > 0)
    print("built", d, flush=True)
names = names[:121]
Xtr = np.concatenate(Xs); del Xs; gc.collect()
ytr = np.concatenate(ys); ztr = np.concatenate(zs)
Xte = np.load(f"{_W}/Xtest_409.npy")[:, :121]

prm = dict(learning_rate=0.05, num_leaves=127, min_data_in_leaf=300,
           feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
           verbose=-1, num_threads=2, seed=_SEED1)
print("seed:", _SEED1, flush=True)
reg = lgb.train(dict(objective="l2", **prm), lgb.Dataset(Xtr, ztr, feature_name=names), num_boost_round=700)
pz = reg.predict(Xte); reg.save_model(f"{_W}/mt1_reg.txt"); del reg; gc.collect()
print("reg done", flush=True)
clf = lgb.train(dict(objective="binary", **prm), lgb.Dataset(Xtr, ytr, feature_name=names), num_boost_round=500)
p = clf.predict(Xte); clf.save_model(f"{_W}/mt1_clf.txt"); del clf; gc.collect()
print("clf done", flush=True)
mb = ytr
size = lgb.train(dict(objective="l2", **prm), lgb.Dataset(Xtr[mb], ztr[mb], feature_name=names), num_boost_round=650)
s = size.predict(Xte); size.save_model(f"{_W}/mt1_size.txt"); del size; gc.collect()
print("size done", flush=True)
np.save(f"{_W}/mt1_test_pz.npy", pz.astype(np.float32))
np.save(f"{_W}/mt1_test_two.npy", (p * s).astype(np.float32))
print("done", flush=True)

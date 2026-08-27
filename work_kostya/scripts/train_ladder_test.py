"""Лестница порогов на тест-сетке (зеркало train_ladder.py)."""
import numpy as np, lightgbm as lgb, gc, os, json
from features import build_features
cube = np.load("/root/work/cube_test.npy", mmap_mode="r")
G = np.load("/root/work/gmv_mat_testgrid.npy")
TG = json.load(open("/root/work/testgrid_days.json"))
TEST_DAY = 409
N = 250000
DAYS = TG[-8:]  # 8 последних срезов тест-сетки (325..374), зеркально 8 вал-срезам
tg2c = {d: i for i, d in enumerate(TG)}
Xtr = np.empty((N * len(DAYS), 125), dtype=np.float32)
ztr = np.empty(N * len(DAYS), dtype=np.float32)
wtr = np.empty(N * len(DAYS), dtype=np.float32)
for j, d in enumerate(DAYS):
    X, names = build_features(d, cube, TEST_DAY)
    sl = slice(j * N, (j + 1) * N)
    Xtr[sl] = X; del X; gc.collect()
    ztr[sl] = np.log1p(G[:, tg2c[d]])
    wtr[sl] = 0.5 ** (((TEST_DAY - d) / 7.0) / 26.0)
    print("built", d, flush=True)
Xte = np.load("/root/work/Xtest_409.npy")
THRESH = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.5]
prm = dict(objective="binary", learning_rate=0.05, num_leaves=160, min_data_in_leaf=400,
           feature_fraction=0.75, bagging_fraction=0.8, bagging_freq=1,
           max_bin=127, verbose=-1, num_threads=2, seed=1)
for t in THRESH:
    out = f"/root/work/lad_test_t{t}.npy"
    if os.path.exists(out):
        continue
    mdl = lgb.train(prm, lgb.Dataset(Xtr, (ztr > t), weight=wtr), num_boost_round=350)
    np.save(out, mdl.predict(Xte).astype(np.float32))
    mdl.save_model(f"/root/work/lad_test_t{t}.txt"); del mdl; gc.collect()
    print("ladder-test t", t, "done", flush=True)
print("all done", flush=True)

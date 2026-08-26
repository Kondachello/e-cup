"""Лестница порогов: P(log1p(gmv) > t) для t=2..8.5 -> E[z] квадратурой. Мои 125 признаков, 8 срезов."""
import numpy as np, lightgbm as lgb, gc, os
from features import build_features
cube = np.load("/root/work/cube_val.npy", mmap_mode="r")
gmv_mat = np.load("/root/work/gmv_mat.npy")
anchor_days = np.load("/root/work/anchor_days.npy")
d2c = {int(d): i for i, d in enumerate(anchor_days)}
TRAIN_DAYS = [295, 302, 309, 316, 323, 330, 337, 344]
VAL_DAY = 379
N = 250000
Xtr = np.empty((N * len(TRAIN_DAYS), 125), dtype=np.float32)
ztr = np.empty(N * len(TRAIN_DAYS), dtype=np.float32)
wtr = np.empty(N * len(TRAIN_DAYS), dtype=np.float32)
for j, d in enumerate(TRAIN_DAYS):
    X, names = build_features(d, cube, 379)
    sl = slice(j * N, (j + 1) * N)
    Xtr[sl] = X; del X; gc.collect()
    ztr[sl] = np.log1p(gmv_mat[:, d2c[d]])
    wtr[sl] = 0.5 ** (((VAL_DAY - d) / 7.0) / 26.0)
    print("built", d, flush=True)
Xv = np.load("/root/work/Xval3_379.npy")
THRESH = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.5]
prm = dict(objective="binary", learning_rate=0.05, num_leaves=160, min_data_in_leaf=400,
           feature_fraction=0.75, bagging_fraction=0.8, bagging_freq=1,
           max_bin=127, verbose=-1, num_threads=2, seed=1)
for t in THRESH:
    out = f"/root/work/lad_val_t{t}.npy"
    if os.path.exists(out):
        continue
    mdl = lgb.train(prm, lgb.Dataset(Xtr, (ztr > t), weight=wtr), num_boost_round=350)
    np.save(out, mdl.predict(Xv).astype(np.float32))
    mdl.save_model(f"/root/work/lad_t{t}.txt"); del mdl; gc.collect()
    print("ladder t", t, "done", flush=True)
print("all done", flush=True)

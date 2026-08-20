"""Two-part + direct GMV model on the val grid.
Train anchors <= 344 (gap>=35 to val 379). Heads: clf P(buy), reg E[log1p gmv], size E[log1p gmv | buy].
Variants: direct reg | p*size | mix. Report raw and 24-bin val-calibrated RMSLE.
"""
import numpy as np, lightgbm as lgb, json, gc
from sklearn.metrics import roc_auc_score
from features import build_features

cube = np.load("/root/work/cube_val.npy", mmap_mode="r")
buy_mat = np.load("/root/work/buy_mat.npy")
gmv_mat = np.load("/root/work/gmv_mat.npy")
anchor_days = np.load("/root/work/anchor_days.npy")
day_to_col = {int(d): i for i, d in enumerate(anchor_days)}

TRAIN_DAYS = [302, 309, 316, 323, 330, 337, 344]
VAL_DAY = 379

Xs, ys, zs = [], [], []
for d in TRAIN_DAYS:
    X, names = build_features(d, cube, 379)
    c = day_to_col[d]
    Xs.append(X); ys.append(buy_mat[:, c].copy()); zs.append(np.log1p(gmv_mat[:, c]).astype(np.float32))
    print("built", d, X.shape, flush=True)
Xtr = np.concatenate(Xs); del Xs; gc.collect()
ytr = np.concatenate(ys); ztr = np.concatenate(zs)
Xv, names = build_features(VAL_DAY, cube, 379)
yv = buy_mat[:, day_to_col[VAL_DAY]]
zv = np.log1p(gmv_mat[:, day_to_col[VAL_DAY]]).astype(np.float32)
gv = gmv_mat[:, day_to_col[VAL_DAY]]
np.save("/root/work/Xval2_379.npy", Xv)
json.dump(names, open("/root/work/feat_names2.json", "w"))
print("train matrix", Xtr.shape, flush=True)

common = dict(learning_rate=0.05, num_leaves=127, min_data_in_leaf=300,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
              verbose=-1, num_threads=2, seed=1)

def rmsle_log(pred_z, z):  # pred already in log1p space
    return float(np.sqrt(np.mean((np.maximum(pred_z, 0) - z) ** 2)))

# 1) direct regression on z
reg = lgb.train(dict(objective="l2", **common), lgb.Dataset(Xtr, ztr, feature_name=names), num_boost_round=700)
reg.save_model("/root/work/reg_z.txt")
pz = reg.predict(Xv).astype(np.float32)
print("direct reg  raw val RMSLE:", round(rmsle_log(pz, zv), 5), flush=True)

# 2) classifier
clf = lgb.train(dict(objective="binary", **common), lgb.Dataset(Xtr, ytr, feature_name=names), num_boost_round=500)
clf.save_model("/root/work/clf2.txt")
p = clf.predict(Xv).astype(np.float32)
print("clf AUC:", round(roc_auc_score(yv, p), 5), flush=True)

# 3) size head on buyers only
mb = ytr
size = lgb.train(dict(objective="l2", **common), lgb.Dataset(Xtr[mb], ztr[mb], feature_name=names), num_boost_round=500)
size.save_model("/root/work/size_z.txt")
s = size.predict(Xv).astype(np.float32)
two = (p * s).astype(np.float32)
print("two-part raw val RMSLE:", round(rmsle_log(two, zv), 5), flush=True)

mix = 0.5 * pz + 0.5 * two
print("mix raw val RMSLE:", round(rmsle_log(mix, zv), 5), flush=True)

# 24-bin calibration fit on val (team convention: compare AFTER calibration)
def calibrate(pred_z, z, bins=24):
    order = np.argsort(pred_z, kind="stable")
    edges = [order[int(len(order) * i / bins)] for i in range(bins)] + [None]
    out = np.empty_like(pred_z)
    cuts, vals = [], []
    for i in range(bins):
        lo = int(len(order) * i / bins); hi = int(len(order) * (i + 1) / bins)
        idx = order[lo:hi]
        vals.append(z[idx].mean()); cuts.append(pred_z[idx].max())
        out[idx] = z[idx].mean()
    # enforce monotone bin values
    vals = np.maximum.accumulate(np.array(vals))
    for i in range(bins):
        lo = int(len(order) * i / bins); hi = int(len(order) * (i + 1) / bins)
        out[order[lo:hi]] = vals[i]
    return out, (np.array(cuts), vals)

for nm, pr in [("direct", pz), ("two-part", two), ("mix", mix)]:
    cal, _ = calibrate(pr, zv)
    print(f"{nm}: calibrated val RMSLE = {round(rmsle_log(cal, zv), 5)}", flush=True)
np.save("/root/work/val_pz.npy", pz); np.save("/root/work/val_two.npy", two)
np.save("/root/work/val_p.npy", p); np.save("/root/work/val_s.npy", s)

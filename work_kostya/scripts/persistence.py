"""Score disjoint-window anchors with the fixed classifier; build P (scores) and B (outcomes)
matrices for persistence analysis of the hard pool."""
import numpy as np, lightgbm as lgb, json
from features import build_features

cube = np.load("/root/work/cube_val.npy", mmap_mode="r")
buy_mat = np.load("/root/work/buy_mat.npy")
gmv_mat = np.load("/root/work/gmv_mat.npy")
anchor_days = np.load("/root/work/anchor_days.npy")
day_to_col = {int(d): i for i, d in enumerate(anchor_days)}
model = lgb.Booster(model_file="/root/work/clf_buy.txt")

DISJOINT = [36, 71, 106, 141, 176, 211, 246, 281, 316, 379]  # windows [T,T+30) pairwise disjoint
P = np.zeros((250000, len(DISJOINT)), dtype=np.float32)
B = np.zeros((250000, len(DISJOINT)), dtype=bool)
G = np.zeros((250000, len(DISJOINT)), dtype=np.float32)
for j, d in enumerate(DISJOINT):
    if d == 379:
        X = np.load("/root/work/Xval_anchor379.npy")
        P[:, j] = np.load("/root/work/p_buy_val.npy")
    else:
        X, _ = build_features(d, cube, 379)
        P[:, j] = model.predict(X)
    B[:, j] = buy_mat[:, day_to_col[d]]
    G[:, j] = gmv_mat[:, day_to_col[d]]
    print("scored", d, "buy_share", round(B[:, j].mean(), 3), "mean_p", round(P[:, j].mean(), 3))
np.save("/root/work/P_disjoint.npy", P)
np.save("/root/work/B_disjoint.npy", B)
np.save("/root/work/G_disjoint.npy", G)
json.dump(DISJOINT, open("/root/work/disjoint_days.json", "w"))
print("saved")

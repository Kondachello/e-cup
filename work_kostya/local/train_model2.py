"""Stronger val-grid model: 10 slices, recency weights, extra features, 2-seed average.
Heads: direct reg on z, clf, size. Output: val preds saved for calibration/analysis.

LOCAL ADAPTATION (delta vs ../scripts/train_model2.py):
1) /root/work -> <repo>/work_kostya/work;
2) SEEDS parameterized via env KSEED: unset -> original [1, 2];
   KSEED=N -> [1 + 1000*N, 2 + 1000*N] (deterministic shift, keeps 2-seed averaging).
No other changes.
"""
import numpy as np, lightgbm as lgb, json, gc, os
from pathlib import Path
from sklearn.metrics import roc_auc_score
from features import build_features

_W = str(Path(__file__).resolve().parents[1] / "work")  # was /root/work
_KSEED = os.environ.get("KSEED", "").strip()

cube = np.load(f"{_W}/cube_val.npy", mmap_mode="r")
buy_mat = np.load(f"{_W}/buy_mat.npy")
gmv_mat = np.load(f"{_W}/gmv_mat.npy")
anchor_days = np.load(f"{_W}/anchor_days.npy")
day_to_col = {int(d): i for i, d in enumerate(anchor_days)}

TRAIN_DAYS = [281, 288, 295, 302, 309, 316, 323, 330, 337, 344]
VAL_DAY = 379

Xs, ys, zs, ws = [], [], [], []
for d in TRAIN_DAYS:
    X, names = build_features(d, cube, 379)
    c = day_to_col[d]
    Xs.append(X); ys.append(buy_mat[:, c].copy())
    zs.append(np.log1p(gmv_mat[:, c]).astype(np.float32))
    wk_back = (VAL_DAY - d) / 7.0
    ws.append(np.full(X.shape[0], 0.5 ** (wk_back / 26.0), dtype=np.float32))
    print("built", d, X.shape, flush=True)
Xtr = np.concatenate(Xs); del Xs; gc.collect()
ytr = np.concatenate(ys); ztr = np.concatenate(zs); wtr = np.concatenate(ws)
Xv, names = build_features(VAL_DAY, cube, 379)
np.save(f"{_W}/Xval3_379.npy", Xv)
json.dump(names, open(f"{_W}/feat_names3.json", "w"))
yv = buy_mat[:, day_to_col[VAL_DAY]]
zv = np.log1p(gmv_mat[:, day_to_col[VAL_DAY]]).astype(np.float32)
print("train", Xtr.shape, flush=True)

def rmsle_log(pz):
    return float(np.sqrt(np.mean((np.maximum(pz, 0) - zv) ** 2)))

base = dict(learning_rate=0.05, num_leaves=160, min_data_in_leaf=400,
            feature_fraction=0.75, bagging_fraction=0.8, bagging_freq=1,
            max_bin=127, verbose=-1, num_threads=2)

pz_acc = np.zeros(250000); p_acc = np.zeros(250000); s_acc = np.zeros(250000)
SEEDS = [1, 2] if not _KSEED else [1 + 1000 * int(_KSEED), 2 + 1000 * int(_KSEED)]
print("SEEDS:", SEEDS, flush=True)
for seed in SEEDS:
    prm = dict(base, seed=seed)
    reg = lgb.train(dict(objective="l2", **prm), lgb.Dataset(Xtr, ztr, weight=wtr, feature_name=names), num_boost_round=800)
    reg.save_model(f"{_W}/m2_reg_s{seed}.txt")
    pz_acc += reg.predict(Xv); del reg; gc.collect()
    print("seed", seed, "reg done", flush=True)
    clf = lgb.train(dict(objective="binary", **prm), lgb.Dataset(Xtr, ytr, weight=wtr, feature_name=names), num_boost_round=550)
    clf.save_model(f"{_W}/m2_clf_s{seed}.txt")
    p_acc += clf.predict(Xv); del clf; gc.collect()
    print("seed", seed, "clf done", flush=True)
    mb = ytr
    size = lgb.train(dict(objective="l2", **prm), lgb.Dataset(Xtr[mb], ztr[mb], weight=wtr[mb], feature_name=names), num_boost_round=650)
    size.save_model(f"{_W}/m2_size_s{seed}.txt")
    s_acc += size.predict(Xv); del size; gc.collect()
    print("seed", seed, "size done", flush=True)

pz = (pz_acc / len(SEEDS)).astype(np.float32)
p = (p_acc / len(SEEDS)).astype(np.float32)
s = (s_acc / len(SEEDS)).astype(np.float32)
two = (p * s).astype(np.float32)
np.save(f"{_W}/m2_val_pz.npy", pz); np.save(f"{_W}/m2_val_two.npy", two)
np.save(f"{_W}/m2_val_p.npy", p); np.save(f"{_W}/m2_val_s.npy", s)
print("clf AUC:", round(roc_auc_score(yv, p), 5), flush=True)
print("raw: direct", round(rmsle_log(pz), 5), "two", round(rmsle_log(two), 5), flush=True)

from sklearn.isotonic import IsotonicRegression
for a in [0.0, 0.4, 0.5, 0.6, 1.0]:
    mix = a * pz + (1 - a) * two
    iso = IsotonicRegression(y_min=0, out_of_bounds="clip").fit(mix, zv)
    ins = rmsle_log(iso.predict(mix).astype(np.float32))
    rng = np.random.default_rng(0); fold = rng.random(len(zv)) < 0.5
    out = np.empty_like(zv)
    for f in [fold, ~fold]:
        i2 = IsotonicRegression(y_min=0, out_of_bounds="clip").fit(mix[~f], zv[~f])
        out[f] = i2.predict(mix[f])
    print(f"mix a={a}: iso in-sample {ins:.5f}  cross-fit {rmsle_log(out):.5f}", flush=True)

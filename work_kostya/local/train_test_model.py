"""Final TEST-grid model: same recipe as train_model2 but on cube_test, anchors 311..374.
Also: P(appear) head on natural anchors 248..283. Outputs test predictions (raw two-part+direct mix,
plus appearance-shaded variant) and val-grid calibration transfer.

LOCAL ADAPTATION (delta vs ../scripts/train_test_model.py):
1) /root/work -> <repo>/work_kostya/work;
2) SEEDS parameterized via env KSEED: unset -> original [1, 2];
   KSEED=N -> [1 + 1000*N, 2 + 1000*N];
3) the auxiliary P(appear) head (NOT part of kostya46; feeds kostya46shade only) can be
   skipped with env KOSTYA_SKIP_APP=1 to save time in seed runs. Default = run (original
   behavior). Its seed stays 1 as in the original — it is not a config-2 head.
No other changes.
"""
import numpy as np, lightgbm as lgb, json, gc, os
from pathlib import Path
from features import build_features

_W = str(Path(__file__).resolve().parents[1] / "work")  # was /root/work
_KSEED = os.environ.get("KSEED", "").strip()

cube = np.load(f"{_W}/cube_test.npy", mmap_mode="r")
G = np.load(f"{_W}/gmv_mat_testgrid.npy")
TRAIN = json.load(open(f"{_W}/testgrid_days.json"))
TEST_DAY = 409

Xs, ys, zs, ws = [], [], [], []
for j, d in enumerate(TRAIN):
    X, names = build_features(d, cube, TEST_DAY)
    Xs.append(X)
    z = np.log1p(G[:, j]).astype(np.float32)
    zs.append(z); ys.append(G[:, j] > 0)
    ws.append(np.full(X.shape[0], 0.5 ** (((TEST_DAY - d) / 7.0) / 26.0), dtype=np.float32))
    print("built", d, flush=True)
Xtr = np.concatenate(Xs); del Xs; gc.collect()
ytr = np.concatenate(ys); ztr = np.concatenate(zs); wtr = np.concatenate(ws)
Xte, names = build_features(TEST_DAY, cube, TEST_DAY)
np.save(f"{_W}/Xtest_409.npy", Xte)
print("train", Xtr.shape, flush=True)

base = dict(learning_rate=0.05, num_leaves=160, min_data_in_leaf=400,
            feature_fraction=0.75, bagging_fraction=0.8, bagging_freq=1,
            max_bin=127, verbose=-1, num_threads=2)
pz_acc = np.zeros(250000); p_acc = np.zeros(250000); s_acc = np.zeros(250000)
SEEDS = [1, 2] if not _KSEED else [1 + 1000 * int(_KSEED), 2 + 1000 * int(_KSEED)]
print("SEEDS:", SEEDS, flush=True)
for seed in SEEDS:
    prm = dict(base, seed=seed)
    reg = lgb.train(dict(objective="l2", **prm), lgb.Dataset(Xtr, ztr, weight=wtr, feature_name=names), num_boost_round=800)
    reg.save_model(f"{_W}/mt_reg_s{seed}.txt"); pz_acc += reg.predict(Xte); del reg; gc.collect()
    print("seed", seed, "reg done", flush=True)
    clf = lgb.train(dict(objective="binary", **prm), lgb.Dataset(Xtr, ytr, weight=wtr, feature_name=names), num_boost_round=550)
    clf.save_model(f"{_W}/mt_clf_s{seed}.txt"); p_acc += clf.predict(Xte); del clf; gc.collect()
    print("seed", seed, "clf done", flush=True)
    mb = ytr
    size = lgb.train(dict(objective="l2", **prm), lgb.Dataset(Xtr[mb], ztr[mb], weight=wtr[mb], feature_name=names), num_boost_round=650)
    size.save_model(f"{_W}/mt_size_s{seed}.txt"); s_acc += size.predict(Xte); del size; gc.collect()
    print("seed", seed, "size done", flush=True)
del Xtr; gc.collect()

pz = (pz_acc/2).astype(np.float32); p = (p_acc/2).astype(np.float32); s = (s_acc/2).astype(np.float32)
two = (p * s).astype(np.float32)
np.save(f"{_W}/mt_test_pz.npy", pz); np.save(f"{_W}/mt_test_two.npy", two)
np.save(f"{_W}/mt_test_p.npy", p); np.save(f"{_W}/mt_test_s.npy", s)

if os.environ.get("KOSTYA_SKIP_APP", "").strip():
    print("P(appear) head skipped (KOSTYA_SKIP_APP set)", flush=True)
else:
    # P(appear) head on natural anchors
    A = np.load(f"{_W}/app_mat_natural.npy")
    APP = json.load(open(f"{_W}/app_anchor_days.json"))
    Xs, ys = [], []
    for j, d in enumerate(APP):
        X, _ = build_features(d, cube, TEST_DAY)
        Xs.append(X); ys.append(A[:, j])
        print("app built", d, flush=True)
    Xa = np.concatenate(Xs); ya = np.concatenate(ys); del Xs; gc.collect()
    app = lgb.train(dict(objective="binary", **dict(base, seed=1)),
                    lgb.Dataset(Xa, ya, feature_name=names), num_boost_round=400)
    app.save_model(f"{_W}/mt_app.txt")
    pa = app.predict(Xte).astype(np.float32)
    np.save(f"{_W}/mt_test_papp.npy", pa)
    print("P_app on test: mean", round(float(pa.mean()), 4), "p10", round(float(np.quantile(pa, 0.1)), 4), flush=True)
print("done", flush=True)

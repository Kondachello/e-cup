"""Союз признаков: мои 125 + их базовый тир (~150). 8 срезов, m2-рецепт, 2 сида.
Вопрос: даёт ли объединение пространств шаг вниз по скору (и вклад сверх насыщения).
"""
import numpy as np, polars as pl, lightgbm as lgb, json, gc
from datetime import date, timedelta
import sys
sys.path.insert(0, "/root/work")
from features import build_features

DAY0 = date(2025, 1, 1)
cube = np.load("/root/work/cube_val.npy", mmap_mode="r")
buy_mat = np.load("/root/work/buy_mat.npy")
gmv_mat = np.load("/root/work/gmv_mat.npy")
anchor_days = np.load("/root/work/anchor_days.npy")
d2c = {int(d): i for i, d in enumerate(anchor_days)}
TRAIN_DAYS = [295, 302, 309, 316, 323, 330, 337, 344]
VAL_DAY = 379

def team_feats(day):
    a = (DAY0 + timedelta(days=day - 1)).isoformat()
    df = pl.read_parquet(f"/root/work/features/anchor={a}.parquet").sort("user_id")
    cols = [c for c in df.columns if c not in ("user_id", "anchor_date", "target")]
    X = df.select(cols).with_columns(pl.all().cast(pl.Float32, strict=False)).fill_null(np.nan).to_numpy()
    return X, cols

N = 250000
# probe feature dims once
Xm, mnames = build_features(TRAIN_DAYS[0], cube, 379)
Xt, tnames = team_feats(TRAIN_DAYS[0])
F = Xm.shape[1] + Xt.shape[1]
names = mnames + [f"tm_{n}" for n in tnames]
Xtr = np.empty((N * len(TRAIN_DAYS), F), dtype=np.float32)
ytr = np.empty(N * len(TRAIN_DAYS), dtype=bool)
ztr = np.empty(N * len(TRAIN_DAYS), dtype=np.float32)
wtr = np.empty(N * len(TRAIN_DAYS), dtype=np.float32)
for j, d in enumerate(TRAIN_DAYS):
    if j > 0:
        Xm, _ = build_features(d, cube, 379)
        Xt, _ = team_feats(d)
    sl = slice(j * N, (j + 1) * N)
    Xtr[sl, :Xm.shape[1]] = Xm
    Xtr[sl, Xm.shape[1]:] = Xt
    del Xm, Xt; gc.collect()
    c = d2c[d]
    ytr[sl] = buy_mat[:, c]; ztr[sl] = np.log1p(gmv_mat[:, c])
    wtr[sl] = 0.5 ** (((VAL_DAY - d) / 7.0) / 26.0)
    print("built", d, flush=True)
Xm, _ = build_features(VAL_DAY, cube, 379)
Xt, _ = team_feats(VAL_DAY)
Xv = np.column_stack([Xm, Xt]).astype(np.float32); del Xm, Xt; gc.collect()
np.save("/root/work/Xval_union.npy", Xv)
json.dump(names, open("/root/work/union_names.json", "w"))
print("train", Xtr.shape, flush=True)

base = dict(learning_rate=0.05, num_leaves=160, min_data_in_leaf=400,
            feature_fraction=0.75, bagging_fraction=0.8, bagging_freq=1,
            max_bin=127, verbose=-1, num_threads=2)
for seed in [1, 2]:
    prm = dict(base, seed=seed)
    reg = lgb.train(dict(objective="l2", **prm), lgb.Dataset(Xtr, ztr, weight=wtr, feature_name=names), num_boost_round=800)
    np.save(f"/root/work/un_val_pz_s{seed}.npy", reg.predict(Xv).astype(np.float32))
    reg.save_model(f"/root/work/un_reg_s{seed}.txt"); del reg; gc.collect()
    print("seed", seed, "reg done", flush=True)
    clf = lgb.train(dict(objective="binary", **prm), lgb.Dataset(Xtr, ytr, weight=wtr, feature_name=names), num_boost_round=550)
    np.save(f"/root/work/un_val_p_s{seed}.npy", clf.predict(Xv).astype(np.float32))
    clf.save_model(f"/root/work/un_clf_s{seed}.txt"); del clf; gc.collect()
    print("seed", seed, "clf done", flush=True)
    mb = ytr
    size = lgb.train(dict(objective="l2", **prm), lgb.Dataset(Xtr[mb], ztr[mb], weight=wtr[mb], feature_name=names), num_boost_round=650)
    np.save(f"/root/work/un_val_s_s{seed}.npy", size.predict(Xv).astype(np.float32))
    size.save_model(f"/root/work/un_size_s{seed}.txt"); del size; gc.collect()
    print("seed", seed, "size done", flush=True)
print("all done", flush=True)

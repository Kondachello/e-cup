"""Задача 3: пробник остатка бленда на МОЁМ признаковом пространстве.
Протокол №1: честный OOF по юзерам против остатка бленда + замер концентрации + плацебо.
Reference blend: interim = NNLS(pack30 + kostya46_v2_cal) full-fit (заменить на свежую
колонку blend после git pull).
"""
import numpy as np, polars as pl, lightgbm as lgb, json
from scipy.optimize import nnls

d = pl.read_parquet("/mnt/user-data/uploads/ozon_cup/work/preds_pack/val_preds.parquet").sort("user_id")
y = np.log1p(d["target"].to_numpy().astype(np.float64))
cols = [c for c in d.columns if c not in ("user_id", "target", "blend")]
M = np.stack([d[c].to_numpy().astype(np.float64) for c in cols], axis=1)

def fit_shifts(lp, ly, bins=24):
    qs = np.quantile(lp, np.linspace(0, 1, bins + 1)); qs[0] -= 1e-9; qs[-1] += 1e-9
    cs, ss = [], []
    for i in range(bins):
        m = (lp > qs[i]) & (lp <= qs[i + 1])
        if m.sum() < 500: continue
        cs.append(lp[m].mean()); ss.append(ly[m].mean() - lp[m].mean())
    return np.array(cs), np.array(ss)

def cal_honest(lp, seed=0):
    r = np.random.default_rng(seed); half = r.permutation(len(y)) < len(y) // 2
    out = np.empty_like(lp)
    for m in (half, ~half):
        c, s = fit_shifts(lp[m], y[m]); out[~m] = np.clip(lp[~m] + np.interp(lp[~m], c, s), 0, None)
    return out

L = lambda f: np.load(f"/root/work/{f}").astype(np.float64)
pz4 = (L("m2_val_pz.npy")*2 + L("m2_val_pz_s3.npy") + L("m2_val_pz_s4.npy"))/4
p4  = (L("m2_val_p.npy")*2 + L("m2_val_p_s3.npy") + L("m2_val_p_s4.npy"))/4
s4  = (L("m2_val_s.npy")*2 + L("m2_val_s_s3.npy") + L("m2_val_s_s4.npy"))/4
tw2 = (L("m2_val_twlog_s1.npy") + L("m2_val_twlog_s2.npy"))/2
m1  = 0.5*L("val_pz.npy") + 0.5*L("val_two.npy")
mine_cal = cal_honest(0.55*(0.4*pz4 + 0.6*p4*s4) + 0.25*tw2 + 0.2*m1)

A = np.column_stack([M, mine_cal])
w, _ = nnls(A, y)
blend = A @ w
sb = float(np.sqrt(np.mean((blend - y) ** 2)))
print(f"interim reference blend (pack30+kostya46, full-fit NNLS): {sb:.6f}", flush=True)
resid = y - blend

X = np.load("/root/work/Xval3_379.npy").astype(np.float32)
extra = np.column_stack([p4, s4, pz4, tw2, m1]).astype(np.float32)
Xf = np.column_stack([X, extra])
print("features:", Xf.shape, flush=True)

def oof_correct(target, seed=7, rounds=500):
    rng = np.random.default_rng(seed)
    fold = rng.integers(0, 4, len(y))
    oof = np.zeros(len(y))
    prm = dict(objective="l2", learning_rate=0.03, num_leaves=63, min_data_in_leaf=500,
               feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
               max_bin=127, verbose=-1, num_threads=2, seed=seed)
    for k in range(4):
        tr = fold != k
        mdl = lgb.train(prm, lgb.Dataset(Xf[tr], target[tr]), num_boost_round=rounds)
        oof[~tr] = mdl.predict(Xf[~tr])
        print(f"  fold {k} done", flush=True)
    return oof

oof = oof_correct(resid)
np.save("/root/work/probe_oof_resid.npy", oof)
for alpha in [0.3, 0.5, 1.0]:
    corrected = blend + alpha * oof
    s = float(np.sqrt(np.mean((corrected - y) ** 2)))
    print(f"alpha={alpha}: {s:.6f}  (delta {sb - s:+.6f})", flush=True)

# concentration of gain (alpha=1)
gain_u = (blend - y) ** 2 - (blend + oof - y) ** 2
pos_total = gain_u.sum()
order = np.argsort(-gain_u)
for frac in [0.001, 0.01]:
    k = int(len(y) * frac)
    share = gain_u[order[:k]].sum() / pos_total if pos_total != 0 else float("nan")
    print(f"top-{frac:.1%} users carry {share:.1%} of net gain (своя доля {frac:.1%}, порог смерти {3*frac:.1%})", flush=True)

# placebo: permuted residuals
rng = np.random.default_rng(99)
oof_p = oof_correct(rng.permutation(resid), seed=8, rounds=500)
for alpha in [1.0]:
    s = float(np.sqrt(np.mean((blend + alpha * oof_p - y) ** 2)))
    print(f"PLACEBO alpha={alpha}: {s:.6f}  (delta {sb - s:+.6f})", flush=True)

# bootstrap SE of delta at alpha=1
eA = (blend + oof - y) ** 2; eB = (blend - y) ** 2
ds = []
for _ in range(300):
    idx = rng.integers(0, len(y), len(y))
    ds.append(np.sqrt(eB[idx].mean()) - np.sqrt(eA[idx].mean()))
print(f"delta(alpha=1) bootstrap: {np.mean(ds):.6f} ± {np.std(ds):.6f}", flush=True)

"""v3 = 0.6*v2 + 0.4*ladder (лестница порогов, квадратура E[z]). Вал+тест зеркально."""
import numpy as np, polars as pl
from scipy.optimize import nnls

L = lambda f: np.load(f"/root/work/{f}").astype(np.float64)
TH = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.5]
ts = np.array([0.0] + TH)

def quad(surv_cols, tail_mean):
    S = np.minimum.accumulate(np.column_stack(surv_cols), axis=1)
    area = ((S[:, :-1] + S[:, 1:]) * 0.5 * np.diff(ts)).sum(axis=1)
    return area + S[:, -1] * tail_mean

# tail mean excess из исторических окон (одна константа для обеих сеток)
gmv_mat = np.load("/root/work/gmv_mat.npy")
zh = np.log1p(gmv_mat[:, :-1]).ravel()
TAIL = float((zh[zh > 8.5] - 8.5).mean())

# ---- val ----
p4 = (L("m2_val_p.npy")*2 + L("m2_val_p_s3.npy") + L("m2_val_p_s4.npy"))/4
pz4 = (L("m2_val_pz.npy")*2 + L("m2_val_pz_s3.npy") + L("m2_val_pz_s4.npy"))/4
s4 = (L("m2_val_s.npy")*2 + L("m2_val_s_s3.npy") + L("m2_val_s_s4.npy"))/4
tw2 = (L("m2_val_twlog_s1.npy") + L("m2_val_twlog_s2.npy"))/2
m1 = 0.5*L("val_pz.npy") + 0.5*L("val_two.npy")
v2v = 0.55*(0.4*pz4 + 0.6*p4*s4) + 0.25*tw2 + 0.2*m1
ladv = quad([p4] + [L(f"lad_val_t{t}.npy") for t in TH], TAIL)
val_mix = 0.6*v2v + 0.4*ladv

# ---- test ----
p4t = (L("mt_test_p.npy")*2 + L("mt_test_p_s3.npy") + L("mt_test_p_s4.npy"))/4
pz4t = (L("mt_test_pz.npy")*2 + L("mt_test_pz_s3.npy") + L("mt_test_pz_s4.npy"))/4
s4t = (L("mt_test_s.npy")*2 + L("mt_test_s_s3.npy") + L("mt_test_s_s4.npy"))/4
tw2t = (L("mt_test_twlog_s1.npy") + L("mt_test_twlog_s2.npy"))/2
m1t = 0.5*L("mt_test_m1pz.npy") + 0.5*L("mt_test_m1two.npy")
v2t = 0.55*(0.4*pz4t + 0.6*p4t*s4t) + 0.25*tw2t + 0.2*m1t
ladt = quad([p4t] + [L(f"lad_test_t{t}.npy") for t in TH], TAIL)
test_mix = 0.6*v2t + 0.4*ladt
papp = L("mt_test_papp.npy")

users = pl.read_parquet("/root/work/users_order.parquet")["user_id"].cast(pl.Int64)
outs = {"kostya46_val": val_mix, "kostya46_test": test_mix,
        "kostya46shade_val": val_mix, "kostya46shade_test": test_mix*papp}
for nm, arr in outs.items():
    pl.DataFrame({"user_id": users, "pred": np.expm1(np.maximum(arr, 0))}
                 ).write_parquet(f"/root/work/v3_{nm}.parquet")

# ---- числа ----
d = pl.read_parquet("/mnt/user-data/uploads/ozon_cup/work/preds_pack/val_preds.parquet").sort("user_id")
y = np.log1p(d["target"].to_numpy().astype(np.float64))
b = d["blend"].to_numpy().astype(np.float64)
sb = np.sqrt(((y-b)**2).mean())
def fit_shifts(lp, ly, bins=24):
    qs = np.quantile(lp, np.linspace(0,1,bins+1)); qs[0]-=1e-9; qs[-1]+=1e-9
    cs, ss = [], []
    for i in range(bins):
        m = (lp>qs[i])&(lp<=qs[i+1])
        if m.sum()<500: continue
        cs.append(lp[m].mean()); ss.append(ly[m].mean()-lp[m].mean())
    return np.array(cs), np.array(ss)
def cal_honest(lp, seed=0):
    r = np.random.default_rng(seed); h = r.permutation(len(y)) < len(y)//2
    out = np.empty_like(lp)
    for m in (h,~h):
        c,s = fit_shifts(lp[m], y[m]); out[~m] = np.clip(lp[~m]+np.interp(lp[~m],c,s),0,None)
    return out
mc = cal_honest(val_mix)
sm = float(np.sqrt(np.mean((mc-y)**2)))
rho = float(np.corrcoef(y-b, y-mc)[0,1])
print(f"kostya46 v3: score {sm:.6f}  corr {rho:.5f}  ЗАПАС {sb/sm-rho:+.5f}")
cols = [c for c in d.columns if c not in ("user_id","target","blend")]
M = np.stack([d[c].to_numpy().astype(np.float64) for c in cols], axis=1)
def honest(mat, seed=0):
    r = np.random.default_rng(seed); h = r.permutation(len(y)) < len(y)//2
    out = np.empty_like(y); ws = []
    for m in (h, ~h):
        w, _ = nnls(mat[~m], y[~m]); ws.append(w)
        out[m] = mat[m] @ w
    return float(np.sqrt(np.mean((out-y)**2))), ws
base, _ = honest(M)
plus, ws = honest(np.column_stack([M, mc]))
print(f"NNLS: {base:.6f} -> {plus:.6f}  gain {base-plus:+.6f}  вес {[round(w[-1],4) for w in ws]}")
print("files: v3_kostya46_{val,test}.parquet, v3_kostya46shade_{val,test}.parquet")

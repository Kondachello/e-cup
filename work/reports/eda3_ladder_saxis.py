# eda3 activity ladder: S-axis bound — momentum composition shift val->test anchor
# x oracle per-bin residual biases => upper bound on transferable mean shift
import numpy as np
import polars as pl

uni = pl.read_parquet("work/reports/eda3_ladder_uni.parquet")
e = (np.log1p(uni["target"].to_numpy()) - uni["blend"].to_numpy().astype(np.float64))
N = len(e)

Lsv = np.load("work/reports/eda3_ladder_val_Ls.npy").astype(np.int16)
Lst = np.load("work/reports/eda3_ladder_test_Ls.npy").astype(np.int16)

def accbins(L):
    acc = L[:, 0].astype(int) - 2 * L[:, 1] + L[:, 2]
    return np.clip((acc + 12) // 3, 0, 8)

bv, bt = accbins(Lsv), accbins(Lst)
shift = 0.0
print("bin | p_val | p_test | dp | mean_e_val")
for b in range(9):
    pv, pt = (bv == b).mean(), (bt == b).mean()
    mu = e[bv == b].mean() if (bv == b).sum() else 0.0
    shift += (pt - pv) * mu
    print(f"{b} | {pv:.4f} | {pt:.4f} | {pt-pv:+.4f} | {mu:+.4f}")
print(f"S-axis transfer bound (mean log shift if oracle biases were real): {shift:+.6f}")
# effect of a global mean shift s on RMSLE: dRMSLE ~ s*mean_e/rmse (first order); bound with s itself
mse0 = np.mean(e ** 2)
print(f"=> dRMSLE bound ~ |shift*mean_e|/rmsle = {abs(shift*e.mean())/np.sqrt(mse0):.7f} (mean_e={e.mean():+.4f})")


cv = Lsv[:, 1].astype(int) * 11 + Lsv[:, 0]
ct = Lst[:, 1].astype(int) * 11 + Lst[:, 0]
shift2 = 0.0
rows = []
for c in range(121):
    pv, pt = (cv == c).mean(), (ct == c).mean()
    if (cv == c).sum() > 500:
        mu = e[cv == c].mean()
        shift2 += (pt - pv) * mu
        rows.append((c // 11, c % 11, round(pv, 4), round(pt, 4), round(pt - pv, 4), round(mu, 3)))
rows.sort(key=lambda r: -abs(r[4]))
print("top cells by |dp| (,L0,p_val,p_test,dp,mean_e):", rows[:8])
print(f"cell-composition transfer bound: {shift2:+.6f}")

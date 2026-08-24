# eda3 activity ladder: residual of blend vs momentum (2nd derivative) + transition cells
# oracle dMSE, honest OOF (2-fold x 20 by users), whale concentration, placebo
import numpy as np
import polars as pl

rng = np.random.default_rng(7)
uni = pl.read_parquet("work/reports/eda3_ladder_uni.parquet")
e = (np.log1p(uni["target"].to_numpy()) - uni["blend"].to_numpy().astype(np.float64))
N = len(e)
mse0 = np.mean(e ** 2)
print("blend rmsle:", round(np.sqrt(mse0), 6), "N", N)

Ls = np.load("work/reports/eda3_ladder_val_Ls.npy").astype(np.int16)
Lg = np.load("work/reports/eda3_ladder_val_Lg.npy").astype(np.int16)
Zs = np.load("work/reports/eda3_ladder_val_zsrch.npy")
Zg = np.load("work/reports/eda3_ladder_val_zgmv.npy")

def bin_report(name, key, min_n=300):
    # key: integer bin id per user (may be -1 = excluded)
    m = key >= 0
    ids = np.unique(key[m])
    rows = []
    dmse_or = 0.0
    for b in ids:
        s = key == b
        n = s.sum()
        if n < min_n:
            continue
        mu = e[s].mean()
        t = mu / (e[s].std(ddof=1) / np.sqrt(n))
        dmse_or += n * mu * mu / N
        rows.append((b, n, round(mu, 4), round(t, 2)))
    dr_or = np.sqrt(mse0) - np.sqrt(mse0 - dmse_or)
    print(f"[{name}] bins:", rows)
    print(f"[{name}] oracle dMSE={dmse_or:.6f} dRMSLE={dr_or:.6f}")
    return dr_or

def oof(name, key, reps=20, min_n=50, placebo=False):
    # 2-fold by users, correct fold B by bin means of fold A; per-user gain for concentration
    m = key >= 0
    gains = []
    conc1, conc01 = [], []
    for r in range(reps):
        k = key.copy()
        if placebo:
            idx = np.where(m)[0]
            k[idx] = k[rng.permutation(idx)]
        fold = rng.random(N) < 0.5
        corr = np.zeros(N)
        for fa, fb in [(fold, ~fold), (~fold, fold)]:
            src = fa & m
            dst = fb & m
            for b in np.unique(k[src]):
                s = src & (k == b)
                if s.sum() < min_n:
                    continue
                corr[dst & (k == b)] = e[s].mean()
        g_user = e ** 2 - (e - corr) ** 2  # per-user MSE gain
        dmse = g_user.mean()
        gains.append(dmse)
        # concentration: share of net gain from top users by |contribution|
        order = np.argsort(-np.abs(g_user))
        tot = g_user.sum()
        if abs(tot) > 1e-12:
            conc1.append(g_user[order[: N // 100]].sum() / tot)
            conc01.append(g_user[order[: N // 1000]].sum() / tot)
    gains = np.array(gains)
    dmse = gains.mean()
    dr = np.sqrt(mse0) - np.sqrt(mse0 - dmse) if dmse < mse0 else float("nan")
    tag = " PLACEBO" if placebo else ""
    print(f"[{name}{tag}] OOF dMSE={dmse:.6f} +-{gains.std():.6f} -> dRMSLE={dr:.6f}; "
          f"conc top1%={np.mean(conc1) if conc1 else float('nan'):.2f} top0.1%={np.mean(conc01) if conc01 else float('nan'):.2f}")
    return dr

acc = Ls[:, 0].astype(int) - 2 * Ls[:, 1] + Ls[:, 2]
bins = np.clip((acc + 12) // 3, 0, 8)  # 9 bins of width 3
bin_report("acc_Ls(binw3)", bins)
oof("acc_Ls(binw3)", bins)
oof("acc_Ls(binw3)", bins, placebo=True)

# ---- 2) first difference dL01 for contrast (trend is in features -> expect 0) ----
d1 = np.clip(Ls[:, 0].astype(int) - Ls[:, 1], -6, 6) + 6
bin_report("dL01_Ls", d1)
oof("dL01_Ls", d1)

# ---- 3) continuous acceleration on log1p searches, decile bins ----
a = Zs[:, 0] - 2 * Zs[:, 1] + Zs[:, 2]
q = np.quantile(a, np.linspace(0, 1, 11))
q[0] -= 1; q[-1] += 1
ab = np.digitize(a, q[1:-1])
bin_report("acc_zsrch(dec)", ab)
oof("acc_zsrch(dec)", ab)

# ---- 4) gmv-ladder momentum ----
accg = Lg[:, 0].astype(int) - 2 * Lg[:, 1] + Lg[:, 2]
bing = np.clip((accg + 12) // 3, 0, 8)
bin_report("acc_Lg(binw3)", bing)
oof("acc_Lg(binw3)", bing)

#121 cells ----
cell = Ls[:, 1].astype(int) * 11 + Ls[:, 0]
bin_report("cell_L1L0_Ls", cell, min_n=2000)
oof("cell_L1L0_Ls", cell, min_n=100)
oof("cell_L1L0_Ls", cell, min_n=100, placebo=True)

# ---- 6) long trajectory: level climb over 6 months L0 - L6 (beyond 90d windows) ----
d6 = np.clip(Ls[:, 0].astype(int) - Ls[:, 6], -8, 8) // 2 + 4
bin_report("dL06_Ls", d6)
oof("dL06_Ls", d6)

# ---- 7) gmv long climb ----
d6g = np.clip(Lg[:, 0].astype(int) - Lg[:, 6], -8, 8) // 2 + 4
bin_report("dL06_Lg", d6g)
oof("dL06_Lg", d6g)

# whale check: mean gmv month0 by acc bin
g0 = np.expm1(Zg[:, 0])
for b in range(9):
    s = bins == b
    if s.sum():
        print("accbin", b, "n", s.sum(), "mean_gmv_m0", round(g0[s].mean(), 1), "mean_e", round(e[s].mean(), 4))

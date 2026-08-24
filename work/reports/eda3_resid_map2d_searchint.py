import os
os.environ.setdefault("POLARS_MAX_THREADS", "2")
import numpy as np
import polars as pl
import json

SCR = "/private/tmp/claude-501/-Users-alexanderkondakov-ozon-cup/b994bee8-2354-4524-9a2b-530595cd5feb/scratchpad"
ax = pl.read_parquet(f"{SCR}/user_axes.parquet")
vp = pl.read_parquet(
    "/Users/alexanderkondakov/ozon-cup/work/preds_pack/val_preds.parquet",
    columns=["user_id", "target", "blend"],
)
df = vp.join(ax, on="user_id", how="left")
y = np.log1p(df["target"].to_numpy())
b = df["blend"].to_numpy().astype(np.float64)
e0 = y - b
N = len(e0)
base = float(np.sqrt(np.mean(e0**2)))

def col(name):
    return df[name].to_numpy().astype(np.float64)

act90 = col("act90"); sd90 = col("search_days90")
search90 = col("search90"); act28 = col("act28")
ord_days90 = col("ord_days90"); ord_days365 = col("ord_days365")
rec_ord = df["rec_ord"].to_numpy().astype(np.float64)
rec_ord = np.where(np.isnan(rec_ord), 999.0, rec_ord)
gmv365 = col("gmv365")
with np.errstate(invalid="ignore"):
    search_int = np.where(act90 > 0, sd90 / act90, np.nan)

def qbin(x, K=10):
    x = np.asarray(x, dtype=np.float64)
    nanm = np.isnan(x)
    xs = x[~nanm]
    qs = np.unique(np.quantile(xs, np.linspace(0, 1, K + 1)))
    codes = np.full(len(x), -1, dtype=np.int32)
    codes[~nanm] = np.searchsorted(qs[1:-1], xs, side="right")
    nb = len(qs) - 1
    codes[nanm] = nb
    return codes, nb + (1 if nanm.any() else 0), qs

# ---- 1. margin detail (10 bins) ----
codes, nc, qs = qbin(search_int, 10)
print("search_int bin edges:", np.round(qs, 3))
print(f"{'bin':>4} {'n':>7} {'mean_e':>8} {'se':>7} {'z':>7}  (last bin = act90==0)")
cnt = np.bincount(codes, minlength=nc).astype(float)
s1 = np.bincount(codes, weights=e0, minlength=nc)
s2 = np.bincount(codes, weights=e0**2, minlength=nc)
m = s1 / np.maximum(cnt, 1)
se = np.sqrt((s2 / np.maximum(cnt, 1) - m**2) / np.maximum(cnt, 1))
for i in range(nc):
    print(f"{i:>4} {int(cnt[i]):>7} {m[i]:+.4f} {se[i]:.4f} {m[i]/se[i]:+7.2f}")

def cv_margin(e, codes, ncodes, fold, nf=5):
    c = np.zeros(len(e))
    for f in range(nf):
        tr = fold != f; te = fold == f
        cntf = np.bincount(codes[tr], minlength=ncodes).astype(float)
        mf = np.bincount(codes[tr], weights=e[tr], minlength=ncodes) / np.maximum(cntf, 1)
        c[te] = mf[codes[te]]
    return c

# ---- 2. robustness: seeds x bins ----
print("\nOOF gain by (fold seed, K):")
gains = {}
for K in (5, 10, 20):
    codesK, ncK, _ = qbin(search_int, K)
    row = []
    for seed in range(5):
        fold = np.random.default_rng(seed).integers(0, 5, N)
        c = cv_margin(e0, codesK, ncK, fold)
        g = base - np.sqrt(((e0 - c) ** 2).mean())
        row.append(g)
    gains[K] = row
    print(f"K={K:>2}: {' '.join(f'{g:+.6f}' for g in row)}  mean {np.mean(row):+.6f} ± {np.std(row):.6f}")

# split-half: fit on half A apply to B and vice versa, 10 random splits
sh = []
for s in range(10):
    half = np.random.default_rng(100 + s).integers(0, 2, N)
    c = cv_margin(e0, codes, nc, half, nf=2)
    sh.append(base - np.sqrt(((e0 - c) ** 2).mean()))
print(f"split-half (10 сплитов): mean {np.mean(sh):+.6f} ± {np.std(sh):.6f}  min {np.min(sh):+.6f}")

# ---- 3. does it survive the linear-rank span of the closed corrector's columns? ----
def ranks(x):
    r = np.argsort(np.argsort(x, kind="stable"), kind="stable").astype(np.float64)
    return (r / (len(x) - 1)) - 0.5

# columns most related: search_days_90, active_days_90, searches_90, ord_days_90, act28, rec
Xcols = [ranks(sd90), ranks(act90), ranks(search90), ranks(ord_days90), ranks(act28),
         ranks(rec_ord), ranks(ord_days365), ranks(np.log1p(gmv365))]
X = np.stack(Xcols, axis=1)
X = np.concatenate([X, np.ones((N, 1))], axis=1)

fold = np.random.default_rng(0).integers(0, 5, N)
def cv_ridge_resid(e, X, fold, alpha=1.0):
    r = np.zeros(len(e))
    for f in range(5):
        tr = fold != f; te = fold == f
        A = X[tr].T @ X[tr] + alpha * np.eye(X.shape[1])
        w = np.linalg.solve(A, X[tr].T @ e[tr])
        r[te] = e[te] - X[te] @ w
    return r

e_lin = cv_ridge_resid(e0, X, fold, alpha=1.0)
g_lin = base - np.sqrt((e_lin**2).mean())
c_m = cv_margin(e_lin, codes, nc, fold)
g_after = np.sqrt((e_lin**2).mean()) - np.sqrt(((e_lin - c_m) ** 2).mean())
print(f"\nlinear-rank span (8 колонок закрытого корректора, cv-ridge): gain {g_lin:+.6f}")
print(f"search_int margin ПОСЛЕ снятия линейного спана: {g_after:+.6f}")

# quadratic span (ranks + pairwise products of the 4 core cols)
core = [ranks(sd90), ranks(act90), ranks(search90), ranks(ord_days90)]
XQ = [x for x in Xcols]
for i in range(4):
    for j in range(i, 4):
        XQ.append(core[i] * core[j])
XQ = np.stack(XQ, axis=1)
XQ = np.concatenate([XQ, np.ones((N, 1))], axis=1)
e_q = cv_ridge_resid(e0, XQ, fold, alpha=1.0)
g_q = base - np.sqrt((e_q**2).mean())
c_mq = cv_margin(e_q, codes, nc, fold)
g_after_q = np.sqrt((e_q**2).mean()) - np.sqrt(((e_q - c_mq) ** 2).mean())
print(f"quad-rank span: gain {g_q:+.6f}; search_int margin после: {g_after_q:+.6f}")

# ---- 4. concentration for the base margin (seed 0), and gain by bin ----
c = cv_margin(e0, codes, nc, fold)
e_new = e0 - c
s_new = np.sqrt((e_new**2).mean());
ci = (e_new**2 - e0**2) / (N * (s_new + base))
delta = ci.sum()
order = np.argsort(-np.abs(ci))
print(f"\nконцентрация margin-корректора: delta {delta:+.6f}, top0.1% {ci[order[:250]].sum()/delta:.2f}, top1% {ci[order[:2500]].sum()/delta:.2f}")
# gain contribution per bin
print("вклад бинов в OOF gain:")
for i in range(nc):
    mask = codes == i
    gi = ci[mask].sum()
    print(f"bin {i}: n={int(cnt[i]):>7} вклад {gi:+.6f} ({100*gi/delta:5.1f}%)")

# ---- 5. churned-heavy pocket split-half ----
pocket = (rec_ord > 90) & (rec_ord < 999) & (ord_days365 >= 23)
print(f"\nchurned-heavy pocket: n={pocket.sum()}, mean_e={e0[pocket].mean():+.4f}, se={e0[pocket].std()/np.sqrt(pocket.sum()):.4f}")
ph = []
for s in range(20):
    half = np.random.default_rng(200 + s).integers(0, 2, N)
    c2 = np.zeros(N)
    for h in (0, 1):
        tr = (half != h) & pocket; te = (half == h) & pocket
        if tr.sum() > 10:
            c2[te] = e0[tr].mean()
    ph.append(base - np.sqrt(((e0 - c2) ** 2).mean()))
print(f"pocket split-half gain: mean {np.mean(ph):+.6f} ± {np.std(ph):.6f} min {np.min(ph):+.6f}")

json.dump({"gains_by_K": {str(k): v for k, v in gains.items()},
           "split_half": sh, "g_lin": g_lin, "g_after_lin": g_after,
           "g_quad": g_q, "g_after_quad": g_after_q,
           "pocket_split_half": ph},
          open(f"{SCR}/searchint_deep.json", "w"), indent=1, default=float)

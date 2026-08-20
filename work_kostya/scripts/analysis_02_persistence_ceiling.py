"""№3: персистентность пула и потолок. Требует: persistence.py уже отработал
(P_disjoint/B_disjoint/G_disjoint.npy). Печатает все числа §2-§3 отчёта."""
import numpy as np, json
from sklearn.metrics import roc_auc_score
from scipy.optimize import minimize
from scipy.special import betaln, gammaln

P = np.load("P_disjoint.npy"); B = np.load("B_disjoint.npy")
NJ = P.shape[1]; val = NJ - 1
pools = np.zeros_like(B)
for j in range(NJ):
    pools[:, j] = P[:, j] <= np.quantile(P[:, j], 0.30)
pool_v = pools[:, val]; yv = B[:, val]
n_past_buys = B[:, :val].sum(1)

print("== конверсия пула по прошлым покупкам (9 непересекающихся окон) ==")
for k in range(4):
    m = pool_v & (n_past_buys == k)
    print(f"  past_buys={k}: n={m.sum()}  P(buy@val)={yv[m].mean():.4f}")

print("\n== внутри децильных бинов p: AUC прошлых покупок (0.5 = ничего сверх признаков) ==")
pv = P[pool_v, val]; yb = yv[pool_v]; nb = n_past_buys[pool_v]
bins = np.quantile(pv, np.linspace(0, 1, 11))
tot_w = tot_a = 0.0
for i in range(10):
    m = (pv >= bins[i]) & (pv < bins[i+1]) if i < 9 else (pv >= bins[i])
    if m.sum() > 500 and 0 < yb[m].mean() < 1 and len(np.unique(nb[m])) > 1:
        a = roc_auc_score(yb[m], nb[m]); tot_w += m.sum(); tot_a += a * m.sum()
print(f"  средневзвешенный AUC: {tot_a/tot_w:.4f}")

def fit_bb(kvec, n):
    cnt = np.bincount(kvec, minlength=n+1)
    def nll(params):
        a, b = np.exp(params)
        lp = np.array([gammaln(n+1)-gammaln(k+1)-gammaln(n-k+1)+betaln(a+k, b+n-k)-betaln(a, b)
                       for k in range(n+1)])
        return -(cnt * lp).sum()
    return np.exp(minimize(nll, np.log([0.5, 3.0]), method="Nelder-Mead").x)

print("\n== бета-биномиальный потолок ==")
rng = np.random.default_rng(1)
for J in [list(range(1, 9)), [5, 6, 7, 8]]:
    n = len(J); k = B[:, J].sum(1)[pool_v].astype(int)
    a, b = fit_bb(k, n)
    lam = rng.beta(a + k, b + n - k)
    ysim = rng.random(len(lam)) < lam
    print(f"  окон={n}: Beta({a:.2f},{b:.2f})  оракул-λ AUC={roc_auc_score(ysim, lam):.4f}")
print(f"  достигнутый моделью AUC в пуле: {roc_auc_score(yb, pv):.4f}")

print("\n== цена λ-оракула в RMSLE (симуляция) ==")
a, b, n = 1.384, 12.490, 4
kdist = np.bincount(B[:, [5, 6, 7, 8]].sum(1)[pool_v].astype(int), minlength=5) / pool_v.sum()
drift = yb.mean() / (a / (a + b))
M = 300000
k = rng.choice(np.arange(5), size=M, p=kdist / kdist.sum())
lam = np.clip(rng.beta(a + k, b + n - k) * drift, 0, 0.95)
post = (a + k) / (a + b + n) * drift
y = rng.random(M) < lam
S = rng.normal(3.5, 1.4, M)
mu = 3.5
L_post = np.mean(np.where(y, (S - post * mu) ** 2, (post * mu) ** 2))
L_lam = np.mean(np.where(y, (S - lam * mu) ** 2, (lam * mu) ** 2))
base = 1.6903
print(f"  оракул-λ: {base:.4f} -> {np.sqrt(base**2 - (L_post - L_lam) * 0.30):.4f} "
      f"(delta {base - np.sqrt(base**2 - (L_post - L_lam) * 0.30):.5f})")

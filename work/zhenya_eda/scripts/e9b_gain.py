"""E9b. Анализ поправок на кэшированных прогнозах (якорь 2025-10-11)."""
import os
import numpy as np
d = np.load(os.environ.get("ZH_OUT", "work/zhenya_eda/out") + "/e9_cache.npz")
lp, pv, yte, vte = d["lp"], d["pv"], d["yte"], d["vte"]
ly = np.log1p(yte)
rm = lambda l: float(np.sqrt(np.mean((l - ly) ** 2)))
rng = np.random.default_rng(0); idx = rng.permutation(len(ly)); h = len(ly)//2; F, G = idx[:h], idx[h:]

base = np.empty_like(lp)
for tr, te in ((F, G), (G, F)):
    b = np.polyfit(lp[tr], ly[tr], 1); base[te] = b[0]*lp[te] + b[1]
print(f"исчезли {100*vte.mean():.2f}%   сырой {rm(lp):.6f}   после глобального аффина {rm(base):.6f}  <- БАЗА")

def corr_on(start, cond, nb=20):
    out = start.copy()
    for tr, te in ((F, G), (G, F)):
        q = np.quantile(cond[tr], np.linspace(0,1,nb+1)); q[0],q[-1] = -np.inf, np.inf
        b1, b2 = np.digitize(cond[tr], q[1:-1]), np.digitize(cond[te], q[1:-1])
        sh = np.array([np.mean(ly[tr][b1==b]-start[tr][b1==b]) if (b1==b).sum()>50 else 0.0 for b in range(nb)])
        out[te] = start[te] + sh[b2]
    return out

lvl = corr_on(base, base)
van = corr_on(base, pv)
print(f"\nпоправка по УРОВНЮ прогноза (это команда делает): {rm(lvl):.6f}  ({rm(base)-rm(lvl):+.6f})")
print(f"поправка по P(ИСЧЕЗНЕТ):                          {rm(van):.6f}  ({rm(base)-rm(van):+.6f})")
seq = corr_on(lvl, pv)
print(f"уровень, ЗАТЕМ P(исчезнет) СВЕРХУ:                {rm(seq):.6f}  сверх уровня {rm(lvl)-rm(seq):+.6f}")

# ПЛАЦЕБО: случайная величина той же размерности, и перемешанный p_vanish
pl_rand = rng.normal(size=len(ly))
pl_perm = pv[rng.permutation(len(pv))]
print(f"\nПЛАЦЕБО случайная величина сверх уровня:          {rm(lvl)-rm(corr_on(lvl, pl_rand)):+.6f}")
print(f"ПЛАЦЕБО перемешанный p_vanish сверх уровня:       {rm(lvl)-rm(corr_on(lvl, pl_perm)):+.6f}")

print("\n=== ЧТО ИМЕННО ПРОИСХОДИТ: остаток по децилям P(исчезнет) ===")
q = np.quantile(pv, np.linspace(0,1,11)); q[0],q[-1] = -np.inf, np.inf
b = np.digitize(pv, q[1:-1])
print(" дец |  P(исч) сред | реально исчезли | средний остаток (факт-прогноз) | вклад в MSE")
tot = np.mean((lvl-ly)**2)
for k in range(10):
    m = b == k
    print(f"  {k:>2} |   {pv[m].mean():.4f}     |     {100*vte[m].mean():6.2f}%     |"
          f"        {np.mean(ly[m]-lvl[m]):+.4f}            |   {100*np.mean((lvl[m]-ly[m])**2)*m.mean()/tot:5.2f}%")

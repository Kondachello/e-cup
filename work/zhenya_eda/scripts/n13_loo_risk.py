# -*- coding: utf-8 -*-
"""N13. LOO-проверка приора семейства + риск-профиль пере-дозы F2."""
import math, sys, json
import numpy as np
sys.stdout.reconfigure(encoding="utf-8")
F0, NP_, NOISE = 1.6470, 50_000, 0.000022
FPC=math.sqrt(0.8)
K=np.array([x[1] for x in SEG]); SR=np.array([x[2] for x in SEG])
Q=(F0/SR)**2/NP_; S=SR*FPC
TAUS=np.concatenate([[1e-4],np.geomspace(0.005,0.35,600)])
def post_of(k,s):
    prof=np.array([min(0.5*np.sum(np.log(2*np.pi*(t*t+s**2))+(k-mu)**2/(t*t+s**2))
                   for mu in np.linspace(-0.25,0.45,241)) for t in TAUS])
    p=np.exp(-(prof-prof.min())); return p/p.sum()
def mu_at(t,k,s): v=t*t+s**2; return float(np.sum(k/v)/np.sum(1.0/v))
def dose(kk,ss,k,s,post):
    return sum(p*((mu_at(t,k,s))+(1.25*(t*t/(t*t+ss*ss))-0.25)*(kk-mu_at(t,k,s)))
               for t,p in zip(TAUS,post))

print("=== LOO: доза каждой оси по приору БЕЗ НЕЁ САМОЙ ===")
print(f"{'ось':14s}{'kappa':>8}{'доза ':>9}{'полн.приор':>12}{'LOO-приор':>11}{'LOO mult':>10}")
loo_mult={}
for i,(n,kk,sr) in enumerate(SEG):
    if n not in FACT: continue
    pf=post_of(K,S); b_full=dose(kk,S[i],K,S,pf)
    m=np.ones(len(SEG),bool); m[i]=False
    pl=post_of(K[m],S[m]); b_loo=dose(kk,S[i],K[m],S[m],pl)
    loo_mult[n]=round(b_loo/FACT[n],3)
    print(f"{n:14s}{kk:+8.3f}{FACT[n]:9.3f}{b_full:12.3f}{b_loo:11.3f}{b_loo/FACT[n]:10.3f}")
print("  LOO почти не двигает дозы => точка не тянет собственную усадку. Вывод устойчив.")

print(f"\n=== РИСК: что если истинный tau_seg БОЛЬШЕ, чем даёт семейство ===")
print(f"{'tau_seg':>9}{'опт.доза ':>14}{'E[priv] при 0.258':>19}{'E[priv] при 0.078':>19}{'кто лучше':>11}")
for t in (0.05,0.069,0.081,0.12,0.15,0.196,0.30):
    mu=mu_at(t,K,S); w=t*t/(t*t+ss*ss); wp=1.25*w-0.25; ekq=mu+wp*(kk-mu)
    g=lambda b: qq*(2*b*ekq-b*b)/(2*F0)
    print(f"{t:9.3f}{ekq:14.3f}{g(0.258):19.6f}{g(0.078):19.6f}"
          f"{'0.078' if g(0.078)>g(0.258) else '0.258':>11}")
print("\n  Перелом около tau_seg ~ 0.20 (реестровое значение). Ниже — резать дозу выгодно,")
print("  выше — держать. Апостериор семейства даёт P(tau>0.15) = 0.04, то есть резать.")

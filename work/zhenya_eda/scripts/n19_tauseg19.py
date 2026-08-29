# -*- coding: utf-8 -*-
"""N19. Финальный апостериор tau_seg на расширенном семействе + сдвиг доз F3/F5."""
import math, sys, json
import numpy as np
sys.stdout.reconfigure(encoding="utf-8")
F0,NP_,NOISE=1.6470,50_000,0.000022
FPC=math.sqrt(0.8)
# (имя, kappa, sigma_отчёт, база) — sigma у Саши БЕЗ fpc
K=np.array([x[1] for x in SEG]); SR=np.array([x[2] for x in SEG]); S=SR*FPC
Q=(F0/SR)**2/NP_
TAUS=np.concatenate([[1e-4],np.geomspace(0.005,0.35,700)])
def fitpost(k,s):
    prof=np.array([min(0.5*np.sum(np.log(2*np.pi*(t*t+s**2))+(k-mu)**2/(t*t+s**2))
                   for mu in np.linspace(-0.3,0.5,321)) for t in TAUS])
    p=np.exp(-(prof-prof.min())); return p/p.sum(), prof
def mu_at(t,k,s): v=t*t+s**2; return float(np.sum(k/v)/np.sum(1.0/v))

for lab,idx in [("10 точек (Часть H)",list(range(10))),("18 точек (все)",list(range(18)))]:
    p,prof=fitpost(K[idx],S[idx]); tml=TAUS[prof.argmin()]
    lo=np.sqrt(0);  ok=TAUS[prof<=prof.min()+1.92]
    print(f"{lab:22s} mu={mu_at(tml,K[idx],S[idx]):+.3f}  tau={tml:.3f} "
          f"[{ok.min():.3f},{ok.max():.3f}]  P(tau<0.05)={p[TAUS<0.05].sum():.2f}")

post,_=fitpost(K,S)
def dose(kk,ss,k,s,post):
    return sum(pp*(mu_at(t,k,s)+(1.25*(t*t/(t*t+ss*ss))-0.25)*(kk-mu_at(t,k,s)))
               for t,pp in zip(TAUS,post))
print(f"\n=== СДВИГ ДОЗ ОТ РАСШИРЕНИЯ СЕМЕЙСТВА (10 -> 18 точек) ===")
post10,_=fitpost(K[:10],S[:10])
print(f"{'ось':14s}{'q':>9}{'kappa':>8}{'доза F3':>9}{'опт(10т)':>10}{'опт(18т)':>10}"
      f"{'сдвиг EV':>11}{'в шумах':>9}")
tot=0.0; newmult={}
for (n,kk,sr,b),qq,ss in zip(SEG,Q,S):
    if n not in FACT: continue
    d10=dose(kk,ss,K[:10],S[:10],post10); d18=dose(kk,ss,K,S,post)
    cur=FACT[n]; dv=qq*((d18-cur)**2)/(2*F0); tot+=dv
    newmult[n]=round(d18,4)
    print(f"{n:14s}{qq:9.5f}{kk:+8.3f}{cur:9.3f}{d10:10.3f}{d18:10.3f}{dv:+11.6f}{dv/NOISE:9.1f}")
print(f"{'ИТОГО сдвиг приватного EV':52s}{tot:+11.6f}{tot/NOISE:9.1f}")
print(f"\n  Порог решения Саши: сдвиг > 2-3 шума привата => пересобирать.")
print(f"  ВЕРДИКТ: {'ПЕРЕСОБИРАТЬ (F6)' if tot>2*NOISE else 'дозы ФИНАЛЬНЫ, пересборка не окупается'}")
json.dump(dict(newdose=newmult,shift=tot),open("out/n19_tauseg.json","w",encoding="utf-8"),
          ensure_ascii=False,indent=1)

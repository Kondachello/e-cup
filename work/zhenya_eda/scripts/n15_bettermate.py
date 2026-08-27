# -*- coding: utf-8 -*-
"""N15. Пара лучше, чем +T3: кандидаты должны расходиться по ГЛАВНОЙ
неопределённости (tau_seg), а не по глубине сборки.

Розыгрыш по апостериору tau_seg: для каждого tau считаем приватный выигрыш
каждого кандидата над T3-цепочкой и берём МИНИМУМ скора (= максимум выигрыша).
"""
import math, sys, json
import numpy as np
sys.stdout.reconfigure(encoding="utf-8")
F0, NP_, NOISE = 1.6470, 50_000, 0.000022
FPC=math.sqrt(0.8)
SEG=[("",-0.076,0.113),("",-0.040,0.113),("",+0.126,0.051),("",-0.117,0.130),
     ("",-0.128,0.120),("",-0.057,0.053),("",+0.081,0.064),("",+0.021,0.060),
     ("",+0.157,0.068),("",+0.286,0.111)]
K=np.array([x[1] for x in SEG]); SR=np.array([x[2] for x in SEG]); S=SR*FPC
TAUS=np.concatenate([[1e-4],np.geomspace(0.005,0.35,600)])
prof=np.array([min(0.5*np.sum(np.log(2*np.pi*(t*t+S**2))+(K-mu)**2/(t*t+S**2))
               for mu in np.linspace(-0.25,0.45,241)) for t in TAUS])
post=np.exp(-(prof-prof.min())); post/=post.sum()
def mu_at(t): v=t*t+S**2; return float(np.sum(K/v)/np.sum(1.0/v))

# три сегментные оси в сборке: (имя, q, kappa, sigma_fpc)
AXES={"":(0.02086,0.126,0.051*FPC),"":(0.01173,0.157,0.068*FPC),
      "":(0.00440,0.286,0.111*FPC)}
BASE = 0.000254        # приватный выигрыш общей части (R8+редоза+микровектор) над T3
D_F2 = {"":0.138,"":0.155,"":0.258}                     # действующий финалист
D_F3 = {"":0.084,"":0.085,"":0.078}                     # мои приватные оптимумы
D_F4 = {"":0.000,"":0.000,"":0.000}                     # сегменты не трогаем вовсе

def gain(doses, tau):
    mu=mu_at(tau); g=BASE
    for n,(q,kp,ss) in AXES.items():
        w=tau*tau/(tau*tau+ss*ss); wp=1.25*w-0.25; ekq=mu+wp*(kp-mu)
        b=doses[n]; g+=q*(2*b*ekq-b*b)/(2*F0)
    return g
def stats(doses):
    g=np.array([gain(doses,t) for t in TAUS])
    return float((post*g).sum()), float(np.sqrt((post*(g-(post*g).sum())**2).sum()))
def pair(d1,d2):
    g1=np.array([gain(d1,t) for t in TAUS]); g2=np.array([gain(d2,t) for t in TAUS])
    best=np.maximum(g1,g2)                      # выигрыш больше = скор ниже = лучше
    return float((post*best).sum())

print("=== ОДИНОЧНЫЕ КАНДИДАТЫ (выигрыш над T3-цепочкой, больше = лучше) ===")
CAND={"F2 (дозы Саши)":D_F2,"F3 (мои приват-оптимумы)":D_F3,"F4 (без сегментных доз)":D_F4}
for n,d in CAND.items():
    m,sd=stats(d); print(f"  {n:26s} E={m:+.6f} = {m/NOISE:+5.1f} шума   sd={sd:.6f}")
m_T3=0.0
print(f"  {'T3 (страховка)':26s} E={m_T3:+.6f} = {0.0:+5.1f} шума   sd=0.000000")

print(f"\n=== ПАРЫ: E[лучший из двух] ===")
print(f"{'пара':38s}{'E[лучший]':>12}{'в шумах':>10}{'выигрыш от 2-го':>17}")
rows=[]
import itertools
opts=dict(CAND); opts["T3"]={k:0.0 for k in AXES}   # T3 = без всей сборки
BASE_T3=BASE
for a,b in itertools.combinations(opts,2):
    da,db=opts[a],opts[b]
    if b=="T3":
        g1=np.array([gain(da,t) for t in TAUS]); g2=np.zeros_like(g1)
    elif a=="T3":
        g2=np.array([gain(db,t) for t in TAUS]); g1=np.zeros_like(g2)
    else:
        g1=np.array([gain(da,t) for t in TAUS]); g2=np.array([gain(db,t) for t in TAUS])
    best=np.maximum(g1,g2); e=float((post*best).sum())
    solo=max(float((post*g1).sum()),float((post*g2).sum()))
    rows.append((f"{a} + {b}",e,e-solo))
for n,e,d in sorted(rows,key=lambda r:-r[1]):
    print(f"{n:38s}{e:+12.6f}{e/NOISE:+10.1f}{d:+17.6f}")
json.dump(dict(rows=rows),open("out/n15_pair.json","w",encoding="utf-8"),
          ensure_ascii=False,indent=1)

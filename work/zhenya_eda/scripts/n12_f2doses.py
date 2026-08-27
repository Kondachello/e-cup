# -*- coding: utf-8 -*-
"""N12. Дозы действующего финалиста  под приором СЕМЕЙСТВА, с интегрированием
по апостериору tau_seg и с поправкой fpc.

sigma в отчётах Саши считаны как F0/sqrt(n*q) — БЕЗ fpc. Значит:
    q_i        = (F0/sigma_rep)^2 / n_pub          (инверсия по той же формуле)
    sigma_true = 0.894 * sigma_rep                 (с fpc)
"""
import math, sys, json
import numpy as np
sys.stdout.reconfigure(encoding="utf-8")
F0, NP_, NOISE = 1.6470, 50_000, 0.000022
FPC = math.sqrt(1-NP_/250_000)

k=np.array([x[1] for x in SEG]); srep=np.array([x[2] for x in SEG])
q=(F0/srep)**2/NP_; s=srep*FPC
print("=== ГЕОМЕТРИЯ СЕМЕЙСТВА (q инвертирован, sigma с fpc) ===")
for (n,kk,sr),qq,ss in zip(SEG,q,s): print(f"  {n:14s} q={qq:.5f}  sigma_rep={sr:.3f} -> sigma={ss:.3f}")

# --- апостериор tau_seg (профиль, mu профилируется)
TAUS=np.concatenate([[1e-4],np.geomspace(0.005,0.35,700)])
def nllt(t):
    return min(0.5*np.sum(np.log(2*np.pi*(t*t+s**2))+(k-mu)**2/(t*t+s**2))
               for mu in np.linspace(-0.25,0.45,281))
prof=np.array([nllt(t) for t in TAUS]); post=np.exp(-(prof-prof.min())); post/=post.sum()
tau_ml=float(TAUS[prof.argmin()])
MUS=np.linspace(-0.25,0.45,281)
def mu_at(t): 
    v=t*t+s**2; return float(np.sum(k/v)/np.sum(1.0/v))     # ML mu при данном tau
print(f"\n  tau_seg ML = {tau_ml:.3f};  P(tau<0.05)={post[TAUS<0.05].sum():.2f}, "
      f"P(tau>0.15)={post[TAUS>0.15].sum():.2f};  mu при ML = {mu_at(tau_ml):+.3f}")

# --- приватно-оптимальная доза, интегрированная по апостериору tau
def dose_post(kk, ss):
    num=0.0
    for t,p in zip(TAUS,post):
        mu=mu_at(t); w=t*t/(t*t+ss*ss); wp=1.25*w-0.25
        num+=p*(mu+wp*(kk-mu))          # E[kappa_Q] усреднённое по tau
    return num
print(f"\n=== ДОЗЫ : факт против приватного оптимума ===")
print(f"{'ось':14s}{'q':>9}{'kappa':>8}{'доза ':>9}{'опт(апост.)':>13}"
      f"{'множитель':>11}{'дельта E[priv]':>16}")
tot=0.0; mult={}
for (n,kk,sr),qq,ss in zip(SEG,q,s):
    if n not in FACT: continue
    b_opt=dose_post(kk,ss); b=FACT[n]
    d=qq*((b_opt-b)**2)/(2*F0); tot+=d; mult[n]=round(b_opt/b,3)
    print(f"{n:14s}{qq:9.5f}{kk:+8.3f}{b:9.3f}{b_opt:13.3f}{b_opt/b:11.3f}{d:+16.6f}")
print(f"{'ИТОГО прибавка к E[private] при перезадании доз':56s}{tot:+.6f} = {tot/NOISE:.1f} шума")
print(f"\n  множители к дозам : " + ", ".join(f"{n.split('_')[0]} ×{m:.2f}" for n,m in mult.items()))
json.dump(dict(tau_ml=tau_ml,mu=mu_at(tau_ml),mult=mult,gain=tot),
          open("out/n12_f2doses.json","w",encoding="utf-8"),ensure_ascii=False,indent=1)

# -*- coding: utf-8 -*-
"""F: прямая оптимизация билета; вариант «две свободные посылки»; чувствительность к смещению."""
import json, os, sys
import numpy as np, scipy.linalg as sla
from scipy.optimize import minimize
sys.path.insert(0,"/Users/alexanderkondakov/ozon-cup/work/scripts")
from p_top3 import Objective, MU_US, SIGMA_US, NOISE
SCR=os.path.dirname(os.path.abspath(__file__))
st=np.load("/Users/alexanderkondakov/ozon-cup/work/reports/lineA/gls_state_eb.npz",allow_pickle=True)
names=[str(x) for x in st["names"]]
Q,cQ,V,Lam,q,bF7=st["Q"],st["cQ"],st["mdl_vivian"],st["Lam"],st["q"],st["doses_F7"]
FS=float(st["F_SCALE"]); n=len(names); RG=1e-9*np.trace(Q)/n*np.eye(n); SAMP=0.0011
def solve(g,lam=1.0): return np.linalg.solve(Q+lam*Lam+RG+g*np.diag(np.diag(Q)),cQ)
def gain(d): return float((2*d@cQ-d@Q@d)/(2*FS))
def gsd(d): return float(np.sqrt(max(1.25**2*d@V@d,0))/FS)
def rms_lp(dd): return float(np.sqrt(max(dd@Q@dd,0)))
vert=bF7+cQ/q
def bias_tot(d):
    b=bF7+d; r=b/vert; cost=q*((b-vert)**2-vert**2)/(2*FS*NOISE); viol=(r<0)|(r>2)
    m=viol&(cost>0); return float(cost[m].sum() if m.any() else 0.0), float(np.abs(r).max()), int((viol&(cost>1)).sum())
d01=solve(0.1); SD_X=1.52*NOISE; GAIN_F8=gain(d01)-0.43*NOISE
SIG_SHARED=float(np.sqrt(SIGMA_US**2-gsd(d01)**2-SD_X**2))
obj=Objective(ns=300_000); rng=np.random.default_rng(31337)
z_sh=obj.z; e_a=obj.w; e_b=rng.standard_normal(obj.ns)
def mu_of(d,h): return MU_US-(gain(d)-GAIN_F8)+h*bias_tot(d)[0]*NOISE
def P_single(d,ex=0.0,h=0.0):
    s=float(np.hypot(gsd(d),ex)); return float((mu_of(d,h)+SIG_SHARED*z_sh+s*e_a<obj.c3).mean())
def P_pair(d1,d2,ex1=0.0,ex2=0.0,h=0.0):
    s1,s2=gsd(d1),gsd(d2); c=float(1.25**2*d1@V@d2/FS**2); ss=SAMP*rms_lp(d1-d2)
    C=np.array([[s1**2+ex1**2,c],[c,s2**2+ex2**2+ss**2]])
    w_,U=np.linalg.eigh(C); w_=np.clip(w_,0,None); A=U@np.diag(np.sqrt(w_))
    g1=mu_of(d1,h)+SIG_SHARED*z_sh+A[0,0]*e_a+A[0,1]*e_b
    g2=mu_of(d2,h)+SIG_SHARED*z_sh+A[1,0]*e_a+A[1,1]*e_b
    return float((np.minimum(g1,g2)<obj.c3).mean())
M=1.25**2*V/FS**2+SAMP**2*Q; wg,Vg=sla.eigh(M,Q)
BV=[Vg[:,-1-j]/np.sqrt(Vg[:,-1-j]@Q@Vg[:,-1-j]) for j in range(6)]
def build(p):
    g=abs(p[0]); lam=abs(p[1]); d=solve(min(g,3.0),min(lam,4.0))
    for j,t in enumerate(p[2:]): d=d+t*BV[j]
    return d
P_F8=P_single(d01,SD_X)
print(f"база: F8 один = {P_F8*100:.2f} %\n")
print("=== F1. ПРЯМАЯ ОПТИМИЗАЦИЯ: max P(пара {новый, F8}) по (γ, Λ, 6 направлений развязки) ===")
for h in [0.0,0.05,0.1,0.25]:
    best=(-1,None)
    for x0 in [np.array([0.0,1.0,0,0,0,0,0,0]),np.array([0.0,0.0,0,0,0,0,0,0]),
               np.array([0.02,1.0,-0.02,0,0,0,0,0]),np.array([0.05,0.5,0.01,0.01,0,0,0,0])]:
        r=minimize(lambda p:-P_pair(build(p),d01,0.0,SD_X,h),x0,method="Nelder-Mead",
                   options=dict(maxiter=1200,xatol=1e-4,fatol=1e-6))
        if -r.fun>best[0]: best=(-r.fun,r.x)
    d=build(best[1]); bt,mr,nm_=bias_tot(d)
    print(f"  хейркат {h:4.2f}: P(пара)={best[0]*100:6.2f} %  "
          f"(vsF8 {(gain(d)-GAIN_F8)/NOISE:+6.2f}ш, sd {gsd(d)/NOISE:5.2f}ш, "
          f"один {P_single(d,0,h)*100:5.2f} %, добавка 2-го {(best[0]-max(P_single(d,0,h),P_single(d01,SD_X,h)))*100:+5.2f} п.п., "
          f"max|r| {mr:.1f}, Σцена {bt:.0f}ш)")
    print(f"              γ={abs(best[1][0]):.4f} Λ×{abs(best[1][1]):.3f} "
          f"t=[{', '.join(f'{v:+.4f}' for v in best[1][2:])}]")
print("\n=== F2. Сравнение с чисто E-фронтовыми файлами (без развязки) при том же хейркате ===")
for h in [0.0,0.05,0.1,0.25]:
    row=[]
    for nm,d in [("γ=0,Λ×0",solve(0,0)),("γ=0",solve(0)),("γ=0.02",solve(0.02)),
                 ("γ=0.05",solve(0.05)),("γ=0.08 (F9)",solve(0.08))]:
        row.append((nm,P_pair(d,d01,0.0,SD_X,h)))
    bb=max(row,key=lambda x:x[1])
    print(f"  хейркат {h:4.2f}: " + "  ".join(f"{nm} {p*100:5.2f}%" for nm,p in row)
          + f"   ЛУЧШИЙ: {bb[0]}")
print("\n=== F3. ВАРИАНТ «ДВЕ СВОБОДНЫЕ ПОСЫЛКИ»: оба финалиста новые ===")
cand={"γ=0,Λ×0":solve(0,0),"γ=0":solve(0),"γ=0.02":solve(0.02),"γ=0.05":solve(0.05),
      "γ=0.08":solve(0.08),"γ=0−0.02·v1":solve(0)+(-0.02)*BV[0],
      "γ=0+0.02·v2":solve(0)+0.02*BV[1],"γ=0−0.015·v3":solve(0)+(-0.015)*BV[2]}
ks=list(cand); best=(-1,None)
for i in range(len(ks)):
    for j in range(i,len(ks)):
        p=P_pair(cand[ks[i]],cand[ks[j]]) if i!=j else P_single(cand[ks[i]])
        if p>best[0]: best=(p,(ks[i],ks[j]))
        if i!=j and p>0.325: print(f"   {ks[i]:14s}+{ks[j]:14s} {p*100:6.2f} %")
print(f"   ЛУЧШАЯ ПАРА ИЗ ДВУХ НОВЫХ: {best[1]} -> {best[0]*100:.2f} % "
      f"(против {P_pair(solve(0,0),d01,0.0,SD_X)*100:.2f} % у «γ=0,Λ0 + F8»)")

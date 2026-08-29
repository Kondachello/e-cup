# -*- coding: utf-8 -*-
"""D: честная добавка второго файла = P(пара) - max(P одиночек), + оптимизация билета + гейт смещения."""
import json, os, sys
import numpy as np, scipy.linalg as sla
sys.path.insert(0,"/Users/alexanderkondakov/ozon-cup/work/scripts")
from p_top3 import Objective, MU_US, SIGMA_US, NOISE
SCR=os.path.dirname(os.path.abspath(__file__))
st=np.load("/Users/alexanderkondakov/ozon-cup/work/reports/lineA/gls_state_eb.npz",allow_pickle=True)
names=[str(x) for x in st["names"]]
Q,cQ,V,Lam,q,bF7=st["Q"],st["cQ"],st["mdl_vivian"],st["Lam"],st["q"],st["doses_F7"]
FS=float(st["F_SCALE"]); n=len(names); RG=1e-9*np.trace(Q)/n*np.eye(n); SAMP=0.0011
def solve(g=0.1,lam=1.0,keep=None):
    A=Q+lam*Lam+RG+g*np.diag(np.diag(Q))
    if keep is None: return np.linalg.solve(A,cQ)
    d=np.zeros(n); k=np.array(sorted(keep)); d[k]=np.linalg.solve(A[np.ix_(k,k)],cQ[k]); return d
def gain(d): return float((2*d@cQ-d@Q@d)/(2*FS))
def gsd(d): return float(np.sqrt(max(1.25**2*d@V@d,0))/FS)
def rms_lp(dd): return float(np.sqrt(max(dd@Q@dd,0)))
vert=bF7+cQ/q
def biasdiag(d):
    b=bF7+d; r=b/vert; cost=q*((b-vert)**2-vert**2)/(2*FS*NOISE)
    viol=(r<0)|(r>2)
    return float(np.abs(r).max()), int(viol.sum()), int((viol&(cost>1)).sum()), \
           float(cost.max()), float(cost[viol&(cost>0)].sum() if (viol&(cost>0)).any() else 0.0)
d01=solve(0.1); SD_X=1.52*NOISE
GAIN_F8=gain(d01)-0.43*NOISE
SIG_SHARED=float(np.sqrt(SIGMA_US**2-gsd(d01)**2-SD_X**2))
obj=Objective(ns=400_000); rng=np.random.default_rng(20260829)
z_sh=obj.z; e_a=obj.w; e_b=rng.standard_normal(obj.ns)
def mu_of(d,hair=0.0):
    _,_,_,_,tot=biasdiag(d)
    return MU_US-(gain(d)-GAIN_F8)+hair*tot*NOISE
def P_single(d,ex=0.0,hair=0.0):
    s=float(np.hypot(gsd(d),ex)); g=mu_of(d,hair)+SIG_SHARED*z_sh+s*e_a
    return float((g<obj.c3).mean())
def P_pair(d1,d2,ex1=0.0,ex2=0.0,hair=0.0):
    s1,s2=gsd(d1),gsd(d2); c=float(1.25**2*d1@V@d2/FS**2); ss=SAMP*rms_lp(d1-d2)
    C=np.array([[s1**2+ex1**2,c],[c,s2**2+ex2**2+ss**2]])
    w_,U=np.linalg.eigh(C); w_=np.clip(w_,0,None); A=U@np.diag(np.sqrt(w_))
    t1=A[0,0]*e_a+A[0,1]*e_b; t2=A[1,0]*e_a+A[1,1]*e_b
    g1=mu_of(d1,hair)+SIG_SHARED*z_sh+t1; g2=mu_of(d2,hair)+SIG_SHARED*z_sh+t2
    return float((np.minimum(g1,g2)<obj.c3).mean())
P_F8=P_single(d01,SD_X); print(f"КАЛИБРОВКА: одиночный F8 = {P_F8*100:.2f} % (цель 20.92 %)\n")

M=1.25**2*V/FS**2+SAMP**2*Q; wg,Vg=sla.eigh(M,Q)
print("=== D0. ИЗ ЧЕГО СОСТОИТ САМОЕ ДЕШЁВОЕ НАПРАВЛЕНИЕ РАЗВЯЗКИ v1 ===")
v1=Vg[:,-1]/np.sqrt(Vg[:,-1]@Q@Vg[:,-1])
for i in np.argsort(-np.abs(v1))[:8]:
    print(f"   {names[i]:6s} вес {v1[i]:+.3f}  q {q[i]:.4f}  доза_F7 {bF7[i]:+.3f}  вершина {vert[i]:+.3f}")
print(f"   отношение дисперсий к семплингу: {wg[-1]/SAMP**2:.1f}x (по sd {np.sqrt(wg[-1])/SAMP:.1f}x)\n")

fam={}
for g in [0.0,0.02,0.05,0.08,0.1,0.2,0.5]: fam[f"γ={g:g}"]=solve(g)
for lam in [0.0,0.5]: fam[f"γ=0,Λ×{lam:g}"]=solve(0.0,lam)
o=np.argsort(-np.abs(cQ))
for k in (10,20,30): fam[f"GLS на топ-{k} осях"]=solve(0.0,1.0,list(o[:k]))
for j in range(3):
    v=Vg[:,-1-j]/np.sqrt(Vg[:,-1-j]@Q@Vg[:,-1-j])
    for t in [-0.03,-0.02,-0.015,-0.01,0.01,0.015,0.02]:
        fam[f"γ=0 {t:+.3f}·v{j+1}"]=solve(0.0)+t*v
print("=== D1. ЧЕСТНАЯ ДОБАВКА ВТОРОГО ФАЙЛА: P(пара) − max(P одиночек) ===")
print("    (пара всегда с УЖЕ ОТПРАВЛЕННЫМ F8 — свободных посылок на два новых файла нет)")
print(f"{'кандидат':22s}{'vsF8,ш':>8}{'sd,ш':>6}{'sdразн':>8}{'P(кандидат)':>13}{'P(пара)':>10}"
      f"{'добавка':>9}{'max|r|':>8}{'мат':>4}{'Σцена':>8}")
print("-"*112)
rr=[]
for nm,d in fam.items():
    ps=P_single(d); pp=P_pair(d,d01,0.0,SD_X)
    mr,nv,nmat,cmax,ctot=biasdiag(d)
    sdd=float(np.hypot(np.sqrt(max(1.25**2*(d-d01)@V@(d-d01),0))/FS,SD_X))/NOISE
    rr.append(dict(name=nm,g=(gain(d)-GAIN_F8)/NOISE,sd=gsd(d)/NOISE,sdd=sdd,ps=ps,pp=pp,
                   add=(pp-max(ps,P_F8))*100,maxr=mr,nmat=nmat,ctot=ctot,d=d))
rr.sort(key=lambda x:-x["pp"])
for r in rr:
    print(f"{r['name']:22s}{r['g']:+8.2f}{r['sd']:6.2f}{r['sdd']:8.2f}{r['ps']*100:12.2f}%"
          f"{r['pp']*100:9.2f}%{r['add']:+9.2f}{r['maxr']:8.1f}{r['nmat']:4d}{r['ctot']:8.1f}")
print("\n=== D2. ГЕЙТ СМЕЩЕНИЯ: та же таблица с ХЕЙРКАТОМ за-вершинной цены ===")
print("    (гейт «за-вершинное СМЕЩЕНИЕ E» из OBJECTIVE §5 остаётся в силе при цели топ-3)")
print(f"{'кандидат':22s}" + "".join(f"{'hair='+str(h):>12}" for h in [0.0,0.1,0.25,0.5,1.0]))
top=[r for r in rr[:12]]
for r in top:
    line=f"{r['name']:22s}"
    for h in [0.0,0.1,0.25,0.5,1.0]:
        line+=f"{P_pair(r['d'],d01,0.0,SD_X,h)*100:11.2f}%"
    print(line)
json.dump(dict(P_F8=P_F8,rows=[{k:v for k,v in r.items() if k!='d'} for r in rr]),
          open(os.path.join(SCR,"partD.json"),"w"),ensure_ascii=False,indent=1)

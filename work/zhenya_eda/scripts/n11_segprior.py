# -*- coding: utf-8 -*-
"""N11. СОБСТВЕННЫЙ приор семейства сегментных зондов (11 точек -seg_chalk).

Мой же §2.1: приор надо брать по ТОЙ ЖЕ популяции. Реестровый N(0.309,0.196^2)
описывает разнородный набор (пересборки бленда, модели, уровни). Сегментные
зонды — однородное семейство одной конструкции. У него свой приор, и именно он
"""
import math, sys, json
import numpy as np
sys.stdout.reconfigure(encoding="utf-8")
NOISE = 0.000022

k = np.array([x[1] for x in SEG]); s = np.array([x[2] for x in SEG])

def fit(k,s,murange=(-0.3,0.5)):
    nll=lambda mu,t2: 0.5*np.sum(np.log(2*np.pi*(t2+s**2))+(k-mu)**2/(t2+s**2))
    MU=np.linspace(*murange,1201); T2=np.concatenate([[0.0],np.geomspace(1e-7,0.2,900)])
    g=np.array([[nll(m,t) for t in T2] for m in MU])
    i,j=np.unravel_index(g.argmin(),g.shape); fm=g.min()
    mo=MU[g.min(1)<=fm+1.92]; to=np.sqrt(T2[g.min(0)<=fm+1.92])
    return float(MU[i]),math.sqrt(float(T2[j])),(mo.min(),mo.max()),(to.min(),to.max())

MU,TAU,mci,tci = fit(k,s)
print(f"=== ПРИОР СЕМЕЙСТВА СЕГМЕНТНЫХ ЗОНДОВ ({len(SEG)} точек) ===")
for n,kk,ss in SEG: print(f"  {n:14s} kappa={kk:+.3f} ± {ss:.3f}")
print(f"\n  Var(kappa)={np.var(k,ddof=1):.5f}  <sigma^2>={np.mean(s**2):.5f}")
print(f"  ML: mu_seg = {MU:+.3f} [{mci[0]:+.3f},{mci[1]:+.3f}]   "
      f"tau_seg = {TAU:.3f} [{tci[0]:.3f},{tci[1]:.3f}]")
print(f"  реестровый приор был N(0.309, 0.196^2) — для этого семейства НЕВЕРЕН:")
print(f"    mu {MU:+.3f} против 0.309  => реестр завышает среднее втрое")
print(f"    tau {TAU:.3f} против 0.196 => реестр завышает разброс вчетверо")

print(f"\n=== ЧТО ЭТО ДАЁТ ДОЗАМ ДЕЙСТВУЮЩЕГО ФИНАЛИСТА F2 ===")
print(f"{'ось':14s}{'kappa':>8}{'sigma':>7}{'w':>7}{'w_p':>7}"
      f"{'доза ':>9}{'доза приор-сем':>15}{'доза приор-реестр':>18}")
rows=[]
for n,kk,ss in SEG:
    w=TAU**2/(TAU**2+ss*ss); wp=1.25*w-0.25
    b_seg = MU + wp*(kk-MU)
    wR=0.196**2/(0.196**2+ss*ss); wpR=1.25*wR-0.25
    b_reg = 0.309 + wpR*(kk-0.309)
    f=FACT.get(n)
    rows.append((n,kk,ss,w,wp,b_seg,b_reg,f))
    if f is not None:
        print(f"{n:14s}{kk:+8.3f}{ss:7.3f}{w:7.3f}{wp:+7.3f}{f:9.3f}{b_seg:15.3f}{b_reg:18.3f}")
print("\n  (остальные оси семейства — нули, дозы не применялись)")

print(f"\n=== ЦЕНА РАСХОЖДЕНИЯ (на ось, при её q) ===")
F0=1.6470; NP_=50_000
print(f"{'ось':14s}{'q(из sigma)':>13}{'дельта дозы':>13}{'потеря E[priv]':>16}")
tot=0.0
for n,kk,ss,w,wp,b_seg,b_reg,f in rows:
    if f is None: continue
    q=(0.894*F0/ss)**2/NP_
    d=(f-b_seg); loss=q*d*d/(2*F0); tot+=loss
    print(f"{n:14s}{q:13.5f}{d:+13.3f}{loss:+16.6f}")
print(f"{'ИТОГО потеря от приора реестра вместо приора семейства':55s}{tot:+.6f} = {tot/NOISE:.1f} шума")
json.dump(dict(mu_seg=MU,tau_seg=TAU,tau_ci=tci,rows=[list(r) for r in rows],loss=tot),
          open("out/n11_segprior.json","w",encoding="utf-8"),ensure_ascii=False,indent=1)

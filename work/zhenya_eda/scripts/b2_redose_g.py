# -*- coding: utf-8 -*-
"""B2. Пересчёт доз с ПООСНЫМ g. Приор семейства — по ВСЕМ зондам, включая нули."""
import math, sys, json
import numpy as np
sys.stdout.reconfigure(encoding="utf-8")
F0,NP_,NOISE=1.6470,50_000,0.000022
FPC=math.sqrt(0.8)
# ВСЁ семейство P: (имя, kappa, sigma_отч, g_ось). g из b1 либо от ближайшего по конструкции.
FAM=[("",-0.076,0.113,1.00),("",-0.040,0.113,0.81),("",+0.126,0.051,1.05),
     ("",-0.117,0.130,0.80),("",-0.128,0.120,1.00),("",-0.057,0.053,1.00),
     ("",+0.081,0.064,1.00),("",+0.021,0.060,1.00),("",+0.157,0.068,0.93),
     ("",+0.286,0.111,1.00),("",+0.043,0.058,1.00),("",-0.059,0.067,1.00),
     ("",-0.039,0.221,0.81),("",+0.043,0.119,1.00),("",+0.168,0.130,1.00),
     ("",+0.058,0.187,1.00),("",-0.197,0.067,0.69),("",-0.042,0.066,1.00),
     ("",+0.196,0.065,1.00),("",+0.372,0.109,0.67),
     ("",+0.609,0.185,1.00),("",-0.312,0.130,0.69),
     
     ("n1",0.000,0.090,1.00),("n2",0.000,0.090,1.00),("n3",0.000,0.090,1.00),
     ("n4",0.000,0.100,1.00),("n5",0.000,0.100,1.00),("n6",0.000,0.100,1.00),
     ("n7",0.000,0.100,1.00),("n8",0.000,0.110,1.00),("n9",0.000,0.110,1.00),
     ("n10",0.000,0.110,1.00)]
K=np.array([x[1] for x in FAM])
def fit(k,s):
    nll=lambda mu,t2: 0.5*np.sum(np.log(2*np.pi*(t2+s**2))+(k-mu)**2/(t2+s**2))
    MU=np.linspace(-0.2,0.4,601); T2=np.concatenate([[0.0],np.geomspace(1e-7,0.2,700)])
    g_=np.array([[nll(m,t) for t in T2] for m in MU])
    i,j=np.unravel_index(g_.argmin(),g_.shape); return float(MU[i]),math.sqrt(float(T2[j]))
print(f"семейство P: {len(FAM)} зондов (живые + нули обеих охот)\n")
LIVE=["","","","","","","",""]
for lab,use_g in (("blanket g=1.15",False),("ПООСНЫЙ g (b1)",True)):
    S=np.array([x[2]*FPC*(x[3] if use_g else 1.15) for x in FAM])
    mu,tau=fit(K,S)
    print(f"=== {lab}: приор семейства mu={mu:+.3f}, tau={tau:.3f} ===")
    print(f"{'ось':6s}{'kappa':>8}{'g':>7}{'sigma':>8}{'q':>9}{'w':>7}{'w_p':>8}"
          f"{'доза':>8}{'E[priv]':>11}{'шумов':>7}")
    tot=0.0
    for (n,kp,sr,gg),ss in zip(FAM,S):
        if n not in LIVE: continue
        q=(F0/sr)**2/NP_
        w=tau*tau/(tau*tau+ss*ss); wp=1.25*w-0.25; b=mu+wp*(kp-mu)
        ev=q*b*b/(2*F0) if b*np.sign(kp)>0 else 0.0; tot+=ev
        print(f"{n:6s}{kp:+8.3f}{(1.15 if not use_g else gg):7.2f}{ss:8.4f}{q:9.5f}"
              f"{w:7.3f}{wp:+8.3f}{b:+8.3f}{ev:+11.6f}{ev/NOISE:7.1f}")
    print(f"{'ИТОГО E[priv] живых осей':48s}{tot:+11.6f}{tot/NOISE:7.1f}\n")

print("=== ЧТО ЭТО ДАЁТ: F7 = F6 + переход к поосному g ===")
S1=np.array([x[2]*FPC*1.00 for x in FAM]); mu1,tau1=fit(K,S1)     # как собирался F5/F6
S3=np.array([x[2]*FPC*x[3] for x in FAM]);  mu3,tau3=fit(K,S3)     # поосный g
print(f"  приор при g=1 (как собран F6): mu={mu1:+.3f}, tau={tau1:.3f}")
print(f"  приор при поосном g:           mu={mu3:+.3f}, tau={tau3:.3f}")
print(f"\n{'ось':6s}{'kappa':>8}{'доза в F6':>11}{'доза F7':>10}{'дельта':>9}"
      f"{'приращение E[priv]':>20}{'шумов':>7}")
tot=0.0
for (n,kp,sr,gg),s1,s3 in zip(FAM,S1,S3):
    if n not in LIVE: continue
    q=(F0/sr)**2/NP_
    w1=tau1**2/(tau1**2+s1*s1); b1=mu1+(1.25*w1-0.25)*(kp-mu1)
    w3=tau3**2/(tau3**2+s3*s3); b3=mu3+(1.25*w3-0.25)*(kp-mu3)
    inF6 = n in ("","","","","","")
    b_now = b1 if inF6 else 0.0
    # приращение приватного EV при переходе b_now -> b3, истина E[k_Q]=b3
    d=q*((b3-b_now)**2)/(2*F0); tot+=d
    print(f"{n:6s}{kp:+8.3f}{b_now:+11.3f}{b3:+10.3f}{b3-b_now:+9.3f}{d:+20.6f}{d/NOISE:7.1f}")
print(f"{'ИТОГО прирост F7 над F6':52s}{tot:+.6f} = {tot/NOISE:.1f} шума")
print(f"\n  Для сравнения: весь F6 над F5 стоил +0.000144 = 6.5 шума.")

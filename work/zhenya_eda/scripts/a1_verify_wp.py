# -*- coding: utf-8 -*-
"""A1. АУДИТ: прямая эмпирическая проверка ядра теории на настоящих данных.

Проверяем три утверждения, на которых висит ВСЁ:
 (1) kappa_Q = kappa_T - (f/(1-f))*(kappa_P - kappa_T),  f = n_pub/N = 0.2
 (2) sigma_kappa = F0/sqrt(n_pub*q) * sqrt(1-f)        [поправка fpc]
 (3) переоценка паблика на ось, дозированную сырой kappa_P, = 2.5*sigma^2*q/(2F0)

Метод: берём НАСТОЯЩИЕ lp-векторы и НАСТОЯЩИЙ таргет (sample_submit = гмв
вал-окна), режем 250k на 50k/200k много раз и меряем всё напрямую.
"""
import json, math, sys
import numpy as np
sys.stdout.reconfigure(encoding="utf-8")
z=np.load("out/lb_full.npz"); meta=json.load(open("out/lb_meta.json"))
t=z["tval"].astype(np.float64)                       # реальный таргет
names=[n for n in meta["names"] if n!="sample"]
L={n:z[f"lp_{n}"].astype(np.float64) for n in names}
N=len(t); NP_=50_000; f=NP_/N
base=L["V3_canon"]                                    # честная база
res=base-t
F0=math.sqrt(float((res*res).mean()))
print(f"база V3_canon, F0 на полном тесте = {F0:.6f}, N={N}, n_pub={NP_}, f={f}")

# несколько настоящих направлений разного масштаба
DIRS={}
for nm,a,b in [("tfm4-ось","T2_tfm4_orth_045","G2_gru_tfm_02"),
               ("gru-ось","G1_gru_tfm_full","V3_canon"),
               ("ridge-ось","R3_ridge","R2_newblend"),
               ("shade-ось","R5_shade","R2_newblend")]:
    d=L[a]-L[b]; d=d-d.mean(); DIRS[nm]=d

rng=np.random.default_rng(4242); NDRAW=800
print(f"\n{'направление':12s}{'q':>10}{'kappa_T':>10}{'sd(k_P) факт':>14}"
      f"{'sd по закону':>14}{'отн.':>7}{'корр(eps,k_Q-k_T)':>19}{'наклон':>9}")
OUT={}
for nm,d in DIRS.items():
    q=float((d*d).mean()); u=d*(t-base)
    kT=float(u.mean())/q
    d2=d*d; su=u.sum(); sd2=d2.sum()
    kp=np.empty(NDRAW); kq=np.empty(NDRAW)
    for i in range(NDRAW):
        P=np.argpartition(rng.random(N),NP_)[:NP_]
        up=u[P].sum(); dp=d2[P].sum()
        kp[i]=up/dp
        kq[i]=(su-up)/(sd2-dp)
    sd_fact=kp.std(); sd_law=F0/math.sqrt(NP_*q)*math.sqrt(1-f)
    eps=kp-kT; dq=kq-kT
    slope=float(np.polyfit(eps,dq,1)[0])
    OUT[nm]=dict(q=q,kT=kT,sd_fact=float(sd_fact),sd_law=sd_law,slope=slope)
    print(f"{nm:12s}{q:10.5f}{kT:+10.3f}{sd_fact:14.5f}{sd_law:14.5f}"
          f"{sd_fact/sd_law:7.3f}{float(np.corrcoef(eps,dq)[0,1]):19.4f}{slope:9.4f}")
print(f"\n  ТЕОРИЯ: наклон d(kappa_Q - kappa_T)/d(eps) должен быть РОВНО -f/(1-f) = {-f/(1-f):.4f}")
print(f"  ФАКТ:   {', '.join(f'{v['slope']:+.4f}' for v in OUT.values())}")
print(f"  ТЕОРИЯ: отношение sd_факт/sd_закон должно быть ~1.00")
json.dump(OUT,open("out/a1_verify.json","w",encoding="utf-8"),ensure_ascii=False,indent=1)

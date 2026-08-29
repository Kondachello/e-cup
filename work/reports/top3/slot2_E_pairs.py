# -*- coding: utf-8 -*-
"""E: шесть заказанных пар + граничное отставание + учёт РАЗНИЦЫ ПЕРЕОЦЕНОК (канал 3)."""
import json, os, sys
import numpy as np
sys.path.insert(0,"/Users/alexanderkondakov/ozon-cup/work/scripts")
from p_top3 import Objective, MU_US, SIGMA_US, NOISE
SCR=os.path.dirname(os.path.abspath(__file__))
z=np.load(os.path.join(SCR,"cands_lp.npz"),allow_pickle=True)
names=[str(x) for x in z["names"]]; LP=z["lp"]; idx={n:i for i,n in enumerate(names)}
lp8=LP[idx["F8_priv"]]; s8=float(np.std(lp8))
# калибровка corr -> rms на локальных файлах
cs,rs_=[],[]
for n in names:
    if n=="F8_priv": continue
    d=LP[idx[n]]-lp8; cs.append(float(np.corrcoef(LP[idx[n]],lp8)[0,1])); rs_.append(float(np.sqrt((d*d).mean())))
cs=np.array(cs); rs_=np.array(rs_)
pred=s8*np.sqrt(np.clip(2*(1-cs),0,None))
k=float(np.median(rs_/pred))
print(f"калибровка rms ≈ {k:.3f} · sd(lp_F8)·sqrt(2(1−corr));  sd(lp_F8)={s8:.4f}; "
      f"разброс {np.percentile(rs_/pred,10):.2f}..{np.percentile(rs_/pred,90):.2f}")
def rms_from_corr(c): return k*s8*np.sqrt(max(2*(1-c),0))

obj=Objective(ns=400_000); rng=np.random.default_rng(555)
z_sh=obj.z; e_a=obj.w
E_F8=1.646203
def P(dE_list, sd_list):
    """пара/одиночка: общая z, индивидуальные добавки."""
    gs=[E_F8+dE+SIGMA_US*z_sh+sd*e for dE,sd,e in zip(dE_list,sd_list,[e_a,rng.standard_normal(obj.ns)])]
    x=gs[0] if len(gs)==1 else np.minimum(*gs)
    return float((x<obj.c3).mean())
BASE=P([0.0],[0.0]); print(f"\nодиночный F8: {BASE*100:.2f} %")

# ---- канал 3: разница ПЕРЕОЦЕНОК (M2). over(T3)=0.000671±0.000277(цепочка)+0.000198±0.000066
# над F8 висит 0.000199 цепочки, над T3 — 0.000474; неопределённость линейна по (kappa−mu)
SD_OVER_CHAIN=0.000277; OV_T3, OV_F8 = 0.000474, 0.000199
sd_over_diff=abs(OV_T3-OV_F8)/OV_T3*SD_OVER_CHAIN
print(f"канал 3 (разница переоценок F8 vs T3): sd = {sd_over_diff:.6f} = {sd_over_diff/NOISE:.1f} шума")

def show(tag, dE_n, sd_n, note=""):
    dE=dE_n*NOISE; sd=sd_n*NOISE
    p=P([0.0,dE],[0.0,sd])
    print(f"{tag:34s}{dE_n:+9.1f}{sd_n:9.2f}{p*100:9.2f}%{(p-BASE)*100:+9.3f}   {note}")
print("\n=== mdl_kyanit. ШЕСТЬ ЗАКАЗАННЫХ ПАР (первый файл = F8, если не сказано иное) ===")
print(f"{'пара':34s}{'2й хуже,ш':>9}{'sd разн,ш':>9}{'P(топ-3)':>9}{'прирост':>9}")
print("-"*95)
rows=[]
def rms_of(nm):
    d=LP[idx[nm]]-lp8; return float(np.sqrt((d*d).mean()))
# T3
r_t3=rms_of("T3_g1_redose_044")
show("F8 + T3 (действующая, только семплинг)",(1.647603-E_F8)/NOISE,0.0011*r_t3/NOISE,"corr 0.9993")
sd_t3_full=float(np.hypot(0.0011*r_t3, sd_over_diff))
show("F8 + T3 (+ разница переоценок)",(1.647603-E_F8)/NOISE,sd_t3_full/NOISE,"sd разн честнее")
# H2_edge_p1 — csv нет, числа из M3 Жени
r_h2=0.0913; E_h2=1.649398
show("F8 + H2_edge_p1",(E_h2-E_F8)/NOISE,0.0011*r_h2/NOISE,"M3: rms 0.0913, corr 0.9985")
# R5_shade — локально есть
r_r5=rms_of("R5_shade"); E_r5=1.647980
show("F8 + R5_shade",(E_r5-E_F8)/NOISE,0.0011*r_r5/NOISE,f"rms {r_r5:.4f}, corr 0.9990")
# A2_probe_s1_gmv — csv нет; corr 0.9950 (M3), паблик 1.6563024
r_a2=rms_from_corr(0.9950); E_a2=1.6563024241+0.000671*0.30
show("F8 + A2_probe_s1_gmv",(E_a2-E_F8)/NOISE,0.0011*r_a2/NOISE,f"corr 0.9950 -> rms≈{r_a2:.3f} САМЫЙ РАЗВЯЗАННЫЙ")
# F9 + T3  и  F9 + F8  (F9 = гамма 0.08: vs F8 +1.10ш, sd разн vs F8 1.51ш)
dF9=-1.10*NOISE
p=P([dF9, (1.647603-E_F8)],[0.0, float(np.hypot(0.0011*r_t3,sd_over_diff))])
p9=P([dF9],[0.0])
print(f"{'F9 + T3':34s}{(1.647603-E_F8)/NOISE-(-1.10):+9.1f}{sd_t3_full/NOISE:9.2f}{p*100:9.2f}%"
      f"{(p-p9)*100:+9.3f}   к одиночному F9 ({p9*100:.2f} %)")
p=P([dF9,0.0],[0.0,1.51*NOISE])
print(f"{'F9 + F8':34s}{1.10:+9.1f}{1.51:9.2f}{p*100:9.2f}%{(p-p9)*100:+9.3f}   к одиночному F9 ({p9*100:.2f} %)")

print("\n=== E2. ГРАНИЧНОЕ ОТСТАВАНИЕ dE*: при каком отставании развязка ещё даёт +1 п.п. ===")
print(f"{'sd разн,ш':>10}{'dE*(+1пп),ш':>13}{'реальный кандидат':>26}{'его dE,ш':>10}{'во сколько раз мимо':>21}")
def dstar(sdn, target=0.01):
    sd=sdn*NOISE
    if P([0.0,0.0],[0.0,sd])-BASE < target: return None
    lo,hi=0.0,0.02
    for _ in range(45):
        m=0.5*(lo+hi)
        if P([0.0,m],[0.0,sd])-BASE>=target: lo=m
        else: hi=m
    return 0.5*(lo+hi)/NOISE
for sdn,cand,dEc in [(0.0011*r_t3/NOISE,"T3 (corr .9993)",(1.647603-E_F8)/NOISE),
                     (sd_t3_full/NOISE,"T3 c переоценкой",(1.647603-E_F8)/NOISE),
                     (0.0011*r_h2/NOISE,"H2_edge_p1",(E_h2-E_F8)/NOISE),
                     (0.0011*r_a2/NOISE,"A2_probe_s1_gmv",(E_a2-E_F8)/NOISE),
                     (4.5,"—",None),(9.1,"—",None),(13.6,"—",None)]:
    ds=dstar(sdn)
    s1=f"{ds:13.1f}" if ds else f"{'нет (<1пп)':>13}"
    if dEc is None: print(f"{sdn:10.2f}{s1}{cand:>26}{'—':>10}{'—':>21}")
    else: print(f"{sdn:10.2f}{s1}{cand:>26}{dEc:10.1f}"
                f"{(dEc/ds if ds else float('inf')):21.1f}")

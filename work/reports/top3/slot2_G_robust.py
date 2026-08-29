# -*- coding: utf-8 -*-
"""G: обратная граница (какая развязка нужна кандидату) + устойчивость по сидам/приорам."""
import json, os, sys
import numpy as np
sys.path.insert(0,"/Users/alexanderkondakov/ozon-cup/work/scripts")
import p_top3 as PT
from p_top3 import Objective, MU_US, SIGMA_US, NOISE
E_F8=1.646203
def mk(ns=300_000, sf=2026, su=777):
    o=Objective(ns=ns,seed_field=sf,seed_us=su); return o
o=mk(); rng=np.random.default_rng(11)
z=o.z; ea=o.w; eb=rng.standard_normal(o.ns)
def Ppair(dE,sd):
    g1=E_F8+SIGMA_US*z; g2=g1+dE+sd*eb
    return float((np.minimum(g1,g2)<o.c3).mean())
BASE=float((E_F8+SIGMA_US*z<o.c3).mean())
print(f"база {BASE*100:.2f} %")
print("\n=== G1. ОБРАТНАЯ ГРАНИЦА: какая развязка нужна кандидату с его отставанием ===")
print(f"{'кандидат':24s}{'dE,ш':>8}{'оценка sd,ш':>13}{'нужно sd,ш':>12}{'во сколько раз':>16}")
def sd_needed(dE,target=0.01):
    lo,hi=0.0,0.02
    if Ppair(dE,hi)-BASE<target: return None
    for _ in range(50):
        m=0.5*(lo+hi)
        if Ppair(dE,m)-BASE>=target: hi=m
        else: lo=m
    return 0.5*(lo+hi)/NOISE
for nm,dEn,sde in [("T3_g1_redose_044",63.6,7.89),("H2_edge_p1",145.2,4.57),
                   ("R5_shade",80.8,3.74),("A2_probe_s1_gmv",468.2,8.22),
                   ("V3_canon",71.9,3.30),("F7_priv",14.7,4.59)]:
    need=sd_needed(dEn*NOISE)
    print(f"{nm:24s}{dEn:8.1f}{sde:13.2f}{need:12.1f}{need/sde:16.1f}")
print("\n=== G2. УСТОЙЧИВОСТЬ по сидам поля/нас (5 сидов) ===")
for tag,dEn,sdn in [("F8+T3",63.6,7.89),("F8+«идеальный» sd=13.6ш при dE=0",0.0,13.6),
                    ("F8+второй ровно как F8 (dE=0,sd=5ш)",0.0,5.0)]:
    vs=[]
    for s in range(5):
        oo=mk(sf=101*(s+3),su=37*(s+3)); r2=np.random.default_rng(500+s)
        z2=oo.z; e2=r2.standard_normal(oo.ns)
        b=float((E_F8+SIGMA_US*z2<oo.c3).mean())
        g1=E_F8+SIGMA_US*z2; g2=g1+dEn*NOISE+sdn*NOISE*e2
        vs.append(float((np.minimum(g1,g2)<oo.c3).mean())-b)
    print(f"  {tag:42s} прирост {min(vs)*100:+.3f}..{max(vs)*100:+.3f} п.п.")
print("\n=== G3. Приоры phi: меняет ли знак вывод по слоту-2? ===")
for tag,(pa,pb) in {"Beta(1.5,12) базовый":(1.5,12.),"Beta(2,10)":(2.,10.),
                    "Beta(1,15)":(1.,15.),"Beta(3,6)":(3.,6.)}.items():
    PT.PHI_A,PT.PHI_B=pa,pb
    oo=Objective(ns=300_000); r2=np.random.default_rng(9)
    z2=oo.z; e2=r2.standard_normal(oo.ns)
    b=float((E_F8+SIGMA_US*z2<oo.c3).mean())
    def pp(dE,sd):
        g1=E_F8+SIGMA_US*z2; g2=g1+dE*NOISE+sd*NOISE*e2
        return float((np.minimum(g1,g2)<oo.c3).mean())
    print(f"  {tag:22s} база {b*100:6.2f} %  F8+T3 {(pp(63.6,7.89)-b)*100:+6.3f}  "
          f"F8+(0,13.6ш) {(pp(0,13.6)-b)*100:+6.2f}  F8+(5ш,6ш) {(pp(5,6)-b)*100:+6.2f}")
PT.PHI_A,PT.PHI_B=1.5,12.

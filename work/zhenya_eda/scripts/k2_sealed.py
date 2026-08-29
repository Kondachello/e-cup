# -*- coding: utf-8 -*-
"""K2. ЗАПЕЧАТАННЫЙ ПРОГНОЗ. Считается ДО вскрытия привата, коммит = печать."""
import math, sys, json
import numpy as np
sys.stdout.reconfigure(encoding="utf-8")
NOISE=0.000022
rng=np.random.default_rng(2908); NS=300_000

F7_PUB, T3_PUB = 1.6458557351, 1.6469321992541033
SHOW11 = 1.6440063524                 # наша витрина, НЕ финалист
GAIN_F7 = 0.0011                      # E[приват F7] ниже E[приват T3] на столько


OVER_T3_LO, OVER_T3_HI = 0.000416, 0.000680
print("=== 1. НАШИ ДВА ФИНАЛИСТА ===")
print(f"  F7_priv паблик {F7_PUB:.7f}   T3 паблик {T3_PUB:.7f}")
for lab,ov in (("низкая",OVER_T3_LO),("высокая",OVER_T3_HI)):
    print(f"  переоценка T3 {lab} {ov:.6f}: E[приват T3]={T3_PUB+ov:.6f}  "
          f"E[приват F7]={T3_PUB+ov-GAIN_F7:.6f}")
OVER=(OVER_T3_LO+OVER_T3_HI)/2
E_T3=T3_PUB+OVER; E_F7=E_T3-GAIN_F7
# дисперсия: sd приватного выигрыша F7 над T3 (из моей дисперсии kappa_Q, Часть J,
# масштабирована на выросшее число осей: 8 живых + редозы + микро)
SD_GAIN=0.00030
print(f"\n  ЦЕНТРАЛЬНЫЙ ПРОГНОЗ: E[приват F7]={E_F7:.6f}, E[приват T3]={E_T3:.6f}")
print(f"  sd выигрыша F7 над T3 = {SD_GAIN:.5f}")
g=rng.normal(GAIN_F7,SD_GAIN,NS)
f7=E_T3-g; t3=np.full(NS,E_T3)
for n,v in (("F7_priv",f7),("T3",t3)):
    if v.std()>0:
        print(f"  {n:8s} 5-95%: [{np.percentile(v,5):.6f}, {np.percentile(v,95):.6f}]")
    else:
        print(f"  {n:8s} точечно {v[0]:.6f} (по построению — это база отсчёта)")
print(f"  P(F7 лучше T3 на привате) = {(f7<t3).mean()*100:.1f}%")

# --- поле
OUR_FAKE=(F7_PUB+OVER)-SHOW11        # согласованно: витрина -> наш ПРИВАТ
BIAS={"A":lambda n:0.0 if n<=9 else OUR_FAKE*math.log(n/9)/math.log(57/9),
      "B":lambda n:0.33*NOISE*n, "C":lambda n:0.0}
PW={"A":0.25,"B":0.55,"C":0.20}
print(f"\n=== 2. МОДЕЛЬ ПОЛЯ (bias откалиброван на нашем разрыве витрина->приват "
      f"{OUR_FAKE:.6f}) ===")
print(f"  вероятности миров: A={PW['A']}, B={PW['B']}, C={PW['C']}")
print(f"  A — поле подгоняло как мы; B — умеренно (0.33*шум*n); C — не подгоняло")
best=np.minimum(f7,t3)
ranks={}
for w in ("A","B","C"):
    cols=[]
    for n,p,s in TEAMS:
        b=BIAS[w](s); cols.append(rng.normal(p+b,max(0.5*b,0.00020),NS))
    for _ in range(8):                       # хвост 8-15 мест
        b=BIAS[w](40); cols.append(rng.normal(1.64655+b,max(0.5*b,0.00020),NS))
    FL=np.column_stack(cols)
    r=1+(FL<best[:,None]).sum(1); ranks[w]=r
    print(f"\n  --- мир {w} ---")
    for n,p,s in TEAMS:
        print(f"    {n:18s} паблик {p:.7f} n={s:3d} -> pop {p+BIAS[w](s):.6f}")
    print(f"    {'МЫ (F7)':18s}                          -> pop {E_F7:.6f}   "
          f"медианный ранг {int(np.median(r))}")

print(f"\n=== 3. ПРОГНОЗ НАШЕГО МЕСТА (смесь миров) ===")
mix=np.concatenate([ranks[w][:int(NS*PW[w])] for w in ("A","B","C")])
for lab,cond in (("P(1 место)",mix==1),("P(2-3)",(mix>=2)&(mix<=3)),
                 ("P(4-5)",(mix>=4)&(mix<=5)),("P(6+)",mix>=6)):
    print(f"  {lab:12s} {cond.mean()*100:5.1f}%")
print(f"  P(топ-3) {(mix<=3).mean()*100:.1f}%   P(топ-5) {(mix<=5).mean()*100:.1f}%   "
      f"медиана места {int(np.median(mix))}")
print(f"  по мирам: A медиана {int(np.median(ranks['A']))}, "
      f"B {int(np.median(ranks['B']))}, C {int(np.median(ranks['C']))}")
json.dump(dict(E_F7=E_F7,E_T3=E_T3,over=OVER,p_top3=float((mix<=3).mean()),
    p_top5=float((mix<=5).mean()),med=int(np.median(mix))),
    open("out/k2_sealed.json","w",encoding="utf-8"),ensure_ascii=False,indent=1)

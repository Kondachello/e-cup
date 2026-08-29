# -*- coding: utf-8 -*-
"""N21. Модель поля соперников: постериор популяционного скора pop_i.

pop_i = public_i + bias(n_i) + eps_i,  bias >= 0 (приват ХУЖЕ паблика).

Калибровка bias — тремя мирами, потому что это НЕНАБЛЮДАЕМОЕ:
     (витрина SHOW10: её преимущество над честным F5 фиктивно целиком).
  B «умеренный»: bias = 0.33 * 0.000022 * n — наша собственная калибровка
     (честная цепочка: переоценка 0.000416 при посылках = 18.9 шума).
  C «поле не подгоняло»: bias = 0 у всех, кроме нас.
"""
import math, sys, json
import numpy as np
sys.stdout.reconfigure(encoding="utf-8")
NOISE=0.000022
SHOW10, F5_PUB = 1.6446514942, 1.6462975719
OUR_FAKE = F5_PUB - SHOW10          # преимущество витрины над честным = фиктивно целиком
print(f"якорь A: наша витрина SHOW10 на {OUR_FAKE:.6f} лучше честного F5 — это чистая подгонка")

def biasA(n): return 0.0 if n<=9 else OUR_FAKE*math.log(n/9)/math.log(57/9)
def biasB(n): return 0.33*NOISE*n
def biasC(n): return 0.0
WORLDS={"A: поле подгоняло как мы":biasA,"B: умеренная подгонка":biasB,"C: поле честное":biasC}

# наши кандидаты: переоценка ОТНОСИТЕЛЬНО T3-цепочки
OVER_T3 = 0.000416                   
GAIN_F5 = 0.000923                   # E[priv выигрыш F5 над T3] (Саша, гросс-GLS)
T3_PUB  = 1.6469322
OVER_F5 = OVER_T3 - (T3_PUB - F5_PUB - GAIN_F5) - (T3_PUB-F5_PUB) + (T3_PUB-F5_PUB)
OVER_F5 = OVER_T3 + (T3_PUB - F5_PUB) - GAIN_F5   # из E[priv F5]=E[priv T3]-GAIN
print(f"ПОПРАВКА К МОЕЙ ЧАСТИ I: я применил переоценку T3-цепочки ко всем файлам.")
print(f"  У F5 она НИЖЕ — редозы её и снимают:")
print(f"  переоценка T3 = {OVER_T3:.6f},  F5 = {OVER_F5:.6f}  (разница {OVER_T3-OVER_F5:+.6f})")
print(f"  E[приват T3] = {T3_PUB+OVER_T3:.7f}")
print(f"  E[приват F5] = {F5_PUB+OVER_F5:.7f}   (в Части I я писал {F5_PUB+OVER_T3:.7f} — завышено)")

print(f"\n=== ПОСТЕРИОРЫ ПОПУЛЯЦИОННОГО СКОРА (меньше = лучше) ===")
for wn,bf in WORLDS.items():
    print(f"\n--- мир {wn} ---")
    rows=[]
    for n,p,s in TEAMS:
        b=bf(s) if "МЫ" not in n else None
        rows.append((n,p,s,b))
    # нас считаем по честному F5, а не по витрине
    out=[]
    for n,p,s,b in rows:
        pop = (F5_PUB+OVER_F5) if "МЫ" in n else p+b
        out.append((n,p,s,pop))
    for n,p,s,pop in sorted(out,key=lambda r:r[3]):
        tag=" <-- МЫ (по честному F5)" if "МЫ" in n else ""
        print(f"  {n:18s} паблик {p:.7f} n={s:3d}  pop {pop:.7f}{tag}")
    ourrank=sorted(out,key=lambda r:r[3]).index([o for o in out if "МЫ" in o[0]][0])+1
    print(f"  НАШ РАНГ в этом мире: {ourrank}")
json.dump(dict(over_f5=OVER_F5,over_t3=OVER_T3),open("out/n21_field.json","w",encoding="utf-8"),
          ensure_ascii=False,indent=1)

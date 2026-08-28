# -*- coding: utf-8 -*-
""" / K1.2. Пересчёт ранга под ИСПРАВЛЕННУЮ переоценку и под F6."""
import math, sys, json
import numpy as np
sys.stdout.reconfigure(encoding="utf-8")
NOISE=0.000022
F5_PUB,F6_PUB,T3_PUB,SHOW10=1.6462975719,1.6459765591,1.6469322,1.6446514942
GAIN_F6_over_F5=0.000144
OVER={"низкая (только +сегменты)":0.000764,"высокая (+8 шагов до )":0.001028,
      "старая (Часть G)":0.000416}
OUR_FAKE=F5_PUB-SHOW10
BIAS={"A":lambda n:0.0 if n<=9 else OUR_FAKE*math.log(n/9)/math.log(57/9),
      "B":lambda n:0.33*NOISE*n,"C":lambda n:0.0}
print("ВАЖНО: наша переоценка вычитается ИЗ НАШЕГО ЖЕ прогноза, а bias(n) соперников")
print("оценивается отдельно. Если наша переоценка выросла вдвое, то и bias(n) поля,")
print("откалиброванный на НАШЕЙ витрине, тоже надо пересмотреть — см. примечание внизу.\n")
for ovn,ov in OVER.items():
    pop_f6=F6_PUB+ov-0  # F6 лучше F5 на приват на 0.000144
    pop_f5=F5_PUB+ov
    pop_f6=pop_f5-GAIN_F6_over_F5
    print(f"=== переоценка: {ovn} = {ov:.6f} ===")
    print(f"  E[приват F5] = {pop_f5:.6f}   E[приват F6] = {pop_f6:.6f}")
    for w in ("A","B","C"):
        rows=[(n,p+BIAS[w](s)) for n,p,s in TEAMS]+[("МЫ (F6)",pop_f6)]
        rows.sort(key=lambda r:r[1])
        rk=[i for i,r in enumerate(rows,1) if "МЫ" in r[0]][0]
        print(f"    мир {w}: ранг {rk}  |  " + " ".join(
            f"{n.split()[0][:8]}={v:.5f}" for n,v in rows[:4]))
    print()
print("ПРИМЕЧАНИЕ О СОГЛАСОВАННОСТИ. bias(n) в мире A откалиброван как разрыв между")
print("нашей витриной SHOW10 и нашим ЧЕСТНЫМ F5. Но если у F5 самого переоценка")
print("0.0008-0.0010, то полный разрыв «витрина -> приват» = 0.00165+0.0009 = 0.0025,")
print("и bias(57) надо брать именно таким. Тогда поле проседает СИЛЬНЕЕ нашего, и")
print("наш ранг в мире A улучшается. Ниже — согласованный вариант.\n")
for ovn,ov in [("низкая",0.000764),("высокая",0.001028)]:
    FULL=OUR_FAKE+ov
    biasA=lambda n: 0.0 if n<=9 else FULL*math.log(n/9)/math.log(57/9)
    pop=F5_PUB+ov-GAIN_F6_over_F5
    rows=[(n,p+biasA(s)) for n,p,s in TEAMS]+[("МЫ (F6)",pop)]
    rows.sort(key=lambda r:r[1])
    rk=[i for i,r in enumerate(rows,1) if "МЫ" in r[0]][0]
    print(f"  согласованный мир A, переоценка {ovn} ({ov:.6f}): bias(57)={FULL:.6f}, "
          f"наш ранг {rk}")
    for n,v in rows[:4]: print(f"      {n:12s} {v:.6f}")

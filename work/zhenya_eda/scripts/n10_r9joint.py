# -*- coding: utf-8 -*-
"""N10. R9 как ОДНО направление (джойнт), а не сумма независимых осей.

c линеен по направлению => c_D = sum a_i c_i = sum a_i kappa_i q_i  (ТОЧНО, без Грама).
q_D замерен = 0.001515. Публичный kappa_D проверяем по факту скоров R8->R9.
Приват: E[c_Q,D] = sum a_i w_p,i kappa_i q_i.
"""
import math, sys, json
import numpy as np
sys.stdout.reconfigure(encoding="utf-8")
NP_, NOISE = 50_000, 0.000022
law = lambda q, F0: 0.894*F0/math.sqrt(NP_*q)
R8, R9 = 1.6468337517, 1.6463209943
Q_D = 0.001515
TAU = 0.0148

# --- публичный c_D из ФАКТА скоров (доза 1.0 применённого джойнта)
c_fact = (R8**2 - R9**2 + Q_D)/2
print("=== R9 КАК ОДНО НАПРАВЛЕНИЕ ===")
print(f"  факт: R8={R8:.7f} -> R9={R9:.7f}, q_D={Q_D:.6f}")
print(f"  c_D (из факта)      = {c_fact:+.6f}   kappa_D = {c_fact/Q_D:+.3f}")

# --- c_D из компонент (проверка тождества линейности)
c_comp = sum(A[n]*Z[n][1]*Z[n][0] for n in A)
print(f"  c_D (из компонент)  = {c_comp:+.6f}   kappa_D = {c_comp/Q_D:+.3f}")
print(f"  расхождение: {abs(c_fact-c_comp):.6f} "
      f"({'тождество держится' if abs(c_fact-c_comp)<0.0002 else 'ОСИ НЕ ТЕ или q_D другой'})")

# --- приватное ожидание c_Q,D по компонентам
print(f"\n  {'ось':5s}{'q':>9}{'kappa':>8}{'w':>7}{'w_p':>8}{'a':>8}{'a*k*q':>11}{'a*wp*k*q':>11}")
cq = 0.0
for n in A:
    qq, kk = Z[n]; sg = law(qq, R8)
    w = TAU**2/(TAU**2+sg*sg); wp = 1.25*w-0.25
    t1 = A[n]*kk*qq; t2 = A[n]*wp*kk*qq; cq += t2
    print(f"  {n:5s}{qq:9.5f}{kk:+8.3f}{w:7.3f}{wp:+8.3f}{A[n]:+8.3f}{t1:+11.6f}{t2:+11.6f}")
print(f"  {'сумма':5s}{'':>39}{c_comp:+11.6f}{cq:+11.6f}")

F0 = R8
g_pub = (2*c_fact - Q_D)/(2*F0)
g_pri = (2*cq - Q_D)/(2*F0)
print(f"\n  публичный урожай (факт):  {g_pub:+.6f} = {g_pub/NOISE:+.1f} шума")
print(f"  ожидание привата:         {g_pri:+.6f} = {g_pri/NOISE:+.1f} шума")
print(f"  Саша: -0.000344  -> {'ПОДТВЕРЖДАЮ' if abs(g_pri+0.000344)<0.00012 else 'расходимся'}")

# --- оптимальный масштаб s на тот же джойнт
s_opt = cq/Q_D
print(f"\n  оптимальный множитель s* на R9-джойнт = c_Q/q_D = {s_opt:+.3f}")
print(f"  E[private] при s*: {(s_opt*cq)/(F0):+.6f} = {(s_opt*cq)/F0/NOISE:+.1f} шума")
print(f"  => при tau_z=0.0148 R9-добавка привату ВРЕДНА; правильный s* близок к нулю"
      if s_opt < 0.15 else "")
json.dump(dict(c_fact=c_fact,c_comp=c_comp,kappa_D=c_fact/Q_D,g_pub=g_pub,g_pri=g_pri,
    s_opt=s_opt), open("out/n10_r9joint.json","w",encoding="utf-8"), ensure_ascii=False, indent=1)

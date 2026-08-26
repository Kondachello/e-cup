# -*- coding: utf-8 -*-
"""N5. Приватная EV-доза финала: пересчёт R9 под максимум E[приват]."""
import math, sys, json
import numpy as np
sys.stdout.reconfigure(encoding="utf-8")
F0, NP_, NA = 1.6470, 50_000, 250_000
FR = NP_/NA; NOISE = 0.000022
sig = lambda q, tau=None: 0.894*F0/math.sqrt(NP_*q)

print("=== ЗАКОН (вывод в отчёте, раздел 3) ===")
print("  kappa_P = kappa_T + eps      (публика 50k)")
print("  kappa_Q = kappa_T - 0.25*eps (приват 200k = ДОПОЛНЕНИЕ публики)")
print("  E[kappa_Q|kappa_P] = mu + w_p (kappa_P - mu),  w_p = 1.25 w - 0.25")
print("  sd[kappa_Q|kappa_P] = 1.25 tau sqrt(1-w)")
print("  при mu=0: доза_приват / доза_паблик = s = 1.25 - 0.25/w")
print("            E[приват]/урожай_паблик   = 0.5(3w-1)/(2-w)\n")

print("=== 1. ЧУВСТВИТЕЛЬНОСТЬ R9 К tau МИКРО-ОСЕЙ (mu = 0) ===")
print("(9 осей разложения; tau_z ОБЯЗАН оцениваться по самим 20 Z-замерам,")
print(" а не браться из реестра предложенных осей — см. раздел 2 отчёта)\n")
print(f"{'q оси':>8} | " + " | ".join(f"tau={t:<5.3f}" for t in (0.196,0.10,0.07,0.05,0.04,0.03)))
print(f"{'':>8} | " + " | ".join(f"{'w / s':>11}" for _ in range(6)))
for q in (0.05, 0.03, 0.02, 0.01, 0.005):
    s_ = sig(q); cells = []
    for tau in (0.196, 0.10, 0.07, 0.05, 0.04, 0.03):
        w = tau*tau/(tau*tau + s_*s_); s = 1.25 - 0.25/w if w > 1e-9 else -9
        cells.append(f"{w:.2f}/{max(s,0):5.2f}")
    print(f"{q:8.3f} | " + " | ".join(f"{c:>11}" for c in cells))

print(f"\n=== 2. ДОЛЯ ПУБЛИЧНОГО УРОЖАЯ, КОТОРАЯ ДОЙДЁТ ДО ПРИВАТА ===")
print(f"{'w':>6}{'s*':>8}{'при дозе паблика':>18}{'при дозе приват':>17}")
for w in (0.99, 0.95, 0.90, 0.85, 0.80, 0.70, 0.60, 0.50, 0.40, 1/3, 0.30, 0.20):
    s = 1.25 - 0.25/w
    r_pub = 0.5*(3*w-1)/(2-w)
    r_pri = max(1.25*w-0.25, 0)**2/(w*(2-w))
    print(f"{w:6.2f}{s:8.3f}{r_pub:18.3f}{r_pri:17.3f}")
print("  w = 1/3 — дозa паблика даёт РОВНО НОЛЬ на привате")
print("  w < 1/3 — дозa паблика даёт ОТРИЦАТЕЛЬНЫЙ приватный EV")

print(f"\n=== 3. R9: сценарии (публичный урожай R8->R9 = 0.000516) ===")
GPUB = 0.000516
print(f"{'сценарий':34s}{'w':>7}{'s*':>7}{'E[приват]':>12}{'в шумах':>9}")
SC = [("tau=0.196 (реестр), q~0.05", 0.05, 0.196), ("tau=0.196, q~0.02", 0.02, 0.196),
      ("tau=0.10, q~0.05", 0.05, 0.10), ("tau=0.07, q~0.05", 0.05, 0.07),
      ("tau=0.05, q~0.05", 0.05, 0.05), ("tau=0.03, q~0.05", 0.05, 0.03)]
rows = []
for nm, q, tau in SC:
    s_ = sig(q); w = tau*tau/(tau*tau + s_*s_)
    s = 1.25 - 0.25/w
    ev = GPUB * max(1.25*w-0.25, 0)**2/(w*(2-w))
    rows.append((nm, w, s, ev))
    print(f"{nm:34s}{w:7.3f}{max(s,0):7.3f}{ev:12.6f}{ev/NOISE:9.1f}")

print(f"\n=== 4. СТОП-ПРАВИЛО ===")
n1 = json.load(open("out/n1_prior.json", encoding="utf-8"))
app = [r for r in n1["rows"] if abs(r["dose"]) > 1e-9]
over_m1 = sum(r["over"] for r in app)
print(f"  переоценка эпохи M1.. + R8 (16 осей):  {over_m1:+.6f} = {over_m1/NOISE:.1f} шума")
for nm, w, s, ev in rows[:1] + rows[3:4]:
    o = GPUB - ev
    print(f"  + R9 при «{nm}»: переоценка {o:+.6f}, ИТОГО {over_m1+o:+.6f} "
          f"= {(over_m1+o)/NOISE:.1f} шума")
print(f"\n  порог доктрины 0.000500 = {0.0005/NOISE:.1f} шума")

print(f"\n=== 5. ИНТЕРВАЛ ПРИВАТА ДЛЯ R9 ===")
print("  Var(приват) = 3.125*w(1-w)*tau^2*q*g_pub / (F0*(2-w))   [вывод в отчёте]")
print(f"{'сценарий':28s}{'E[приват]':>12}{'sd':>10}{'95% интервал':>24}")
for nm, q, tau in SC:
    s_ = sig(q); w = tau*tau/(tau*tau + s_*s_)
    ev = GPUB * max(1.25*w-0.25, 0)**2/(w*(2-w))
    var = 3.125*w*(1-w)*tau*tau*q*GPUB/(F0*(2-w))
    sd = math.sqrt(var)
    print(f"{nm:28s}{ev:12.6f}{sd:10.6f}   [{ev-1.96*sd:+.6f}, {ev+1.96*sd:+.6f}]")

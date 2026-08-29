# -*- coding: utf-8 -*-
"""N14. Финальная пара через E[min] с sigma_d из МОЕЙ дисперсии kappa_Q
и сверка с законом трека 4 (sigma_d = 0.0011*rms).

Замечание по природе sigma_d: обе посылки судятся на ОДНОЙ приватной 200k,
поэтому их разность ДЕТЕРМИНИРОВАНА. sigma_d — это наша НЕОПРЕДЕЛЁННОСТЬ о ней,
и она равна дисперсии приватного выигрыша по разделяющим осям.
"""
import math, sys, json
import numpy as np
from math import erf, exp, pi, sqrt
sys.stdout.reconfigure(encoding="utf-8")
F0, NP_, NOISE = 1.6470, 50_000, 0.000022
FPC = math.sqrt(0.8)
Phi = lambda z: 0.5*(1+erf(z/sqrt(2)))
phi = lambda z: exp(-z*z/2)/sqrt(2*pi)

#(имя, q, kappa, доза, tau приора)
TAU_REG, TAU_SEG, TAU_Z = 0.196, 0.081, 0.0148
# (имя, q, kappa, b_старое, b_новое, tau) — инкрементальный переход b_old -> b_new
AX=[("mdl_talc (шаг R8)",   0.02490, 0.113, 0.000, 0.124, TAU_REG),
    ("редоза mdl_amber",     0.00066, 0.803, 0.803, 0.413, TAU_REG),
    ("редоза mdl_halite",     0.00066, 0.756, 0.756, 0.403, TAU_REG),
    ("редоза mdl_gneis2",     0.00007, 1.317, 1.002, 0.130, TAU_REG),
    (" ядро",      0.02086, 0.126, 0.000, 0.138, TAU_SEG),
    (" спящие",    0.01173, 0.157, 0.000, 0.155, TAU_SEG),
    (" промо",     0.00440, 0.286, 0.000, 0.258, TAU_SEG)]
MICRO = 0.00004    # микровектор: приватно-оптимальные дозы 9 осей (Саша, по моим формулам)
def sig(q): return 0.894*F0/math.sqrt(NP_*q)

print("=== ПРИВАТНАЯ БУХГАЛТЕРИЯ  ПРОТИВ T3 (мои приоры) ===")
print(f"{'ось':14s}{'q':>9}{'доза':>8}{'w':>7}{'E[kQ]':>8}{'E[приват]':>11}{'sd':>10}")
mu=0.0; var=0.0
for n,q,kp,bo,bn,tau in AX:
    s_=sig(q); w=tau*tau/(tau*tau+s_*s_); wp=1.25*w-0.25
    prior_mu = 0.041 if tau==TAU_SEG else 0.309
    ekq=prior_mu+wp*(kp-prior_mu); db=bn-bo
    g=q*(2*ekq*db-(bn*bn-bo*bo))/(2*F0); mu+=g          # ИНКРЕМЕНТАЛЬНО
    v=(db*q/F0)**2 * 1.5625*(1-w)*tau*tau; var+=v
    print(f"{n:14s}{q:9.5f}{db:+8.3f}{w:7.3f}{ekq:+8.3f}{g:+11.6f}{math.sqrt(v):10.6f}")
mu+=MICRO
print(f"{'микровектор':14s}{0.00152:9.5f}{'':>8}{'':>7}{'':>8}{MICRO:+11.6f}{0.000002:10.6f}")
sd=math.sqrt(var)
print(f"{'ИТОГО':14s}{'':>32}{mu:+11.6f}{sd:10.6f}")
print(f"  E[private выигрыш  над T3] = {mu:+.6f} = {mu/NOISE:+.1f} шума, sd = {sd:.6f}")
print(f"  (Саша заявлял +0.00045; расхождение — из-за приора семейства для //)")

print(f"\n=== E[min] ПАРЫ (меньше = лучше, RMSLE) ===")
def Emin(d, sd_):
    """d = mu_2 - mu_1 (насколько 2-й хуже); возвращает E[min]-mu_1"""
    if sd_<1e-12: return min(0.0,d)
    z=d/sd_
    return -(d*Phi(-z)+sd_*phi(z)) + d*Phi(-z) - sd_*phi(z) + 0  # разложим явно ниже
def emin_gain(d, sd_):
    """ожидаемый выигрыш от наличия ВТОРОГО кандидата (насколько E[min] ниже лучшего)"""
    if sd_<1e-12: return 0.0
    z=d/sd_
    return sd_*phi(z) - d*Phi(-z)
SD_MY=sd
RMS_D=0.0  # оценим rms разности из q осей: rms^2 = sum b_i^2 q_i (оси примерно ортогональны)
RMS_D=math.sqrt(sum(((bn-bo)**2*q) for _,q,_,bo,bn,_ in AX))
SD_OLYA=0.0011*RMS_D
print(f"  rms разности -T3 (из доз и q) = {RMS_D:.4f}")
print(f"  sigma_d по моей дисперсии kappa_Q = {SD_MY:.6f}")
print(f"  sigma_d по закону трека 4 (0.0011*rms) = {SD_OLYA:.6f}  "
      f"-> отношение {SD_MY/max(SD_OLYA,1e-12):.2f}")
print(f"\n{'пара':28s}{'разрыв d':>11}{'sigma_d':>10}{'выигрыш от 2-го':>17}")
for lab,d,sdd in [(" + T3 (моя sigma_d)", mu, SD_MY),
                  (" + T3 (sigma_d Оли)", mu, SD_OLYA)]:
    print(f"{lab:28s}{d:+11.6f}{sdd:10.6f}{emin_gain(d,sdd):+17.6f}")
print(f"\n  d — насколько T3 ХУЖЕ  по ожиданию ({mu:+.6f}).")
print(f"  Выигрыш от второго кандидата = sigma_d*phi(z) - d*Phi(-z), z=d/sigma_d.")
json.dump(dict(mu=mu,sd=SD_MY,rms=RMS_D,sd_olya=SD_OLYA,
    gain_my=emin_gain(mu,SD_MY),gain_olya=emin_gain(mu,SD_OLYA)),
    open("out/n14_pair.json","w",encoding="utf-8"),ensure_ascii=False,indent=1)

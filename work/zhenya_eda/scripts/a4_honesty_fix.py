# -*- coding: utf-8 -*-
"""A4. АУДИТ, аккуратная версия: q выводится из ЗАЯВЛЕННОЙ значимости, а не гадается.
   sigma_отч = kappa / (число сигм);  q = (FPC*F0/sigma_true)^2 / n_pub."""
import math, sys, json
import numpy as np
sys.stdout.reconfigure(encoding="utf-8")
F0,NP_,NOISE=1.6470,50_000,0.000022
FPC=math.sqrt(0.8)
MU_S,TAU_S=0.013,0.086          # приор сегментного семейства (18 точек)

def q_from_sigma(sig_rep):      # sigma в отчётах Саши — БЕЗ fpc
    return (F0/sig_rep)**2/NP_
def dose(kp,sig_rep,mu=MU_S,tau=TAU_S):
    s=sig_rep*FPC; w=tau*tau/(tau*tau+s*s); return mu+(1.25*w-0.25)*(kp-mu), w

# (имя, kappa, сигм) — из predict_lb/коммитов; sigma = kappa/сигм
SEG=[(" ядро recency",0.126,2.47),(" спящие",0.157,2.31),(" промо",0.286,2.58),
     (" никогда-не-покупавшие",-0.197,2.94),(" спящие-браузеры",0.372,3.4),
     (" горячие корзины",0.196,3.0)]
print("=== СЕГМЕНТНЫЕ ДОЗЫ: q выведен из значимости ===")
print(f"{'ось':26s}{'k_P':>8}{'сигм':>6}{'sigma':>8}{'q':>9}{'w':>7}{'доза':>8}{'переоц':>10}{'шумов':>7}")
tot=0.0
for n,kp,ns in SEG:
    sig=abs(kp)/ns; q=q_from_sigma(sig); b,w=dose(kp,sig)
    ov=q*abs(b)*abs(kp-b)/F0; tot+=ov
    print(f"{n:26s}{kp:+8.3f}{ns:6.1f}{sig:8.4f}{q:9.5f}{w:7.3f}{b:+8.3f}{ov:10.6f}{ov/NOISE:7.1f}")
print(f"{'ИТОГО сегментные (6 осей)':26s}{'':>38}{tot:10.6f}{tot/NOISE:7.1f}")
print("  (метрономы -0.150 и фаза цикла +0.138 без заявленной значимости — не считаю)")

per=1.25*(1-0.2)*F0/NP_          # безусловная переоценка на одно подогнанное направление, g=1
print(f"\n=== ПОЛНЫЙ СЧЁТ (консервативно, g=1) ===")
M1=0.000416
print(f"  эпоха M1 (посчитано в Части G, условно по замеренным k_P): {M1:.6f} = {M1/NOISE:4.1f} шума")
print(f"  сегментные дозы F1..F6 (6 осей, условно):                  {tot:.6f} = {tot/NOISE:4.1f} шума")
for k,lab in ((8,"все восемь шагов до  подогнаны по LB"),(4,"половина из них"),(0,"ни один")):
    T=M1+tot+k*per
    print(f"  + до-M1: {lab:38s} {T:.6f} = {T/NOISE:4.1f} шума")
print(f"\n  безусловная цена одного подогнанного направления = {per:.3e} = {per/NOISE:.1f} шума")
print(f"  порог доктрины = 0.000500 = 22.7 шума")
T_lo, T_hi = M1+tot, M1+tot+8*per
print(f"\n  ДИАПАЗОН НАКОПЛЕННОЙ ПЕРЕОЦЕНКИ: {T_lo:.6f} … {T_hi:.6f}  "
      f"({T_lo/NOISE:.0f} … {T_hi/NOISE:.0f} шума)")
print(f"  В Части G стояло {M1:.6f} — занижено минимум ВДВОЕ.")
print(f"\n  E[приват F5]: 1.6462976 + переоценка = {1.6462976+T_lo:.6f} … {1.6462976+T_hi:.6f}")
print(f"  (в Части J стояло 1.6464252)")
json.dump(dict(seg=tot,m1=M1,per=per,lo=T_lo,hi=T_hi),
          open("out/a4_honesty.json","w",encoding="utf-8"),ensure_ascii=False,indent=1)

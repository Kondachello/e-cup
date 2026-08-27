# -*- coding: utf-8 -*-
"""N7. Чистое стоп-правило (первичные оси, без двойного счёта /) + коэффициенты
передозировки mdl_amber/mdl_gabbro/mdl_halite/mdl_realgr/mdl_flint/mdl_gypsum/mdl_gneis2 под приватный оптимум.

X1(де-шринк P до ПОЛНОЙ kappa) -> X2(JS-дозы U-семейства) -> V -> T -> mdl_talc/R8.
"""
import math, sys, json
import numpy as np
sys.stdout.reconfigure(encoding="utf-8")
F0, NP_, NA = 1.6470, 50_000, 250_000
FR = NP_/NA; NOISE = 0.000022
MU0, TAU = 0.309, 0.196; TAU2 = TAU*TAU
law = lambda q: 0.894*F0/math.sqrt(NP_*q)

# первичные оси: (имя, q, kappa, финальная доза, приор_mu)
#U/W — JS-дозы ~ публичная усадка;
AX = [
 ("R2_blend",    0.00229,  0.601, 0.618, MU0),
 ("R3_ridge",    0.00280,  0.307, 0.308, MU0),
 ("R5_shade",    0.00007,  1.317, 1.002, MU0),
 ("W1_e_new",    0.00068,  0.102, None,  MU0),
 ("R7_reblend",  0.02490,  0.113, 0.124, MU0),
 ("T_tfm4",      0.00304,  0.459, 0.450, MU0),
]

def w_of(q): sg=law(q); return TAU2/(TAU2+sg*sg)
def bpriv(kp,q): w=w_of(q); return MU0+(1.25*w-0.25)*(kp-MU0)     # приватный оптимум дозы
def bpub(kp,q):  w=w_of(q); return MU0+w*(kp-MU0)                  # публичный оптимум
def gain(b,kp_true,q): return q*(2*b*kp_true - b*b)/(2*F0)

print("=== ЧИСТОЕ СТОП-ПРАВИЛО (первичные оси, финальные дозы) ===")
print(f"{'ось':13s}{'q':>9}{'kappa':>7}{'w':>6}{'доза':>7}{'опт_ПР':>8}{'урож_П':>9}{'ожид_ПР':>9}{'переоц':>9}")
tot_over=tot_pub=tot_pri=0.0; rows=[]
for nm,q,kp,dose,mu in AX:
    w=w_of(q); ekq=bpriv(kp,q)
    b = dose if dose is not None else bpub(kp,q)
    g_pub=gain(b,kp,q); g_pri=gain(b,ekq,q); over=g_pub-g_pri
    tot_over+=over; tot_pub+=g_pub; tot_pri+=g_pri
    rows.append((nm,q,kp,w,b,ekq,g_pub,g_pri,over))
    print(f"{nm:13s}{q:9.5f}{kp:7.3f}{w:6.3f}{b:7.3f}{ekq:8.3f}{g_pub:9.6f}{g_pri:9.6f}{over:+9.6f}")
print(f"{'ИТОГО':13s}{'':>29}{tot_pub:9.6f}{tot_pri:9.6f}{tot_over:+9.6f}")
print(f"\n  накопленная переоценка паблика: {tot_over:.6f} = {tot_over/NOISE:.1f} шума  (порог 0.0005 = 22.7 шума)")
print(f"  вердикт: {'СТОП' if tot_over>0.0005 else 'под порогом, но впритык'}")
# доля сырых доз
raw=[r for r in rows if r[0][0] in 'PR' and r[4]>bpub(r[2],r[1])+1e-6]
print(f"  доля переоценки от осей с СЫРОЙ дозой (P/): "
      f"{sum(r[8] for r in raw)/tot_over*100:.0f}%")

print(f"\n=== КОЭФФИЦИЕНТЫ ПЕРЕДОЗИРОВКИ под приватный оптимум ===")
print(f"{'ось':13s}{'доза сейчас':>12}{'приват-опт':>12}{'множитель':>11}{'+E[приват]':>12}")
red_tot=0.0; coeffs={}
for nm,q,kp,w,b,ekq,g_pub,g_pri,over in rows:
    if b<=1e-9: continue
    g_now=gain(b,ekq,q)                # приватный EV при текущей дозе
    g_opt=gain(ekq,ekq,q) if ekq>0 else 0.0   # приватный EV при приват-оптимуме
    d=g_opt-g_now
    if d>5e-7:
        mult=ekq/b if b>1e-9 else 0.0
        coeffs[nm]=round(mult,3); red_tot+=d
        print(f"{nm:13s}{b:12.3f}{ekq:12.3f}{mult:11.3f}{d:+12.6f}")
print(f"{'ИТОГО прибавка к ожиданию привата':45s}{red_tot:+12.6f} = {red_tot/NOISE:.1f} шума")
json.dump(dict(over=tot_over, redose_gain=red_tot, coeffs=coeffs, rows=[
    dict(ax=r[0],q=r[1],kappa=r[2],w=r[3],dose=r[4],priv_opt=r[5],over=r[8]) for r in rows]),
    open("out/n7_redose.json","w",encoding="utf-8"), ensure_ascii=False, indent=1)

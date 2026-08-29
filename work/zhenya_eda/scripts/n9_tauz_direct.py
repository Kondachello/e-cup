# -*- coding: utf-8 -*-
"""N9. tau_z по ПРЯМЫМ 20 парам Z-разложения (данные Саши, f_reply_sasha_2708.md).
Проверка его расчёта + пересмотр моей оценки 0.078 (была по сегментным зондам).
"""
import math, sys, json
import numpy as np
sys.stdout.reconfigure(encoding="utf-8")
F0, NP_, NA = 1.6470, 50_000, 250_000
FR = NP_/NA; NOISE = 0.000022
law = lambda q: 0.894*F0/math.sqrt(NP_*q)      # с fpc

k = np.array([z[2] for z in Z]); q = np.array([z[1] for z in Z]); s = np.array([law(x) for x in q])

print("=== ПРОВЕРКА tau_z ПО 20 ПРЯМЫМ ПАРАМ ===")
print(f"  Var(kappa) = {np.var(k,ddof=1):.6f}   <sigma^2> = {np.mean(s**2):.6f}")
mom = math.sqrt(max(0.0, np.var(k,ddof=1)-np.mean(s**2)))
print(f"  метод моментов: tau_z = {mom:.4f}   (Саша: 0.0126)")

def nll(mu,t2): v=t2+s**2; return 0.5*np.sum(np.log(2*np.pi*v)+(k-mu)**2/v)
MU=np.linspace(-0.10,0.10,801); T2=np.concatenate([[0.0],np.geomspace(1e-7,0.02,900)])
g=np.array([[nll(m,t) for t in T2] for m in MU])
i,j=np.unravel_index(g.argmin(),g.shape); fm=g.min()
mu_z, tau_z = float(MU[i]), math.sqrt(float(T2[j]))
t_ok=np.sqrt(T2[g.min(0)<=fm+1.92])
print(f"  ML: mu_z = {mu_z:+.4f}   tau_z = {tau_z:.4f}  95%: [{t_ok.min():.4f}, {t_ok.max():.4f}]")
print(f"  Саша: mu_z=-0.011, tau_z=0.0148 [0;0.02]  -> {'СОВПАДАЕТ' if abs(tau_z-0.0148)<0.006 else 'РАСХОДИТСЯ'}")

print(f"\n=== ПОЧЕМУ МОИ 0.078 БЫЛИ НЕВЕРНЫ ===")
print("  сегментные зонды -seg_epidot — ДРУГАЯ популяция: это ПРЕДЛОЖЕННЫЕ оси")
print("  (кто-то выбрал сегмент, веря в него), а Z-разложение — ИСЧЕРПЫВАЮЩЕЕ.")
print("  По моей же §2.1 приор предложенных осей несёт частоту попаданий")
print("  предлагателя и НЕ переносится на разложение. Я нарушил собственное правило.")
print(f"  seg_nickel (+0.126, реальный сигнал) один тянул tau вверх: без него было 0.000.")

print(f"\n=== ПОСЛЕДСТВИЯ ДЛЯ R9 ===")
w = tau_z**2/(tau_z**2+s**2); wp = 1.25*w-0.25
print(f"  средний w = {w.mean():.3f}, средний w_p = {wp.mean():+.3f}")
print(f"  осей в анти-информативной зоне (w<0.20): {(w<0.20).sum()} из 20")
gp=gq=0.0
print(f"\n  {'ось':5s}{'q':>9}{'kappa':>8}{'w':>7}{'w_p':>8}{'доза':>8}{'урож_П':>10}{'ожид_ПР':>10}")
for nm,qq,kk in Z:
    if nm not in DOSE: continue
    ww=tau_z**2/(tau_z**2+law(qq)**2); wpp=1.25*ww-0.25
    b=DOSE[nm]; ekq=wpp*kk
    a=qq*(2*b*kk-b*b)/(2*F0); c=qq*(2*b*ekq-b*b)/(2*F0); gp+=a; gq+=c
    print(f"  {nm:5s}{qq:9.5f}{kk:+8.3f}{ww:7.3f}{wpp:+8.3f}{b:+8.3f}{a:+10.6f}{c:+10.6f}")
print(f"  {'ИТОГО':5s}{'':>32}{gp:+18.6f}{gq:+10.6f}")
print(f"\n  публичный урожай R9-доз: {gp:+.6f}   ожидание привата: {gq:+.6f} = {gq/NOISE:+.1f} шума")
print(f"  Саша: приват −0.000344 -> {'ПОДТВЕРЖДАЮ' if abs(gq+0.000344)<0.00015 else 'РАСХОДИМСЯ'}")
json.dump(dict(tau_z=tau_z,mu_z=mu_z,tau_moments=mom,w_mean=float(w.mean()),
    r9_pub=gp,r9_priv=gq), open("out/n9_tauz.json","w",encoding="utf-8"),
    ensure_ascii=False, indent=1)

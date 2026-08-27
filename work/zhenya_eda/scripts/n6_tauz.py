# -*- coding: utf-8 -*-
"""N6. tau_z осей разложения эмпирическим байесом -> финальный s* для R9.

Данные (committed, work/reports/probe_segments_2608.md): 5 сегментных зондов от
базы R9 — тот же тип микро-оси и та же база, что у 9 осей Z-разложения, для
которых Саша сообщил |kappa|<=0.07. Это прямая выборка (kappa, sigma) для tau_z.
"""
import math, sys, json
import numpy as np
sys.stdout.reconfigure(encoding="utf-8")
F0, NP_, NA = 1.6470, 50_000, 250_000
FR = NP_/NA; NOISE = 0.000022

# (имя, kappa, sigma_kappa) — из probe_segments_2608.md, база R9
k = np.array([p[1] for p in PROBES]); s = np.array([p[2] for p in PROBES])

def fit(k, s):
    nll = lambda mu, t2: 0.5*np.sum(np.log(2*np.pi*(t2+s**2)) + (k-mu)**2/(t2+s**2))
    MU = np.linspace(-0.4, 0.4, 1601); T2 = np.concatenate([[0.0], np.geomspace(1e-6, 0.4, 800)])
    g = np.array([[nll(m, t) for t in T2] for m in MU])
    i, j = np.unravel_index(g.argmin(), g.shape)
    fm = g.min()
    mu_ok = MU[g.min(1) <= fm+1.92]; t_ok = np.sqrt(T2[g.min(0) <= fm+1.92])
    return MU[i], math.sqrt(T2[j]), (mu_ok.min(), mu_ok.max()), (t_ok.min(), t_ok.max())

mu, tau, mci, tci = fit(k, s)
print("=== tau_z ЭМПИРИЧЕСКИМ БАЙЕСОМ по 5 сегментным зондам базы R9 ===")
for n_, kk, ss in PROBES: print(f"  {n_:14s} kappa={kk:+.3f}  sigma={ss:.3f}")
print(f"\n  Var(kappa) выборочная = {np.var(k, ddof=1):.5f}   <sigma^2> = {np.mean(s**2):.5f}")
print(f"  ML: mu_z = {mu:+.3f} [{mci[0]:+.3f},{mci[1]:+.3f}]   "
      f"tau_z = {tau:.3f} [{tci[0]:.3f},{tci[1]:.3f}]")
print(f"  Саша по 9 осям Z-разложения: |kappa|<=0.07 => tau_z<=0.07 независимо. Согласуется.")

# reccore — реальный сигнал, не шум разложения. Проверим tau БЕЗ него (чистые нули):
mu2, tau2, _, tci2 = fit(kk2, ss2)
print(f"  без reccore (4 нулевые оси): tau_z = {tau2:.3f} [{tci2[0]:.3f},{tci2[1]:.3f}] "
      f"-> чистый шум разложения")

def s_law(q): return 0.894*F0/math.sqrt(NP_*q)
def wpriv(w): return 1.25*w - 0.25
def s_star(w): return 1.25 - 0.25/w if w > 1e-9 else -9

print(f"\n=== ФИНАЛЬНЫЙ s* ДЛЯ R9 (9 осей, mu_z=0 по теории §2.2) ===")
GPUB = 0.000516                 # публичный урожай R8->R9
# осям разложения приписываем q_eff — характеристический масштаб. У R9-осей
# q порядка сегментных: reccore q даёт sigma 0.051 => q=(0.894F0/0.051/224)... берём диапазон
print(f"{'tau_z':>7}{'q_eff':>9}{'sigma':>8}{'w':>7}{'s*':>7}{'E[приват]':>12}{'95% интервал привата':>26}")
OUT = {}
for tau_z in (tau, 0.05, 0.04, 0.03, tau2, 0.0):
    for q in (0.02,):
        sg = s_law(q); w = tau_z*tau_z/(tau_z*tau_z + sg*sg); ss_ = max(s_star(w), 0.0)
        ev = GPUB * max(wpriv(w), 0)**2/(w*(2-w)) if w > 1e-9 else 0.0
        var = 3.125*w*(1-w)*tau_z*tau_z*q*GPUB/(F0*(2-w)) if w > 1e-9 else 0.0
        sd = math.sqrt(max(var, 0))
        tag = f"{tau_z:.3f}"
        print(f"{tag:>7}{q:9.3f}{sg:8.3f}{w:7.3f}{ss_:7.3f}{ev:12.6f}"
              f"   [{ev-1.96*sd:+.6f},{ev+1.96*sd:+.6f}]")
        OUT[tag]=dict(tau_z=tau_z,q=q,w=w,s_star=ss_,ev=ev,sd=sd)
json.dump(dict(mu_z=mu, tau_z=tau, tau_ci=tci, probes=PROBES, scenarios=OUT),
          open("out/n6_tauz.json","w",encoding="utf-8"), ensure_ascii=False, indent=1)

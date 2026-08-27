# -*- coding: utf-8 -*-
"""N8. Финальный s* под подпись: argmax E[private] по АПОСТЕРИОРУ tau_z, с полом
положительности. Апостериор tau_z — из 5 сегментных зондов (профиль правдоподобия).
"""
import math, sys, json
import numpy as np
sys.stdout.reconfigure(encoding="utf-8")
F0, NP_ = 1.6470, 50_000
law = lambda q: 0.894*F0/math.sqrt(NP_*q)
GPUB = 0.000516                       # публичный урожай R8->R9
Q_R9 = 0.020                          # эффективный q джойнта R9 (масштаб reccore-оси)
PROBES = [(-0.076,0.113),(-0.040,0.113),(0.126,0.051),(-0.117,0.130),(-0.128,0.120)]
k = np.array([p[0] for p in PROBES]); s = np.array([p[1] for p in PROBES])

# --- профиль правдоподобия по tau (mu профилируется) ---
TAUS = np.concatenate([[0.0], np.geomspace(0.005, 0.4, 400)])
def nll_tau(t2):
    return min(0.5*np.sum(np.log(2*np.pi*(t2+s**2))+(k-mu)**2/(t2+s**2))
              for mu in np.linspace(-0.3,0.3,241))
prof = np.array([nll_tau(t*t) for t in TAUS])
post = np.exp(-(prof-prof.min())); post/=post.sum()      # апостериор ~ profile likelihood

# --- EV(s) при данном tau и приватном законе (mu_z=0) ---
def ev_of(s_mult, tau):
    # E[private](s) = GPUB/(2-w) * [2s(1.25w-0.25) - s^2 w];  вывод в отчёте (§3.1)
    sg=law(Q_R9); w=tau*tau/(tau*tau+sg*sg)
    if w<=1e-9: return 0.0
    A=1.25*w-0.25
    return GPUB/(2.0-w) * (2*s_mult*A - s_mult*s_mult*w)

SGRID=np.linspace(0.0,1.3,131)
# ожидание EV по апостериору tau
EV_bar=np.array([sum(post[i]*ev_of(sm,TAUS[i]) for i in range(len(TAUS))) for sm in SGRID])
s_argmax=SGRID[EV_bar.argmax()]
# пол положительности: min по tau в 90% массы апостериора, EV>0
order=np.argsort(prof); cum=np.cumsum(post[order]); keep=set(order[cum<=0.90])
def worst_ev(sm): return min(ev_of(sm,TAUS[i]) for i in keep)
pos=[sm for sm in SGRID if worst_ev(sm)>0]
s_floor=max(pos) if pos else 0.0

tau_ml=TAUS[prof.argmin()]
print("=== АПОСТЕРИОР tau_z (5 сегментных зондов, профиль) ===")
print(f"  tau_z ML = {tau_ml:.3f};  P(tau_z<0.03)={post[TAUS<0.03].sum():.2f}, "
      f"P(tau_z<0.05)={post[TAUS<0.05].sum():.2f}, P(tau_z>0.10)={post[TAUS>0.10].sum():.2f}")
print(f"\n=== ФИНАЛЬНЫЙ s* ===")
print(f"  argmax E[private] по апостериору:      s* = {s_argmax:.2f}")
print(f"  верхняя граница положительности (90%): s <= {s_floor:.2f}")
print(f"  РЕКОМЕНДАЦИЯ ПОД ПОДПИСЬ:               s* = 0.70")
for sm,lab in [(s_argmax,'argmax'),(0.70,'подпись'),(0.95,'старая')]:
    ev=EV_bar[np.abs(SGRID-sm).argmin()]
    print(f"    s={sm:.2f} ({lab:7s}): E[private]={ev:+.6f} = {ev/0.000022:.1f} шума, "
          f"худший в 90%={worst_ev(sm):+.6f}")
json.dump(dict(tau_ml=float(tau_ml), s_argmax=float(s_argmax), s_floor=float(s_floor),
    p_tau_lt_003=float(post[TAUS<0.03].sum()), p_tau_lt_005=float(post[TAUS<0.05].sum())),
    open("out/n8_sstar.json","w",encoding="utf-8"), ensure_ascii=False, indent=1)

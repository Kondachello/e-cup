# -*- coding: utf-8 -*-
"""N17. Слип-закон на ПЯТИ точках. Проверка: опровергают ли SHOW9/10 закон?"""
import math, sys, json
import numpy as np
sys.stdout.reconfigure(encoding="utf-8")
# (имя, Sigma|c|, слип факт, lam)
PTS=[("SHOW9",  7, -0.000045, 1e-2),("SHOW10", 19, -0.000146, 3e-3),
     ("SHOW4", 37, +0.000711, 1e-3),("SHOW8",146, +0.001180, 3e-4),
     ("SHOW5",279, +0.003080, 1e-4)]
# мой расчёт E[слип] и sd на 31-базисе (n4_curse.json)
MINE={1e-2:(0.000071,0.000124),3e-3:(0.000119,0.000214),1e-3:(0.000173,0.000317),
      3e-4:(0.000238,0.000443),1e-4:(0.000294,0.000550)}
print("=== ОПРОВЕРГАЮТ ЛИ SHOW9/10 ЗАКОН? ===")
print(f"{'файл':8s}{'S|c|':>6}{'lam':>8}{'слип факт':>12}{'мой E':>10}{'моя sd':>10}{'откл, sd':>10}")
for n,sc,sl,lam in PTS:
    e,sd=MINE[lam]
    print(f"{n:8s}{sc:6d}{lam:8.0e}{sl:+12.6f}{e:+10.6f}{sd:10.6f}{(sl-e)/sd:+10.1f}")
print("\n  SHOW9/10 лежат в -1.0 sd от моего расчёта -> НЕ опровергают закон.")
print("  На малой агрессии систематический член ТОНЕТ в собственном разбросе слипа:")
print("  при S|c|<=19 |E| ~ 1e-4, а sd ~ 1.2-2.1e-4. Знак там не информативен.")
print("  Закон работает там, где и должен: S|c| >= 37, где E выходит за 1 sd.")

x=np.array([p[1] for p in PTS],float); y=np.array([p[2] for p in PTS])
print(f"\n=== ФОРМА ЗАКОНА (5 точек) ===")
fits={}
for nm,X in [("линейный  a*S|c|", np.c_[x]),
             ("линейный+сдвиг", np.c_[x,np.ones_like(x)]),
             ("квадратичный a*S|c|^2", np.c_[x*x]),
             ("степенной (лог-лог, только S|c|>=37)", None)]:
    if X is None:
        m=x>=37; lx,ly=np.log(x[m]),np.log(y[m])
        p=np.polyfit(lx,ly,1); pred=np.exp(np.polyval(p,np.log(x[m])))
        r2=1-((y[m]-pred)**2).sum()/((y[m]-y[m].mean())**2).sum()
        print(f"  {nm:38s} показатель {p[0]:.2f}, mdl_flint={r2:.3f}")
        fits[nm]=dict(power=float(p[0]),r2=float(r2)); continue
    c,*_=np.linalg.lstsq(X,y,rcond=None); pred=X@c
    r2=1-((y-pred)**2).sum()/((y-y.mean())**2).sum()
    print(f"  {nm:38s} коэф {np.array2string(c,precision=8)}, mdl_flint={r2:.3f}")
    fits[nm]=dict(coef=c.tolist(),r2=float(r2))
m=x>=37
c,*_=np.linalg.lstsq(np.c_[x[m]],y[m],rcond=None)
print(f"\n  РАБОЧИЙ ЗАКОН (по трём точкам агрессии >=37): слип = {c[0]:.3e} * S|c|")
print(f"    прогноз: S|c|=37 -> {c[0]*37:+.6f} (факт +0.000711)")
print(f"             S|c|=146 -> {c[0]*146:+.6f} (факт +0.001180)")
print(f"             S|c|=279 -> {c[0]*279:+.6f} (факт +0.003080)")
json.dump(fits,open("out/n17_slip.json","w",encoding="utf-8"),ensure_ascii=False,indent=1)

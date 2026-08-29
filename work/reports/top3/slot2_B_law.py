# -*- coding: utf-8 -*-
"""Часть B: закон «развязка стоит квадратично» и граничное отставание."""
import json, os, sys
import numpy as np
sys.path.insert(0, "/Users/alexanderkondakov/ozon-cup/work/scripts")
from p_top3 import Objective, MU_US, SIGMA_US, NOISE

SCR = os.path.dirname(os.path.abspath(__file__))
A = json.load(open(os.path.join(SCR, "partA.json")))
rows = A["rows"]; F0 = 1.646

print("=== B1. ПРОВЕРКА ЗАКОНА  dE = rms^2/(2*F0)  на замеренных файлах ===")
print(f"{'файл':22s}{'rms':>9}{'dE факт,ш':>11}{'dE закон,ш':>12}{'факт/закон':>12}")
sel = sorted([r for r in rows if not r["show"] and r["rms"] > 0.02], key=lambda r: -r["rms"])
rat = []
for r in sel:
    law = r["rms"] ** 2 / (2 * F0) / NOISE
    rat.append(r["dE_n"] / law)
for r in sel[:8] + sel[-8:]:
    law = r["rms"] ** 2 / (2 * F0) / NOISE
    print(f"{r['name']:22s}{r['rms']:9.4f}{r['dE_n']:11.1f}{law:12.1f}{r['dE_n']/law:12.3f}")
rat = np.array(rat)
print(f"\n  отношение факт/закон по {len(rat)} честным файлам: медиана {np.median(rat):.3f}, "
      f"квартили {np.percentile(rat,25):.3f}..{np.percentile(rat,75):.3f}, "
      f"мин {rat.min():.3f}")
print("  => закон работает как НИЖНЯЯ ГРАНИЦА цены развязки (факт >= закона почти всегда)")

print("\n=== B2. ФРОНТ «развязка через непохожесть прогнозов» ===")
print("  sd_d = 0.0011*rms (ЗАМЕР, 15 пар) и dE >= rms^2/(2F0)  =>  dE[ш] = 5.524*sd_d[ш]^2")
obj = Objective(ns=400_000)
g1 = MU_US + SIGMA_US * obj.z
base = float((g1 < obj.c3).mean())
print(f"  одиночный F8: {base*100:.2f} %\n")
print(f"{'rms':>9}{'sd_d,ш':>9}{'dE,ш':>10}{'P(топ3)':>10}{'прирост,п.п.':>14}")
best = (-9, None)
for rms in [0.002, 0.005, 0.01, 0.02, 0.03, 0.05, 0.0595, 0.08, 0.12, 0.2, 0.3, 0.5, 0.8]:
    sd = min(0.0011 * rms, 0.0009); dE = rms ** 2 / (2 * F0)
    g2 = g1 + dE + sd * obj.w
    p = float((np.minimum(g1, g2) < obj.c3).mean())
    if (p - base) > best[0]: best = (p - base, rms)
    print(f"{rms:9.4f}{sd/NOISE:9.2f}{dE/NOISE:10.1f}{p*100:9.2f}%{(p-base)*100:+14.4f}")
# тонкая сетка около максимума
gr = np.geomspace(1e-4, 0.3, 400)
gains = []
for rms in gr:
    sd = min(0.0011 * rms, 0.0009); dE = rms ** 2 / (2 * F0)
    g2 = g1 + dE + sd * obj.w
    gains.append(float((np.minimum(g1, g2) < obj.c3).mean()) - base)
gains = np.array(gains); j = int(np.argmax(gains))
print(f"\n  МАКСИМУМ ПО ВСЕМУ ФРОНТУ: rms={gr[j]:.4f}, sd_d={0.0011*gr[j]/NOISE:.2f}ш, "
      f"dE={gr[j]**2/(2*F0)/NOISE:.1f}ш, прирост {gains[j]*100:+.4f} п.п.")
print("  => канал «непохожесть прогнозов» физически НЕ МОЖЕТ дать 1 п.п.")

print("\n=== B3. ГРАНИЧНОЕ ОТСТАВАНИЕ: при каком dE развязка ещё окупается (+1 п.п.) ===")
print(f"{'sd_d,ш':>9}{'sd_d':>10}{'нужен rms':>11}{'dE* (+1пп),ш':>14}{'dE закона,ш':>13}{'вердикт':>12}")
res = []
for sdn in [1, 2, 3, 4.5, 6, 9.1, 13.6, 20, 30, 45, 68, 100, 150]:
    sd = sdn * NOISE
    lo, hi = 0.0, 0.02
    def gainf(dE):
        g2 = g1 + dE + sd * obj.w
        return float((np.minimum(g1, g2) < obj.c3).mean()) - base
    if gainf(0.0) < 0.01:
        res.append(dict(sd_n=sdn, dE_star=None)); 
        print(f"{sdn:9.1f}{sd:10.6f}{sd/0.0011:11.4f}{'нет':>14}{5.524*sdn**2:13.1f}{'НИКОГДА':>12}")
        continue
    for _ in range(45):
        m = 0.5 * (lo + hi)
        if gainf(m) >= 0.01: lo = m
        else: hi = m
    dE = 0.5 * (lo + hi); law = 5.524 * sdn ** 2
    res.append(dict(sd_n=sdn, dE_star=dE / NOISE))
    ok = "ДОСТИЖИМО" if law <= dE / NOISE else "невозможно"
    print(f"{sdn:9.1f}{sd:10.6f}{sd/0.0011:11.4f}{dE/NOISE:14.1f}{law:13.1f}{ok:>12}")
json.dump(dict(base=base, breakeven=res), open(os.path.join(SCR, "partB.json"), "w"), indent=1)

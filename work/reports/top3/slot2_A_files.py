# -*- coding: utf-8 -*-
"""Часть A: все локальные кандидаты в слот-2, пересчитанные по критерию P(топ-3).

Модель пары (одна общая случайность, часть J / M3.2):
    g1 = mu1 + sigma_shared*z + t1        первый финалист
    g2 = mu2 + sigma_shared*z + t2        второй
    X  = min(g1,g2)                        зачётный приват
Здесь t_i — индивидуальная компонента; для эмпирических файлов
sd(t1-t2) = sigma_d = 0.0011*rms(разница lp)  (ЗАМЕР, трек 4, 15 пар, final_pair_memo.md).
"""
import json, os, sys
import numpy as np
sys.path.insert(0, "/Users/alexanderkondakov/ozon-cup/work/scripts")
from p_top3 import Objective, MU_US, SIGMA_US, NOISE

SCR = os.path.dirname(os.path.abspath(__file__))
z = np.load(os.path.join(SCR, "cands_lp.npz"), allow_pickle=True)
names = [str(x) for x in z["names"]]
LP = z["lp"]; SCORES = z["scores"]; POS = z["posidx"]
idx = {n: i for i, n in enumerate(names)}
F8N, T3N = "F8_priv", "T3_g1_redose_044"
OVER_T3 = 0.000671
E_F8, E_T3 = 1.646203, 1.647603

pos_t3 = POS[idx[T3N]]
def over_of(n):
    i = idx[n]
    if n.startswith("SHOW"):
        return OVER_T3 + max(SCORES[idx[F8N]] - SCORES[i], 0.0)
    return OVER_T3 * min(1.0, POS[i] / max(pos_t3, 1))

lp8 = LP[idx[F8N]]
rows = []
for n in names:
    if n == F8N: continue
    i = idx[n]
    d = LP[i] - lp8
    rms = float(np.sqrt((d * d).mean()))
    cr = float(np.corrcoef(LP[i], lp8)[0, 1])
    e = E_T3 if n == T3N else SCORES[i] + over_of(n)
    sd_d = min(0.0011 * rms, 0.0009)
    rows.append(dict(name=n, pub=float(SCORES[i]), over=float(over_of(n)), E=float(e),
                     rms=rms, corr=cr, sd_d=sd_d,
                     dE=e - E_F8, dE_n=(e - E_F8) / NOISE, sd_d_n=sd_d / NOISE,
                     show=n.startswith("SHOW")))

# ------- P(топ-3): одиночный F8 и каждая пара
obj = Objective(ns=400_000)
SIG_SHARED = SIGMA_US  # для эмпирических файлов вся индивидуальность сидит в sd_d
g1 = MU_US + SIG_SHARED * obj.z
base = float((g1 < obj.c3).mean())
for r in rows:
    g2 = g1 + r["dE"] + r["sd_d"] * obj.w
    r["p_pair"] = float((np.minimum(g1, g2) < obj.c3).mean())
    r["gain_pp"] = (r["p_pair"] - base) * 100

rows.sort(key=lambda r: -r["gain_pp"])
out = dict(base=base, n_files=len(names), rows=rows)
json.dump(out, open(os.path.join(SCR, "partA.json"), "w"), ensure_ascii=False, indent=1)

print(f"кандидатов с локальным lp и замером: {len(names)} (у Жени было 35)")
print(f"одиночный F8: P(топ-3) = {base*100:.2f} %\n")
print(f"{'файл':26s}{'паблик':>12}{'E[priv]':>11}{'dE,ш':>8}{'rms':>8}{'corr':>8}"
      f"{'sd_d,ш':>8}{'P(топ3)':>9}{'прирост':>9}")
print("-" * 100)
for r in rows[:25]:
    tag = " ВИТРИНА" if r["show"] else ""
    print(f"{r['name']:26s}{r['pub']:12.7f}{r['E']:11.6f}{r['dE_n']:+8.1f}{r['rms']:8.4f}"
          f"{r['corr']:8.4f}{r['sd_d_n']:8.2f}{r['p_pair']*100:8.2f}%{r['gain_pp']:+9.3f}{tag}")
print("...")
for nm in [T3N, "H2_edge_p1", "R5_shade", "A2_probe_s1_gmv"]:
    rr = [r for r in rows if r["name"] == nm]
    if rr:
        r = rr[0]
        print(f"{r['name']:26s}{r['pub']:12.7f}{r['E']:11.6f}{r['dE_n']:+8.1f}{r['rms']:8.4f}"
              f"{r['corr']:8.4f}{r['sd_d_n']:8.2f}{r['p_pair']*100:8.2f}%{r['gain_pp']:+9.3f}")
    else:
        print(f"{nm:26s}  -- нет локального lp/замера --")

print("\n--- самые РАЗВЯЗАННЫЕ (макс rms) среди честных ---")
hon = sorted([r for r in rows if not r["show"]], key=lambda r: -r["rms"])[:12]
for r in hon:
    print(f"{r['name']:26s} rms {r['rms']:8.4f} corr {r['corr']:8.5f} "
          f"dE {r['dE_n']:+9.1f}ш sd_d {r['sd_d_n']:6.2f}ш прирост {r['gain_pp']:+7.3f} п.п.")

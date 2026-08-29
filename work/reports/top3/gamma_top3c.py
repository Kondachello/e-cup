# -*- coding: utf-8 -*-
"""GAMMA, часть 3: эмпирическая калибровка дисконта по реплею F8 + внутренний оптимум
   в честном (мягкий дисконт + смещение сторожа) чтении."""
from __future__ import annotations
import json, math, sys
import numpy as np
sys.path.insert(0, "/Users/alexanderkondakov/ozon-cup/work/scripts")
from p_top3 import Objective, MU_US, SIGMA_US   # noqa

R = "/Users/alexanderkondakov/ozon-cup/work/reports/lineA/"
S = np.load(R + "a5_loo_state_a5.npz", allow_pickle=True)
E = np.load(R + "gls_state_eb.npz", allow_pickle=True)
A4A = np.load(R + "gls_state_a4all.npz", allow_pickle=True)
A4B = np.load(R + "gls_state_a4base.npz", allow_pickle=True)
Q, cQ, mu_c, Lam, fam = S["Q"], S["cQ"], S["mu_c"], S["Lam"], S["fam"]
dF8 = S["d_F8"]; F = float(S["F_SCALE"]); NZ = float(S["NOISE"]); V = E["mdl_vivian"]
F7_PUB = float(S["F0"])
dc = A4A["cQ"][:46] - A4B["cQ"]
c_soft = np.where(np.isin(fam, ["model", "decomp"]), cQ, mu_c)
g = lambda d, c: float((2 * d @ c - d @ Q @ d) / (2 * F))
dose = lambda gm: np.linalg.solve(Q + Lam + gm * np.diag(np.diag(Q)), cQ)
sd_gain = lambda d: float(1.25 * math.sqrt(d @ V @ d) / F)
wd = lambda d: float(-d @ dc / F)
g_in_F8, g_soft_F8, g_oof_F8, wd_F8 = g(dF8, cQ), g(dF8, c_soft), g(dF8, mu_c), wd(dF8)
sd_ref = sd_gain(dose(0.1)); VAR_SHARED = SIGMA_US ** 2 - sd_ref ** 2
obj = Objective(ns=400_000); P_F8 = obj.P_top3(MU_US, SIGMA_US)

print("=" * 100)
print("E. ЭМПИРИЧЕСКАЯ КАЛИБРОВКА ДИСКОНТА ПО ЕДИНСТВЕННОМУ ВНЕШНЕМУ ЗАМЕРУ (реплей F8)")
print("=" * 100)
PUB_ALG_F8, PUB_MEAS_F8 = 1.6457652, 1.6458057389
claim = F7_PUB - PUB_ALG_F8; real = F7_PUB - PUB_MEAS_F8
print(f"  паблик F7 (база)                     {F7_PUB:.10f}")
print(f"  паблик F8 по алгебре                 {PUB_ALG_F8:.10f}   заявленный выигрыш "
      f"{claim:.3e} = {claim/NZ:.2f} шума")
print(f"  паблик F8 замеренный                 {PUB_MEAS_F8:.10f}   реализованный выигрыш "
      f"{real:.3e} = {real/NZ:.2f} шума")
print(f"  РЕАЛИЗОВАННАЯ ДОЛЯ = {real/claim:.3f}  =>  ЭМПИРИЧЕСКИЙ ДИСКОНТ x{claim/real:.2f}")
print(f"  (мягкий дисконт A5 = x1.72; полный OOF потребовал бы отрицательной реализации)")
print(f"  оптимизм, который полный OOF приписывает самому F8: "
      f"{(g_in_F8-g_oof_F8)/NZ:.2f} шума против продемонстрированных {(claim-real)/NZ:.2f}")

DISC_EMP = claim / real
GR = sorted(set([round(x, 4) for x in np.arange(0, 0.4001, 0.0025)] + [0.0765]))

def row(gm, disc):
    d = dose(gm); sd = sd_gain(d); wex = wd(d) - wd_F8
    dE = (g(d, cQ) - g_in_F8) / disc
    sig = math.sqrt(VAR_SHARED + sd ** 2)
    return dict(gamma=gm, dE=dE, wex=wex, sd=sd, sig=sig,
                p_nobias=obj.P_top3(MU_US - dE, sig),
                p_bias=obj.P_top3(MU_US - dE + wex, sig),
                p_half=obj.P_top3(MU_US - dE + 0.5 * wex, sig))

print()
print("=" * 100)
print("F. ЧЕСТНОЕ ЧТЕНИЕ: мягкий дисконт (эмпирический x%.2f) + СМЕЩЕНИЕ СТОРОЖА" % DISC_EMP)
print("=" * 100)
print(f"{'gamma':>7}{'dE,ш':>8}{'сторож сверх F8,ш':>19}{'нетто,ш':>9}{'||d||':>8}"
      f"{'P без смещ':>12}{'P со смещ':>11}{'P полсмещ':>11}{'к F8 (со смещ)':>16}")
best = None
tab = []
for gm in GR:
    r = row(gm, DISC_EMP); tab.append(r)
    if best is None or r["p_bias"] > best["p_bias"]: best = r
    if gm in (0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.0765, 0.08, 0.1, 0.12, 0.15, 0.2, 0.3, 0.4):
        print(f"{gm:7.4f}{r['dE']/NZ:+8.2f}{r['wex']/NZ:+19.2f}{(r['dE']-r['wex'])/NZ:+9.2f}"
              f"{np.linalg.norm(dose(gm)):8.3f}{r['p_nobias']*100:11.2f}%{r['p_bias']*100:10.2f}%"
              f"{r['p_half']*100:10.2f}%{(r['p_bias']-P_F8)*100:+16.2f}")
print(f"  ||d_F8|| = {np.linalg.norm(dF8):.3f}   сторож на F8 = {wd_F8/NZ:.2f} шума")
print(f"  F8 (слот не тратим): {P_F8*100:.2f} %")
print(f"  ВНУТРЕННИЙ ОПТИМУМ честного чтения: gamma = {best['gamma']:.4f}  "
      f"P = {best['p_bias']*100:.2f} %  ({(best['p_bias']-P_F8)*100:+.2f} п.п. к F8)")
bh = max(tab, key=lambda r: r["p_half"])
bn = max(tab, key=lambda r: r["p_nobias"])
print(f"  при ПОЛОВИННОМ доверии сторожу:      gamma = {bh['gamma']:.4f}  "
      f"P = {bh['p_half']*100:.2f} %  ({(bh['p_half']-P_F8)*100:+.2f} п.п.)")
print(f"  при НУЛЕВОМ доверии сторожу:         gamma = {bn['gamma']:.4f}  "
      f"P = {bn['p_nobias']*100:.2f} %  ({(bn['p_nobias']-P_F8)*100:+.2f} п.п.)")

print()
print("  доля доверия сторожу theta, при которой оптимум уезжает с gamma=0:")
for th in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0):
    vals = [(r["gamma"], obj.P_top3(MU_US - r["dE"] + th * r["wex"], r["sig"])) for r in tab]
    b = max(vals, key=lambda t: t[1])
    print(f"    theta={th:4.1f}:  gamma*={b[0]:6.4f}  P={b[1]*100:6.2f} %  "
          f"({(b[1]-P_F8)*100:+.2f} п.п. к F8)")

json.dump(dict(disc_emp=DISC_EMP, claim=claim, real=real, P_F8=P_F8,
               tab=[{k: v for k, v in r.items()} for r in tab]),
          open("/private/tmp/claude-501/-Users-alexanderkondakov-ozon-cup/0b55ab9f-3777-4ebc-bd91-937895c0e355/scratchpad/gamma_top3c.json", "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)

print()
print("=" * 100)
print("G. МИНИМАКС ПО ДОВЕРИЮ СТОРОЖУ theta in [0,1] — gamma, устойчивая к тому, чего мы не знаем")
print("=" * 100)
THS = np.linspace(0, 1, 11)
scan = []
for r in tab:
    ps = [obj.P_top3(MU_US - r["dE"] + th * r["wex"], r["sig"]) for th in THS]
    scan.append((r["gamma"], min(ps), max(ps), ps))
b = max(scan, key=lambda t: t[1])
print(f"{'gamma':>8}{'мин по theta':>14}{'макс по theta':>15}{'мин к F8, п.п.':>17}")
for gm, mn, mx, _ in scan:
    if gm in (0.0, 0.01, 0.02, 0.03, 0.05, 0.0765, 0.08, 0.1, 0.12, 0.155, 0.2, 0.3):
        print(f"{gm:8.4f}{mn*100:13.2f}%{mx*100:14.2f}%{(mn-P_F8)*100:+17.2f}")
print(f"  МИНИМАКС: gamma = {b[0]:.4f}, гарантированные {b[1]*100:.2f} % "
      f"({(b[1]-P_F8)*100:+.2f} п.п. к F8) при любом доверии сторожу")
for gm in (0.0765, 0.08):
    s = [t for t in scan if abs(t[0] - gm) < 1e-9][0]
    print(f"  gamma={gm}: гарантия {s[1]*100:.2f} % ({(s[1]-P_F8)*100:+.2f} п.п.), "
          f"потолок {s[2]*100:.2f} % ({(s[2]-P_F8)*100:+.2f} п.п.)")

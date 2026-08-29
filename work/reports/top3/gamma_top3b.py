# -*- coding: utf-8 -*-
"""GAMMA, часть 2: смешение режимов зачёта, минимакс, широкая сетка."""
from __future__ import annotations
import json, math, sys
import numpy as np
sys.path.insert(0, "/Users/alexanderkondakov/ozon-cup/work/scripts")
from p_top3 import Objective, MU_US, SIGMA_US, NOISE   # noqa

R = "/Users/alexanderkondakov/ozon-cup/work/reports/lineA/"
S = np.load(R + "a5_loo_state_a5.npz", allow_pickle=True)
E = np.load(R + "gls_state_eb.npz", allow_pickle=True)
A4A = np.load(R + "gls_state_a4all.npz", allow_pickle=True)
A4B = np.load(R + "gls_state_a4base.npz", allow_pickle=True)
GB = np.load(R + "gram_boot_a2.npz", allow_pickle=True)
Q, cQ, mu_c, Lam = S["Q"], S["cQ"], S["mu_c"], S["Lam"]
Sp, Se, kap, dot, fam = S["Sig_p"], S["Sig_e"], S["kap"], S["dot"], S["fam"]
dF8 = S["d_F8"]; F = float(S["F_SCALE"]); NZ = float(S["NOISE"]); V = E["mdl_vivian"]
dc_watch = A4A["cQ"][:46] - A4B["cQ"]
c_soft = np.where(np.isin(fam, ["model", "decomp"]), cQ, mu_c)
g = lambda d, c: float((2 * d @ c - d @ Q @ d) / (2 * F))
dose = lambda gam: np.linalg.solve(Q + Lam + gam * np.diag(np.diag(Q)), cQ)
sd_gain = lambda d: float(1.25 * math.sqrt(d @ V @ d) / F)
wd = lambda d: float(-d @ dc_watch / F)
g_in_F8, g_soft_F8, g_oof_F8, wd_F8 = g(dF8, cQ), g(dF8, c_soft), g(dF8, mu_c), wd(dF8)
sd_ref = sd_gain(dose(0.1))
VAR_SHARED = SIGMA_US ** 2 - sd_ref ** 2
DISC = 1.72
obj = Objective(ns=400_000)
P_F8 = obj.P_top3(MU_US, SIGMA_US)


def gram_p95(gammas, amp=5.5):
    QB, DB = GB["QB"], GB["DOTB"]; SpSe = Sp + Se
    acc = {gm: [] for gm in gammas}
    for b in range(QB.shape[0]):
        Qh = Q + amp * (QB[b] - Q); dh = dot + amp * (DB[b] - dot)
        qh = np.diag(Qh); cPh = qh * kap - dh
        mh = mu_c + Sp @ np.linalg.solve(SpSe, cPh - mu_c)
        cQh = 1.25 * mh - 0.25 * cPh
        for gm in gammas:
            acc[gm].append(g(np.linalg.solve(Qh + Lam + gm * np.diag(qh), cQh), cQ))
    return {gm: float(np.percentile(g(dose(gm), cQ) - np.array(v), 95)) for gm, v in acc.items()}

WIDE = sorted(set([round(x, 4) for x in np.arange(0, 0.501, 0.005)] +
                  [0.0765, 0.6, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0]))
GRID = [0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.0765, 0.08, 0.1]
GP95 = gram_p95(GRID)

REG = ("raw", "soft", "poof", "oof")
def deltas(gam):
    d = dose(gam)
    return dict(raw=g(d, cQ) - g_in_F8,
                soft=(g(d, cQ) - g_in_F8) / DISC,
                poof=g(d, c_soft) - g_soft_F8,
                oof=g(d, mu_c) - g_oof_F8), sd_gain(d), wd(d) - wd_F8

def P(gam, reg, bias=False, var=False):
    dd, sd, wex = deltas(gam)
    mu = MU_US - dd[reg] + (wex if bias else 0.0)
    ex = 0.0
    if var:
        gp = GP95.get(round(gam, 4), 0.0)
        ex = math.hypot(gp / 1.645, abs(wex))
    return obj.P_top3(mu, math.sqrt(VAR_SHARED + sd ** 2 + ex ** 2))

print("=" * 104)
print("A. ШИРОКАЯ СЕТКА: где полный OOF догоняет F8 (и догоняет ли)")
print("=" * 104)
print(f"{'gamma':>7}{'RAW,ш':>9}{'SOFT,ш':>9}{'pOOF,ш':>9}{'OOF,ш':>9}{'P raw':>9}{'P soft':>9}{'P pOOF':>9}{'P oof':>9}")
wide_rows = []
for gm in WIDE:
    dd, sd, wex = deltas(gm)
    ps = {r: P(gm, r) for r in REG}
    wide_rows.append(dict(gamma=gm, **{("d_" + r): dd[r] / NZ for r in REG},
                          **{("p_" + r): ps[r] for r in REG}))
    if gm in (0.0, 0.02, 0.05, 0.0765, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0, 2.0, 5.0):
        print(f"{gm:7.4f}{dd['raw']/NZ:+9.2f}{dd['soft']/NZ:+9.2f}{dd['poof']/NZ:+9.2f}"
              f"{dd['oof']/NZ:+9.2f}{ps['raw']*100:8.2f}%{ps['soft']*100:8.2f}%"
              f"{ps['poof']*100:8.2f}%{ps['oof']*100:8.2f}%")
print(f"  F8 (слот не тратим) во всех режимах по построению: {P_F8*100:.2f} %")
best_oof = max(wide_rows, key=lambda r: r["p_oof"])
print(f"  argmax OOF по [0,5]: gamma={best_oof['gamma']}  P={best_oof['p_oof']*100:.2f} % "
      f"(F8 = {P_F8*100:.2f} %) -> полный OOF НЕ достигает F8 ни при какой gamma: "
      f"{best_oof['p_oof'] < P_F8}")

print()
print("=" * 104)
print("B. СМЕСЬ РЕЖИМОВ (неопределённость дисконта — это тоже дисперсия; P смеси = взвешенная сумма P)")
print("=" * 104)
WS = {"штаб A5 (0.25/0.50/0.25)": (0.25, 0.50, 0.25),
      "оптимист (0.50/0.40/0.10)": (0.50, 0.40, 0.10),
      "пессимист (0.10/0.40/0.50)": (0.10, 0.40, 0.50),
      "равные трети": (1/3, 1/3, 1/3)}
mix = {}
for nm, (wr, ws, wo) in WS.items():
    print(f"  веса {nm}:")
    row = []
    for gm in GRID:
        p = wr * P(gm, "raw") + ws * P(gm, "soft") + wo * P(gm, "oof")
        pb = wr * P(gm, "raw", bias=True) + ws * P(gm, "soft", bias=True) + wo * P(gm, "oof", bias=True)
        row.append((gm, p, pb))
    mix[nm] = row
    print("    gamma:  " + "".join(f"{gm:8.4f}" for gm, _, _ in row))
    print("    P смеси:" + "".join(f"{p*100:7.2f}%" for _, p, _ in row))
    print("    +смещ.: " + "".join(f"{pb*100:7.2f}%" for _, _, pb in row))
    b = max(row, key=lambda t: t[1]); bb = max(row, key=lambda t: t[2])
    print(f"    лучшая gamma = {b[0]} (P={b[1]*100:.2f} %, {(b[1]-P_F8)*100:+.2f} п.п. к F8); "
          f"со смещением {bb[0]} (P={bb[2]*100:.2f} %, {(bb[2]-P_F8)*100:+.2f} п.п.)")

print()
print("  ПОРОГ: при каком весе полного OOF gamma=0 перестаёт бить F8 (raw/soft делят остаток 1:2)")
for gm in (0.0, 0.02, 0.05, 0.08, 0.1):
    lo, hi = 0.0, 1.0
    for _ in range(60):
        w = (lo + hi) / 2
        rest = 1 - w
        p = rest / 3 * P(gm, "raw") + 2 * rest / 3 * P(gm, "soft") + w * P(gm, "oof")
        if p > P_F8: lo = w
        else: hi = w
    print(f"    gamma={gm:6.4f}: критический вес OOF = {(lo+hi)/2*100:.1f} %")

print()
print("=" * 104)
print("C. МИНИМАКС (худший из трёх режимов) и правило приёмки задания")
print("=" * 104)
print(f"{'gamma':>7}{'raw':>9}{'soft':>9}{'oof':>9}{'ХУДШИЙ':>10}{'к F8':>9}   вердикт")
for gm in GRID:
    ps = [P(gm, "raw"), P(gm, "soft"), P(gm, "oof")]
    w = min(ps)
    print(f"{gm:7.4f}{ps[0]*100:8.2f}%{ps[1]*100:8.2f}%{ps[2]*100:8.2f}%{w*100:9.2f}%"
          f"{(w-P_F8)*100:+9.2f}   {'ПРОХОДИТ' if w >= P_F8 else 'НЕ проходит правило'}")

print()
print("=" * 104)
print("D. ЦЕНА АГРЕССИИ: дисперсия против смещения (мягкий режим x1.72)")
print("=" * 104)
print(f"{'gamma':>7}{'P soft':>9}{'как ДИСПЕРСИЯ':>16}{'как СМЕЩЕНИЕ':>15}{'разрыв':>9}"
      f"{'сторож,ш':>11}{'сверх F8,ш':>12}{'грам p95,ш':>12}")
for gm in GRID:
    p0 = P(gm, "soft"); pv = P(gm, "soft", var=True); pb = P(gm, "soft", bias=True)
    _, _, wex = deltas(gm)
    print(f"{gm:7.4f}{p0*100:8.2f}%{(pv-p0)*100:+15.2f}{(pb-p0)*100:+15.2f}"
          f"{(pv-pb)*100:+9.2f}{wd(dose(gm))/NZ:11.2f}{wex/NZ:12.2f}{GP95[gm]/NZ:12.2f}")

print()
print("  ПРОВЕРКА ПОРОГА ИЗ OBJECTIVE §5: «gamma=0 бьёт gamma=0.08, пока скрытое")
print("  за-вершинное смещение меньше 4.4 шума».")
w0, w8 = wd(dose(0.0)), wd(dose(0.08))
print(f"    замеренное сторожем избыточное смещение gamma=0 над gamma=0.08 = "
      f"{(w0-w8)/NZ:.2f} шума (порог 4.4)")
print(f"    => порог {'НЕ пройден (gamma=0 отклоняется)' if (w0-w8)/NZ>4.4 else 'пройден с запасом ' + f'{4.4-(w0-w8)/NZ:.2f} шума'}")
print(f"    контроль в вероятностях: raw+смещение  gamma=0 {P(0.0,'raw',bias=True)*100:.2f} % "
      f"против gamma=0.08 {P(0.08,'raw',bias=True)*100:.2f} %")
print(f"                             soft+смещение gamma=0 {P(0.0,'soft',bias=True)*100:.2f} % "
      f"против gamma=0.08 {P(0.08,'soft',bias=True)*100:.2f} %")

json.dump(dict(P_F8=P_F8, wide=wide_rows, mix={k: [[a, b, c] for a, b, c in v] for k, v in mix.items()},
               gram_p95={str(k): v for k, v in GP95.items()},
               wd={str(gm): wd(dose(gm)) for gm in GRID}, wd_F8=wd_F8),
          open("/private/tmp/claude-501/-Users-alexanderkondakov-ozon-cup/0b55ab9f-3777-4ebc-bd91-937895c0e355/scratchpad/gamma_top3b.json", "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)

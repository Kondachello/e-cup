# -*- coding: utf-8 -*-
"""N4. Проклятие победителя — ТОЧНО и БЕЗ таргета.

delta(x) = mean_250k(x) - mean_public(x). Для файла спана lp = lp_a + c0 + sum c_i d_i
    слип_f2 = delta(u),  u = sum_i c_i d_i^2 - (c0 + sum_i c_i d_i)^2 - 2 c0 lp_a
Таргет сокращается полностью: psi восстановлена из ПУБЛИЧНЫХ скоров, а mean(lp^2)
берётся по 250k. Оптимизатор видит psi + Delta/2 и эксплуатирует Delta.

Раскладывая, на каждый раскол хватает трёх k-мерных объектов:
    S = delta(d_i d_j),  g = delta(lp_a d_i),  h = delta(d_i),  A = delta(lp_a)
    Delta_i = delta(lp_i^2 - lp_a^2) = 2 g_i + S_ii
    слип    = c'diag(S) - 2 c0 (c'h) - c' S c - 2 c0 A
"""
import json, math, sys
import numpy as np
sys.stdout.reconfigure(encoding="utf-8")

NP_, F0R, MEAN_T = 50_000, 1.6470, 2.3275
ANCH, DROP = "A1_gram7_shift", {"sample"}
z = np.load("out/lb_full.npz"); meta = json.load(open("out/lb_meta.json"))
names = [n for n in meta["names"] if n not in DROP]
L = np.vstack([z[f"lp_{n}"].astype(np.float64) for n in names])
fsc = np.array([meta["sc"][n] for n in names])
N = L.shape[1]; a = names.index(ANCH); lp_a = L[a]; D = L - lp_a; k = len(names)
Bf = np.vstack([np.ones(N), D]); G = Bf @ Bf.T / N; mv = Bf @ lp_a / N
qd = (L*L).mean(1)
psi0 = np.concatenate([[MEAN_T], ((qd - qd[a]) - (fsc**2 - fsc[a]**2))/2])
S_all = D @ D.T / N; g_all = D @ lp_a / N; h_all = D.mean(1); A_all = lp_a.mean()
print(f"базис {k} файлов, {N} юзеров, якорь {ANCH}")

rng = np.random.default_rng(11)
NDRAW = 80
SPL = []
for _ in range(NDRAW):
    P = rng.choice(N, NP_, replace=False)
    Dp = D[:, P]; ap = lp_a[P]
    SPL.append((S_all - Dp @ Dp.T / NP_, g_all - Dp @ ap / NP_,
                h_all - Dp.mean(1), A_all - ap.mean()))

def slip(c, S, g, h, A):
    c0, ci = c[0], c[1:]
    return float(ci @ np.diag(S) - 2*c0*(ci @ h) - ci @ S @ ci - 2*c0*A)

def run(idx, lam, psi, Gm, mm):
    R = np.eye(len(mm)); R[0, 0] = 1e-4; M = Gm + lam*R
    c_star = np.linalg.solve(M, psi - mm)
    so, sf, sc_ = [], [], []
    for S, g, h, A in SPL:
        Ss, gs, hs = S[np.ix_(idx, idx)], g[idx], h[idx]
        Dl = np.concatenate([[0.0], 2*gs + np.diag(Ss)])
        c = np.linalg.solve(M, psi + Dl/2 - mm)
        so.append(slip(c, Ss, gs, hs, A)); sf.append(slip(c_star, Ss, gs, hs, A))
        sc_.append(float(np.abs(c[1:]).sum()))
    return np.mean(so)/(2*F0R), np.std(so)/(2*F0R), np.mean(sf)/(2*F0R), np.mean(sc_)

ALL = list(range(k))
print(f"\n=== СЛИП ПРИ ОПТИМИЗАЦИИ ({NDRAW} расколов 50k/250k, базис {k}) ===")
print(f"{'lam':>9}{'S|c|':>8}{'E[слип]':>11}{'sd':>10}{'при фикс. c':>13}")
res = {}
for lam in (1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4):
    e, sd, ef, sc_ = run(ALL, lam, psi0, G, mv)
    res[f"{lam:.0e}"] = dict(sumc=sc_, slip=e, sd=sd, slip_fixed=ef)
    print(f"{lam:9.0e}{sc_:8.1f}{e:+11.6f}{sd:10.6f}{ef:+13.6f}")

print(f"\n=== ЗАВИСИМОСТЬ ОТ РАЗМЕРА БАЗИСА (lam = 1e-3) ===")
print(f"{'k':>5}{'S|c|':>8}{'E[слип]':>11}{'на ось':>11}")
oth = [i for i in range(k) if i != a]
dim = {}
for kk in (4, 8, 12, 16, 20, 24, 30):
    ee, cc = [], []
    for _ in range(8):
        sub = sorted(rng.choice(oth, size=min(kk, len(oth)), replace=False))
        Bs = np.vstack([np.ones(N), D[sub]]); Gs = Bs @ Bs.T / N
        e, _, _, sc_ = run(sub, 1e-3, np.concatenate([[MEAN_T], psi0[1:][sub]]),
                           Gs, Bs @ lp_a / N)
        ee.append(e); cc.append(sc_)
    dim[kk] = float(np.mean(ee))
    print(f"{kk:5d}{np.mean(cc):8.1f}{np.mean(ee):+11.6f}{np.mean(ee)/kk:+11.7f}")
json.dump(dict(lam=res, dim=dim), open("out/n4_curse.json", "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)

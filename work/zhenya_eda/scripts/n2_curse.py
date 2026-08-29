# -*- coding: utf-8 -*-
"""N2. Winner's curse спан-арбитража, ИЗ ПЕРВЫХ ПРИНЦИПОВ.

Тождество. Пусть delta(u) = mean_250k(u) - mean_public(u).  Предсказатель берёт
phi_i из ПУБЛИЧНОГО скора, но mean(lp^2) считает по 250k.  Тогда для любого
файла спана lp = lp_a + c0 + sum c_i d_i  (d_i = lp_i - lp_a):

    слип = факт - расчёт = delta(u),
    u = sum_i c_i*d_i^2 - (c0 + sum_i c_i*d_i)^2 - 2*c0*lp_a

u зависит ТОЛЬКО от lp-векторов, таргет сокращается => слип считается локально.

1) при ФИКСИРОВАННОМ c:  E[слип]=0,  sd = sd_250k(u)*sqrt((1-f)/n_pub).
2) когда c ПОДБИРАЕТСЯ под замеры, оптимизатор эксплуатирует delta:
       E[слип] ~ 0.5*tr[(G+lam*R)^-1 * C],   C = Cov_250k(v)*(1-f)/n_pub,
   v_i = lp_i^2 - lp_a^2.  Это и есть проклятие победителя.
"""
import json, math, sys
import numpy as np
sys.stdout.reconfigure(encoding="utf-8")

F0R, N_PUB, N_ALL = 1.6470, 50_000, 250_000
FR = N_PUB/N_ALL; VARF = (1.0-FR)/N_PUB          # множитель дисперсии delta
MEAN_T = 2.3275
ANCH = "A1_gram7_shift"

z = np.load("out/lb_full.npz"); meta = json.load(open("out/lb_meta.json"))
names = meta["names"]; sc = meta["sc"]
L = np.vstack([z[f"lp_{n}"].astype(np.float64) for n in names])
f = np.array([sc[n] for n in names])
a = names.index(ANCH); lp_a = L[a]; N = L.shape[1]
D = L - lp_a                                       # направления
V = L*L - lp_a*lp_a                                # v_i, для ковариации delta
print(f"базис {len(names)} файлов, {N} юзеров, якорь {ANCH} f={f[a]:.7f}")

SHOWS = ["SHOW_maxpub", "SHOW2_aggr", "SHOW3_maxpub", "SHOW3b_safe"]
show_i = [names.index(s) for s in SHOWS]
idx = [i for i in range(len(names)) if i not in show_i]      # базис БЕЗ SHOW

def parts(idx):
    B = np.vstack([np.ones(N), D[idx]])
    G = B @ B.T / N
    m = B @ lp_a / N
    qd = (L*L).mean(1)
    psi = ((qd - qd[a]) - (f**2 - f[a]**2))/2
    return B, G, m, np.concatenate([[MEAN_T], psi[idx]])

B, G, m, psi = parts(idx)
# --- ковариация delta по базису (v_i центрируем по 250k)
Vc = V[idx] - V[idx].mean(1, keepdims=True)
C = (Vc @ Vc.T / N) * VARF                          # k x k, без строки константы
C = np.pad(C, ((1,0),(1,0)))                        # константа: delta(1)=0 точно
print(f"sd(delta(v_i)) по базису: медиана {np.sqrt(np.diag(C)[1:]).mean():.6f}")

def slip_of(c):
    """точная величина u и её sd для данного вектора c (первый элемент — c0)"""
    c0 = c[0]; ci = c[1:]
    d = ci @ D[idx]
    u = (ci[:,None]*(D[idx]**2)).sum(0) - (c0 + d)**2 - 2*c0*lp_a
    return float(np.std(u)*math.sqrt(VARF))

def project(lp):
    Gd = G + 1e-9*np.trace(G)/len(G)*np.eye(len(G))
    c = np.linalg.solve(Gd, B @ (lp - lp_a)/N)
    q = float((lp*lp).mean()); qa = float((lp_a*lp_a).mean())
    fsq = f[a]**2 + (q-qa) - 2*float(c @ psi)
    r = lp - lp_a - B.T@c
    return math.sqrt(max(fsq,1e-12)), float(np.abs(c[1:]).sum()), c, float(np.sqrt((r*r).mean()))

print(f"\n=== 1. ФИКСИРОВАННЫЙ c: слип должен быть НУЛЕВЫМ в среднем ===")
print(f"{'файл':16s}{'расчёт':>11}{'факт':>11}{'промах':>10}{'sd(слип)':>10}{'сигм':>7}{'S|c|':>7}{'ост':>8}")
obs = []
for s in SHOWS:
    lp = L[names.index(s)]
    pred, w1, c, res = project(lp)
    miss = sc[s] - pred
    sd = slip_of(c)/(2*F0R)
    obs.append((s, w1, miss, sd, len(idx)))
    print(f"{s:16s}{pred:11.7f}{sc[s]:11.7f}{miss:+10.6f}{sd:10.6f}{miss/max(sd,1e-12):7.1f}"
          f"{w1:7.1f}{res:8.4f}")

print(f"\n=== 2. ПРОКЛЯТИЕ ПОБЕДИТЕЛЯ: E[слип] = 0.5*tr[(G+lam R)^-1 C] / (2*F0) ===")
R = np.eye(len(m)); R[0,0] = 1e-4
print(f"{'lam':>10}{'E[слип]':>12}{'S|c|':>9}{'расчёт':>12}")
for lam in (3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5):
    A = np.linalg.solve(G + lam*R, C)
    e = 0.5*np.trace(A)/(2*F0R)
    c = np.linalg.solve(G + lam*R, psi - m)
    fsq = f[a]**2 + 2*c@m + c@G@c - 2*c@psi
    print(f"{lam:10.1e}{e:12.6f}{np.abs(c[1:]).sum():9.1f}{math.sqrt(max(fsq,1e-12)):12.7f}")

print(f"\n=== 3. МАСШТАБИРОВАНИЕ ПО РАЗМЕРНОСТИ СПАНА (lam=1e-3) ===")
print(f"{'k':>5}{'E[слип]':>12}{'на ось':>12}")
rng = np.random.default_rng(0)
for k in (4, 8, 12, 16, 20, 24, 28):
    ee = []
    for _ in range(12):
        sub = [0] + list(1 + rng.choice(len(idx), size=min(k, len(idx)), replace=False))
        Gs, Cs = G[np.ix_(sub,sub)], C[np.ix_(sub,sub)]
        Rs = np.eye(len(sub)); Rs[0,0] = 1e-4
        ee.append(0.5*np.trace(np.linalg.solve(Gs + 1e-3*Rs, Cs))/(2*F0R))
    print(f"{k:5d}{np.mean(ee):12.6f}{np.mean(ee)/k:12.7f}")
json.dump(dict(obs=[(o[0],o[1],o[2],o[3]) for o in obs]),
          open("out/n2_curse.json","w",encoding="utf-8"), ensure_ascii=False, indent=1)

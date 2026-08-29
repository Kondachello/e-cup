# -*- coding: utf-8 -*-
"""N3. ПРЯМАЯ симуляция всего конвейера: замер -> оптимизация -> факт паблик/приват.

Геометрия lp-векторов настоящая (замеренных файла, 250k юзеров). Роль
неизвестного тестового таргета играет ВАЛИДАЦИОННЫЙ (tval) — механизм от этого
не меняется, меняется только абсолютный уровень F0.

Каждый розыгрыш:
  1. случайные 50k = "публика", остальные 200k = "приват";
  2. считаем "замеренные скоры" файлов на публике (это то, что даёт платформа);
  3. предсказатель восстанавливает psi из ЭТИХ скоров, но mean(lp^2) берёт по 250k;
  4. ридж-оптимизация -> c -> расчётный скор;
  5. факт на публике (слип = проклятие победителя) и факт на ПРИВАТЕ (что достанется).
"""
import json, math, sys
import numpy as np
sys.stdout.reconfigure(encoding="utf-8")

NP_, NA = 50_000, 250_000
ANCH = "A1_gram7_shift"
DROP = {"sample"}   # sample_submit = сам вал-таргет: в спане даёт фиктивную утечку
z = np.load("out/lb_full.npz"); meta = json.load(open("out/lb_meta.json"))
names = [n for n in meta["names"] if n not in DROP]
L = np.vstack([z[f"lp_{n}"].astype(np.float64) for n in names])
t = z["tval"].astype(np.float64)
N = L.shape[1]; a = names.index(ANCH); lp_a = L[a]; D = L - lp_a
qd = (L*L).mean(1)                       # по 250k — так и делает предсказатель
rms = lambda v, i: math.sqrt(float(np.mean((v[i]-t[i])**2)))
print(f"F0 якоря на вал-таргете (250k): {math.sqrt(float(np.mean((lp_a-t)**2))):.6f}")

B_all = np.vstack([np.ones(N), D])
G = B_all @ B_all.T / N
m = B_all @ lp_a / N
LAMS = (3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4)
NDRAW = 40
rng = np.random.default_rng(7)
acc = {l: [] for l in LAMS}
for it in range(NDRAW):
    P = rng.choice(N, NP_, replace=False)
    msk = np.zeros(N, bool); msk[P] = True; Q = np.where(~msk)[0]
    fP = np.array([rms(L[i], P) for i in range(len(names))])
    psi = ((qd - qd[a]) - (fP**2 - fP[a]**2))/2
    psiv = np.concatenate([[float(t[P].mean())], psi])
    for lam in LAMS:
        R = np.eye(len(m)); R[0,0] = 1e-4
        c = np.linalg.solve(G + lam*R, psiv - m)
        fsq = fP[a]**2 + 2*c@m + c@G@c - 2*c@psiv
        calc = math.sqrt(max(fsq, 1e-12))
        lp = lp_a + B_all.T @ c
        acc[lam].append((calc, rms(lp, P), rms(lp, Q), float(np.abs(c[1:]).sum()),
                         rms(lp_a, P), rms(lp_a, Q)))
print(f"\n=== {NDRAW} розыгрышей раскола 50k/200k, базис {len(names)} ===")
print(f"{'lam':>9}{'S|c|':>8}{'расчёт':>10}{'ФАКТ пабл':>11}{'СЛИП':>10}"
      f"{'ФАКТ прив':>11}{'П-ПР':>10}{'выигр.ПР':>10}")
out = {}
for lam in LAMS:
    A = np.array(acc[lam])
    calc, fp, fq, w1, ap, aq = A.T
    slip = fp - calc
    print(f"{lam:9.0e}{w1.mean():8.1f}{calc.mean():10.6f}{fp.mean():11.6f}"
          f"{slip.mean():+10.6f}{fq.mean():11.6f}{(fq-fp).mean():+10.6f}"
          f"{(fq-aq).mean():+10.6f}")
    out[f"{lam:.0e}"] = dict(sum_c=w1.mean(), calc=calc.mean(), pub=fp.mean(),
                             slip=slip.mean(), slip_sd=float(slip.std()),
                             priv=fq.mean(), priv_gain=float((fq-aq).mean()),
                             pub_gain=float((fp-ap).mean()))
print("\nСЛИП = ФАКТ_паблик - расчёт  (проклятие победителя)")
print("выигр.ПР = ФАКТ_приват - приват якоря  (отрицательное = реально лучше)")
print(f"\n{'lam':>9}{'выигр.П':>11}{'выигр.ПР':>11}{'перенос':>10}{'слип sd':>10}")
for lam in LAMS:
    o = out[f"{lam:.0e}"]
    tr = o["priv_gain"]/o["pub_gain"] if abs(o["pub_gain"]) > 1e-12 else float("nan")
    print(f"{lam:9.0e}{o['pub_gain']:+11.6f}{o['priv_gain']:+11.6f}{tr:10.3f}{o['slip_sd']:10.6f}")
json.dump(out, open("out/n3_sim.json","w",encoding="utf-8"), ensure_ascii=False, indent=1)

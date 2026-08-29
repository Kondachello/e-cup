#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
A6 — связь правки приора A1 (tau_model 0.196 -> 0.2855) с ранговым треком: k(F8).

Считает k(F8) шестью маршрутами R2c (work/reports/rank/r2c_countersign_f8.md)
под разными приорами, из выгрузок матриц солвера линии A.

Вход:  work/reports/lineA/gls_state_base.npz  (прогон со СТАРЫМ приором = F8 бит-в-бит)
       work/reports/lineA/gls_state_eb.npz    (прогон с НОВЫМ приором A1)
Выход: work/reports/lineA/a6_k_link.json

Контроль: маршруты под старым приором обязаны воспроизвести R2c
          (A 37.0, A' 44.4, B 16.2, C 23.0, D 19.2, E 23.3).

Запуск:  OMP_NUM_THREADS=4 .venv/bin/python work/reports/lineA/a6_k_link.py
Ничего не пишет вне work/reports/lineA/. Солвер не трогает.
"""
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
LA = os.path.join(ROOT, "work", "reports", "lineA")

MODEL = ["mdl_amber", "mdl_gabbro", "mdl_halite", "mdl_marble", "mdl_realgr", "mdl_tektit", "mdl_olivin", "mdl_flint", "mdl_gypsum",
         "mdl_gneis2", "mdl_malach", "", "mdl_vivian", "mdl_corund", "mdl_larvik", "mdl_talc"]

UNIT = 3.29e-5          # цена одного «фейкового направления» на приватной шкале (R2a/R2c)
PUB_F8 = 1.6458057389   # замеренный паблик F8_priv
NOISE = 2.2e-5

zb = np.load(os.path.join(LA, "gls_state_base.npz"), allow_pickle=True)
ze = np.load(os.path.join(LA, "gls_state_eb.npz"), allow_pickle=True)

names = [str(x) for x in zb["names"]]
n = len(names)
q = zb["q"]
Q = zb["Q"]
F = float(zb["F_SCALE"])
u = (0.8 / 50000) * F ** 2      # единица «фейка» в единицах c, выведена в R2c §2


def taus(tau_model, tau_decomp=0.0148, tau_seg=0.1410):
    return np.array([tau_model if x in MODEL else
                     (tau_decomp if x.startswith("Z") else tau_seg) for x in names])


# Sigma_eps восстанавливается из V прогона: V = (Sp^-1 + Se^-1)^-1
t0 = taus(0.196)
Sp0 = np.diag((q * t0) ** 2)
Se = np.linalg.inv(np.linalg.inv(zb["mdl_vivian"]) - np.linalg.inv(Sp0))

# сверка: Se не должна зависеть от приора
t1 = taus(0.2855)
Se_check = np.linalg.inv(np.linalg.inv(ze["mdl_vivian"]) - np.linalg.inv(np.diag((q * t1) ** 2)))
se_rel = float(np.abs(Se - Se_check).max() / np.abs(Se).max())

sigma = np.sqrt(np.diag(Se)) / q      # поосная sigma в шкале kappa


def routes(tau_model, tau_decomp=0.0148, tau_seg=0.1410, gamma=0.1):
    t = taus(tau_model, tau_decomp, tau_seg)
    Sp = np.diag((q * t) ** 2)
    V = np.linalg.inv(np.linalg.inv(Sp) + np.linalg.inv(Se))
    K = np.eye(n) - V @ np.linalg.inv(Sp)
    w = t ** 2 / (t ** 2 + sigma ** 2)
    Lam = np.diag(q * (1 - w) * t ** 2)
    M = Q + Lam + gamma * np.diag(np.diag(Q))
    C = 1.25 * K - 0.25 * np.eye(n)
    return {
        "A":  float(np.trace(np.linalg.solve(M, Q))),
        "Ap": float(np.trace(np.linalg.solve(M, Se)) / u),
        "B":  float(np.trace(np.linalg.solve(M, C @ Se)) / u),
        "C":  float(1.25 * np.trace(K) - 0.25 * n),
        "D":  float(np.sum(1.25 * w - 0.25)),
        "E":  float(np.trace(np.linalg.solve(Q, C @ Se)) / u),
        "trK": float(np.trace(K)),
        "sum_w": float(w.sum()),
    }


SCEN = [
    ("registry",        "реестр 0.196/0.0148/0.141",             0.196, 0.0148, 0.1410),
    ("A1",              "A1: model 0.2855",                      0.2855, 0.0148, 0.1410),
    ("A1_A4d",          "A1 + decomp 0.0209 (д)",              0.2855, 0.0209, 0.1410),
    ("A1_A5",           "A1 + decomp 0.030 ()",                0.2855, 0.0300, 0.1410),
    ("all_x1457",       "все семьи x1.457 (калибровка R2c)",     0.196 * 1.4566, 0.0148 * 1.4566, 0.1410 * 1.4566),
    ("A5_full",         " полн.: 0.2855 / 0.030 / 0.141x1.35", 0.2855, 0.0300, 0.1410 * 1.35),
]

out = {
    "task": "A6 — k(F8) под правкой приора A1",
    "unit_priv_per_k": UNIT,
    "pub_F8": PUB_F8,
    "noise": NOISE,
    "n_axes": n,
    "sigma_eps_prior_independent_relerr": se_rel,
    "scenarios": {},
    "gamma_effect": {},
}

for key, label, tm, td, ts in SCEN:
    r = routes(tm, td, ts, gamma=0.1)
    r["label"] = label
    r["priv_F8_route_C"] = PUB_F8 + UNIT * r["C"]
    out["scenarios"][key] = r

base = out["scenarios"]["registry"]
for key in out["scenarios"]:
    s = out["scenarios"][key]
    s["dk_C_vs_registry"] = s["C"] - base["C"]
    s["dpriv_vs_registry"] = UNIT * (s["C"] - base["C"])
    s["dpriv_in_noises"] = UNIT * (s["C"] - base["C"]) / NOISE

# влияние gamma (маршрут B — единственный, где gamma входит; C/D/E от gamma не зависят)
for g in [0.0, 0.03, 0.05, 0.0765, 0.08, 0.1, 0.2, 0.3]:
    out["gamma_effect"]["%.4f" % g] = {
        "B_old_prior": routes(0.196, gamma=g)["B"],
        "B_new_prior": routes(0.2855, gamma=g)["B"],
        "A_old_prior": routes(0.196, gamma=g)["A"],
    }

# что это делает с порогами витрин (R2a)
SHOWS = {"SHOW9_l1e2": (1.6446539, 58.5), "SHOW10_l3e3": (1.6446515, 66.0),
         "SHOW11_hull4": (1.6440064, 96.0)}
out["showcases"] = {}
for nm, (pub, keff) in SHOWS.items():
    row = {"pub": pub, "k_eff_own": keff}
    for key in ("registry", "A1"):
        priv_f8 = out["scenarios"][key]["priv_F8_route_C"]
        row[key] = {"k_breakeven": (priv_f8 - pub) / UNIT,
                    "show_minus_F8": (pub + UNIT * keff) - priv_f8}
    out["showcases"][nm] = row

with open(os.path.join(LA, "a6_k_link.json"), "w") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)

print("Sigma_eps prior-independent, rel.err = %.2e" % se_rel)
print("%-38s %6s %6s %6s %6s %6s %6s" % ("сценарий", "A", "A'", "B(.1)", "C", "D", "E"))
for key, label, *_ in SCEN:
    s = out["scenarios"][key]
    print("%-38s %6.2f %6.2f %6.2f %6.2f %6.2f %6.2f" %
          (label, s["A"], s["Ap"], s["B"], s["C"], s["D"], s["E"]))
print()
for key, label, *_ in SCEN:
    s = out["scenarios"][key]
    print("%-38s k_C=%6.2f  dk=%+5.2f  E[priv F8]=%.7f  d=%+.2e (%.1f шума)" %
          (label, s["C"], s["dk_C_vs_registry"], s["priv_F8_route_C"],
           s["dpriv_vs_registry"], s["dpriv_in_noises"]))
print("\nJSON: %s" % os.path.join(LA, "a6_k_link.json"))

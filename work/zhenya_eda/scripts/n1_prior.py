# -*- coding: utf-8 -*-
"""mdl_wulfen. Приор kappa на обновлённом реестре + ПРИВАТНАЯ усадка + налог + переоценка.

Ключевая новая вещь: публика (50k) — ПОДМНОЖЕСТВО тех же 250k, поэтому ошибка
публичного замера и приватный остаток АНТИКОРРЕЛИРОВАНЫ. Отсюда приватный вес
усадки не равен публичному.
"""
import json, math, sys
import numpy as np
sys.stdout.reconfigure(encoding="utf-8")

F0, N_PUB, N_ALL = 1.6470, 50_000, 250_000
FR = N_PUB / N_ALL                       # 0.2
NOISE = 0.000022
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
REG = json.load(open(ROOT / "work/zhenya_eda/kappa_registry.json", encoding="utf-8"))

# --- новые точки вечера 26.08 (в реестре Саши их ещё нет) ---------------------
NEW = [
    dict(axis="R7_reblend_full", cls="пересборка бленда", q=0.0249, kappa=0.113,
         prov="зонд полным шагом", dose=0.124, dose_src="усадка w=0.95 -> R8"),
    dict(axis="T_tfm4", cls="новая модель", q=0.003041, kappa=0.459,
         prov="АЛГЕБРА: парабола по двум дозам T1/T2", dose=0.45, dose_src="T2"),
]

DOSE = {"R2_blend_delta": (0.618, ", сырая k"), "R3_ridge_stack": (0.308, ", сырая k"),
        "X2_crumbs_bundle": (1.301, "сырая k"), "E1_erafix": (0.0, "ось закрыта"),
        "R7_reblend_full": (0.124, "усадка w=0.96 -> R8"), "T_tfm4": (0.45, "T2")}

def sigma_law(q, fpc=True):
    s = F0 / math.sqrt(N_PUB * q)
    return s * math.sqrt(1.0 - FR) if fpc else s

pts = []
for a in REG["axes"]:
    d, src = DOSE.get(a["axis"], (None, "?"))
    pts.append(dict(axis=a["axis"], cls=a["cls"], q=a["q"], kappa=a["kappa"],
                    sigma_reg=a["sigma"], prov="зонд против базы", dose=d, dose_src=src))
have = {p["axis"] for p in pts}
pts += [a for a in NEW if a["axis"] not in have]
for p in pts:
    p["sigma"] = sigma_law(p["q"])
    p["sigma_nofpc"] = sigma_law(p["q"], fpc=False)

k = np.array([p["kappa"] for p in pts]); s = np.array([p["sigma"] for p in pts])
nll = lambda mu, t2: 0.5*np.sum(np.log(2*np.pi*(t2+s**2)) + (k-mu)**2/(t2+s**2))
MUS, T2S = np.linspace(-0.4, 1.0, 1401), np.geomspace(1e-5, 2.0, 600)
grid = np.array([[nll(m, t) for t in T2S] for m in MUS])
i, j = np.unravel_index(grid.argmin(), grid.shape)
MU0, TAU2 = float(MUS[i]), float(T2S[j]); TAU = math.sqrt(TAU2)
fmin = grid.min()
mu_ok = MUS[(grid.min(1) <= fmin+1.92)]; ta_ok = np.sqrt(T2S[(grid.min(0) <= fmin+1.92)])
print(f"=== ПЛОСКИЙ ПРИОР, ML по {len(pts)} осям (sigma по закону с fpc) ===")
print(f"mu0 = {MU0:.3f}  [{mu_ok.min():.3f}, {mu_ok.max():.3f}]   "
      f"tau = {TAU:.3f}  [{ta_ok.min():.3f}, {ta_ok.max():.3f}]")
print(f"рабочий приор был N(0.333, 0.204^2) — подтверждён, tau чуть уже")
print(f"\nПоправка на конечную популяцию: sigma_закон в реестре завышена в "
      f"{1/math.sqrt(1-FR):.3f} раза (публика — ПОДМНОЖЕСТВО 250k, а не независимая выборка)")

print("\n=== ЗАКОН ПРИВАТНОЙ УСАДКИ ===")
print("  kappa_T (все 250k) ~ N(mu0, tau^2);  замер kappa_P = kappa_T + eps, sd(eps)=sigma")
print("  приват = дополнение публики  =>  kappa_Q = kappa_T - (f/(1-f))*eps = kappa_T - 0.25*eps")
print("  E[kappa_Q | kappa_P] = mu0 + w_priv*(kappa_P - mu0),  w_priv = 1.25*w - 0.25")
print("  sd[kappa_Q | kappa_P] = 1.25*tau*sqrt(1-w)")
print("  * w = 0.20  -> w_priv = 0   замер БЕСПОЛЕЗЕН для привата")
print("  * w < 0.20  -> w_priv < 0   замер АНТИ-информативен (двигать дозу ПРОТИВ чтения)")
print("  * при mu0=0 публичная доза w*k имеет E[приват] = 0.5*k^2*w*(3w-1) -> нужен w > 1/3")

rows = []
for p in pts:
    sg, q, kp = p["sigma"], p["q"], p["kappa"]
    w = TAU2/(TAU2 + sg*sg); wp = 1.25*w - 0.25
    b_opt_pub = MU0 + w*(kp - MU0)
    b_opt_pri = MU0 + wp*(kp - MU0)
    b = p["dose"] if p["dose"] is not None else b_opt_pub      # JS-дозы = усадка
    ekq = b_opt_pri
    g_pub = q*(2*b*kp - b*b)/(2*F0)
    g_pri = q*(2*b*ekq - b*b)/(2*F0)
    g_best = q*(ekq*ekq)/(2*F0) if ekq > 0 else 0.0
    rows.append(dict(axis=p["axis"], q=q, kappa=kp, sigma=sg, w=w, w_priv=wp,
                     dose=b, dose_src=p["dose_src"], b_opt_pub=b_opt_pub, b_opt_pri=b_opt_pri,
                     gain_pub=g_pub, gain_priv=g_pri, over=g_pub-g_pri,
                     regret=g_best-g_pri))

print(f"\n{'ось':22s}{'q':>9}{'kappa':>7}{'sigma':>7}{'w':>6}{'w_pr':>7}"
      f"{'доза':>7}{'опт_ПР':>8}{'урож_П':>9}{'ожид_ПР':>9}{'переоц':>9}")
for r in sorted(rows, key=lambda r: -r["q"]):
    print(f"{r['axis']:22s}{r['q']:9.5f}{r['kappa']:7.3f}{r['sigma']:7.3f}{r['w']:6.3f}"
          f"{r['w_priv']:7.3f}{r['dose']:7.3f}{r['b_opt_pri']:8.3f}"
          f"{r['gain_pub']:9.6f}{r['gain_priv']:9.6f}{r['over']:9.6f}")

app = [r for r in rows if abs(r["dose"]) > 1e-9]
print(f"\nПРИМЕНЁННЫХ ОСЕЙ: {len(app)} из {len(rows)}")
print(f"  публичный урожай (сумма):        {sum(r['gain_pub'] for r in app):+.6f}")
print(f"  ожидаемый приватный (сумма):     {sum(r['gain_priv'] for r in app):+.6f}")
print(f"  ПЕРЕОЦЕНКА ПАБЛИКА:              {sum(r['over'] for r in app):+.6f} "
      f"= {sum(r['over'] for r in app)/NOISE:.1f} шума")
print(f"  сожаление к приватному оптимуму: {sum(r['regret'] for r in app):+.6f} "
      f"= {sum(r['regret'] for r in app)/NOISE:.1f} шума")
json.dump(dict(mu0=MU0, tau=TAU, rows=rows),
          open(ROOT / "work/zhenya_eda/out/n1_prior.json", "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)

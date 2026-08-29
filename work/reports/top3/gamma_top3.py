# -*- coding: utf-8 -*-
"""GAMMA — сетка gamma под цель ТОП-3.  Линия TOP-3, 29.08.2026.

Всё считается из выгруженных матриц солвера (a5_loo_state_a5.npz — конфигурация
кандидата F9: 46 осей, новый приор EB, база F7), поэтому каждое число
"""
from __future__ import annotations
import json, math, sys, os
import numpy as np

sys.path.insert(0, "/Users/alexanderkondakov/ozon-cup/work/scripts")
from p_top3 import Objective, MU_US, SIGMA_US, NOISE   # noqa: E402

R = "/Users/alexanderkondakov/ozon-cup/work/reports/lineA/"
S = np.load(R + "a5_loo_state_a5.npz", allow_pickle=True)
E = np.load(R + "gls_state_eb.npz", allow_pickle=True)      # тот же прогон, но с V
A4A = np.load(R + "gls_state_a4all.npz", allow_pickle=True)
A4B = np.load(R + "gls_state_a4base.npz", allow_pickle=True)
GB = np.load(R + "gram_boot_a2.npz", allow_pickle=True)

Q, cQ, cP, mu_c = S["Q"], S["cQ"], S["cP"], S["mu_c"]
Lam, Sp, Se = S["Lam"], S["Sig_p"], S["Sig_e"]
q, kap, dot, fam = S["q"], S["kap"], S["dot"], S["fam"]
dF8, dF9 = S["d_F8"], S["d_F9"]
F = float(S["F_SCALE"]); NZ = float(S["NOISE"])
V = E["mdl_vivian"]
names = S["names"]
assert abs(E["Q"] - Q).max() == 0.0 and abs(E["cQ"] - cQ).max() < 1e-17

dQ = np.diag(np.diag(Q))


def g(d, c):
    """алгебра E[priv] - F7 для вектора доз d при кредите c"""
    return float((2 * d @ c - d @ Q @ d) / (2 * F))


def dose(gam, Qm=Q, cm=cQ, Lm=Lam):
    return np.linalg.solve(Qm + Lm + gam * np.diag(np.diag(Qm)), cm)


def sd_gain(d):
    """sd алгебры: c_Q = 1.25 m* - 0.25 c_P, апостериорная var(m*) = V"""
    return float(1.25 * math.sqrt(d @ V @ d) / F)


#кредит по замеру только у групп, переживших групповой LOO
soft_mask = np.isin(fam, ["model", "decomp"])
c_soft = np.where(soft_mask, cQ, mu_c)

#досыпка 38 осей меняет ТОЛЬКО c_Q на старом блоке (Грам-минор тождественен)
assert abs(A4A["Q"][:46, :46] - A4B["Q"]).max() < 1e-13
dc_watch = A4A["cQ"][:46] - A4B["cQ"]          # сдвиг кредита от досыпки Z-осей


def watchdog(d):
    """просадка E при честном (расширенном) спане, в единицах скора (>0 = потеря)"""
    return float(-d @ dc_watch / F)


# ---------------------------------------------------------------- бутстрап Грама
def gram_ensemble(gammas, amp=1.0):
    """A2: дозы считаются по кривому Грам-у, выигрыш зачитывается по истинному Q."""
    QB, DB = GB["QB"], GB["DOTB"]
    nb = QB.shape[0]
    out = {gam: [] for gam in gammas}
    SpSe = Sp + Se
    for b in range(nb):
        Qh = Q + amp * (QB[b] - Q)
        dh = dot + amp * (DB[b] - dot)
        qh = np.diag(Qh)
        cPh = qh * kap - dh
        mh = mu_c + Sp @ np.linalg.solve(SpSe, cPh - mu_c)
        cQh = 1.25 * mh - 0.25 * cPh
        for gam in gammas:
            try:
                d_b = np.linalg.solve(Qh + Lam + gam * np.diag(qh), cQh)
            except np.linalg.LinAlgError:
                continue
            out[gam].append(g(d_b, cQ))          # реализация по ИСТИННОМУ кредиту/Граму
    return {k: np.array(v) for k, v in out.items()}


# ---------------------------------------------------------------- сетка
GRID = [0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.0765, 0.08, 0.1]
FINE = sorted(set([round(x, 5) for x in np.arange(0.0, 0.2001, 0.0025)] + GRID))

DISC_SOFT = 1.72          # мягкий дисконт A5 (буквальный)

g_in_F8, g_soft_F8, g_oof_F8 = g(dF8, cQ), g(dF8, c_soft), g(dF8, mu_c)
wd_F8 = watchdog(dF8)

# сверка с подписанными числами
CHK = dict(
    g_in_F9=g(dF9, cQ), g_oof_F9=g(dF9, mu_c),
    F9_minus_F8_in_noises=(g(dF9, cQ) - g_in_F8) / NZ,
    F9_minus_F8_oof_noises=(g(dF9, mu_c) - g_oof_F8) / NZ,
    watchdog_F8=wd_F8, watchdog_F8_pct=100 * wd_F8 / g(dF8, cQ),
    sum_opt_A5=g(dF9, cQ) - g(dF9, mu_c),
    soft_share_F9=(g(dF9, c_soft)) / g(dF9, cQ),
)

A3 = json.load(open(R + "a3_gamma_path.json", encoding="utf-8"))
A3P = {round(p["gamma"], 6): p for p in A3["path"]}

obj = Objective(ns=400_000)
P_F8_now = obj.P_top3(MU_US, SIGMA_US)

sd_ref = None
rows = []
for gam in FINE:
    d = dose(gam)
    r = dict(gamma=gam, g_in=g(d, cQ), g_soft=g(d, c_soft), g_oof=g(d, mu_c),
             sd=sd_gain(d), wd=watchdog(d), norm=float(np.linalg.norm(d)),
             maxabs=float(np.abs(d).max()))
    rows.append(r)
    if abs(gam - 0.1) < 1e-9:
        sd_ref = r["sd"]


chk_path = []
for r in rows:
    p = A3P.get(round(r["gamma"], 6))
    if p:
        chk_path.append(dict(gamma=r["gamma"],
                             d_gain=r["g_in"] - p["gain"],
                             d_sd=r["sd"] - p["sd"],
                             d_vsF8=(r["g_in"] - g_in_F8) / NZ - p["vs_F8_noises"]))
CHK["path_max_abs_err"] = dict(
    gain=max(abs(c["d_gain"]) for c in chk_path),
    sd=max(abs(c["d_sd"]) for c in chk_path),
    vsF8=max(abs(c["d_vsF8"]) for c in chk_path))

VAR_SHARED = SIGMA_US ** 2 - sd_ref ** 2      # общая (не-дозовая) часть нашей сигмы


def sigma_of(sd, extra=0.0, mode="nested"):
    if mode == "nested":
        v = VAR_SHARED + sd ** 2
    else:                                      # аддитивная сенситивность
        v = SIGMA_US ** 2 + sd ** 2
    return math.sqrt(v + extra ** 2)


# ---------------------------------------------------------------- модельный риск
gam_boot_nat = gram_ensemble(GRID, amp=1.0)
gam_boot_amp = gram_ensemble(GRID, amp=5.5)   #структура бутстрапа x5.5 = 10 % сдвиг

MR = {}
for gam in GRID:
    d = dose(gam)
    base = g(d, cQ)
    nat = base - gam_boot_nat[gam]
    amp = base - gam_boot_amp[gam]
    MR[gam] = dict(
        regret_p95_nat=float(np.percentile(nat, 95)), regret_max_nat=float(nat.max()),
        regret_p95_amp=float(np.percentile(amp, 95)), regret_max_amp=float(amp.max()),
        sd_amp=float(gam_boot_amp[gam].std()),
        wd=watchdog(d), wd_excess=watchdog(d) - wd_F8,
    )

# ---------------------------------------------------------------- сборка
def pack(r, regime, mrisk=None, mode="nested"):
    if regime == "raw":
        delta = r["g_in"] - g_in_F8
    elif regime == "soft_flat":
        delta = (r["g_in"] - g_in_F8) / DISC_SOFT
    elif regime == "soft_exact":
        delta = r["g_soft"] - g_soft_F8
    elif regime == "oof":
        delta = r["g_oof"] - g_oof_F8
    mu = MU_US - delta
    extra = 0.0
    if mrisk == "var":                    # генеровое чтение: модельная ошибка = симметричный разброс
        m = MR[round(r["gamma"], 6)]
        extra = math.hypot(m["regret_p95_amp"] / 1.645, abs(m["wd_excess"]))
    elif mrisk == "bias":                 # честное чтение: знак известен, это смещение
        m = MR[round(r["gamma"], 6)]
        mu = mu + m["wd_excess"] + m["regret_p95_amp"] / 1.645 * 0.0
    elif mrisk == "both":
        m = MR[round(r["gamma"], 6)]
        mu = mu + m["wd_excess"]
        extra = m["regret_p95_amp"] / 1.645
    sig = sigma_of(r["sd"], extra, mode)
    return mu, sig, delta


out_grid = []
for gam in GRID:
    r = [x for x in rows if abs(x["gamma"] - gam) < 1e-9][0]
    e = dict(gamma=gam, sd=r["sd"], sd_noises=r["sd"] / NZ, wd=r["wd"],
             wd_excess=r["wd"] - wd_F8, wd_excess_noises=(r["wd"] - wd_F8) / NZ,
             pub_algebra=A3P[round(gam, 6)]["pub_algebra"] if round(gam, 6) in A3P else None,
             max_db=A3P[round(gam, 6)]["max_delta_b"] if round(gam, 6) in A3P else None,
             n_material=len(A3P[round(gam, 6)]["material"]) if round(gam, 6) in A3P else None)
    for reg in ("raw", "soft_flat", "soft_exact", "oof"):
        mu, sig, delta = pack(r, reg)
        e[reg] = dict(delta_vs_F8=delta, delta_noises=delta / NZ, mu=mu, sigma=sig,
                      p3=obj.P_top3(mu, sig), p5=obj.P_topk(5, mu, sig))
        mu2, sig2, _ = pack(r, reg, mode="add")
        e[reg]["p3_sigma_additive"] = obj.P_top3(mu2, sig2)
    for mrisk in ("var", "bias", "both"):
        mu, sig, delta = pack(r, "soft_flat", mrisk=mrisk)
        e["mr_" + mrisk] = dict(mu=mu, sigma=sig, p3=obj.P_top3(mu, sig))
        mu, sig, _ = pack(r, "raw", mrisk=mrisk)
        e["mr_" + mrisk + "_raw"] = dict(mu=mu, sigma=sig, p3=obj.P_top3(mu, sig))
    e["gram"] = MR[gam]
    out_grid.append(e)

# тонкая сетка — поиск gamma*
fine = []
for r in rows:
    row = dict(gamma=r["gamma"])
    for reg in ("raw", "soft_flat", "soft_exact", "oof"):
        mu, sig, delta = pack(r, reg)
        row[reg] = obj.P_top3(mu, sig)
    fine.append(row)

res = dict(check=CHK, base_P=P_F8_now, MU_US=MU_US, SIGMA_US=SIGMA_US, NOISE=NZ,
           sd_ref=sd_ref, var_shared_sigma=math.sqrt(VAR_SHARED),
           g_in_F8=g_in_F8, g_soft_F8=g_soft_F8, g_oof_F8=g_oof_F8, wd_F8=wd_F8,
           grid=out_grid, fine=fine)
with open("/private/tmp/claude-501/-Users-alexanderkondakov-ozon-cup/0b55ab9f-3777-4ebc-bd91-937895c0e355/scratchpad/gamma_top3.json", "w", encoding="utf-8") as f:
    json.dump(res, f, ensure_ascii=False, indent=1)

# ---------------------------------------------------------------- печать
print("=" * 100)
print("СВЕРКА С ПОДПИСАННЫМИ ОТЧЁТАМИ")
print("=" * 100)
for k, v in CHK.items():
    print(f"  {k:26s} {v}")
print(f"  база P(топ-3) при F8: {P_F8_now*100:.2f} %   (Женя 20.81 %)")
print(f"  sd(gamma=0.1) = {sd_ref:.6g}   общая часть sigma = {math.sqrt(VAR_SHARED):.6g}")

print()
print("=" * 100)
print("1-2. P(ТОП-3) ПО СЕТКЕ gamma В ЧЕТЫРЁХ РЕЖИМАХ ЗАЧ�ем")
print("=" * 100)
hdr = f"{'gamma':>7}{'sd,ш':>7}{'RAW dE,ш':>10}{'P raw':>8}{'x1.72':>8}{'P soft':>8}{'softX':>8}{'P softX':>9}{'OOF dE,ш':>10}{'P oof':>8}"
print(hdr)
for e in out_grid:
    print(f"{e['gamma']:7.4f}{e['sd']/NZ:7.2f}{e['raw']['delta_noises']:+10.2f}"
          f"{e['raw']['p3']*100:7.2f}%{e['soft_flat']['delta_noises']:+8.2f}"
          f"{e['soft_flat']['p3']*100:7.2f}%{e['soft_exact']['delta_noises']:+8.2f}"
          f"{e['soft_exact']['p3']*100:8.2f}%{e['oof']['delta_noises']:+10.2f}{e['oof']['p3']*100:7.2f}%")

print()
print("=" * 100)
print("4. МОДЕЛЬНЫЙ РИСК ПО gamma (бутстрап Грама A2 + сторож )")
print("=" * 100)
print(f"{'gamma':>7}{'||d||':>8}{'max|db|':>9}{'нар.':>6}{'регрет p95 нат,ш':>18}"
      f"{'регрет p95 x5.5,ш':>19}{'сторож,ш':>10}{'сверх F8,ш':>12}")
for e in out_grid:
    m = e["gram"]
    r = [x for x in rows if abs(x["gamma"] - e["gamma"]) < 1e-9][0]
    print(f"{e['gamma']:7.4f}{r['norm']:8.3f}{e['max_db']:9.3f}{e['n_material']:6d}"
          f"{m['regret_p95_nat']/NZ:18.3f}{m['regret_p95_amp']/NZ:19.3f}"
          f"{e['wd']/NZ:10.2f}{e['wd_excess']/NZ:12.2f}")

print()
print(f"{'gamma':>7}{'P soft':>9}{'+дисп(var)':>12}{'+смещ(bias)':>13}{'оба':>9}"
      f"{'|| P raw':>10}{'+дисп':>8}{'+смещ':>8}{'оба':>8}")
for e in out_grid:
    print(f"{e['gamma']:7.4f}{e['soft_flat']['p3']*100:8.2f}%{e['mr_var']['p3']*100:11.2f}%"
          f"{e['mr_bias']['p3']*100:12.2f}%{e['mr_both']['p3']*100:8.2f}%"
          f"{e['raw']['p3']*100:9.2f}%{e['mr_var_raw']['p3']*100:7.2f}%"
          f"{e['mr_bias_raw']['p3']*100:7.2f}%{e['mr_both_raw']['p3']*100:7.2f}%")

print()
print("=" * 100)
print("3. МОНОТОННОСТЬ И gamma* (тонкая сетка 0..0.2 шаг 0.0025)")
print("=" * 100)
for reg in ("raw", "soft_flat", "soft_exact", "oof"):
    ps = np.array([f[reg] for f in fine]); gs = np.array([f["gamma"] for f in fine])
    i = int(np.argmax(ps))
    mono = bool(np.all(np.diff(ps) <= 1e-9))
    print(f"  {reg:11s} argmax gamma = {gs[i]:.4f}   P = {ps[i]*100:.2f} %   "
          f"монотонно убывает по gamma: {mono}   P(0)={ps[0]*100:.2f}%  P(0.1)={ps[gs.tolist().index(0.1)]*100:.2f}%")

print()
print("сенситивность к разбиению sigma (аддитивная вместо вложенной):")
for e in out_grid:
    print(f"  gamma={e['gamma']:6.4f}  soft: вложенная {e['soft_flat']['p3']*100:6.2f}%  "
          f"аддитивная {e['soft_flat']['p3_sigma_additive']*100:6.2f}%")

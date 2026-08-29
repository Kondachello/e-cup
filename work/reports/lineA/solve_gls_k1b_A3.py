"""solve_gls_k1b_A1.py — КОПИЯ  для линии A (задача A1).
задаётся ключом --prior-model MU,TAU; (3) добавлен блок эмпирического байеса
МОДЕЛЬНОГО семейства (--eb-model) по ВСЕМ модельным осям, включая нулевые
Оригинал НЕ ТРОНУТ.

Исходная шапка:

σ_u каждой оси — ЛОКАЛЬНО из валидационных остатков: g_i = sd(h_i·r)/(F0v·√q_i),
σ_i = g_i·F0·√(0.8/(50000·q_i)); ковариация шума замера между осями — из тех же
остатков: Σ_ε = (0.8/n_pub)·(F0/F0v)²·Cov_val(h_i·r, h_j·r) (диагональ тождественно
воспроизводит поосную формулу K1b §6). Сегментный приор — свежий эмпирический

work/features/anchor=2026-01-14.parquet (val-таргет), MEASURED из predict_lb.

lp-массив кандидата F8 в скретчпаде (НЕ эмитится).

"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("POLARS_MAX_THREADS", "4")

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[3]
SUB = ROOT / "submissions"
CANON = SUB / "canonical"
OUT = Path(__file__).resolve().parent
SCRATCH = Path(os.environ.get(
    "GLS_SCRATCH",
    "/private/tmp/claude-501/-Users-alexanderkondakov-ozon-cup/"
    "9e300804-ff07-49ee-bd53-a9ebbe1be2db/scratchpad"))

sys.path.insert(0, str(ROOT / "work" / "scripts"))
import predict_lb as plb  # noqa: E402

MEAS = {n: s for n, _, s in plb.MEASURED}
N_PUB = 50_000
FPC2 = 0.8
NOISE = 0.000022                     # «шум» LB для вердикта в шумах
F_SCALE = 1.646                      # 2F-масштаб перевода Δ(S²) -> Δ(S)
GAMMA_STAR = 0.1                       #переопределяется ключом --gamma

PRIOR_MODEL = (0.309, 0.196)          # реестровый приор (дефолт = рецепт F8)
PRIOR_DECOMP = (0.0, 0.0148)
TAG = ""
EB_MODEL = False                      # считать ли EB модельного семейства
EB_APPLY = False                      # подставить ли EB-оценку в PRIOR_MODEL
EB_MU_FIX = None                      # если задано — mu приора фиксируется


# формо-дозы F7 сверх F6 (координатор, 29.08) — для сверки разложения
F7_DOSES = {"": 0.1477, "": -0.0208, "": 0.0708, "": -0.0157, "": 0.0392}


def lp(fn: str) -> np.ndarray:
    for p in (SUB / fn, CANON / fn):
        if p.exists():
            d = pl.read_csv(p, schema_overrides={"user_id": pl.Int64}).sort("user_id")
            return np.log1p(np.clip(d["predict"].to_numpy().astype(np.float64), 0, None))
    raise FileNotFoundError(fn)



def m(x): return float(np.mean(x))
def qq(x): return float(np.mean(x * x))


# ------------------------- эмпирический байес семейства (A1) -------------------------
def _nll(mu, tau, k, s):
    v = tau ** 2 + s ** 2
    return float(np.sum(0.5 * np.log(v) + 0.5 * (k - mu) ** 2 / v))


def _nll_prof(tau, k, s):
    """-logL с профилированным (GLS-оптимальным) mu."""
    v = tau ** 2 + s ** 2
    mu = float(np.sum(k / v) / np.sum(1.0 / v))
    return _nll(mu, tau, k, s), mu


def _reml(tau, k, s):
    v = tau ** 2 + s ** 2
    w = 1.0 / v
    mu = float(np.sum(k * w) / np.sum(w))
    return float(np.sum(0.5 * np.log(v) + 0.5 * (k - mu) ** 2 / v) + 0.5 * np.log(np.sum(w)))


def eb_fit(k, s, grid_hi=1.2, ngrid=2401):
    """Возвращает словарь оценок tau: моменты (простые и DL), ML, REML, CI по профилю."""
    k = np.asarray(k, float); s = np.asarray(s, float)
    n = len(k)
    # 1) простой метод моментов: Var(k) = tau^2 + E[s^2]
    mu_m = float(k.mean())
    var_k = float(k.var(ddof=1))
    mean_s2 = float((s ** 2).mean())
    tau2_mom = var_k - mean_s2
    tau_mom = float(np.sqrt(max(tau2_mom, 0.0)))
    # 2) DerSimonian-Laird (взвешенный момент)
    w0 = 1.0 / s ** 2
    mu_w0 = float(np.sum(w0 * k) / np.sum(w0))
    Q = float(np.sum(w0 * (k - mu_w0) ** 2))
    denom = float(np.sum(w0) - np.sum(w0 ** 2) / np.sum(w0))
    tau_dl = float(np.sqrt(max((Q - (n - 1)) / denom, 0.0)))
    # 3) ML по сетке tau с профилированным mu
    taus = np.linspace(0.0, grid_hi, ngrid)
    vals = [_nll_prof(t, k, s) for t in taus]
    nlls = np.array([v[0] for v in vals])
    j = int(np.argmin(nlls))
    tau_ml, mu_ml, nll_ml = float(taus[j]), float(vals[j][1]), float(nlls[j])
    # 4) REML
    rl = np.array([_reml(t, k, s) for t in taus])
    tau_reml = float(taus[int(np.argmin(rl))])
    # 5) профильный 95% CI по ML: 2*(nll - nll_min) <= 3.841
    ok = np.where(2 * (nlls - nll_ml) <= 3.8415)[0]
    ci = (float(taus[ok[0]]), float(taus[ok[-1]])) if len(ok) else (tau_ml, tau_ml)
    ok68 = np.where(2 * (nlls - nll_ml) <= 1.0)[0]
    ci68 = (float(taus[ok68[0]]), float(taus[ok68[-1]])) if len(ok68) else (tau_ml, tau_ml)
    return dict(n=n, mu_mom=mu_m, tau_mom=tau_mom, tau2_mom_raw=tau2_mom,
                var_k=var_k, mean_s2=mean_s2, mu_dl=mu_w0, tau_dl=tau_dl, Q=Q,
                mu_ml=mu_ml, tau_ml=tau_ml, nll_ml=nll_ml,
                tau_reml=tau_reml, ci95=list(ci), ci68=list(ci68))


def eb_boot(k, s, nb=4000, seed=17):
    """непараметрический бутстрап по осям: разброс tau_ML и tau_MoM."""
    k = np.asarray(k, float); s = np.asarray(s, float)
    rng = np.random.default_rng(seed)
    n = len(k)
    tm, tl = [], []
    taus = np.linspace(0.0, 1.2, 241)
    for _ in range(nb):
        idx = rng.integers(0, n, n)
        kk, ss = k[idx], s[idx]
        v = kk.var(ddof=1) - (ss ** 2).mean()
        tm.append(np.sqrt(max(v, 0.0)))
        nl = np.array([_nll_prof(t, kk, ss)[0] for t in taus])
        tl.append(taus[int(np.argmin(nl))])
    tm = np.array(tm); tl = np.array(tl)
    return dict(tau_mom_mean=float(tm.mean()), tau_mom_ci=[float(np.quantile(tm, .025)),
                                                           float(np.quantile(tm, .975))],
                tau_ml_mean=float(tl.mean()), tau_ml_ci=[float(np.quantile(tl, .025)),
                                                         float(np.quantile(tl, .975))])


def main() -> None:
    print("=" * 100)
    print("K1b-ПЕРЕСЧЁТ: локальные σ_u, свежий сегментный приор, GLS от F7")
    print("=" * 100)

    # ---------------------------------------------------------------- lp состояния
    S_names = {"Q1": "Q1_probes5", "mdl_flint": "R2_newblend",
               "mdl_gypsum": "R3_ridge", "mdl_gneis2": "R5_shade", "V3": "V3_canon", "G1": "G1_gru_tfm_full",
               "G2": "G2_gru_tfm_02", "T2": "T2_tfm4_orth_045", "T3": "T3_g1_redose_044",
               "mdl_talc": "R7_zreopt", "R8": "R8_zharvest_012", "R9": "R9_zharv2",
               "F1": "F1_priv", "F3": "F3_priv",
               "F5": "F5_priv", "F6": "F6_priv", "F7": "F7_priv"}
    L = {k: lp(v + ".csv") for k, v in S_names.items()}
    S = {k: MEAS[v] for k, v in S_names.items()}
    S.update({"mdl_amber": 1.647843925, "mdl_gabbro": 1.6479838892, "mdl_halite": 1.6478627751,
              "mdl_marble": 1.6482580065, "mdl_realgr": 1.6480249422})
    n = len(L["F7"])

    DELTA_U3 = m(L[""] - L[""])
    C1_R6 = (S[""] ** 2 - S[""] ** 2 + DELTA_U3 ** 2) / (2 * DELTA_U3)

    def c1_at(base: str) -> float:
        return C1_R6 + m(L[""]) - m(L[base])

    # ------------------------------------------------- оси: 32 снапшота + новые зонды
    OLD32 = [("mdl_amber", "", "model"), ("mdl_gabbro", "", "model"), ("mdl_halite", "", "model"),
             ("mdl_marble", "", "model"), ("mdl_realgr", "", "model"),
             ("mdl_tektit", "model"), ("mdl_olivin", "", "model"),
             ("mdl_flint", "Q1", "model"), ("mdl_gypsum", "mdl_flint", "model"), ("mdl_gneis2", "mdl_flint", "model"),
             ("mdl_malach", "", "model"), ("", "", "model"), ("mdl_vivian", "", "model"),
             ("mdl_corund", "V3", "model"), ("mdl_larvik", "G2", "model"), ("mdl_talc", "T3", "model"),
             ("", "R9", "segment"), ("", "F1", "segment"), ("", "F1", "segment"),
             ("R8", "decomp"), ("R8", "decomp"), ("R8", "decomp"),
             ("R8", "decomp"), ("R8", "decomp"), ("R8", "decomp"),
             ("R8", "decomp"), ("R8", "decomp"), ("R8", "decomp"),
             ("", "F3", "segment"), ("", "F3", "segment"), ("seg_realgr", "F3", "segment"),
             ("", "F3", "segment")]
    NEW_F5 = []
    NEW_F6 = []

    AX = {}      # name -> dict(d, base, fam)
    uid_ref = None
    for fn in NEW_F5:
        k = fn.split("_")[0]
        AX[k] = dict(d=lp(fn + ".csv") - L["F5"], base="F5", fam="segment", probe=fn)
        S[k] = MEAS[fn]
    for fn in NEW_F6:
        k = fn.split("_")[0]
        AX[k] = dict(d=lp(fn + ".csv") - L["F6"], base="F6", fam="segment", probe=fn)
        S[k] = MEAS[fn]
    AX["mdl_wulfen"] = dict(d=lp("N1_ktpp.csv") - L["F6"], base="F6", fam="model", probe="N1_ktpp")
    S["mdl_wulfen"] = MEAS["N1_ktpp"]
    names = list(AX)
    D = np.stack([AX[k]["d"] for k in names])
    q_vec = (D * D).mean(1)
    print(f"осей в GLS: {len(names)} (P54 залит не был — без замера, исключён)")

    # ------------------------------------------------------- вал-остатки и локальные g
    v = pl.read_parquet(ROOT / "work" / "features" / "anchor=2026-01-14.parquet",
                        columns=["user_id", "target"]).sort("user_id")
    assert np.array_equal(v["user_id"].to_numpy(), uid_ref), "val-юниверс не совпал"
    bv = pl.read_parquet(ROOT / "work" / "preds" / "blend_opt_val.parquet").sort("user_id")
    assert np.array_equal(bv["user_id"].to_numpy(), uid_ref), "blend_opt_val юниверс"
    tval = np.log1p(np.clip(v["target"].to_numpy().astype(np.float64), 0, None))
    pval = np.log1p(np.clip(bv["pred"].to_numpy().astype(np.float64), 0, None))
    r = tval - pval
    F0v = float(np.sqrt(qq(r)))
    print(f"вал-остаток: F0v = {F0v:.6f} (RMSLE бленда на вале)")

    H = D * r                                  # h_i·r поюзерно
    Hc = H - H.mean(1, keepdims=True)
    g_vec = np.sqrt((Hc * Hc).mean(1)) / (F0v * np.sqrt(q_vec))
    C_val = (Hc @ Hc.T) / n                    # Cov(h_i r, h_j r)

    # ------------------------------------------------------------- каппы парной параболой
    kap = {}
    for i, k in enumerate(names):
        a = AX[k]
        if a["probe"] is None:
            PROBE32 = {"mdl_amber": ("", 1.0), "mdl_gabbro": ("", 1.0),
                       "mdl_halite": ("", 1.0), "mdl_marble": ("", 1.0),
                       "mdl_realgr": ("", 1.0), "mdl_tektit": (0.894),
                       "mdl_olivin": ("", 1.0), "mdl_flint": ("R2_newblend", "Q1", 1.0),
                       "mdl_gypsum": ("R3_ridge", "mdl_flint", 1.0), "mdl_gneis2": ("R5_shade", "mdl_flint", 1.0),
                       "mdl_malach": ("", 1.0), "": ("", 1.0),
                       "mdl_vivian": ("V3_canon", "", 0.5289), "mdl_corund": ("G1_gru_tfm_full", "V3", 1.0),
                       "mdl_larvik": ("T2_tfm4_orth_045", "G2", 0.45), "mdl_talc": ("R7_zreopt", "T3", 1.0),
                       "": ("R9", 1.0), "": ("F1", 1.0),
                       "": ("F1", 1.0), "": ("F3", 1.0),
                       "": ("F3", 1.0), "seg_realgr": ("P46_silcarrier", "F3", 1.0),
                       "": ("F3", 1.0)}
            pf, bk, b_pr = PROBE32[k]
            s_step = lp(pf + ".csv") - L[bk]
            s_probe = MEAS[pf] if pf in MEAS else S[k.split("_")[0]] if False else None
            s_probe = MEAS.get(pf, {}.get(pf))
        else:
            pf, bk, b_pr = a["probe"], a["base"], 1.0
            s_step = a["d"]
            s_probe = MEAS[pf]
        delta = m(s_step)
        c_step = (S[bk] ** 2 - s_probe ** 2 + qq(s_step)) / 2
        c_d = (c_step - delta * c1_at(bk)) / b_pr
        kap[k] = c_d / q_vec[i]

    # ------------------------------------- поосные σ (K1b) и сравнение со старым законом
    F0b = {k: S[AX[k]["base"]] for k in names}
    F0b["mdl_olivin"] = 1.6473309766
    sig = {k: g_vec[i] * F0b[k] * np.sqrt(FPC2 / (N_PUB * q_vec[i]))
           for i, k in enumerate(names)}

    # ------------------- свежий эмпирический байес сегментной семьи (ВСЕ точки, поосный g)
    print("\n--- сегментный приор: эмпирический байес по всем сегментным точкам ---")
    extra_seg = [("R9"), ("R9"), ("R9"),
                 ("R9"), ("F1"), ("F1"),
                 ("F1"), ("F1"), (""),
                 (""), (""), (""),
                 ("F3"), ("F3"), ("F3")]
    seg_pts = []
    for k in names:
        if AX[k]["fam"] == "segment":
            seg_pts.append((k, kap[k], sig[k]))
    for fn, bk in extra_seg:
        d = lp(fn + ".csv") - L[bk]
        qd = qq(d)
        delta = m(d)
        c_d = (S[bk] ** 2 - MEAS[fn] ** 2 + qd) / 2 - delta * c1_at(bk)
        hk = d * r
        g_ = float(np.std(hk) / (F0v * np.sqrt(qd)))
        seg_pts.append((fn.split("_")[0], c_d / qd, g_ * S[bk] * np.sqrt(FPC2 / (N_PUB * qd))))
    ks = np.array([p[1] for p in seg_pts])
    ss = np.array([p[2] for p in seg_pts])

    def nll(mu, tau):
        vv = tau ** 2 + ss ** 2
        return float(np.sum(0.5 * np.log(vv) + 0.5 * (ks - mu) ** 2 / vv))
    taus = np.linspace(0.0, 0.35, 351)
    mus = np.linspace(-0.15, 0.25, 401)
    best = min(((nll(mu, t), mu, t) for t in taus for mu in mus))
    MU_SEG, TAU_SEG = best[1], best[2]
    print(f"  {len(seg_pts)} точек; ML: mu_seg = {MU_SEG:+.4f}, tau_seg = {TAU_SEG:.4f} "
          f"(K1b на 32 точках: +0.026 / 0.121)")

    # ---------------- A1: ЭМПИРИЧЕСКИЙ БАЙЕС ПРИОРА МОДЕЛЬНОГО СЕМЕЙСТВА ----------------
    eb_model = None
    prior_model_used = PRIOR_MODEL

    print(f"\nПРИОР МОДЕЛЬНОГО СЕМЕЙСТВА В РАСЧЁТЕ: mu={prior_model_used[0]:+.5f}, "
          f"tau={prior_model_used[1]:.5f}")

    PRIORS = {"model": prior_model_used, "segment": (MU_SEG, TAU_SEG), "decomp": PRIOR_DECOMP}
    mu_vec = np.array([PRIORS[AX[k]["fam"]][0] for k in names])
    tau_vec = np.array([PRIORS[AX[k]["fam"]][1] for k in names])
    w_vec = np.array([tau_vec[i] ** 2 / (tau_vec[i] ** 2 + sig[k] ** 2)
                      for i, k in enumerate(names)])

    # ---------------------------------------------------- таблица g с флагами
    print("\n--- локальные g по осям (флаг: |g−1| > 0.15 — прежнее допущение g=1 ломается) ---")
    for i, k in enumerate(names):
        fl = "  <-- ФЛАГ" if abs(g_vec[i] - 1) > 0.15 else ""
        print(f"  {k:6s} [{AX[k]['fam']:7s}] q={q_vec[i]:.3e}  κ_pair={kap[k]:+.3f}  "
              f"g={g_vec[i]:.3f}  σ_лок={sig[k]:.4f}  w={w_vec[i]:.3f}{fl}")

    # ------------------------------------------------- фактические дозы: F5, F6, F7
    print("\n--- разложение шагов F3->F5->F6->F7 (LSQ, фактические дозы) ---")
    one = np.ones(n)

    def decomp(y, keys, tag):
        A = np.vstack([np.stack([AX[k]["d"] for k in keys]), one])
        G = A @ A.T / n
        c = np.linalg.solve(G + 1e-12 * np.trace(G) / len(G) * np.eye(len(G)), A @ y / n)
        res = y - c @ A
        print(f"  {tag}: резид rms {np.sqrt(qq(res)):.2e}; " +
              " ".join(f"{k} {v:+.4f}" for k, v in zip(keys, c[:-1]) if abs(v) > 5e-3))
        return dict(zip(keys, c[:-1]))

    d65 = decomp(L["F6"] - L["F5"], [k for k, *_ in OLD32] +
                 [f.split("_")[0] for f in NEW_F5], "F6-F5")
    d76 = decomp(L["F7"] - L["F6"], ["", "", "", "", ""], "F7-F6")
    print("  формо-дозы координатора:", F7_DOSES)

    # суммарные дозы в F7 (все применённые оси)
    applied = [k for k, *_ in OLD32] + [f.split("_")[0] for f in NEW_F5] + ["", ""]
    bF7 = decomp(L["F7"] - L[""], applied, "F7-M1 (итог)")
    bF7["mdl_tektit"] = bF7.get("mdl_tektit", 0) + 0.894
    bF7["mdl_olivin"] = bF7.get("mdl_olivin", 0) + 0.65 / 0.9055014
    for k in names:
        bF7.setdefault(k, 0.0)

    # ------------------------------------------------------------- когерентный GLS от F7
    dot = np.array([m(AX[k]["d"] * (L["F7"] - L[AX[k]["base"]])) for k in names])
    cP = np.array([q_vec[i] * kap[k] for i, k in enumerate(names)]) - dot
    Q = D @ D.T / n
    F0_state = S["F7"]
    Sig_e = (FPC2 / N_PUB) * (F0_state / F0v) ** 2 * C_val
    # поосная сверка: diag(Sig_e) обязана равняться (q·σ)²
    dchk = np.abs(np.sqrt(np.diag(Sig_e)) -
                  np.array([q_vec[i] * sig[k] for i, k in enumerate(names)]))
    print(f"\n  сверка Σ_ε: max|sqrt(diag)−q·σ_лок| = {dchk.max():.2e} (тождество)")
    Sig_p = np.diag((q_vec * tau_vec) ** 2)
    mu_c = mu_vec * q_vec - dot
    K = Sig_p @ np.linalg.inv(Sig_p + Sig_e)
    m_star = mu_c + K @ (cP - mu_c)
    V_star = (np.eye(len(names)) - K) @ Sig_p
    cQ = 1.25 * m_star - 0.25 * cP

    Lam = np.diag(q_vec * (1 - w_vec) * tau_vec ** 2)
    ridge = 1e-9 * np.trace(Q) / len(Q) * np.eye(len(Q))
    db = {g: np.linalg.solve(Q + Lam + ridge + g * np.diag(np.diag(Q)), cQ)
          for g in sorted({0.0, 0.03, 0.1, 0.3, 1.0, GAMMA_STAR})}

    def gain(d_): return float((2 * d_ @ cQ - d_ @ Q @ d_) / (2 * F_SCALE))
    def gsd(d_): return float(np.sqrt(1.25 ** 2 * d_ @ V_star @ d_) / F_SCALE)

    print("\n--- GLS от F7 (поосные σ, свежий сегментный приор) ---")
    for g in db:
        print(f"  γ={g:4.2f}: E[priv F8−F7] {gain(db[g]):+.6f} (sd {gsd(db[g]):.6f}) "
              f"= {gain(db[g])/NOISE:+.1f} шума  max|Δb| {np.abs(db[g]).max():.3f}")
    dstar = db[GAMMA_STAR]
    gn, gs = gain(dstar), gsd(dstar)

    print(f"\n--- Δдозы к F7 (γ={GAMMA_STAR}), топ-15 по |Δ| ---")
    order = np.argsort(-np.abs(dstar))
    for i in order[:15]:
        k = names[i]
        print(f"  {k:6s} доза_F7 {bF7[k]:+.4f}  Δ {dstar[i]:+.4f} -> F8 {bF7[k]+dstar[i]:+.4f}"
              f"   (g={g_vec[i]:.2f}, κ̂резид {cQ[i]/q_vec[i]:+.3f})")

    # кандидат F8: lp в скретчпад, публичный прогноз алгеброй (+predict_lb контроль)
    f8 = L["F7"] + dstar @ D
    f8 = np.clip(f8 - (m(f8) - m(L["F7"])), 0, None)
    pub_alg = float(np.sqrt(max(S["F7"] ** 2 - 2 * dstar @ cP + dstar @ Q @ dstar, 0)))
    np.save(SCRATCH / f"F8_k1b_lp{TAG}.npy", f8)
    basis = plb.load_basis()
    P = plb.LBPredictor(basis)
    r8 = P.predict(f8)
    print(f"\n--- кандидат F8 (НЕ эмитится) ---")
    print(f"  E[priv]−F7 = {gn:+.6f} ± {gs:.6f}  ({gn/NOISE:+.1f} шума)")
    print(f"  паблик: алгебра {pub_alg:.7f} (Δ {pub_alg-S['F7']:+.6f})  "
          f"predict_lb {r8['pred']:.7f} ±{r8['sigma68']:.5f} (novelty {r8['novelty']:.1e})")
    print(f"  mean {m(f8):.6f} (F7 {m(L['F7']):.6f})  sd {float(np.std(f8)):.6f} "
          f"(F7 {float(np.std(L['F7'])):.6f})  клипов {int((L['F7'] + dstar @ D < 0).sum())}")
    print(f"  lp: {SCRATCH}/F8_k1b_lp{TAG}.npy")

    verdict = ("ДА: оптимум двигается на {:+.1f} шума (> 2-3), пересборка F8 окупается"
               if gn / NOISE > 3 else
               "НА ГРАНИЦЕ: {:+.1f} шума — по правилу K1b решают слот подтверждения и риск сборки"
               if gn / NOISE > 2 else
               "НЕТ: {:+.1f} шума (< 2-3) — оставаться на F7").format(gn / NOISE)
    print(f"\nВЕРДИКТ: {verdict}")

    out = dict(
        mu_seg=MU_SEG, tau_seg=TAU_SEG, n_seg_points=len(seg_pts), F0v=F0v,
        g={k: float(g_vec[i]) for i, k in enumerate(names)},
        g_flags=[k for i, k in enumerate(names) if abs(g_vec[i] - 1) > 0.15],
        kappa_pair={k: float(kap[k]) for k in names},
        sigma_local={k: float(sig[k]) for k in names},
        w={k: float(w_vec[i]) for i, k in enumerate(names)},
        doses_F7={k: float(bF7[k]) for k in names},
        f6_doses_lsq={k: float(v) for k, v in d65.items() if abs(v) > 5e-3},
        f7_form_doses_lsq={k: float(v) for k, v in d76.items()},
        delta_doses_F8={names[i]: float(dstar[i]) for i in range(len(names))},
        ridge_path={str(g): dict(gain=gain(db[g]), sd=gsd(db[g]),
                                 max_abs=float(np.abs(db[g]).max())) for g in db},
        gain_priv_vs_F7=gn, gain_priv_sd=gs, gain_in_noise=gn / NOISE,
        pub_pred_algebra=pub_alg, pub_pred_lb=r8["pred"], pub_novelty=r8["novelty"],
        f8_mean=m(f8), f8_sd=float(np.std(f8)),
        verdict=verdict, lp_path=str(SCRATCH / f"F8_k1b_lp{TAG}.npy"),
        prior_model_used=list(prior_model_used), prior_model_registry=list(PRIOR_MODEL),
        eb_model=eb_model,
    )
    (OUT / f"k1b_result{TAG}.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(f"итоги: {OUT/f'k1b_result{TAG}.json'}")

    # ------------- ФИНАЛ (решение координатора 29.08): ось mdl_wulfen ИСКЛЮЧЕНА, γ=0.1 -------------
    keep = [i for i, k in enumerate(names) if k != "mdl_wulfen"]
    kn = [names[i] for i in keep]
    Qk = Q[np.ix_(keep, keep)]
    Sig_e_k = Sig_e[np.ix_(keep, keep)]
    Sig_p_k = Sig_p[np.ix_(keep, keep)]
    mu_k, cP_k = mu_c[keep], cP[keep]
    Kk = Sig_p_k @ np.linalg.inv(Sig_p_k + Sig_e_k)
    m_k = mu_k + Kk @ (cP_k - mu_k)
    V_k = (np.eye(len(keep)) - Kk) @ Sig_p_k
    cQ_k = 1.25 * m_k - 0.25 * cP_k
    Lam_k = Lam[np.ix_(keep, keep)]
    ridge_k = 1e-9 * np.trace(Qk) / len(Qk) * np.eye(len(Qk))
    d_fin = np.linalg.solve(Qk + Lam_k + ridge_k + GAMMA_STAR * np.diag(np.diag(Qk)), cQ_k)
    gain_f = float((2 * d_fin @ cQ_k - d_fin @ Qk @ d_fin) / (2 * F_SCALE))
    sd_f = float(np.sqrt(1.25 ** 2 * d_fin @ V_k @ d_fin) / F_SCALE)
    Dk = D[keep]
    f8f_raw = L["F7"] + d_fin @ Dk
    f8f_raw = f8f_raw - (m(f8f_raw) - m(L["F7"]))
    nclip_f = int((f8f_raw < 0).sum())
    f8f = np.clip(f8f_raw, 0, None)
    pub_f = float(np.sqrt(max(S["F7"] ** 2 - 2 * d_fin @ cP_k + d_fin @ Qk @ d_fin, 0)))
    np.save(SCRATCH / f"F8_final_lp{TAG}.npy", f8f)
    print("\n" + "=" * 100)
    print("F8_FINAL (без mdl_wulfen, γ=0.1)")
    print(f"  E[priv]−F7 = {gain_f:+.6f} ± {sd_f:.6f}  ({gain_f/NOISE:+.1f} шума)")
    print(f"  паблик (алгебра) = {pub_f:.7f}  (Δ к F7 {pub_f - S['F7']:+.6f})")
    print(f"  mean {m(f8f):.6f} (F7 {m(L['F7']):.6f})  sd {float(np.std(f8f)):.6f} "
          f"(F7 {float(np.std(L['F7'])):.6f})  клипов {nclip_f}")
    print(f"  lp: {SCRATCH}/F8_final_lp{TAG}.npy")
    print("  топ-10 Δдоз к F7:")
    of = np.argsort(-np.abs(d_fin))
    for j in of[:10]:
        k = kn[j]
        print(f"    {k:6s} доза_F7 {bF7[k]:+.4f}  Δ {d_fin[j]:+.4f} -> F8 {bF7[k]+d_fin[j]:+.4f}")
    # A1: выгрузка матриц прогона — чтобы кросс-приорные сравнения считать без пересчёта
    np.savez(OUT / f"gls_state{TAG}.npz", names=np.array(kn), Q=Qk, cQ=cQ_k, cP=cP_k,
             V=V_k, Lam=Lam_k, d_fin=d_fin, q=q_vec[keep],
             doses_F7=np.array([bF7[k] for k in kn]),
             prior_model=np.array(prior_model_used), F0=S["F7"], F_SCALE=F_SCALE,
             NOISE=NOISE, gamma=GAMMA_STAR)
    print(f"  матрицы прогона: {OUT/f'gls_state{TAG}.npz'}")

    # -------- A1: сравнение нового решения с ДЕЙСТВУЮЩИМ F8 под ОДНИМ И ТЕМ ЖЕ c_Q --------
    cmp = None

    (OUT / f"k1b_final{TAG}.json").write_text(json.dumps(dict(
        excluded="mdl_wulfen", gamma=GAMMA_STAR,
        prior_model_used=list(prior_model_used), prior_model_registry=list(PRIOR_MODEL),
        delta_doses={kn[j]: float(d_fin[j]) for j in range(len(kn))},
        doses_F8={kn[j]: float(bF7[kn[j]] + d_fin[j]) for j in range(len(kn))},
        gain_priv_vs_F7=gain_f, gain_priv_sd=sd_f, gain_in_noise=gain_f / NOISE,
        pub_pred_algebra=pub_f, f8_mean=m(f8f), f8_sd=float(np.std(f8f)), f8_clips=nclip_f,
        lp_path=str(SCRATCH / f"F8_final_lp{TAG}.npy"), vs_F8=cmp,
        eb_model=eb_model), ensure_ascii=False, indent=1))
    print(f"  json: {OUT/f'k1b_final{TAG}.json'}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--prior-model", default=None,
                    help="MU,TAU модельного приора (по умолчанию реестровый 0.309,0.196)")
    ap.add_argument("--eb-model", action="store_true", help="считать EB модельного приора")
    ap.add_argument("--eb-apply", action="store_true", help="подставить EB-оценку в приор")
    ap.add_argument("--eb-mu", type=float, default=None, help="зафиксировать mu приора")
    ap.add_argument("--gamma", type=float, default=None, help=": gamma гребня")
    ap.add_argument("--tag", default="", help="суффикс выходных файлов")
    a = ap.parse_args()
    if a.prior_model:
        mu_, tau_ = (float(x) for x in a.prior_model.split(","))
        PRIOR_MODEL = (mu_, tau_)
    EB_MODEL = a.eb_model or a.eb_apply
    EB_APPLY = a.eb_apply
    EB_MU_FIX = a.eb_mu
    TAG = a.tag
    if a.gamma is not None:
        GAMMA_STAR = a.gamma
    main()

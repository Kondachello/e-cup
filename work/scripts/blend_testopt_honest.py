"""Итоговая честная сводка: три базы сравнения, две оценки переобучения, SE симуляции."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
import predict_lb as PL
from blend_testopt import (Phi, build_pool, load_lp, solve_nonneg, solve_simplex,
                           solve_unconstrained, W_VAL, SIGMA2, N_PUB)

ROOT = Path(__file__).resolve().parents[2]
basis = PL.load_basis(); P = PL.LBPredictor(basis); F = Phi(P)
uid, N = P.uid, P.N
pool = build_pool(); K = len(pool)
L = np.stack([load_lp(n, uid, "test") for n in pool])
M = L@L.T/N; m = L.mean(1); phi = F(L)
phi_a = float(F(P.lp_a)); const = P.f_a**2 - P.q_a + 2*phi_a
Mc = M - np.outer(m, m); phic = phi - PL.MEAN_T*m; const_c = const - PL.MEAN_T**2
fc = lambda w: float(np.sqrt(max(w@Mc@w - 2*w@phic + const_c, 1e-12)))
wv = np.zeros(K)
for n, v in W_VAL.items(): wv[pool.index(n)] = v
lp_val = wv@L

# --- 1. переоценка избытка риска в симуляции с SE ------------------------------------
tv = basis["tval"]
Lv = np.stack([load_lp(n, uid, "val") for n in pool])
Mv = Lv@Lv.T/N; mv = Lv.mean(1); Mcv = Mv - np.outer(mv, mv)
mt = float(tv.mean()); ccv = float((tv**2).mean()) - mt**2
phicv = Lv@tv/N - mt*mv
fv = lambda w: float(np.sqrt(max(w@Mcv@w - 2*w@phicv + ccv, 1e-12)))
rng = np.random.default_rng(3); R = 200
S_all = [rng.choice(N, N_PUB, replace=False) for _ in range(R)]
print("=== избыток истинного риска от подгонки φ под 50k (симуляция на val, с SE) ===")
EX = {}
for name, slv in [("unconstrained", solve_unconstrained), ("nonneg", solve_nonneg),
                  ("simplex", lambda A,b: solve_simplex(A,b)),
                  ("affine2", None)]:
    if name == "affine2":      # база (c): 2 параметра — наклон и сдвиг val-бленда
        lpv0 = wv@Lv; X = np.vstack([lpv0, np.ones(N)]); A2 = X@X.T/N
        ex = []
        opt_c = np.linalg.solve(A2, np.array([float(X[0]@tv/N), mt]))
        f_opt = float(np.sqrt(max(((opt_c@X)**2).mean() - 2*(opt_c@X)@tv/N + float((tv**2).mean()), 1e-12)))
        for S in S_all:
            b = np.array([float(X[0][S]@tv[S]/N_PUB), float(tv[S].mean())])
            c = np.linalg.solve(A2, b); z = c@X
            ex.append(float(np.sqrt(max((z**2).mean() - 2*z@tv/N + float((tv**2).mean()),1e-12))) - f_opt)
    else:
        f_opt = fv(slv(Mcv, phicv)); ex = []
        for S in S_all:
            pS = Lv[:,S]@tv[S]/N_PUB - float(tv[S].mean())*mv
            ex.append(fv(slv(Mcv, pS)) - f_opt)
    ex = np.array(ex)
    EX[name] = dict(excess=float(ex.mean()), se=float(ex.std()/np.sqrt(R)), opt=f_opt)
    print(f"   {name:15s} избыток {ex.mean():+.6f} ± {ex.std()/np.sqrt(R):.6f}  "
          f"(оптимум симуляции {f_opt:.6f})  → оптимизм f² {4*1.667*ex.mean():.3e}")

# --- 2. шум φ (остаточный механизм), правильный знак ---------------------------------
Res = F.resid(L); Rm = Res@Res.T/N; gamma = 1.65*PL.KAPPA68
Le = np.linalg.cholesky(gamma**2*Rm + 1e-16*np.trace(gamma**2*Rm)/K*np.eye(K))
rng2 = np.random.default_rng(5)
def phi_noise(slv, nmc=600):
    g = []
    for _ in range(nmc):
        e = Le@rng2.standard_normal(K); w = slv(Mc, phic+e); g.append(2*float(w@e))
    return float(np.mean(g))

# --- 3. три базы сравнения + кандидаты ------------------------------------------------
print("\n=== честная сводка ===")
def affine_of(lp):
    X = np.vstack([lp, np.ones(N)]); A2 = X@X.T/N
    c = np.linalg.solve(A2, np.array([float(F(lp)), PL.MEAN_T]))
    z = c@X
    fsq = P.f_a**2 + (z**2).mean() - P.q_a - 2*(float(F(z)) - phi_a)
    return float(c[0]), float(c[1]), float(np.sqrt(max(fsq,1e-12)))

rows = []
f_a_raw = float(np.sqrt(max(wv@M@wv - 2*wv@phi + const, 1e-12)))
rows.append(("val-NNLS как есть (0 подгон. парам.)", f_a_raw, 0, None, 0.0))
rows.append(("val-NNLS + глоб. сдвиг (1 парам.)", fc(wv), 1, None, 0.0))
sl, sh, f_aff = affine_of(lp_val)
rows.append((f"val-NNLS + аффин. наклон {sl:.4f}/сдвиг (2 парам.)", f_aff, 2, "affine2", 0.0))
for tag, slv, k in [("test-opt simplex+сдвиг", lambda A,b: solve_simplex(A,b), 5),
                    ("test-opt nonneg+сдвиг", solve_nonneg, 10),
                    ("test-opt свободные+сдвиг", solve_unconstrained, 22)]:
    key = {"test-opt simplex+сдвиг":"simplex","test-opt nonneg+сдвиг":"nonneg",
           "test-opt свободные+сдвиг":"unconstrained"}[tag]
    w = slv(Mc, phic); rows.append((tag, fc(w), k, key, phi_noise(slv)))
res = {}
print(f"{'вариант':44s}{'прогноз':>10}{'k':>4}{'шум φ':>11}{'честно(форм)':>14}{'честно(эмп)':>13}")
for lab, fp, k, key, dphi in rows:
    d_form = 2*k*SIGMA2/N_PUB
    d_emp = 4*1.667*EX[key]["excess"] if key else 4*1.667*0.0
    if key is None: d_emp = d_form
    h_f = float(np.sqrt(fp**2 + dphi + d_form)); h_e = float(np.sqrt(fp**2 + dphi + d_emp))
    res[lab] = dict(pred=fp, k=k, d_phi=dphi, d_form=d_form, d_emp=d_emp,
                    honest_formula=h_f, honest_emp=h_e)
    print(f"{lab:44s}{fp:10.6f}{k:4d}{dphi:+11.2e}{h_f:14.6f}{h_e:13.6f}")
res["_excess_sim"] = EX
res["_affine"] = dict(slope=sl, shift=sh, pred=f_aff)
Path(ROOT/"work"/"reports"/"blend_testopt_honest.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
print("\nсохранено work/reports/blend_testopt_honest.json")

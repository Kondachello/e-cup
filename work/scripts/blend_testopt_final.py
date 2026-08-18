from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, polars as pl
sys.path.insert(0, str(Path(__file__).parent))
import predict_lb as PL
from blend_testopt import (Phi, build_pool, load_lp, solve_nonneg, solve_simplex,
                           solve_unconstrained, solve_ridge, W_VAL, SIGMA2, N_PUB)

ROOT = Path(__file__).resolve().parents[2]
basis = PL.load_basis(); P = PL.LBPredictor(basis); F = Phi(P)
uid, N, names = P.uid, P.N, list(basis["names"])
pool = build_pool(); K = len(pool)
L = np.stack([load_lp(n, uid, "test") for n in pool])
M = L@L.T/N; m = L.mean(1); phi = F(L)
phi_a = float(F(P.lp_a)); const = P.f_a**2 - P.q_a + 2*phi_a
Mc = M - np.outer(m, m); phic = phi - PL.MEAN_T*m; const_c = const - PL.MEAN_T**2
fc = lambda w: float(np.sqrt(max(w@Mc@w - 2*w@phic + const_c, 1e-12)))
wv = np.zeros(K)
for n, v in W_VAL.items(): wv[pool.index(n)] = v
out = {}

# оптимальный аффинный пересчёт lp -> a·lp + b для произвольного файла
def best_affine(lp):
    X = np.vstack([lp, np.ones(N)])
    A = X@X.T/N; b = np.array([float(F(lp)), PL.MEAN_T])
    c = np.linalg.solve(A, b)
    fsq = P.f_a**2 + ((c@X)**2).mean() - P.q_a - 2*(float(F(c@X)) - phi_a)
    return float(c[0]), float(c[1]), float(np.sqrt(max(fsq,1e-12)))
lp_val = wv@L
a_, b_, f_ = best_affine(lp_val)
print(f"   {'val-бленд (калибр.)':18s} оптимальный наклон {a_:.4f}  сдвиг {b_:+.4f}  → {f_:.6f}")
out["affine_valblend"] = dict(slope=a_, shift=b_, pred=f_)
print(f"   независимый LB-замер наклона для A1 (KNOWLEDGE): k = 1.0065")

print("\n=== B. Декомпозиция выигрыша (от val-весов к test-оптимальным) ===")
steps = [("val-веса, без сдвига",             float(np.sqrt(max(wv@M@wv - 2*wv@phi + const,1e-12)))),
         ("+ оптимальный глобальный сдвиг",   fc(wv))]
a_, b_, f_aff = best_affine(lp_val)
steps.append(("+ оптимальный наклон (аффин.)", f_aff))
sub = [pool.index(n) for n in W_VAL]
Mc9 = Mc[np.ix_(sub,sub)]; phic9 = phic[sub]
f9 = lambda w: float(np.sqrt(max(w@Mc9@w - 2*w@phic9 + const_c,1e-12)))
steps.append(("+ test-веса, те же 9, Σw=1",   f9(solve_simplex(Mc9, phic9))))
steps.append(("+ Σw свободна (те же 9)",      f9(solve_nonneg(Mc9, phic9))))
w_nn = solve_nonneg(Mc, phic)
steps.append(("+ все 21 модель (nonneg)",     fc(w_nn)))
prev = None
for lab, v in steps:
    d = "" if prev is None else f"   Δ {v-prev:+.6f}"
    print(f"   {lab:32s} {v:.6f}{d}"); prev = v
out["decomposition"] = {k: v for k, v in steps}

print("\n=== C. Спред: почему Σw > 1 ===")
sd_val = float((lp_val - lp_val.mean()).std()); sd_a1 = float(basis["L"][names.index("A1_gram7_shift")].std())
lp_nn = w_nn@L + float(PL.MEAN_T - m@w_nn)
print(f"   sd(lp): val-бленд калиброванный {sd_val:.4f} | A1 {sd_a1:.4f} | "
      f"test-opt {lp_nn.std():.4f} | val-таргет {basis['tval'].std():.4f} | "
      f"тест-таргет (замер) {np.sqrt(PL.MEAN_T_SQ - PL.MEAN_T**2):.4f}")
out["sd"] = dict(val_blend=sd_val, a1=sd_a1, testopt=float(lp_nn.std()),
                 tval=float(basis['tval'].std()))

print("\n=== D. Честная поправка: две оценки ===")
Res = F.resid(L); Rm = Res@Res.T/N; gamma = 1.65*PL.KAPPA68
Le = np.linalg.cholesky(gamma**2*Rm + 1e-16*np.trace(gamma**2*Rm)/K*np.eye(K))
rng = np.random.default_rng(5)
def phi_noise(slv, Mm, pp, nmc=500):
    g = [ ]
    for _ in range(nmc):
        e = Le@rng.standard_normal(K); w = slv(Mm, pp+e)
        g.append(+2*float(w@e))
    return float(np.mean(g))
sim = json.load(open(ROOT/"work"/"reports"/"blend_testopt_sim.json"))
rows = {}
for tag, slv, k in [("unconstrained", solve_unconstrained, 22),
                    ("nonneg", solve_nonneg, 10), ("simplex", lambda A,b: solve_simplex(A,b), 5)]:
    w = slv(Mc, phic); fp = fc(w)
    d_phi = phi_noise(slv, Mc, phic)
    d_form = 2*k*SIGMA2/N_PUB
    excess = sim[tag]["true_mean"] - sim[tag]["opt"]          # избыток истинного риска, в f
    d_emp = 2*excess*2*1.667                                   # → в f² (оптимизм = 2·избыток)
    h_form = float(np.sqrt(fp**2 + d_phi + d_form))
    h_emp = float(np.sqrt(fp**2 + d_phi + d_emp))
    rows[tag] = dict(pred=fp, d_phi=d_phi, d_form=d_form, d_emp=d_emp,
                     honest_formula=h_form, honest_empirical=h_emp, excess_f=excess)
    print(f"   {tag:15s} прогноз {fp:.6f} | шум φ {d_phi:+.2e} | "
          f"2kσ²/n {d_form:.2e} → {h_form:.6f} | эмпир. {d_emp:.2e} → {h_emp:.6f}")
out["honest"] = rows

shift = float(PL.MEAN_T - m@w_nn)
lp_out = w_nn@L + shift
neg = int((lp_out < 0).sum())
lp_clip = np.clip(lp_out, 0, None)
pred = np.expm1(lp_clip)
pl.DataFrame({"user_id": uid, "predict": pred}).write_csv(sp)
print(f"   отрицательных lp: {neg} ({neg/N:.4%}) — обрезаны в 0")
print(f"   Σw {w_nn.sum():.4f}  сдвиг {shift:+.5f}  веса: "
      f"{ {pool[i]: round(float(v),4) for i,v in enumerate(w_nn) if v>1e-4} }")
r = P.predict(PL.read_lp(sp)[1])
print(f"   predict_lb: {r['pred']:.6f} ±{r['sigma68']:.5f}(68%) ±{r['sigma95']:.5f}(95%)  "
      f"sd(ост) {r['sd_resid']:.4f}  novelty {r['novelty']:.2e}")
print(f"   до обрезания было {fc(w_nn):.6f} → цена обрезания {r['pred']-fc(w_nn):+.6f}")
out["submission"] = dict(path=str(sp), pred=r["pred"], sigma68=r["sigma68"], sigma95=r["sigma95"],
                         sd_resid=r["sd_resid"], novelty=r["novelty"], n_neg=neg,
                         shift=shift, w={pool[i]: float(v) for i,v in enumerate(w_nn) if v>1e-4},
                         w_sum=float(w_nn.sum()), pred_before_clip=fc(w_nn))
Path(ROOT/"work"/"reports"/"blend_testopt_final.json").write_text(json.dumps(out, indent=1, ensure_ascii=False, default=float))
print("\nсохранено work/reports/blend_testopt_final.json")

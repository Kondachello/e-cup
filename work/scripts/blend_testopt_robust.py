"""Стресс-тесты решения blend_testopt: устойчивость φ, устойчивость M, честная поправка."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, polars as pl
from scipy.optimize import nnls, minimize
sys.path.insert(0, str(Path(__file__).parent))
import predict_lb as PL
from blend_testopt import (Phi, build_pool, load_lp, solve_nonneg, solve_simplex,
                           solve_unconstrained, solve_ridge, W_VAL, SIGMA2, N_PUB)

ROOT = Path(__file__).resolve().parents[2]
rng = np.random.default_rng(11)

basis = PL.load_basis(); P = PL.LBPredictor(basis); F = Phi(P)
uid, N = P.uid, P.N
pool = build_pool(); K = len(pool)
L = np.stack([load_lp(n, uid, "test") for n in pool])
M = L @ L.T / N; m = L.mean(1); phi = F(L)
phi_a = float(F(P.lp_a)); const = P.f_a**2 - P.q_a + 2*phi_a
Mc = M - np.outer(m, m); phic = phi - PL.MEAN_T*m; const_c = const - PL.MEAN_T**2
fc = lambda w, Mm=Mc, pp=phic, cc=const_c: float(np.sqrt(max(w@Mm@w - 2*w@pp + cc, 1e-12)))
wv = np.zeros(K)
for n, v in W_VAL.items(): wv[pool.index(n)] = v

out = {}
print("=== 1. Проверка оптимального сдвига прямым перебором (val-веса) ===")
lpv = wv @ L
for c0 in [0.0, 0.10, 0.15, PL.MEAN_T - float(m@wv), 0.20, 0.25]:
    print(f"   сдвиг {c0:+.5f}  predict {P.predict(lpv+c0)['pred']:.6f}")
out["shift_opt"] = float(PL.MEAN_T - m@wv)

print("\n=== 2. Джекнайф по базису замеренных файлов (устойчивость φ) ===")
# выбрасываем по одному ПОЗДНЕМУ замеренному файлу (кроме якоря) и пересчитываем всё
names = list(basis["names"]); anch = names.index(PL.ANCHOR)
jk = []
for drop in range(len(names)):
    if drop == anch or names[drop] == "sample_submit":
        continue
    use = [i for i in range(len(names)) if i != drop]
    Pj = PL.LBPredictor(basis, use_idx=use); Fj = Phi(Pj)
    phij = Fj(L); phia_j = float(Fj(Pj.lp_a))
    constj = Pj.f_a**2 - Pj.q_a + 2*phia_j
    Mcj, phicj, ccj = Mc, phij - PL.MEAN_T*m, constj - PL.MEAN_T**2
    wn = solve_nonneg(Mcj, phicj)
    jk.append(dict(drop=names[drop], pred=fc(wn, Mcj, phicj, ccj),
                   pred_at_full=fc(wn), val=fc(wv, Mcj, phicj, ccj),
                   w=wn, dphi=float(np.abs(phij-phi).max())))
pr = np.array([j["pred"] for j in jk]); pf = np.array([j["pred_at_full"] for j in jk])
vv = np.array([j["val"] for j in jk])
print(f"   выброшено по одному из {len(jk)} файлов")
print(f"   прогноз nonneg+shift : {pr.mean():.6f} ± {pr.std():.6f}  [{pr.min():.6f}, {pr.max():.6f}]")
print(f"   тот же w на полном φ : {pf.mean():.6f} ± {pf.std():.6f}  max {pf.max():.6f}")
print(f"   val-веса             : {vv.mean():.6f} ± {vv.std():.6f}")
print(f"   разрыв val − test-opt: {(vv-pf).mean():.6f} ± {(vv-pf).std():.6f}  min {(vv-pf).min():.6f}")
worst = sorted(jk, key=lambda j: -(j["pred_at_full"]))[:3]
for j in worst: print(f"     худший при выбросе {j['drop']:20s} → {j['pred_at_full']:.6f}")
out["jackknife"] = dict(pred_mean=float(pr.mean()), pred_sd=float(pr.std()),
                        at_full_mean=float(pf.mean()), at_full_sd=float(pf.std()),
                        at_full_max=float(pf.max()),
                        val_mean=float(vv.mean()), gap_mean=float((vv-pf).mean()),
                        gap_min=float((vv-pf).min()), gap_sd=float((vv-pf).std()),
                        n=len(jk))

print("\n=== 3. Устойчивость M к подвыборке юзеров (5 фолдов по 50k) ===")
idx = rng.permutation(N); folds = np.array_split(idx, 5)
fold_res = []
for i, fo in enumerate(folds):
    Lf = L[:, fo]; Mf = Lf@Lf.T/len(fo); mf = Lf.mean(1)
    Mcf = Mf - np.outer(mf, mf)
    wn = solve_nonneg(Mcf, phic)          # φ тот же (публичная величина)
    fold_res.append(dict(fold=i, w=wn, pred_at_full=fc(wn)))
    print(f"   фолд {i}: прогноз на полном M {fc(wn):.6f}   Σw {wn.sum():.4f}")
pf2 = np.array([f["pred_at_full"] for f in fold_res])
print(f"   разброс {pf2.std():.6f}, потеря против полного решения "
      f"{pf2.mean()-fc(solve_nonneg(Mc,phic)):+.6f}")
out["m_folds"] = dict(sd=float(pf2.std()), mean=float(pf2.mean()))

print("\n=== 4. Эмпирическая поправка на публичный сабсет (структурная, вместо 2kσ²/n) ===")
tv = basis["tval"]
Z = L * tv                                    # (K,N) — вклад lp_i·t в среднее
Cz = np.cov(Z)                                # ковариация по юзерам
Sp = (1.0/N_PUB - 1.0/N) * Cz                 # Cov(φ_pub − φ_full)
Lp = np.linalg.cholesky(Sp + 1e-14*np.trace(Sp)/K*np.eye(K))
def mc_gap(solver, S_chol, nmc=400):
    g = []
    for _ in range(nmc):
        e = S_chol @ rng.standard_normal(K)
        w = solver(Mc, phic + e)
        g.append((w@Mc@w - 2*w@phic + const_c) - (w@Mc@w - 2*w@(phic+e) + const_c))
    return float(np.mean(g))
for tag, slv in [("unconstrained", solve_unconstrained), ("nonneg", solve_nonneg),
                 ("simplex", lambda A,b: solve_simplex(A,b))]:
    g = mc_gap(slv, Lp)
    w = slv(Mc, phic); k = max(int((np.abs(w)>1e-8).sum()), 1)
    print(f"   {tag:15s} эмпирич. {g:+.3e}  против формулы 2kσ²/n = {2*k*SIGMA2/N_PUB:.3e} (k={k})")
    out[f"emp_pub_{tag}"] = g

print("\n=== 5. Тот же пул из 9 моделей: val-веса против test-оптимальных ===")
sub = [pool.index(n) for n in W_VAL]
Mc9 = Mc[np.ix_(sub, sub)]; phic9 = phic[sub]
w9_val = np.array([W_VAL[pool[i]] for i in sub])
w9_nn = solve_nonneg(Mc9, phic9); w9_sx = solve_simplex(Mc9, phic9)
f9 = lambda w: float(np.sqrt(max(w@Mc9@w - 2*w@phic9 + const_c, 1e-12)))
print(f"   val-веса           {f9(w9_val):.6f}")
print(f"   test-opt nonneg    {f9(w9_nn):.6f}   Σw {w9_nn.sum():.4f}")
print(f"   test-opt simplex   {f9(w9_sx):.6f}")
print("   веса:", {pool[i]: round(float(v),3) for i,v in zip(sub, w9_nn) if abs(v)>1e-4})
out["pool9"] = dict(val=f9(w9_val), nonneg=f9(w9_nn), simplex=f9(w9_sx),
                    w_nn={pool[i]: float(v) for i,v in zip(sub,w9_nn)},
                    w_sx={pool[i]: float(v) for i,v in zip(sub,w9_sx)})

Mc2 = M2 - np.outer(m2, m2); phic2 = phi2 - PL.MEAN_T*m2
f2 = lambda w: float(np.sqrt(max(w@Mc2@w - 2*w@phic2 + const_c, 1e-12)))
for tag, slv in [("nonneg", solve_nonneg), ("simplex", lambda A,b: solve_simplex(A,b)),
                 ("unconstr", solve_unconstrained)]:
    w2 = slv(Mc2, phic2)
    nz = {pool2[i]: round(float(v),3) for i,v in enumerate(w2) if abs(v)>0.004}
    print(f"   {tag:10s} {f2(w2):.6f}  сдвиг {PL.MEAN_T-float(m2@w2):+.4f}  Σw {w2.sum():.3f}")
    print(f"              {nz}")
    out[f"aug_{tag}"] = dict(pred=f2(w2), w=nz, shift=float(PL.MEAN_T-m2@w2))

print("\n=== 7. Leave-one-model-out (nonneg+shift) ===")
base = fc(solve_nonneg(Mc, phic))
loo = []
for i in range(K):
    keep = [j for j in range(K) if j != i]
    Mck = Mc[np.ix_(keep, keep)]; pck = phic[keep]
    w = solve_nonneg(Mck, pck)
    v = float(np.sqrt(max(w@Mck@w - 2*w@pck + const_c, 1e-12)))
    loo.append((pool[i], v - base))
for n, dv in sorted(loo, key=lambda t: -t[1])[:6]:
    print(f"   без {n:22s} {base+dv:.6f}  ({dv:+.6f})")
out["loo"] = {n: float(v) for n, v in loo}
out["base_nonneg_shift"] = base

Path(ROOT/"work"/"reports"/"blend_testopt_robust.json").write_text(json.dumps(out, indent=1, ensure_ascii=False, default=float))
print("\nсохранено work/reports/blend_testopt_robust.json")

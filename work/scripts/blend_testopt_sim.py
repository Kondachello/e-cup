"""Симуляция канала «подгон φ под 50k → истина на 250k», ЧИСТО на валидационном окне.

Только val-предсказания против val-таргета: тестовые предсказания использовать нельзя —
их фичи включают окно val-таргета (утечка, RMSLE 1.05 вместо 1.67).

Считается ровно w-зависимая часть завышения: gap = 2·ŵ'(φ_S − φ_full).
Константа (mean(t²), mean(t)) в реальной схеме сокращается через f²(якоря), поэтому
её выборочный шум в оптимизм не входит.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
import predict_lb as PL
from blend_testopt import build_pool, load_lp, solve_nonneg, solve_simplex, solve_unconstrained, solve_ridge, SIGMA2, N_PUB

ROOT = Path(__file__).resolve().parents[2]
basis = PL.load_basis(); P = PL.LBPredictor(basis)
uid, N = P.uid, P.N
tv = basis["tval"]
pool = build_pool(); K = len(pool)
Lv = np.stack([load_lp(n, uid, "val") for n in pool])
M = Lv @ Lv.T / N; m = Lv.mean(1); Mc = M - np.outer(m, m)
T2 = float((tv**2).mean()); mt = float(tv.mean()); cc = T2 - mt**2
phic_full = Lv @ tv / N - mt * m
f = lambda w: float(np.sqrt(max(w@Mc@w - 2*w@phic_full + cc, 1e-12)))

rng = np.random.default_rng(3); R = 200
print(f"=== чистая симуляция на val ({K} калиброванных моделей, n_pub={N_PUB}, {R} розыгрышей) ===")
print(f"   оптимум на полных 250k: unconstr {f(solve_unconstrained(Mc,phic_full)):.6f}   "
      f"nonneg {f(solve_nonneg(Mc,phic_full)):.6f}   simplex {f(solve_simplex(Mc,phic_full)):.6f}")
res = {}
variants = [("unconstrained", solve_unconstrained), ("nonneg", solve_nonneg),
            ("simplex", lambda A,b: solve_simplex(A,b))]
for rel in (1e-3, 1e-2):
    variants.append((f"ridge{rel:g}", lambda A,b,r=rel: solve_ridge(A,b,r*np.trace(Mc)/K)))
S_all = [rng.choice(N, N_PUB, replace=False) for _ in range(R)]
for name, slv in variants:
    gaps, trues, ks = [], [], []
    for S in S_all:
        phic_S = Lv[:, S] @ tv[S] / N_PUB - float(tv[S].mean()) * m
        w = slv(Mc, phic_S)
        gaps.append(2 * float(w @ (phic_S - phic_full)))
        trues.append(f(w)); ks.append(int((np.abs(w) > 1e-8).sum()))
    g = float(np.mean(gaps)); se = float(np.std(gaps)/np.sqrt(R)); kk = float(np.mean(ks))
    form = 2*kk*SIGMA2/N_PUB
    # эквивалент в единицах f при f≈1.65
    print(f"   {name:15s} k̄ {kk:5.1f}  завышение f² {g:+.3e} ±{se:.1e}  "
          f"(формула {form:.3e}, ×{g/form:5.1f})  в f: {g/(2*1.65):+.6f}   "
          f"истина ŵ {np.mean(trues):.6f} (оптимум {f(slv(Mc,phic_full)):.6f})")
    res[name] = dict(gap_f2=g, se=se, k=kk, formula=form, ratio=g/form,
                     gap_f=g/(2*1.65), true_mean=float(np.mean(trues)),
                     opt=f(slv(Mc, phic_full)))
Path(ROOT/"work"/"reports"/"blend_testopt_sim.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
print("\nсохранено work/reports/blend_testopt_sim.json")

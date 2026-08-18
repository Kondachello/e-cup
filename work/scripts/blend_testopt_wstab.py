"""Устойчивость самих ВЕСОВ: джекнайф по базису + подвыборки юзеров + возмущение φ."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
import predict_lb as PL
from blend_testopt import Phi, build_pool, load_lp, solve_nonneg, W_VAL

ROOT = Path(__file__).resolve().parents[2]
basis = PL.load_basis(); P = PL.LBPredictor(basis); F = Phi(P)
uid, N, names = P.uid, P.N, list(basis["names"])
pool = build_pool(); K = len(pool)
L = np.stack([load_lp(n, uid, "test") for n in pool])
M = L@L.T/N; m = L.mean(1); phi = F(L)
Mc = M - np.outer(m, m); phic = phi - PL.MEAN_T*m
W = []
# (a) джекнайф по замеренным файлам
anch = names.index(PL.ANCHOR)
for d in range(len(names)):
    if d == anch: continue
    Pj = PL.LBPredictor(basis, use_idx=[i for i in range(len(names)) if i != d])
    W.append(solve_nonneg(Mc, Phi(Pj)(L) - PL.MEAN_T*m))
# (b) подвыборки юзеров для M
rng = np.random.default_rng(1)
for _ in range(30):
    S = rng.choice(N, 125_000, replace=False)
    Ls = L[:, S]; Ms = Ls@Ls.T/len(S); ms = Ls.mean(1)
    W.append(solve_nonneg(Ms - np.outer(ms, ms), phic))
# (c) возмущение φ калиброванным шумом предиктора
Res = F.resid(L); Rm = Res@Res.T/N; g = 1.65*PL.KAPPA68
Le = np.linalg.cholesky(g**2*Rm + 1e-16*np.trace(g**2*Rm)/K*np.eye(K))
for _ in range(200):
    W.append(solve_nonneg(Mc, phic + Le@rng.standard_normal(K)))
W = np.array(W)
w0 = solve_nonneg(Mc, phic)
CC = np.corrcoef(L)
print(f"{'модель':22s}{'val':>7}{'test':>8}{'сред':>8}{'sd':>8}{'доля≠0':>9}{'макс.корр с':>22}")
rowsj = {}
for i, n in enumerate(pool):
    o = np.argsort(-CC[i]); j = o[1]
    rowsj[n] = dict(val=W_VAL.get(n,0.0), test=float(w0[i]), mean=float(W[:,i].mean()),
                    sd=float(W[:,i].std()), freq=float((W[:,i]>1e-6).mean()),
                    nearest=pool[j], corr=float(CC[i,j]))
    print(f"{n:22s}{W_VAL.get(n,0):7.3f}{w0[i]:8.3f}{W[:,i].mean():8.3f}{W[:,i].std():8.3f}"
          f"{(W[:,i]>1e-6).mean():9.2f}   {pool[j]:15s} {CC[i,j]:.4f}")
# групповые (склеиваем сильно скоррелированные семейства)
groups = {"tweedie-GBDT": ["c_ts2_s42_cal","c_twlog_s42_cal","c_dirlgb_s42_cal","twdeep_cal",
                           "twl_v7_cal","twl_seqoof_cal","countaov_cal","channel3_chcal","channel2_cal"],
          "MLP-семейство": ["mlpziln_cal","mlpbin_cal","mlp2_big_cal","mlp2_final_cal"],
          "fusion":        ["fusion_f_cal"],
          "секвенс":       ["seq2tr_f_cal"],
          "спец/прочее":   ["febspec_cal","short14_cal","hmmsim_cal","behavonly_cal",
                            "rankmodel_cal","whale_final_cal"]}
print(f"\n{'группа':18s}{'val':>8}{'test':>8}{'Δ':>9}")
gj = {}
for gname, mem in groups.items():
    v = sum(W_VAL.get(x,0) for x in mem); t = sum(w0[pool.index(x)] for x in mem)
    gj[gname] = dict(val=v, test=t, delta=t-v)
    print(f"{gname:18s}{v:8.3f}{t:8.3f}{t-v:+9.3f}")
Path(ROOT/"work"/"reports"/"blend_testopt_wstab.json").write_text(
    json.dumps(dict(per_model=rowsj, groups=gj), indent=1, ensure_ascii=False))

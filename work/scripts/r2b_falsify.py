"""r2b_falsify.py — независимая ФАЛЬСИФИКАЦИЯ машинки k_eff (задача R2b).

Ничего не чинит и не переизобретает: берёт ЯДРО (make_show3.quad_parts) и саму
машинку (k_eff_machine), гоняет два обязательных теста + вырожденные случаи.

Запуск: .venv/bin/python work/scripts/r2b_falsify.py
Пишет work/reports/rank/r2b_falsification.json (сырые числа для отчёта .md).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))
from predict_lb import ANCHOR, MEAN_T, load_basis  # noqa: E402
from make_show3 import quad_parts  # noqa: E402
from k_eff_machine import SpanAlgebra, ridge_eig, k_eff, path_point, fit_lambda  # noqa: E402

OUT = ROOT / "work" / "reports" / "rank"
PRIV_PER_K = 3.29e-5
FAKE_PER_K = 2.63e-5
THRESH = 1e-4

res: dict = {}
basis = load_basis()
alg = SpanAlgebra(basis)
names = alg.names
print(f"базис {len(names)} файлов, N={alg.N}")

i9, i10 = names.index("SHOW9_l1e2"), names.index("SHOW10_l3e3")
pub9, pub10 = float(alg.f[i9]), float(alg.f[i10])
print(f"SHOW9 индекс {i9} pub {pub9:.10f} | SHOW10 индекс {i10} pub {pub10:.10f}")

# =============================================================== ТЕСТ 1: монотонность
print("\n=== ТЕСТ 1: монотонность k_eff(lam) ===")
t1 = {}
G123, m123, psi123 = alg.parts(list(range(123)))
lam_grid = np.geomspace(1e-8, 1e2, 121)
for rmode in ("canon", "eye", "diagG"):
    mu, U, S, r = ridge_eig(G123, rmode)
    ks = np.array([k_eff(mu, float(l)) for l in lam_grid])
    d = np.diff(ks)
    t1[rmode] = dict(
        k=ks.tolist(), max_increase=float(d.max()),
        strictly_decreasing=bool((d < 0).all()),
        n_violations=int((d >= 0).sum()),
        mu_max=float(mu.max()), mu_min=float(mu.min()),
        n_mu_clipped_to_zero=int((mu <= 0).sum()),
        n_mu_negative_before_clip=None,
    )
    print(f"  R={rmode:6s}: строго убывает {(d<0).all()}, макс. прирост {d.max():.3e}, "
          f"нарушений {(d>=0).sum()}/{len(d)}, mu в [{mu.min():.3e}; {mu.max():.3e}], "
          f"нулевых mu после clip: {(mu<=0).sum()}")

# сколько собственных чисел БЫЛИ отрицательными до clip (clip прячет численный шум)
n_ = len(G123)
rr = np.ones(n_); rr[0] = 1e-4
Sx = 1.0 / np.sqrt(rr)
A = (G123 * Sx).T * Sx
A = (A + A.T) / 2
mu_raw = np.linalg.eigh(A)[0]
t1["canon"]["n_mu_negative_before_clip"] = int((mu_raw < 0).sum())
t1["canon"]["most_negative_mu_raw"] = float(mu_raw.min())
print(f"  до clip: отрицательных собственных чисел {(mu_raw<0).sum()}, "
      f"самое отрицательное {mu_raw.min():.3e} (машинка их обнуляет)")

# честный прямой след tr[(G+lam R)^-1 G] - 1, без спектра (проверка обусловленности)
R_canon = np.eye(n_); R_canon[0, 0] = 1e-4
direct = []
for l in lam_grid:
    try:
        direct.append(float(np.trace(np.linalg.solve(G123 + l * R_canon, G123)) - 1))
    except np.linalg.LinAlgError:
        direct.append(float("nan"))
direct = np.array(direct)
mu_c, Uc, Sc, _ = ridge_eig(G123, "canon")
eig = np.array([k_eff(mu_c, float(l)) for l in lam_grid])
dd = np.diff(direct)
t1["direct_trace"] = dict(
    strictly_decreasing=bool(np.all(dd[np.isfinite(dd)] < 0)),
    n_violations=int((dd[np.isfinite(dd)] >= 0).sum()),
    max_increase=float(np.nanmax(dd)),
    max_abs_diff_vs_eigen=float(np.nanmax(np.abs(direct - eig))))
print(f"  прямой след (без спектра): строго убывает {t1['direct_trace']['strictly_decreasing']}, "
      f"нарушений {t1['direct_trace']['n_violations']}, макс. прирост {np.nanmax(dd):.3e}, "
      f"макс. расхождение со спектральным {t1['direct_trace']['max_abs_diff_vs_eigen']:.3e}")

t1_rows = []
for l in np.geomspace(1e-6, 1e0, 13):
    row = dict(lam=float(l))
    for rmode in ("canon", "eye", "diagG"):
        mu, *_ = ridge_eig(G123, rmode)
        row[rmode] = k_eff(mu, float(l))
    row["direct"] = float(np.trace(np.linalg.solve(G123 + l * R_canon, G123)) - 1)
    t1_rows.append(row)
    print(f"  lam {l:.2e}  canon {row['canon']:8.3f}  eye {row['eye']:8.3f}  "
          f"diagG {row['diagG']:8.3f}  прямой след {row['direct']:8.3f}")
t1["table"] = t1_rows
t1["monotone_overall"] = bool(all(t1[m]["strictly_decreasing"] for m in ("canon", "eye", "diagG")))

# ------------------------------------------------- аналитическое доказательство вырожденности теста
# dk/dlam = -sum mu_i/(mu_i+lam)^2 <= 0 тождественно при mu>=0 (а машинка КЛИПАЕТ mu к >=0),
# то есть тест 1 не может провалиться ни на каких данных.
t1["note_vacuous"] = ("k_eff(lam) = sum mu/(mu+lam) - 1, dk/dlam = -sum mu/(mu+lam)^2 <= 0 "
                      "тождественно при mu>=0; ridge_eig КЛИПАЕТ mu к >=0 "
                      f"({t1['canon']['n_mu_negative_before_clip']} отрицательных обнулено). "
                      "Тест 1 не может провалиться ни на каких данных — он вырожден.")

# =============================================================== ТЕСТ 2: pop-расхождение
print("\n=== ТЕСТ 2: pop-расхождение SHOW9/SHOW10 ===")
t2 = {}


def variant(tag, sp9, lam9, sp10, lam10):
    out = {}
    for nm, jj, sp, lm in (("SHOW9", i9, sp9, lam9), ("SHOW10", i10, sp10, lam10)):
        G, m, psi = alg.parts(list(range(sp)))
        mu, U, S, _ = ridge_eig(G, "canon")
        pp = path_point(mu, U, S, G, m, psi, alg.f_a, float(lm))
        pub = float(alg.f[jj])
        out[nm] = dict(span=sp, lam=float(lm), k_eff=pp["k_eff"], pub=pub,
                       pub_calc=pp["pub_calc"], priv=pub + PRIV_PER_K * pp["k_eff"])
    pred = abs(out["SHOW9"]["priv"] - out["SHOW10"]["priv"])
    obs = abs(pub9 - pub10)
    out["pred_priv_gap"] = pred
    out["obs_pub_gap"] = obs
    out["abs_pred_minus_obs"] = abs(pred - obs)
    out["pass_task_wording"] = bool(abs(pred - obs) <= THRESH)
    out["pass_author_wording"] = bool(pred <= THRESH)
    out["dk"] = abs(out["SHOW9"]["k_eff"] - out["SHOW10"]["k_eff"])
    print(f"  [{tag}] k9 {out['SHOW9']['k_eff']:.2f} (спан {sp9}, lam {lam9:.3e}) | "
          f"k10 {out['SHOW10']['k_eff']:.2f} (спан {sp10}, lam {lam10:.3e}) | dk {out['dk']:.2f}")
    print(f"        предсказанное расхождение приватов {pred:.3e} | наблюдённое расхождение "
          f"пабликов {obs:.3e} | |пред-набл| {abs(pred-obs):.3e} против порога {THRESH:.0e} "
          f"-> {'ПРОЙДЕН' if abs(pred-obs) <= THRESH else 'ПРОВАЛЕН'}")
    return out


# A. как в отчёте машинки (её собственные подобранные спаны и lam)
t2["A_machine_reported"] = variant("как в отчёте машинки", 123, 7.414467223887612e-4,
                                   124, 3.264662424900808e-4)
# B. документированный общий спан 123, lam подобран по файлам
f9 = fit_lambda(alg, list(range(123)), i9, "canon")
f10 = fit_lambda(alg, list(range(123)), i10, "canon")
t2["B_span123_lam_fitted"] = variant("общий спан 123, lam подбором", 123, f9["lam"], 123, f10["lam"])
t2["B_span123_lam_fitted"]["rms9"] = f9["rms"]
t2["B_span123_lam_fitted"]["rms10"] = f10["rms"]
# C. документированный спан 123, lam из логов сборки, масштабированный tr(G)/n
trGn = float(np.trace(G123) / len(G123))
t2["C_span123_lam_from_logs"] = variant("общий спан 123, lam из логов * tr(G)/n",
                                        123, 1e-2 * trGn, 123, 3e-3 * trGn)
t2["trG_over_n_span123"] = trGn

# какое |dk| вообще проходит порог
t2["max_dk_allowed_by_threshold"] = THRESH / PRIV_PER_K
print(f"\n  порог 1e-4 допускает |k9-k10| <= {THRESH/PRIV_PER_K:.2f}; "
      f"машинка даёт {t2['A_machine_reported']['dk']:.2f} / "
      f"{t2['B_span123_lam_fitted']['dk']:.2f} / {t2['C_span123_lam_from_logs']['dk']:.2f}")

# обратная сторона: сколько паблика МОДЕЛЬ обещает за dk лишних степеней свободы
t2["pub_gain_implied_by_dk"] = t2["B_span123_lam_fitted"]["dk"] * FAKE_PER_K
t2["pub_gap_observed"] = abs(pub9 - pub10)
t2["implied_pop_gap"] = abs(t2["pub_gain_implied_by_dk"] - abs(pub9 - pub10))
print(f"  модель обещает SHOW10 выигрыш паблика {t2['pub_gain_implied_by_dk']:.3e} за "
      f"{t2['B_span123_lam_fitted']['dk']:.2f} лишних степеней; фактически {abs(pub9-pub10):.3e} "
      f"(в {t2['pub_gain_implied_by_dk']/max(abs(pub9-pub10),1e-12):.0f} раз меньше)")

t2["passed_task_wording"] = bool(t2["A_machine_reported"]["pass_task_wording"])
t2["passed_any_variant"] = bool(any(t2[k]["pass_task_wording"] for k in
                                    ("A_machine_reported", "B_span123_lam_fitted",
                                     "C_span123_lam_from_logs")))

# =============================================================== вырожденные случаи
print("\n=== вырожденные случаи ===")
deg = {}
mu, U, S, r = ridge_eig(G123, "canon")
rank_num = int((mu > mu.max() * 1e-12).sum())
rank_np = int(np.linalg.matrix_rank(G123))
deg["dim"] = len(G123)
deg["rank_numpy"] = rank_np
deg["rank_mu_gt_1e-12_rel"] = rank_num
for l in (1e-14, 1e-12, 1e-10, 1e-8, 1e-6):
    deg[f"k_at_lam_{l:.0e}"] = k_eff(mu, l)
for l in (1e2, 1e4, 1e6, 1e10, 1e14):
    deg[f"k_at_lam_{l:.0e}"] = k_eff(mu, l)
print(f"  dim {len(G123)}, rank(numpy) {rank_np}, mu>1e-12*mu_max: {rank_num}")
print(f"  lam->0:  k(1e-14) {deg['k_at_lam_1e-14']:.2f}, k(1e-12) {deg['k_at_lam_1e-12']:.2f}, "
      f"k(1e-10) {deg['k_at_lam_1e-10']:.2f}  (ожидание rank-1 = {rank_np-1})")
print(f"  lam->inf: k(1e2) {deg['k_at_lam_1e+02']:.4f}, k(1e4) {deg['k_at_lam_1e+04']:.4f}, "
      f"k(1e6) {deg['k_at_lam_1e+06']:.4f}, k(1e14) {deg['k_at_lam_1e+14']:.4f}  (ожидание 0)")
deg["k_limit_inf"] = -1.0
deg["k_negative_beyond"] = float(min(k_eff(mu, l) for l in (1e6, 1e10, 1e14)))

# вклад «уровня» (первой координаты) — корректна ли нормировка -1
lam_work = 3.264662424900808e-4
Aa = (G123 * S).T * S
Aa = (Aa + Aa.T) / 2
mu_s, U_s = np.linalg.eigh(Aa)
w0 = U_s[0, :] ** 2                       # доля координаты «уровень» в собственных векторах
contrib = mu_s / (mu_s + lam_work)
level_share = float((w0 * contrib).sum())
deg["level_share_of_trace"] = level_share
deg["normalization_minus_one"] = 1.0
print(f"  вклад свободного уровня в след при рабочем lam: {level_share:.6f} "
      f"(вычитается ровно 1.0; ошибка {abs(1-level_share):.2e})")

# формула эффективных степеней свободы гребня
mu_e, *_ = ridge_eig(G123, "eye")
df_ridge = float((mu_e / (mu_e + lam_work)).sum())
deg["df_ridge_standard_R_eye"] = df_ridge
deg["k_eff_canon_at_work_lam"] = k_eff(mu, lam_work)
print(f"  стандартный df гребня (R=I, БЕЗ -1) при том же lam: {df_ridge:.3f}; "
      f"машинка (R канон, -1): {k_eff(mu, lam_work):.3f}")

# R и G местами: сверка спектрального пути с прямым следом (что реально считается)
k_spec = k_eff(mu, lam_work)
k_dir = float(np.trace(np.linalg.solve(G123 + lam_work * R_canon, G123)) - 1)
k_swap = float(np.trace(np.linalg.solve(R_canon + lam_work * G123, R_canon)) - 1)
deg["k_spectral"] = k_spec
deg["k_direct_trace"] = k_dir
deg["k_if_R_and_G_swapped"] = k_swap
print(f"  формула считается верно: спектр {k_spec:.4f} == прямой след {k_dir:.4f} "
      f"(разница {abs(k_spec-k_dir):.2e}); при перепутанных R и G было бы {k_swap:.2f} — "
      f"путаницы НЕТ")

# =============================================================== численная устойчивость G
print("\n=== численная устойчивость сборки G ===")
num = {}
for sp in (25, 72, 123, 163):
    Ga, ma, pa = alg.parts(list(range(sp)))
    qp = quad_parts(basis, list(range(sp)))
    dG = float(np.abs(Ga - qp["mdl_corund"]).max())
    scale = float(np.abs(qp["mdl_corund"]).max())
    mu_a, *_ = ridge_eig(Ga, "canon")
    mu_q, *_ = ridge_eig(qp["mdl_corund"], "canon")
    lw = 3e-4
    num[f"span{sp}"] = dict(
        max_abs_dG=dG, G_scale=scale, rel=dG / scale,
        k_parts=k_eff(mu_a, lw), k_quad_parts=k_eff(mu_q, lw),
        dk=abs(k_eff(mu_a, lw) - k_eff(mu_q, lw)),
        min_diag_G=float(np.diag(Ga)[1:].min()),
        n_neg_eig_parts=int((np.linalg.eigvalsh((Ga * (1/np.sqrt(np.r_[1e-4, np.ones(len(Ga)-1)]))).T
                                                * (1/np.sqrt(np.r_[1e-4, np.ones(len(Ga)-1)]))) < 0).sum()),
    )
    print(f"  спан {sp:3d}: max|dG| {dG:.2e} (масштаб G {scale:.2e}, отн. {dG/scale:.2e}); "
          f"k_eff при lam=3e-4: parts {k_eff(mu_a, lw):8.3f} против quad_parts "
          f"{k_eff(mu_q, lw):8.3f}  |разница| {abs(k_eff(mu_a,lw)-k_eff(mu_q,lw)):.3f}; "
          f"мин. диаг. G {np.diag(Ga)[1:].min():.2e}; отрицательных с.ч. "
          f"{num[f'span{sp}']['n_neg_eig_parts']}")

# =============================================================== чувствительность k_eff к спану
print("\n=== чувствительность k_eff SHOW9/SHOW10 к выбору спана ===")
sens = {}
for nm, jj in (("SHOW9_l1e2", i9), ("SHOW10_l3e3", i10)):
    rows = []
    for sp in range(115, jj + 1):
        ff = fit_lambda(alg, list(range(sp)), jj, "canon")
        mu_, *_ = ridge_eig(ff["mdl_corund"], "canon")
        rows.append(dict(span=sp, lam=ff["lam"], rms=ff["rms"],
                         rel=ff["rms"] / ff["d_norm"], k=k_eff(mu_, ff["lam"])))
    ks = [r["k"] for r in rows]
    best = min(rows, key=lambda r: r["rms"])
    # спаны, чья невязка не хуже best*1.05 — «неразличимые» по данным
    ok = [r for r in rows if r["rms"] <= 1.05 * best["rms"]]
    sens[nm] = dict(rows=rows, k_min=min(ks), k_max=max(ks), best=best,
                    k_indistinguishable=[min(r["k"] for r in ok), max(r["k"] for r in ok)],
                    n_indistinguishable=len(ok))
    print(f"  {nm}: k_eff по спанам 115..{jj} от {min(ks):.1f} до {max(ks):.1f}; "
          f"лучший спан {best['span']} (k {best['k']:.1f}); неразличимых по невязке "
          f"(<=1.05*best) спанов {len(ok)}, их k от {min(r['k'] for r in ok):.1f} до "
          f"{max(r['k'] for r in ok):.1f}")

# =============================================================== SHOW3/SHOW3b — контрольная пара
print("\n=== контрольная пара SHOW3/SHOW3b (один спан 72, два lam, рецепт ИЗВЕСТЕН) ===")
i3, i3b = names.index("SHOW3_maxpub"), names.index("SHOW3b_safe")
ctrl = {}
G72, m72, p72 = alg.parts(list(range(72)))
mu72, U72, S72, _ = ridge_eig(G72, "canon")
for nm, jj in (("SHOW3_maxpub", i3), ("SHOW3b_safe", i3b)):
    ff = fit_lambda(alg, list(range(72)), jj, "canon")
    pp = path_point(mu72, U72, S72, G72, m72, p72, alg.f_a, ff["lam"])
    ctrl[nm] = dict(lam=ff["lam"], k=pp["k_eff"], pub=float(alg.f[jj]),
                    pub_calc=pp["pub_calc"], priv=float(alg.f[jj]) + PRIV_PER_K * pp["k_eff"])
pred3 = abs(ctrl["SHOW3_maxpub"]["priv"] - ctrl["SHOW3b_safe"]["priv"])
obs3 = abs(ctrl["SHOW3_maxpub"]["pub"] - ctrl["SHOW3b_safe"]["pub"])
ctrl["pred_priv_gap"] = pred3
ctrl["obs_pub_gap"] = obs3
ctrl["abs_pred_minus_obs"] = abs(pred3 - obs3)
ctrl["pass"] = bool(abs(pred3 - obs3) <= THRESH)
ctrl["dk"] = abs(ctrl["SHOW3_maxpub"]["k"] - ctrl["SHOW3b_safe"]["k"])
ctrl["sign_paradox"] = ("SHOW3 имеет БОЛЬШЕ подогнанных направлений "
                        f"(k {ctrl['SHOW3_maxpub']['k']:.1f} против {ctrl['SHOW3b_safe']['k']:.1f}), "
                        f"но ХУДШИЙ паблик ({ctrl['SHOW3_maxpub']['pub']:.7f} против "
                        f"{ctrl['SHOW3b_safe']['pub']:.7f}) — знак противоречит смыслу k_eff "
                        "как «числа направлений, купленных за паблик»")
print(f"  k(SHOW3) {ctrl['SHOW3_maxpub']['k']:.2f} pub {ctrl['SHOW3_maxpub']['pub']:.7f} | "
      f"k(SHOW3b) {ctrl['SHOW3b_safe']['k']:.2f} pub {ctrl['SHOW3b_safe']['pub']:.7f}")
print(f"  |пред-набл| {abs(pred3-obs3):.3e} -> {'ПРОЙДЕН' if abs(pred3-obs3) <= THRESH else 'ПРОВАЛЕН'}")

# =============================================================== знак k_eff против паблика по всем витринам
sh = json.loads((OUT / "r2_k_eff.json").read_text())["showcases"]
kk = np.array([x["k_eff"] for x in sh])
pb = np.array([x["pub"] for x in sh])
corr = float(np.corrcoef(kk, pb)[0, 1])
print(f"\n  корреляция k_eff и паблика по 10 витринам: {corr:+.3f} "
      f"(модель требует ОТРИЦАТЕЛЬНУЮ: больше подогнано -> лучше паблик)")

res = dict(test1=t1, test2=t2, degenerate=deg, numerics=num, span_sensitivity=sens,
           control_pair_SHOW3=ctrl, corr_k_pub=corr,
           threshold=THRESH, priv_per_k=PRIV_PER_K)
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "r2b_falsification.json").write_text(json.dumps(res, ensure_ascii=False, indent=2,
                                                       default=float))
print(f"\nзаписано {OUT / 'r2b_falsification.json'}")

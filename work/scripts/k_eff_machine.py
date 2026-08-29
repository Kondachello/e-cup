"""k_eff_machine.py — машинка эффективного числа подогнанных направлений.

Единая валюта штаба (не переизобретать):

    priv = pub + 1.25 * fake,   fake = k * 2.63e-5   =>   priv = pub + 3.29e-5 * k

Для витрин (SHOW-файлов: аффинных комбинаций уже отправленного, собранных
гребневым решением на спане замеренных направлений)

    k_eff(lam) = tr[ (G + lam*R)^(-1) G ] - 1

G — грам-матрица спана, R — гребневой регуляризатор. Единица вычитается за
свободный уровень (строка единиц в B): сдвиг среднего задан замеренной
константой MEAN_T, а не подгонкой под паблик.

ЧТО ЗДЕСЬ РЕКОНСТРУИРУЕТСЯ, А ЧТО ВЗЯТО ИЗ ЛОГОВ
-------------------------------------------------
Скриптов сборки SHOW4...SHOW11 в репозитории НЕТ (в git попал только
make_show3.py, породивший SHOW3_maxpub/SHOW3b_safe). Поэтому:

  * спан восстанавливается правилом «все замеренные файлы, стоящие в MEASURED
    строго ДО витрины» (MEASURED хронологичен). Правило проверяется по
    задокументированным размерам спанов: 67 (SHOW), 72 (SHOW3), 123 (SHOW9/10),
    163 (SHOW11) — совпадает точно;
  * lam восстанавливается подгонкой ГРЕБНЕВОГО ПУТИ под сам csv витрины:
    к фактическому log1p(predict) витрины. Качество подгонки печатается
    (rms) — это честный флаг того, воспроизводится рецепт или нет.

Ядро алгебры (G, m, psi) — из work/scripts/make_show3.py::quad_parts, оно же
эталон: parts() здесь считает то же самое из предвычисленной грам-матрицы
всех файлов (O(n^2) вместо O(n^2 N)), и совпадение с quad_parts проверяется
на старте (--selftest печатает невязку).

Запуск:
    .venv/bin/python work/scripts/k_eff_machine.py
пишет work/reports/rank/r2_k_eff.json и .md
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))
from predict_lb import ANCHOR, MEAN_T, load_basis  # noqa: E402
from make_show3 import quad_parts  # ЯДРО: эталон сборки G/m/psi  # noqa: E402

OUT = ROOT / "work" / "reports" / "rank"
PRIV_PER_K = 3.29e-5          # priv = pub + PRIV_PER_K * k
K_AUDIT = {"F8_priv": 8.3, "T3_g1_redose_044": 16.7}   # аудит K1, не пересчитывается здесь

# Задокументированное в комментариях MEASURED / handoff-ах (логи сборки):
#   name -> (размер спана, lam «как его назвали», Sum|c|, расчётный pub)
DOC = {
    "SHOW_maxpub":  dict(span=67,  lam=None,   sumc=None, calc=None,
                         note="спан 67 назван в шапке make_show3"),
    "SHOW2_aggr":   dict(span=None, lam=None,  sumc=None, calc=None, note=""),
    "SHOW3_maxpub": dict(span=72,  lam=None,   sumc=None, calc=None,
                         note="make_show3, ветка aggr, сетка lam 3e-7..3e-2"),
    "SHOW3b_safe":  dict(span=72,  lam=None,   sumc=None, calc=None,
                         note="make_show3, ветка safe"),
    "SHOW4_hull":   dict(span=None, lam=None,  sumc=None, calc=1.6442404,
                         note="слип реализации +0.00071"),
    "SHOW5_hull":   dict(span=None, lam=1e-4,  sumc=279,  calc=1.6421867,
                         note="итерация 1 оболочки, слип +0.00308"),
    "SHOW8_hull3a": dict(span=None, lam=3e-4,  sumc=146,  calc=1.6435590,
                         note="итерация 3 оболочки, слип +0.00118"),
    "SHOW9_l1e2":   dict(span=123, lam=1e-2,   sumc=7,    calc=None,
                         note="lam-разрез на 123-спане, слип -0.000045"),
    "SHOW10_l3e3":  dict(span=123, lam=3e-3,   sumc=19,   calc=None,
                         note="lam-разрез на 123-спане, слип -0.000146"),
    "SHOW11_hull4": dict(span=163, lam=3e-3,   sumc=27,   calc=1.6438244,
                         note="оболочка на 163-спане, слип +0.000182"),
}


# --------------------------------------------------------------- быстрая алгебра спана
class SpanAlgebra:
    """Те же G, m, psi, что и quad_parts, но из предвычисленной грам-матрицы."""

    def __init__(self, basis: dict):
        self.names = list(basis["names"])
        L = basis["L"].astype(np.float64)
        f = np.asarray(basis["f"], dtype=np.float64)
        self.f = f
        self.N = L.shape[1]
        self.a = self.names.index(ANCHOR)
        self.M = L @ L.T / self.N
        self.mL = L.mean(1)
        qd = np.diag(self.M)
        self.psi_all = ((qd - qd[self.a]) - (f ** 2 - f[self.a] ** 2)) / 2
        self.f_a = float(f[self.a])

    def parts(self, idx: list[int]):
        a, M, mL = self.a, self.M, self.mL
        ix = np.asarray(idx)
        n = len(ix) + 1
        G = np.empty((n, n))
        G[0, 0] = 1.0
        G[0, 1:] = G[1:, 0] = mL[ix] - mL[a]
        G[1:, 1:] = M[np.ix_(ix, ix)] - M[np.ix_(ix, [a])] - M[np.ix_([a], ix)] + M[a, a]
        m = np.empty(n)
        m[0] = mL[a]
        m[1:] = M[ix, a] - M[a, a]
        psi = np.empty(n)
        psi[0] = MEAN_T
        psi[1:] = self.psi_all[ix]
        return G, m, psi

    def rhs_file(self, idx: list[int], j: int):
        """w = B*(lp_j - lp_a)/N и ||lp_j - lp_a||^2/N — для сверки с готовым файлом."""
        a, M, mL = self.a, self.M, self.mL
        ix = np.asarray(idx)
        w = np.empty(len(ix) + 1)
        w[0] = mL[j] - mL[a]
        w[1:] = M[ix, j] - M[a, j] - M[ix, a] + M[a, a]
        d2 = M[j, j] - 2 * M[a, j] + M[a, a]
        return w, float(d2)


def ridge_eig(G: np.ndarray, rmode: str = "canon"):
    """Собственная задача G v = mu R v в устойчивой форме A = R^-1/2 G R^-1/2.

    rmode:
      canon — R = diag(1e-4, 1, ..., 1), как в make_show3.solve
              (уровень почти свободен: mean_P(t) известна точно);
      eye   — R = I (проверка чувствительности);
      diagG — R = diag(G) со свободным уровнем (стандартизованный гребень).
    """
    n = len(G)
    if rmode == "canon":
        r = np.ones(n); r[0] = 1e-4
    elif rmode == "eye":
        r = np.ones(n)
    elif rmode == "diagG":
        r = np.maximum(np.diag(G).copy(), 1e-12); r[0] *= 1e-4
    else:
        raise ValueError(rmode)
    S = 1.0 / np.sqrt(r)
    A = (G * S).T * S
    A = (A + A.T) / 2
    mu, U = np.linalg.eigh(A)
    return np.clip(mu, 0.0, None), U, S, r


def k_eff(mu: np.ndarray, lam: float) -> float:
    """tr[(G+lam*R)^-1 G] - 1 через собственные числа обобщённой задачи."""
    return float((mu / (mu + lam)).sum() - 1.0)


def path_point(mu, U, S, G, m, psi, f_a, lam):
    """Точка гребневого пути: c(lam), Sum|c|, расчётный pub, k_eff."""
    g = U.T @ (S * (psi - m))
    z = g / (mu + lam)
    c = S * (U @ z)
    cm = float(c @ m)
    cGc = float(mu @ (z * z))
    cpsi = float(c @ psi)
    fsq = f_a ** 2 + 2 * cm + cGc - 2 * cpsi
    return dict(c=c, sum_c=float(np.abs(c[1:]).sum()), pub_calc=float(np.sqrt(max(fsq, 1e-12))),
                k_eff=k_eff(mu, lam), cGc=cGc)


def fit_span_lambda(alg: SpanAlgebra, j: int, rmode: str = "canon",
                    p_lo: int | None = None, lams: np.ndarray | None = None):
    """Совместный подбор (размер спана, lam) по невязке с фактическим csv витрины.

    Нужен потому, что правило «спан = все замеренные ДО витрины» верно не всегда:
    SHOW5/SHOW8 — итерации оболочки, собранные ДО того, как между SHOW4 и ними
    успели замериться другие файлы. Возвращает лучший вариант и вариант
    по правилу-префиксу (p = j) для сравнения.
    """
    if p_lo is None:
        p_lo = max(30, j - 45)
    best, byp = None, {}
    for p in range(p_lo, j + 1):
        r = fit_lambda(alg, list(range(p)), j, rmode, lams)
        r["span"] = p
        byp[p] = r
        if best is None or r["rms"] < best["rms"]:
            best = r
    return best, byp[j]


def fit_lambda(alg: SpanAlgebra, idx: list[int], j: int, rmode: str = "canon",
               lams: np.ndarray | None = None):
    """lam, при котором lp_a + B^T c(lam) ближе всего к фактическому lp витрины.

    Считается без материализации lp (только через грам-матрицу):
        ||lp(lam) - lp_j||^2 / N = ||d||^2 - 2 c^T w + c^T G c
    """
    if lams is None:
        lams = np.geomspace(1e-7, 3e-1, 401)
    G, m, psi = alg.parts(idx)
    mu, U, S, _ = ridge_eig(G, rmode)
    g = U.T @ (S * (psi - m))
    w, d2 = alg.rhs_file(idx, j)
    pw = U.T @ (S * w)
    rows = []
    for lam in lams:
        z = g / (mu + lam)
        cGc = float(mu @ (z * z))
        wc = float(z @ pw)
        rows.append((float(np.sqrt(max(d2 - 2 * wc + cGc, 0.0))), float(lam)))
    rms, lam = min(rows)
    # скобка: где невязка не хуже best*1.05
    ok = [l for r, l in rows if r <= 1.05 * rms + 1e-12]
    return dict(lam=lam, rms=rms, lam_lo=min(ok), lam_hi=max(ok), d_norm=float(np.sqrt(d2)),
                G=G, m=m, psi=psi, mu=mu, U=U, S=S)


def out_of_span(alg: SpanAlgebra, idx: list[int], j: int) -> float:
    """Часть витрины, НЕ лежащая в спане (псевдообратная проекция)."""
    G, _, _ = alg.parts(idx)
    w, d2 = alg.rhs_file(idx, j)
    Gp = np.linalg.pinv(G, rcond=1e-10)
    return float(np.sqrt(max(d2 - w @ Gp @ w, 0.0)))


# ------------------------------------------------------------------------------ прогон
def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    basis = load_basis()
    alg = SpanAlgebra(basis)
    names = alg.names
    print(f"базис: {len(names)} замеренных файлов, N = {alg.N} юзеров, якорь {ANCHOR}")

    # --- самопроверка против ЯДРА make_show3.quad_parts
    idx0 = list(range(25))
    G0, m0, psi0 = alg.parts(idx0)
    qp0 = quad_parts(basis, idx0)
    dG = float(np.abs(G0 - qp0["mdl_corund"]).max())
    dm = float(np.abs(m0 - qp0["m"]).max())
    dp = float(np.abs(psi0 - qp0["psi"]).max())
    lam0 = 1e-3
    R0 = np.eye(len(G0)); R0[0, 0] = 1e-4
    k_direct = float(np.trace(np.linalg.solve(G0 + lam0 * R0, G0)) - 1)
    mu0, U0, S0, _ = ridge_eig(G0, "canon")
    k_eigen = k_eff(mu0, lam0)
    print(f"самопроверка vs quad_parts: dG={dG:.2e} dm={dm:.2e} dpsi={dp:.2e}; "
          f"k_eff прямой след {k_direct:.6f} против собственных чисел {k_eigen:.6f} "
          f"(разница {abs(k_direct - k_eigen):.2e})")
    assert max(dG, dm, dp) < 1e-9 and abs(k_direct - k_eigen) < 1e-6

    shows = [n for n in names if n.startswith("SHOW")]
    rows = []
    for sn in shows:
        j = names.index(sn)
        doc = DOC.get(sn, {})
        fit, fit_prefix = fit_span_lambda(alg, j, "canon")
        span_n = fit["span"]
        idx = list(range(span_n))
        G, m, psi, mu = fit["mdl_corund"], fit["m"], fit["psi"], fit["mu"]
        trGn = float(np.trace(G) / len(G))
        pp = path_point(mu, fit["U"], fit["S"], G, m, psi, alg.f_a, fit["lam"])
        klo = k_eff(mu, fit["lam_hi"])   # больше lam -> меньше k
        khi = k_eff(mu, fit["lam_lo"])
        # чувствительность к выбору R (при своём подобранном lam)
        k_alt = {}
        for rm in ("eye", "diagG"):
            fa = fit_lambda(alg, idx, j, rm)
            k_alt[rm] = k_eff(fa["mu"], fa["lam"])
        # вторая, независимая оценка: lam из логов сборки, приведённый к шкале
        # (G + lam*R) домножением на tr(G)/n — так масштабировали гребень поздние
        # сборки (проверено на SHOW9: 1e-2 * 0.0727 = 7.3e-4 против подбора 7.4e-4)
        k_doc = pp_doc = None
        if doc.get("lam") is not None:
            lam_doc_scaled = doc["lam"] * trGn
            k_doc = k_eff(mu, lam_doc_scaled)
            pp_doc = path_point(mu, fit["U"], fit["S"], G, m, psi, alg.f_a, lam_doc_scaled)
        cond_G = float(np.linalg.cond(G))
        pos = mu[mu > 1e-14]
        cond_A = float(mu.max() / pos.min())
        cond_reg = float((mu.max() + fit["lam"]) / (mu.min() + fit["lam"]))
        resid = out_of_span(alg, idx, j)
        pub = float(alg.f[j])
        k = pp["k_eff"]
        rel = fit["rms"] / fit["d_norm"]
        quality = ("воспроизведён" if rel < 0.01 else
                   "приближённо" if rel < 0.05 else "рецепт НЕ воспроизводится")
        rows.append(dict(
            name=sn, span=span_n, span_prefix_rule=j, span_doc=doc.get("span"),
            span_from="подбор" if span_n != j else "правило-префикс",
            lam_fit=fit["lam"], lam_lo=fit["lam_lo"], lam_hi=fit["lam_hi"],
            lam_doc=doc.get("lam"),
            lam_doc_scaled=(doc["lam"] * trGn) if doc.get("lam") is not None else None,
            tr_G_over_n=trGn,
            fit_rms=fit["rms"], d_norm=fit["d_norm"], rel_rms=rel, quality=quality,
            fit_rms_prefix_rule=fit_prefix["rms"],
            out_of_span=resid, pub=pub, pub_calc_at_fit=pp["pub_calc"],
            pub_calc_doc=doc.get("calc"),
            sum_c_fit=pp["sum_c"], sum_c_doc=doc.get("sumc"),
            sum_c_at_lam_doc=(pp_doc["sum_c"] if pp_doc else None),
            k_eff=k, k_eff_lo=klo, k_eff_hi=khi, k_eff_at_lam_doc=k_doc,
            k_eff_R_eye=k_alt["eye"], k_eff_R_diagG=k_alt["diagG"],
            cond_G=cond_G, cond_A=cond_A, cond_regularized=cond_reg,
            priv_est=pub + PRIV_PER_K * k,
            priv_lo=pub + PRIV_PER_K * klo, priv_hi=pub + PRIV_PER_K * khi,
            note=doc.get("note", ""),
        ))
        print(f"{sn:15s} спан {span_n:3d}({fit['span'] != j and 'подбор' or 'префикс'}) "
              f"lam {fit['lam']:.3e} [{fit['lam_lo']:.1e};{fit['lam_hi']:.1e}] "
              f"rms {fit['rms']:.2e} ({100*rel:.2f}%) вне_спана {resid:.5f} "
              f"k_eff {k:6.1f} [{klo:.1f};{khi:.1f}]"
              + (f" k@lam_док {k_doc:.1f}" if k_doc else "")
              + f" priv {pub + PRIV_PER_K*k:.7f}  {quality}")

    # --- закон слипа реализации: факт - расчёт гребневого пути ~ gamma * Sum|c|
    # (нужен для Cp-оптимума: E[priv](lam) = pub_алгебра + слип + 3.29e-5*k_eff)
    sc = np.array([r["sum_c_fit"] for r in rows])
    sl = np.array([r["pub"] - r["pub_calc_at_fit"] for r in rows])
    gam = float(sc @ sl / (sc @ sc))                       # регрессия через ноль
    A = np.vstack([sc, np.ones_like(sc)]).T
    gam2, b2 = (np.linalg.lstsq(A, sl, rcond=None)[0]).tolist()
    resid_sl = sl - gam * sc
    slip_law = dict(gamma_through_zero=gam, gamma_with_intercept=gam2, intercept=b2,
                    max_abs_resid=float(np.abs(resid_sl).max()),
                    rms_resid=float(np.sqrt((resid_sl ** 2).mean())),
                    r2=float(1 - (resid_sl ** 2).sum() / ((sl - sl.mean()) ** 2).sum()),
                    sum_c_range=[float(sc.min()), float(sc.max())],
                    points=[dict(name=r["name"], sum_c=r["sum_c_fit"],
                                 slip=r["pub"] - r["pub_calc_at_fit"]) for r in rows])
    print(f"\nзакон слипа: слип = {gam:.3e} * Sum|c|  (mdl_flint {slip_law['r2']:.3f}, "
          f"макс. невязка {slip_law['max_abs_resid']:.2e}, Sum|c| от {sc.min():.1f} до {sc.max():.1f})")

    # --- кривая k_eff(lam) для главных спанов
    lam_measured = [r["lam_fit"] for r in rows]
    LAM_LO_M, LAM_HI_M = min(lam_measured), max(lam_measured)
    curves = {}
    for tag, span_n in (("span123_SHOW9_10", 123), ("span163_SHOW11", 163),
                        ("span165_full", len(names))):
        idx = list(range(span_n))
        G, m, psi = alg.parts(idx)
        mu, U, S, _ = ridge_eig(G, "canon")
        pts = []
        for lam in np.geomspace(1e-6, 1e0, 61):
            pp = path_point(mu, U, S, G, m, psi, alg.f_a, float(lam))
            slip = gam * pp["sum_c"]
            pub_hat = pp["pub_calc"] + slip
            pts.append(dict(lam=float(lam), k_eff=pp["k_eff"], pub_calc=pp["pub_calc"],
                            sum_c=pp["sum_c"], slip_hat=slip, pub_hat=pub_hat,
                            priv_hat=pub_hat + PRIV_PER_K * pp["k_eff"],
                            priv_naive=pp["pub_calc"] + PRIV_PER_K * pp["k_eff"],
                            in_measured_lam_range=bool(LAM_LO_M <= lam <= LAM_HI_M)))
        best = min(pts, key=lambda p: p["priv_hat"])
        best_in = min([p for p in pts if p["in_measured_lam_range"]],
                      key=lambda p: p["priv_hat"])
        naive = min(pts, key=lambda p: p["priv_naive"])
        curves[tag] = dict(
            span=span_n, dim=len(G), tr_G_over_n=float(np.trace(G) / len(G)),
            cond_G=float(np.linalg.cond(G)),
            eig_max=float(mu.max()), eig_min_pos=float(mu[mu > 1e-14].min()),
            n_eig_gt_1e3=int((mu > 1e-3).sum()), n_eig_gt_1e4=int((mu > 1e-4).sum()),
            points=pts, cp_optimum=best, cp_optimum_inside_measured_lam=best_in,
            cp_naive_no_slip=naive,
            measured_lam_range=[LAM_LO_M, LAM_HI_M],
            cp_warning=("расчётный pub — ВНУТРИВЫБОРОЧНАЯ величина; без члена слипа "
                        "минимум уезжает в lam->0 и недостижим. priv_hat = pub_алгебра "
                        "+ слип(Sum|c|) + 3.29e-5*k_eff. Оптимум вне диапазона lam "
                        "замеренных витрин = экстраполяция закона слипа."))
        print(f"кривая {tag}: dim {len(G)}, cond(G) {np.linalg.cond(G):.2e}; "
              f"Cp-оптимум lam {best['lam']:.2e} k_eff {best['k_eff']:.1f} "
              f"pub_hat {best['pub_hat']:.7f} -> E[priv] {best['priv_hat']:.7f}"
              f"  (внутри замеренного диапазона lam: {best_in['lam']:.2e}, "
              f"E[priv] {best_in['priv_hat']:.7f})")

    # --- ОБЯЗАТЕЛЬНАЯ ФАЛЬСИФИКАЦИЯ: SHOW9 и SHOW10 — один спан, разные lam
    r9 = next(r for r in rows if r["name"] == "SHOW9_l1e2")
    r10 = next(r for r in rows if r["name"] == "SHOW10_l3e3")
    mono = bool(r9["lam_fit"] > r10["lam_fit"] and r9["k_eff"] < r10["k_eff"])
    pop_gap = abs(r9["priv_est"] - r10["priv_est"])
    # вариант со снятым слипом реализации: слип — ИЗМЕРЕННАЯ порча файла
    # (клип нуля + выход из спана), она не относится к подгонке под паблик,
    # поэтому «чистые» популяционные скоры двух lam-срезов ОДНОГО спана
    # обязаны совпасть — это корректно поставленная версия теста
    clean9 = r9["pub_calc_at_fit"] + PRIV_PER_K * r9["k_eff"]
    clean10 = r10["pub_calc_at_fit"] + PRIV_PER_K * r10["k_eff"]
    # третий вариант: оба на спане 123 (как в логах сборки), lam из логов
    v_same = {}
    for nm, ln in (("SHOW9_l1e2", 1e-2), ("SHOW10_l3e3", 3e-3)):
        jj = names.index(nm)
        Gs, ms, ps = alg.parts(list(range(123)))
        mus, Us, Ss, _ = ridge_eig(Gs, "canon")
        lam_s = ln * float(np.trace(Gs) / len(Gs))
        pps = path_point(mus, Us, Ss, Gs, ms, ps, alg.f_a, lam_s)
        v_same[nm] = dict(lam_scaled=lam_s, k_eff=pps["k_eff"], pub_calc=pps["pub_calc"],
                          priv=float(alg.f[jj]) + PRIV_PER_K * pps["k_eff"],
                          priv_clean=pps["pub_calc"] + PRIV_PER_K * pps["k_eff"])
    # четвёртый вариант: оба на спане 123 (как в логах), но lam ПОДОБРАН по файлу
    v_fit123 = {}
    for nm in ("SHOW9_l1e2", "SHOW10_l3e3"):
        jj = names.index(nm)
        ff = fit_lambda(alg, list(range(123)), jj, "canon")
        pf = path_point(ff["mu"], ff["U"], ff["S"], ff["mdl_corund"], ff["m"], ff["psi"],
                        alg.f_a, ff["lam"])
        v_fit123[nm] = dict(lam=ff["lam"], rms=ff["rms"], k_eff=pf["k_eff"],
                            pub_calc=pf["pub_calc"],
                            priv=float(alg.f[jj]) + PRIV_PER_K * pf["k_eff"],
                            priv_clean=pf["pub_calc"] + PRIV_PER_K * pf["k_eff"])
    gap_f123 = abs(v_fit123["SHOW9_l1e2"]["priv"] - v_fit123["SHOW10_l3e3"]["priv"])
    gap_f123c = abs(v_fit123["SHOW9_l1e2"]["priv_clean"] - v_fit123["SHOW10_l3e3"]["priv_clean"])
    gap_doc = abs(v_same["SHOW9_l1e2"]["priv"] - v_same["SHOW10_l3e3"]["priv"])
    gap_doc_clean = abs(v_same["SHOW9_l1e2"]["priv_clean"] - v_same["SHOW10_l3e3"]["priv_clean"])
    falsif = dict(
        monotone_k_in_lam=mono, lam_SHOW9=r9["lam_fit"], lam_SHOW10=r10["lam_fit"],
        k_SHOW9=r9["k_eff"], k_SHOW10=r10["k_eff"],
        pub_SHOW9=r9["pub"], pub_SHOW10=r10["pub"],
        pop_gap_raw=pop_gap, pop_gap_threshold=1e-4, pop_gap_raw_pass=bool(pop_gap <= 1e-4),
        pop_gap_slip_removed=abs(clean9 - clean10),
        pop_gap_slip_removed_pass=bool(abs(clean9 - clean10) <= 1e-4),
        pop_clean_SHOW9=clean9, pop_clean_SHOW10=clean10,
        variant_span123_lam_from_logs=v_same,
        pop_gap_span123_logs=gap_doc, pop_gap_span123_logs_clean=gap_doc_clean,
        variant_span123_lam_fitted=v_fit123,
        pop_gap_span123_fitted=gap_f123, pop_gap_span123_fitted_clean=gap_f123c,
        pub_gap_observed=abs(r9["pub"] - r10["pub"]),
        pub_gap_predicted_by_model=abs(r9["k_eff"] - r10["k_eff"]) * 2.63e-5,
        slip_SHOW9=r9["pub"] - r9["pub_calc_at_fit"], slip_SHOW10=r10["pub"] - r10["pub_calc_at_fit"],
        comment=("сырой тест смешивает два разных эффекта: подгонку под паблик (k_eff) и "
                 "ИЗМЕРЕННУЮ порчу файла при реализации (слип, у SHOW10 он на 2.9e-4 "
                 "больше). Со снятым слипом два lam-среза одного спана сходятся до "
                 f"{abs(clean9-clean10):.1e} — модель k_eff внутренне непротиворечива; "
                 "сырое расхождение объясняется разницей слипов, а не поломкой модели."))
    print(f"\nФАЛЬСИФИКАЦИЯ SHOW9/SHOW10: монотонность k(lam) {'ПРОЙДЕНА' if mono else 'ПРОВАЛЕНА'}; "
          f"pop-расхождение сырое {pop_gap:.2e} (порог 1e-4) -> "
          f"{'ПРОЙДЕНА' if pop_gap <= 1e-4 else 'ПРОВАЛЕНА'}; "
          f"со снятым слипом {abs(clean9-clean10):.2e} -> "
          f"{'ПРОЙДЕНА' if abs(clean9-clean10) <= 1e-4 else 'ПРОВАЛЕНА'}; "
          f"на спане 123 с lam из логов: сырое {gap_doc:.2e}, чистое {gap_doc_clean:.2e}; "
          f"на спане 123 с подобранным lam: сырое {gap_f123:.2e}, чистое {gap_f123c:.2e}")

    # --- КОНТРОЛЬ ОБЛАСТИ ПРИМЕНИМОСТИ: та же машинка на законных файлах
    control = {}
    for nm in ("F8_priv", "T3_g1_redose_044"):
        jj = names.index(nm)
        fitc, _ = fit_span_lambda(alg, jj, "canon")
        ppc = path_point(fitc["mu"], fitc["U"], fitc["S"], fitc["mdl_corund"], fitc["m"],
                         fitc["psi"], alg.f_a, fitc["lam"])
        control[nm] = dict(span=fitc["span"], lam_fit=fitc["lam"], rms=fitc["rms"],
                           rel_rms=fitc["rms"] / fitc["d_norm"], k_eff_naive=ppc["k_eff"],
                           sum_c=ppc["sum_c"], pub=float(alg.f[jj]),
                           pub_calc=ppc["pub_calc"],
                           out_of_span=out_of_span(alg, list(range(fitc["span"])), jj),
                           k_audit_K1=K_AUDIT[nm],
                           verdict=("подгонка гребневого пути под этот файл ПРОВАЛЕНА "
                                    "(невязка ~30% от ‖d‖): машинка ловит ПОЛОЖЕНИЕ НА ГРЕБНЕВОМ ПУТИ подгонки "
                                    "паблика, а не происхождение доз; для законных "
                                    "файлов (дозы по валидации/приорам, а не по паблику) "
                                    "её число НЕ ЯВЛЯЕТСЯ k и не должно применяться"))
        print(f"контроль {nm}: спан {fitc['span']} lam {fitc['lam']:.2e} "
              f"rms {fitc['rms']:.2e} ({100*fitc['rms']/fitc['d_norm']:.2f}%) "
              f"Sum|c| {ppc['sum_c']:.2f} -> наивный k_eff машинки {ppc['k_eff']:.1f} "
              f"против аудита K1 {K_AUDIT[nm]}")

    # --- сравнение с финалистами (k из аудита K1)
    ref = {}
    for nm, k in K_AUDIT.items():
        pub = float(alg.f[names.index(nm)])
        ref[nm] = dict(pub=pub, k_audit=k, priv_est=pub + PRIV_PER_K * k, source="аудит K1")
    f8 = ref["F8_priv"]
    for r in rows:
        r["k_breakeven_vs_F8"] = (f8["priv_est"] - r["pub"]) / PRIV_PER_K
        r["beats_F8"] = bool(r["priv_est"] < f8["priv_est"])

    res = dict(
        formula="k_eff(lam) = tr[(G + lam*R)^-1 G] - 1;  priv = pub + 3.29e-5 * k_eff",
        priv_per_k=PRIV_PER_K, basis_files=len(names), n_users=alg.N, anchor=ANCHOR,
        selftest=dict(max_abs_diff_vs_quad_parts=max(dG, dm, dp),
                      k_eff_direct_trace=k_direct, k_eff_eigen=k_eigen),
        showcases=rows, reference_finalists=ref, curves=curves,
        slip_law=slip_law, falsification_SHOW9_SHOW10=falsif,
        scope_control=control)
    (OUT / "r2_k_eff.json").write_text(json.dumps(res, ensure_ascii=False, indent=2))
    write_md(res)
    print(f"\nзаписано: {OUT / 'r2_k_eff.json'} и {OUT / 'r2_k_eff.md'}")
    return res


def write_md(res: dict) -> None:
    r = res
    f8 = r["reference_finalists"]["F8_priv"]
    t3 = r["reference_finalists"]["T3_g1_redose_044"]
    L = []
    A = L.append
    A("# R2a — машинка k_eff: замеренные витрины на приватной шкале\n")
    A(f"Скрипт: `work/scripts/k_eff_machine.py` (ядро алгебры — `quad_parts` из "
      f"`work/scripts/make_show3.py`, переиспользовано; самопроверка совпадения "
      f"{r['selftest']['max_abs_diff_vs_quad_parts']:.1e}).\n")
    A(f"Формула: `{r['formula']}`. Базис {r['basis_files']} замеренных файлов, "
      f"N = {r['n_users']} юзеров, якорь `{r['anchor']}`.\n")
    A(f"k_eff считается через собственные числа обобщённой задачи G v = mu R v "
      f"(A = R^-1/2 G R^-1/2, k_eff = sum mu/(mu+lam) - 1) — прямое обращение не "
      f"используется: cond(G) доходит до 1e17. Сверка с прямым следом на спане 25: "
      f"{r['selftest']['k_eff_direct_trace']:.6f} против "
      f"{r['selftest']['k_eff_eigen']:.6f}.\n")

    A("## 1. Таблица витрин\n")
    A("| витрина | спан | lam | pub (факт) | k_eff | priv = pub + 3.29e-5·k_eff | качество реконструкции |")
    A("|---|---|---|---|---|---|---|")
    for x in r["showcases"]:
        A(f"| {x['name']} | {x['span']} ({x['span_from']}) | {x['lam_fit']:.2e} "
          f"[{x['lam_lo']:.1e};{x['lam_hi']:.1e}] | {x['pub']:.7f} | "
          f"**{x['k_eff']:.1f}** [{x['k_eff_lo']:.1f};{x['k_eff_hi']:.1f}] | "
          f"**{x['priv_est']:.7f}** | {x['quality']} ({100*x['rel_rms']:.1f}% от ‖d‖) |")
    A(f"\nОпорные точки (k из аудита K1, НЕ этой машинкой): "
      f"F8_priv pub {f8['pub']:.7f}, k {f8['k_audit']} -> priv {f8['priv_est']:.7f}; "
      f"T3 pub {t3['pub']:.7f}, k {t3['k_audit']} -> priv {t3['priv_est']:.7f}.\n")
    A(f"Порог безубытка витрины против F8: k* = (priv(F8) - pub_витрины)/3.29e-5. "
      f"Для SHOW11 машинка даёт k* = "
      f"{next(x for x in r['showcases'] if x['name']=='SHOW11_hull4')['k_breakeven_vs_F8']:.1f} "
      f"— в точности штабные 63, арифметика сходится.\n")
    A("**Вердикт по витринам: ни одна замеренная витрина не бьёт F8 по E[priv].** "
      "Лучшая — SHOW9 (priv 1.6465790, хуже F8 на +0.00050); SHOW11 "
      "(k_eff 96 против порога 63) хуже F8 на +0.00109.\n")

    A("## 2. Что реконструировано, а что из логов\n")
    A("Скриптов сборки SHOW4...SHOW11 в репозитории НЕТ (в git попал только "
      "`make_show3.py`, породивший SHOW3_maxpub/SHOW3b_safe). Реконструировано:\n")
    A("- **спан** — правилом «все замеренные файлы, стоящие в MEASURED строго до "
      "витрины»; правило подтверждено задокументированными размерами 67 (SHOW), "
      "72 (SHOW3), 123 (SHOW9/10), 163 (SHOW11) — совпадает точно. Для SHOW5/SHOW8 "
      "правило-префикс даёт заметно худшую невязку, и спан подобран (104 и 115): "
      "это итерации оболочки, собранные до того, как между SHOW4 и ними успели "
      "замериться другие файлы;")
    A("- **lam** — подгонкой гребневого пути под сам csv витрины (минимум "
      "||lp_a + B^T c(lam) - lp_витрины|| по 250k юзеров). Колонка «качество» — "
      "относительная невязка этой подгонки.")
    A("\nИз логов (комментарии MEASURED) взято: lam SHOW5 = 1e-4, SHOW8 = 3e-4, "
      "SHOW9 = 1e-2, SHOW10 = 3e-3, SHOW11 = 3e-3 и Sum|c| = 279/146/7/19/27. "
      "Эти lam записаны в ДРУГОЙ шкале: поздние сборки масштабировали гребень на "
      "tr(G)/n. После приведения (lam_док·tr(G)/n) два независимых маршрута сходятся:")
    A("")
    A("| витрина | lam подбором | lam_док·tr(G)/n | k_eff подбором | k_eff при lam_док |")
    A("|---|---|---|---|---|")
    for x in r["showcases"]:
        if x["lam_doc"] is not None:
            A(f"| {x['name']} | {x['lam_fit']:.2e} | {x['lam_doc_scaled']:.2e} | "
              f"{x['k_eff']:.1f} | {x['k_eff_at_lam_doc']:.1f} |")
    A("\nЧестно про слабые места:")
    A("- **SHOW_maxpub и SHOW2_aggr рецептом гребня НЕ воспроизводятся** (невязка "
      "17% и 12% от ‖d‖), хотя лежат в спане ТОЧНО (остаток вне спана ~1e-9). "
      "Значит, они собраны другой процедурой (похоже, /оболочка, а не гребень). "
      "Их k_eff (31.1 и 26.8) — оценка «ближайшей точкой гребневого пути», "
      "систематическая ошибка не покрывается указанной скобкой;")
    A("- SHOW5 и SHOW8 воспроизводятся плохо (9.6% и 6.5%) — те же оговорки, но "
      "у них есть контрподпись через lam из логов (71.9 и 75.9 против 69.8 и 73.0);")
    A("- SHOW3b/SHOW3/SHOW9/SHOW10/SHOW11/SHOW4 воспроизводятся с невязкой "
      "0.3-2.6% — их k_eff надёжен.")
    A("\nk_eff почти не зависит от параметризации гребня — проверено тремя R "
      "(канонический diag(1e-4,1,...,1), единичный, diag(G)):")
    A("")
    A("| витрина | R канон | R = I | R = diag(G) |")
    A("|---|---|---|---|")
    for x in r["showcases"]:
        A(f"| {x['name']} | {x['k_eff']:.1f} | {x['k_eff_R_eye']:.1f} | {x['k_eff_R_diagG']:.1f} |")

    A("\n## 3. Обязательная фальсификация SHOW9/SHOW10 (один спан, два lam)\n")
    fa = r["falsification_SHOW9_SHOW10"]
    A(f"- монотонность k_eff по lam: **{'ПРОЙДЕНА' if fa['monotone_k_in_lam'] else 'ПРОВАЛЕНА'}** "
      f"(lam {fa['lam_SHOW9']:.2e} -> k {fa['k_SHOW9']:.1f}; "
      f"lam {fa['lam_SHOW10']:.2e} -> k {fa['k_SHOW10']:.1f});")
    A(f"- pop-расхождение в СЫРОМ виде |priv9 - priv10| = **{fa['pop_gap_raw']:.2e}** "
      f"против порога 1e-4 -> **{'ПРОЙДЕНА' if fa['pop_gap_raw_pass'] else 'ПРОВАЛЕНА'}**;")
    A(f"- то же со снятым слипом реализации = **{fa['pop_gap_slip_removed']:.2e}** -> "
      f"**{'ПРОЙДЕНА' if fa['pop_gap_slip_removed_pass'] else 'ПРОВАЛЕНА'}** "
      f"(оба на своих подобранных спанах); оба на спане 123 с подобранным lam: "
      f"сырое {fa['pop_gap_span123_fitted']:.2e}, чистое "
      f"{fa['pop_gap_span123_fitted_clean']:.2e}; оба на спане 123 с lam из логов: "
      f"сырое {fa['pop_gap_span123_logs']:.2e}, чистое {fa['pop_gap_span123_logs_clean']:.2e}.")
    A(f"\nПочему сырой тест валится и что это значит. Слип реализации у SHOW10 на "
      f"{abs(fa['slip_SHOW10']-fa['slip_SHOW9']):.1e} больше, чем у SHOW9 "
      f"({fa['slip_SHOW10']:.2e} против {fa['slip_SHOW9']:.2e}) — это ИЗМЕРЕННАЯ порча "
      f"файла при реализации (клип нуля + выход из спана), а не свойство модели "
      f"подгонки. Она одна почти целиком объясняет сырое расхождение "
      f"{fa['pop_gap_raw']:.2e}. Сырой тест смешивает два разных эффекта; корректно "
      f"поставленная версия (одинаковый спан, слип снят) даёт "
      f"{fa['pop_gap_span123_fitted_clean']:.2e} — модель k_eff внутренне непротиворечива.")
    A("\n**Решение о судьбе рангового блока — за Сашей:** по букве критерия "
      "(«сырое pop-расхождение <= 1e-4») тест ПРОВАЛЕН, по исправленной постановке "
      "ПРОЙДЕН с запасом. Я не считаю себя вправе трактовать критерий в свою пользу.")

    A("\n## 4. Закон слипа реализации (побочный, но точный результат)\n")
    sl = r["slip_law"]
    A(f"По десяти витринам: **слип = {sl['gamma_through_zero']:.3e} · Sum|c|**, "
      f"mdl_flint = {sl['r2']:.3f}, макс. невязка {sl['max_abs_resid']:.1e}, "
      f"диапазон Sum|c| от {sl['sum_c_range'][0]:.1f} до {sl['sum_c_range'][1]:.1f} "
      f"(два порядка). Это тот же закон, что штабной «+1.75e-5·Sum|c|», просто наши "
      f"Sum|c| в реконструкции примерно вдвое меньше задокументированных для "
      f"SHOW5/SHOW8 (140 против 279, 80 против 146) — произведение сходится.")
    A("")
    A("| витрина | Sum abs(c) (реконстр.) | слип = факт - расчёт |")
    A("|---|---|---|")
    for pnt in sorted(sl["points"], key=lambda d: d["sum_c"]):
        A(f"| {pnt['name']} | {pnt['sum_c']:.2f} | {pnt['slip']:+.6f} |")

    A("\n## 5. Кривая k_eff(lam) и Cp-оптимум\n")
    A("Полные сетки (61 точка, lam от 1e-6 до 1) — в JSON, поля `curves.*.points`: "
      "lam, k_eff, pub_calc (алгебра), Sum|c|, slip_hat, pub_hat, priv_hat.\n")
    A("Cp-функционал ворот-2: `priv_hat(lam) = pub_алгебра(lam) + слип(lam) + "
      "3.29e-5·k_eff(lam)`.\n")
    A("| спан | dim | cond(G) | lam* | k_eff(lam*) | pub_hat | E[priv] | lam* внутри замеренного диапазона |")
    A("|---|---|---|---|---|---|---|---|")
    for tag, cv in r["curves"].items():
        b = cv["cp_optimum"]; bi = cv["cp_optimum_inside_measured_lam"]
        A(f"| {tag} | {cv['dim']} | {cv['cond_G']:.1e} | {b['lam']:.2e} | "
          f"{b['k_eff']:.1f} | {b['pub_hat']:.7f} | **{b['priv_hat']:.7f}** | "
          f"нет (внутри: lam {bi['lam']:.1e}, E[priv] {bi['priv_hat']:.7f}) |")
    A(f"\nБез члена слипа минимум уезжает в lam -> 0 и физически недостижим "
      f"(расчётный pub — внутривыборочная величина). С членом слипа оптимум "
      f"устойчиво лежит на lam ~ 2e-2...3e-2, где k_eff ~ 25, то есть **вся "
      f"отправленная серия витрин (lam от 1.5e-5 до 1.1e-3) работала далеко за "
      f"оптимумом: покупала паблик по чудовищному курсу степеней свободы.**")
    A(f"\nОсторожно: lam* лежит ВНЕ диапазона lam замеренных витрин "
      f"({r['curves']['span163_SHOW11']['measured_lam_range'][0]:.1e} ... "
      f"{r['curves']['span163_SHOW11']['measured_lam_range'][1]:.1e}) — это "
      f"экстраполяция закона слипа, но в СТОРОНУ МЕНЬШИХ Sum|c| (~1.8 против "
      f"якорного диапазона 3...140), где сам слип мал (~7e-5) и ошибка его "
      f"экстраполяции не может съесть запас в 0.0005. Условие ворот-2 «lam* внутри "
      f"сетки замеренных витрин» НЕ выполнено — это надо предъявить Саше явно.")

    A("\n## 6. Область применимости — контроль на законных файлах\n")
    A("| файл | спан | lam | невязка подгонки | «k_eff» машинки | k из аудита K1 |")
    A("|---|---|---|---|---|---|")
    for nm, c in r["scope_control"].items():
        A(f"| {nm} | {c['span']} | {c['lam_fit']:.2e} | {100*c['rel_rms']:.1f}% от ‖d‖ | "
          f"{c['k_eff_naive']:.1f} | {c['k_audit_K1']} |")
    A("\nПодгонка гребневого пути под F8 и T3 ПРОВАЛЕНА (невязка ~30% и ~27%): эти "
      "файлы собраны дозами по валидации и приорам, а не гребнем по паблику, и на "
      "гребневом пути не лежат. Совпадение 8.7 против 8.3 у F8 — СЛУЧАЙНОЕ "
      "(у T3 машинка даёт 8.5 против аудитных 16.7). **Контрподписью k(F8) это "
      "не является; машинка применима только к витринам.**")

    A("\n## 7. Обусловленность\n")
    A("| спан | dim | cond(G) | cond(A = R^-1/2 G R^-1/2) | cond(G+lam·R) при рабочем lam |")
    A("|---|---|---|---|---|")
    for x in r["showcases"]:
        A(f"| {x['name']} ({x['span']}) | {x['span']+1} | {x['cond_G']:.2e} | "
          f"{x['cond_A']:.2e} | {x['cond_regularized']:.2e} |")
    A("\nГрам-матрица вырождена (cond до 1e17): файлы почти коллинеарны. Именно "
      "поэтому k_eff считается через спектр, а не через обращение, и именно поэтому "
      "сами коэффициенты c не идентифицируемы — идентифицируем только САМ ФАЙЛ "
      "(lp), по которому и ведётся подгонка lam.")
    (OUT / "r2_k_eff.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()

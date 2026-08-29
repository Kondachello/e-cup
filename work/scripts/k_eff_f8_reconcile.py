"""R2c, часть 2 — примирение двух путей: ожидание (df) против реализации (аудит K1).

Аудит K1 меряет ФАКТИЧЕСКУЮ переоценку конкретного файла (over(T3)=0.000548 и
далее по цепочке), то есть величину, ОБУСЛОВЛЕННУЮ наблюдёнными замерами.
Витринная формула k_eff = tr[(G+lam R)^-1 G] меряет ОЖИДАНИЕ по шуму замеров.
Это разные условные распределения одной и той же величины.

Здесь считаем обе для шага F7->F8 и смотрим, объясняет ли разница цифру 8.3.

  реализация:  over_step = 1.25 * d.(cP - m_star) / F      (оценка солвера по данным)
  ожидание:    E[over_step] = 1.25 * tr(W Sig_e) / F,  W = M^-1 (1.25K - 0.25I)
  тождество:   E[d.(cP - m_star)] = tr(W Sig_e)   -- проверяется алгебраически

Плюс сценарии кумулятива: k(F8) = tr[(W + (I - W Q) B7) Sig_e]/u для разных
предположений о карте B7 = d(дозы F7)/d(c) прежней цепочки.

Запуск: OMP_NUM_THREADS=4 .venv/bin/python work/scripts/k_eff_f8_reconcile.py
"""
from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")

import numpy as np

SCR = Path("/private/tmp/claude-501/-Users-alexanderkondakov-ozon-cup/"
           "0b55ab9f-3777-4ebc-bd91-937895c0e355/scratchpad")
OUT = Path("/Users/alexanderkondakov/ozon-cup/work/reports/rank")
PUB_F8, PUB_F7 = 1.6458057389, 1.6458557351
K_AUDIT_F8, K_AUDIT_T3 = 8.3, 16.7


def main() -> None:
    z = np.load(SCR / "k1b_matrices.npz", allow_pickle=False)
    keep = z["keep"]
    names = [str(s) for s in z["keep_names"]]
    dim = len(names)
    Q, Sig_e, Sig_p, K = z["Qk"], z["Sig_e_k"], z["Sig_p_k"], z["Kk"]
    Lam, ridge, gamma = z["Lam_k"], z["ridge_k"], float(z["gamma"])
    cP, mu_c, d_fin = z["cP_k"], z["mu_c"][keep], z["d_fin"]
    F = float(z["F0_state"])
    u = float(z["FPC2"]) / int(z["N_PUB"]) * F ** 2
    unit = float(z["FPC2"]) * F / int(z["N_PUB"])
    I = np.eye(dim)
    M = Q + Lam + gamma * np.diag(np.diag(Q)) + ridge
    W = np.linalg.solve(M, 1.25 * K - 0.25 * I)

    m_star = mu_c + K @ (cP - mu_c)
    print("=" * 96)
    print("R2c ч.2: ожидание против реализации на шаге F7 -> F8")
    print("=" * 96)

    real_dot = float(d_fin @ (cP - m_star))
    over_real = 1.25 * real_dot / F
    # сверка тождества с самим солвером: gain - (pub_alg - pub_F7)
    cQ = 1.25 * m_star - 0.25 * cP
    gain = float((2 * d_fin @ cQ - d_fin @ Q @ d_fin) / (2 * 1.646))
    pub_alg = float(np.sqrt(float(z["S_F7"]) ** 2 - 2 * d_fin @ cP + d_fin @ Q @ d_fin))
    dpub = pub_alg - float(z["S_F7"])
    print(f"  gain (E[priv F8]-F7, солвер) = {-gain:+.6f};  Δpub (алгебра) = {dpub:+.6f}")
    print(f"  Δ(priv - pub) реализованный = {-gain - dpub:+.6f}"
          f"   (через d.(cP-m*): {-over_real:+.6f})")
    print(f"  в единицах k:  Δk_шага(реализация) = {over_real/(1.25*unit):+.2f}"
          f"   (знак -: шаг ВОЗВРАЩАЕТ переоценку)")

    # --- ожидание
    exp_dot = float(np.trace(W @ Sig_e))
    over_exp = 1.25 * exp_dot / F
    S_tot = Sig_p + Sig_e
    A = 0.5 * (W.T @ (I - K) + (I - K).T @ W)
    # тождество: E[d.(cP-m*)] = tr(W' (I-K) S_tot) = tr(W Sig_e)
    idn = float(np.trace(W.T @ (I - K) @ S_tot))
    # d = W x + M^-1 mu_c,  x = cP - mu_c ~ N(0, S_tot):
    #   d.(cP-m*) = x'W'(I-K)x + b.x,  b = (I-K)' M^-1 mu_c
    b = (I - K).T @ np.linalg.solve(M, mu_c)
    var_q = 2 * float(np.trace((A @ S_tot) @ (A @ S_tot)))
    var_l = float(b @ S_tot @ b)
    sd_dot = float(np.sqrt(max(var_q + var_l, 0)))
    print(f"\n  E[d.(cP-m*)] = tr(W Sig_e) = {exp_dot:.3e}"
          f"   (сверка через S_tot: {idn:.3e}, расх. {abs(idn-exp_dot):.1e})")
    print(f"  sd: квадратичная часть {np.sqrt(var_q)/u:.2f} + линейная "
          f"{np.sqrt(var_l)/u:.2f} -> итого {sd_dot/u:.2f} (в k)")
    print(f"  Δk_шага(ожидание) = {exp_dot/u:+.2f} +- {sd_dot/u:.2f}")
    zsc = (real_dot - exp_dot) / sd_dot if sd_dot > 0 else float("nan")
    print(f"  реализация {real_dot/u:+.2f} против ожидания {exp_dot/u:+.2f} -> z = {zsc:+.2f}")

    # --- калибровка приорно-шумовой модели: махаланобис (cP - mu_c) по S_tot
    x = cP - mu_c
    chi2 = float(x @ np.linalg.solve(S_tot, x))
    chi2_e = float(x @ np.linalg.solve(Sig_e, x))
    print(f"\n  калибровка модели: (cP-mu)' S_tot^-1 (cP-mu) = {chi2:.1f} при dim = {dim}"
          f"  (ожидание {dim}, sd {np.sqrt(2*dim):.1f}) -> z = "
          f"{(chi2-dim)/np.sqrt(2*dim):+.2f}")
    print(f"  тот же махаланобис только по шуму: {chi2_e:.1f} "
          f"(во столько раз данные шире чистого шума: {chi2_e/dim:.1f}x)")

    # --- поосный разбор реализации
    per = d_fin * (cP - m_star) / u
    print(f"\n  поосный вклад в реализованный k шага (сумма {per.sum():+.2f}):")
    for i in np.argsort(-np.abs(per))[:10]:
        print(f"    {names[i]:6s} w={float(z['w_vec'][keep][i]):.3f} "
              f"Δдоза {d_fin[i]:+.4f}  вклад {per[i]:+6.2f}")

    # --- перекалибровка приора: во сколько раз надо расширить Sig_p, чтобы модель сошлась
    print("\n--- перекалибровка приора (chi2 -> dim) и её влияние на k ---")
    def chi2_f(f_):
        return float(x @ np.linalg.solve(f_ * Sig_p + Sig_e, x))
    lo, hi = 1.0, 200.0
    for _ in range(80):
        mid = np.sqrt(lo * hi)
        if chi2_f(mid) > dim:
            lo = mid
        else:
            hi = mid
    f_cal = float(np.sqrt(lo * hi))
    print(f"  множитель на Sig_p: f = {f_cal:.2f} (tau в {np.sqrt(f_cal):.2f} раза шире),"
          f" chi2 = {chi2_f(f_cal):.1f}")
    recal = {}
    for tag, f_ in (("как в солвере", 1.0), ("перекалиброванный", f_cal)):
        Sp = f_ * Sig_p
        Kc = Sp @ np.linalg.inv(Sp + Sig_e)
        wc = np.diag(Kc)
        Lc = np.diag(np.diag(Q) * 0)  # Lam = diag(q (1-w) tau^2) пересобираем поосно
        qv, tv = z["q_vec"][keep], z["tau_vec"][keep] * np.sqrt(f_)
        Lc = np.diag(qv * (1 - wc) * tv ** 2)
        Mc = Q + Lc + gamma * np.diag(np.diag(Q)) + ridge
        Wc = np.linalg.solve(Mc, 1.25 * Kc - 0.25 * I)
        kexp = float(np.trace(Wc @ Sig_e)) / u
        mc = mu_c + Kc @ (cP - mu_c)
        dc = Wc @ cP + np.linalg.solve(Mc, 1.25 * (I - Kc) @ mu_c)
        kreal = float(dc @ (cP - mc)) / u
        kfix = float(np.trace(np.linalg.solve(Q, 1.25 * Kc - 0.25 * I) @ Sig_e)) / u
        recal[tag] = dict(f=f_, k_expected_step=kexp, k_realized_step=kreal,
                          k_fixpoint=kfix, trK=float(np.trace(Kc)))
        print(f"  {tag:18s}: tr K = {np.trace(Kc):5.1f}  k_шага(ожид) {kexp:6.2f}  "
              f"k_шага(реализ) {kreal:+6.2f}  k_неподв.точки {kfix:6.2f}")

    # --- сценарии кумулятива
    print("\n--- кумулятивный k(F8) = tr[(W + (I - W Q) B7) Sig_e]/u при разных B7 ---")
    WQ = W @ Q
    evWQ = np.linalg.eigvals(WQ).real
    print(f"  спектр W Q: [{evWQ.min():+.3f}; {evWQ.max():+.3f}], след {evWQ.sum():.2f}")
    scen = {}
    Qi = np.linalg.inv(Q)
    for tag, B7, note in (
            ("B7 = 0 (F7 не подогнан по паблику)", np.zeros((dim, dim)), "нижняя грань"),
            ("B7 = W (F7 -- один такой же шаг)", W, "цепочка из 2 шагов"),
            ("B7 = Q^-1(1.25K-0.25I) (сошлась)", Qi @ (1.25 * K - 0.25 * I),
             "неподвижная точка"),
            ("B7 = Q^-1 (чистая подгонка паблика)", Qi, "верхняя грань")):
        Btot = W + (I - WQ) @ B7
        k = float(np.trace(Btot @ Sig_e)) / u
        scen[tag] = k
        print(f"  {tag:42s} k = {k:6.2f}   [{note}]")

    # --- что означает 8.3 в терминах цепочки
    print("\n--- обратный ход: какой k(F7) нужен, чтобы получить аудитные 8.3 ---")
    base = float(np.trace(W @ Sig_e)) / u
    print(f"  k(F8) = k_шага + вклад(F7);  k_шага(ожидание) = {base:.2f}")
    print(f"  => вклад F7 должен быть {K_AUDIT_F8 - base:+.2f} (отрицательный!)")
    print(f"  при реализации шага ({real_dot/u:+.2f}) вклад F7 = "
          f"{K_AUDIT_F8 - real_dot/u:+.2f} -- совместимо с k(T3)={K_AUDIT_T3}")

    out = dict(
        realized_over_step=float(over_real), realized_k_step=float(real_dot / u),
        expected_over_step=float(over_exp), expected_k_step=float(exp_dot / u),
        sd_k_step=float(sd_dot / u), z_realized_vs_expected=float(zsc),
        gain_solver=float(-gain), dpub_algebra=float(dpub),
        chain_scenarios={k: float(v) for k, v in scen.items()},
        eig_WQ=dict(min=float(evWQ.min()), max=float(evWQ.max()), trace=float(evWQ.sum())),
        implied_k_F7_for_audit_expected=float(K_AUDIT_F8 - base),
        implied_k_F7_for_audit_realized=float(K_AUDIT_F8 - real_dot / u),
    )
    (OUT / "r2c_reconcile.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(f"\njson: {OUT/'r2c_reconcile.json'}")


if __name__ == "__main__":
    main()

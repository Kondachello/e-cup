"""R2c — контрподпись k(F8) вторым, независимым путём: через матрицы солвера K1b.

Аудит K1 дал k(F8)=8.3 из ЗАМЕРОВ (over(T3)=0.000548). Здесь k считается из
Q (грам 46 осей), Sig_e (ковариация шума публичных замеров в c-векторе),
Sig_p (приорная ковариация), K = Sig_p(Sig_p+Sig_e)^-1 (калмановский гейн),
V_star = (I-K)Sig_p, Lam = diag(q(1-w)tau^2), гребень gamma*diag(Q).

Единая валюта штаба выводится, а не постулируется:
  шаг F7->F8 в квадрате скора: pub^2 меняется на -2 d.cP + d'Q d,
  priv^2 -- на -2 d.(1.25 c_full - 0.25 cP) + d'Q d  (публика = 20% окна),
  => (priv - pub) = 2.5 d.e / (2F), e = cP - c_full, Cov(e) = Sig_e.
  Для d = W cP:  E[priv-pub] = 1.25 * tr(W Sig_e)/F.
  Штаб: priv - pub = 1.25 * k * 2.63e-5.  Значит
      k = tr(W Sig_e) / u,   u = (0.8/50000)*F^2,   и 2.63e-5 = 0.8*F/50000.
  Одна свободно подогнанная по паблику ось (W=Q^-1, Sig_e=uQ) даёт ровно k=1.

Маршруты:
  A  буквальная формула витрин:      k = tr[(Q + Lam + gamma diag Q)^-1 Q]
  A' то же, но с фактическим Sig_e:  k = tr[M^-1 Sig_e]/u
  B  ШАГ F7->F8 честно (с байесом):  k = tr[M^-1 (1.25K - 0.25I) Sig_e]/u
  C  через V_star (маршрут Жени):    k = dim - 1.25*tr[V_star Sig_p^-1] = 1.25 tr K - 0.25 dim
  D  поосная (диагональная) сверка C: k = sum(1.25 w_i - 0.25)
  E  неподвижная точка цепочки шагов: k = tr[Q^-1 (1.25K - 0.25I) Sig_e]/u
     (гребень и Lam из неподвижной точки ВЫПАДАЮТ: они замедляют сходимость,
      но не сдвигают оптимум -- это и есть кумулятивный k файла)

Запуск: OMP_NUM_THREADS=4 .venv/bin/python work/scripts/k_eff_f8_countersign.py
Матрицы: скретчпад/k1b_matrices.npz (дамп инструментированной копии солвера).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")

import numpy as np

SCR = Path("/private/tmp/claude-501/-Users-alexanderkondakov-ozon-cup/"
           "0b55ab9f-3777-4ebc-bd91-937895c0e355/scratchpad")
ROOT = Path("/Users/alexanderkondakov/ozon-cup")
OUT = ROOT / "work" / "reports" / "rank"

PUB_F8 = 1.6458057389
PUB_T3 = 1.6469321993
K_AUDIT_F8 = 8.3


def sym(A):
    return 0.5 * (A + A.T)


def tr_solve(M, X):
    """tr(M^-1 X) устойчиво: через симметричное решение."""
    return float(np.trace(np.linalg.solve(M, X)))


def spd_tr_inv(A, X):
    """tr(A^-1 X) для симметричной A через собственный базис (A может быть плохо обусловлена)."""
    w, U = np.linalg.eigh(sym(A))
    Xt = U.T @ X @ U
    return float(np.sum(np.diag(Xt) / w)), float(w.max() / w.min())


def main() -> None:
    z = np.load(SCR / "k1b_matrices.npz", allow_pickle=False)
    names = [str(s) for s in z["keep_names"]]
    keep = z["keep"]
    dim = len(names)
    Q, Sig_e, Sig_p = z["Qk"], z["Sig_e_k"], z["Sig_p_k"]
    K, V = z["Kk"], z["V_k"]
    Lam, ridge = z["Lam_k"], z["ridge_k"]
    gamma = float(z["gamma"])
    q_vec, w_vec, tau_vec = z["q_vec"][keep], z["w_vec"][keep], z["tau_vec"][keep]
    fam = [str(s) for s in z["fam"]]
    fam = [fam[i] for i in keep]
    F = float(z["F0_state"])          # S(F7) -- база пересчёта
    N_PUB, FPC2 = int(z["N_PUB"]), float(z["FPC2"])
    d_fin = z["d_fin"]

    u = FPC2 / N_PUB * F ** 2         # единица: одна свободная ось = tr = u
    unit_fake = FPC2 * F / N_PUB      # «2.63e-5» штаба, выведенное
    print("=" * 96)
    print("R2c: контрподпись k(F8) через матрицы солвера K1b")
    print("=" * 96)
    print(f"осей (без mdl_wulfen): {dim};  F(F7) = {F:.7f};  n_pub = {N_PUB}, FPC = {FPC2}")
    print(f"единица «фейка» из алгебры: 0.8*F/n_pub = {unit_fake:.4e}  "
          f"(штаб: 2.63e-5);  1.25*ед = {1.25*unit_fake:.4e} (штаб 3.29e-5)")

    Rg = gamma * np.diag(np.diag(Q)) + ridge
    M = Q + Lam + Rg
    I = np.eye(dim)
    Wmap = np.linalg.solve(M, 1.25 * K - 0.25 * I)      # d = Wmap @ cP + приорная часть
    # контроль: воспроизводим d_fin из дампа
    d_chk = Wmap @ z["cP_k"] + np.linalg.solve(M, 1.25 * (I - K) @ z["mu_c"][keep])
    print(f"контроль карты: max|W cP + приор - d_fin| = {np.abs(d_chk - d_fin).max():.2e}")

    # --- насколько Sig_e пропорциональна Q (допущение витринной формулы)
    ev = np.linalg.eigvals(np.linalg.solve(Q, Sig_e) / u).real
    print(f"\nSig_e / (u*Q): собств. числа  медиана {np.median(ev):.3f}  "
          f"[{ev.min():.3f}; {ev.max():.3f}]  (=1 <=> Sig_e = u*Q)")
    print(f"  tr(Sig_e)/(u*tr Q) = {np.trace(Sig_e)/(u*np.trace(Q)):.3f}")

    res = {}

    # ---------------- A: буквальная витринная формула на осях F8
    kA = tr_solve(M, Q)
    kA_noLam = tr_solve(Q + Rg, Q)
    kA_free = tr_solve(Q + Lam + ridge, Q)
    res["A_ridge_df"] = kA
    print("\n--- A. Буквально формула витрин  k = tr[(Q + Lam + g*diagQ)^-1 Q] ---")
    print(f"  полный регуляризатор (Lam + gamma={gamma}):   k = {kA:.2f}")
    print(f"  только гребень gamma*diag(Q) (без приора):   k = {kA_noLam:.2f}")
    print(f"  только приорный демпфер Lam (gamma=0):       k = {kA_free:.2f}")
    print("  (минус-1 витринной формулы здесь НЕ нужен: в спане F8 нет строки-единицы,")
    print("   уровень выставляется отдельно и по паблику не подгоняется)")

    # ---------------- A': тот же след, но с фактическим Sig_e вместо u*Q
    kAp = tr_solve(M, Sig_e) / u
    res["Ap_ridge_df_SigE"] = kAp
    print(f"\n--- A'. То же, но перенос шума честный: k = tr[M^-1 Sig_e]/u = {kAp:.2f}")

    # ---------------- B: шаг F7->F8 целиком (с байесовым сжатием и членом -0.25 cP)
    kB = float(np.trace(Wmap @ Sig_e)) / u
    res["B_step_full"] = kB
    print(f"\n--- B. ШАГ F7->F8 честно: k = tr[M^-1 (1.25K - 0.25I) Sig_e]/u = {kB:.2f}")
    kB_noneg = float(np.trace(np.linalg.solve(M, 1.25 * K) @ Sig_e)) / u
    print(f"     без «кредита» -0.25*cP (только сжатие K):            {kB_noneg:.2f}")

    # ---------------- C: маршрут через V_star
    trK = float(np.trace(K))
    trV = float(np.trace(V @ np.linalg.inv(Sig_p)))
    kC = 1.25 * trK - 0.25 * dim
    res["C_via_Vstar"] = kC
    print("\n--- C. Через V_star (маршрут ворот-4) ---")
    print(f"  tr K = {trK:.2f} из {dim}  (эфф. число осей, где паблик пересилил приор)")
    print(f"  tr[V_star Sig_p^-1] = {trV:.2f};  dim - tr = {dim - trV:.2f} (= tr K, тождество)")
    print(f"  k = 1.25*tr K - 0.25*dim = dim - 1.25*tr[V_star Sig_p^-1] = {kC:.2f}")

    # ---------------- D: поосная сверка
    kD = float(np.sum(1.25 * w_vec - 0.25))
    res["D_diag_w"] = kD
    print(f"\n--- D. Поосная сверка: sum(1.25 w_i - 0.25) = {kD:.2f}  (sum w = {w_vec.sum():.2f})")

    # ---------------- E: неподвижная точка цепочки одинаковых шагов
    kE_raw, condQ = spd_tr_inv(Q, (1.25 * K - 0.25 * I) @ Sig_e)
    kE = kE_raw / u
    res["E_chain_fixpoint"] = kE
    print(f"\n--- E. Кумулятив: неподвижная точка повторяемого шага ---")
    print(f"  k = tr[Q^-1 (1.25K - 0.25I) Sig_e]/u = {kE:.2f}   (cond Q = {condQ:.1e})")
    print("  гребень и Lam в неподвижную точку НЕ входят: они замедляют сходимость,")
    print("  но не двигают оптимум -- поэтому кумулятив цепочки много больше шага.")

    # ---------------- зависимость шага от gamma
    print("\n--- k шага по gamma (по маршруту B) ---")
    gpath = {}
    for g in (0.0, 0.03, 0.1, 0.3, 1.0):
        Mg = Q + Lam + g * np.diag(np.diag(Q)) + ridge
        kg = float(np.trace(np.linalg.solve(Mg, 1.25 * K - 0.25 * I) @ Sig_e)) / u
        kAg = tr_solve(Mg, Q)
        gpath[str(g)] = dict(k_B=kg, k_A=kAg)
        print(f"  gamma={g:4.2f}:  B (шаг) {kg:6.2f}   A (сырой df) {kAg:6.2f}")

    # ---------------- вклад по семьям (маршрут C/D)
    print("\n--- вклад в k (поосно, 1.25w-0.25) по семьям ---")
    byfam = {}
    for f_ in ("model", "segment", "decomp"):
        idx = [i for i in range(dim) if fam[i] == f_]
        c = float(np.sum(1.25 * w_vec[idx] - 0.25))
        byfam[f_] = dict(n=len(idx), k=c, mean_w=float(w_vec[idx].mean()))
        print(f"  {f_:8s} осей {len(idx):2d}  ср. w {w_vec[idx].mean():.3f}  вклад {c:+6.2f}")
    print("  топ-8 осей по вкладу:")
    contr = 1.25 * w_vec - 0.25
    for i in np.argsort(-contr)[:8]:
        print(f"    {names[i]:6s} [{fam[i]:7s}] w={w_vec[i]:.3f} вклад {contr[i]:+.3f}")
    print("  антивклад (приор пересилил паблик):")
    for i in np.argsort(contr)[:5]:
        print(f"    {names[i]:6s} [{fam[i]:7s}] w={w_vec[i]:.3f} вклад {contr[i]:+.3f}")

    # ---------------- перевод в E[priv]
    print("\n" + "=" * 96)
    print("ПЕРЕВОД В E[priv F8]  (priv = pub + 3.29e-5*k, pub = %.7f)" % PUB_F8)
    table = {}
    for tag, kk, what in (("K1 (аудит, замеры)", K_AUDIT_F8, "весь файл"),
                          ("B  шаг F7->F8", kB, "только последний шаг"),
                          ("C  V_star", kC, "цепочка одинаковых шагов"),
                          ("D  поосная", kD, "цепочка одинаковых шагов"),
                          ("E  неподв. точка", kE, "цепочка одинаковых шагов"),
                          ("A  сырой df", kA, "верхняя оценка (без байеса)")):
        pr = PUB_F8 + 1.25 * unit_fake * kk
        table[tag] = dict(k=kk, priv=pr, scope=what)
        print(f"  {tag:22s} k = {kk:6.2f}  ->  E[priv] = {pr:.6f}   [{what}]")
    kstar = (table["C  V_star"]["priv"] - 1.6440063524) / (1.25 * unit_fake)
    print(f"\n  порог безубытка SHOW11 при k(F8)=k_C: k* = {kstar:.1f} "
          f"(при k(F8)=8.3 было 63.0)")

    out = dict(
        dim=dim, gamma=gamma, F_base=F, unit_fake=unit_fake, u=u,
        k_routes=dict(A_ridge_df=kA, A_ridge_df_noLam=kA_noLam, A_ridge_df_gamma0=kA_free,
                      Ap_ridge_df_SigE=kAp, B_step_full=kB, B_step_noCredit=kB_noneg,
                      C_via_Vstar=kC, D_diag_w=kD, E_chain_fixpoint=kE),
        trK=trK, tr_Vstar_Sigp_inv=trV, sum_w=float(w_vec.sum()),
        SigE_over_uQ=dict(median=float(np.median(ev)), min=float(ev.min()),
                          max=float(ev.max()),
                          trace_ratio=float(np.trace(Sig_e) / (u * np.trace(Q)))),
        gamma_path=gpath, by_family=byfam,
        per_axis={names[i]: dict(w=float(w_vec[i]), fam=fam[i], q=float(q_vec[i]),
                                 tau=float(tau_vec[i]), contrib=float(contr[i]),
                                 d_fin=float(d_fin[i])) for i in range(dim)},
        priv_table={k: v for k, v in table.items()},
        k_star_SHOW11_at_kC=float(kstar),
        pub_F8=PUB_F8, k_audit_K1=K_AUDIT_F8,
    )
    (OUT / "r2c_countersign_f8.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1))
    print(f"\njson: {OUT/'r2c_countersign_f8.json'}")


if __name__ == "__main__":
    main()

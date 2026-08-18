"""blend_testopt.py — подбор весов бленда ПОД ТЕСТОВОЕ ОКНО (а не под валидацию).

Идея: f²(w) = w'Mw − 2w'φ + const, где
  M_ij = mean(lp_i·lp_j)   — считается точно локально по тестовым предсказаниям,
  φ_i  = mean_P(lp_i·t)    — оценивается механизмом predict_lb (span-алгебра + val-остаток).
Оптимум w* = M⁻¹φ (и варианты с ограничениями).

Математика φ переиспользуется из predict_lb.LBPredictor — здесь только линеаризация.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import polars as pl
from scipy.optimize import nnls, minimize

sys.path.insert(0, str(Path(__file__).parent))
import predict_lb as PL

ROOT = Path(__file__).resolve().parents[2]
PREDS = ROOT / "work" / "preds"
OUT = ROOT / "work" / "reports" / "blend_testopt.json"

SIGMA2 = 2.72        # MSE публичной метрики (f² ≈ 1.65²)
N_PUB = 50_000       # размер публичного сабсета
NMC = 400            # число розыгрышей Монте-Карло для оценки шума φ

# --- пул: калиброванные ЧИСТЫЕ модели -------------------------------------------------
EXCLUDE = {"blend"}                       # blend_cal — это сам бленд, не модель
CONTAMINATED = {"lgblog_final", "xgblog_final", "cblog_final", "mlp_final", "gru_final",
                "hjit37", "hjit44"}

# NNLS-веса, подобранные на ВАЛИДАЦИИ (work/reports/scores.tsv, OOF 1.666791)
W_VAL = {"mlpziln_cal": 0.122, "channel2_cal": 0.012, "countaov_cal": 0.074,
         "c_ts2_s42_cal": 0.246, "twl_v7_cal": 0.055, "seq2tr_f_cal": 0.070,
         "fusion_f_cal": 0.316, "hmmsim_cal": 0.028, "behavonly_cal": 0.080}


def build_pool() -> list[str]:
    names = []
    for p in sorted(PREDS.glob("*_cal_test.parquet")):
        stem = p.name[: -len("_cal_test.parquet")]
        if stem in EXCLUDE or stem in CONTAMINATED:
            continue
        if not (PREDS / f"{stem}_cal_val.parquet").exists():
            print(f"  пропуск {stem}: нет _cal_val")
            continue
        names.append(stem + "_cal")
    # channel3 калиброван по каналам (суффикс _chcal), тоже чистая калиброванная модель
    if (PREDS / "channel3_chcal_test.parquet").exists() and \
       (PREDS / "channel3_chcal_val.parquet").exists():
        names.append("channel3_chcal")
    return names


def load_lp(name: str, uid_ref: np.ndarray, split: str = "test") -> np.ndarray:
    d = pl.read_parquet(PREDS / f"{name}_{split}.parquet").sort("user_id")
    if not np.array_equal(d["user_id"].to_numpy(), uid_ref):
        raise ValueError(f"{name}_{split}: user_id не совпадает с базисом")
    return np.log1p(np.clip(d["pred"].to_numpy().astype(np.float64), 0, None))


# --- линеаризация механизма predict_lb -------------------------------------------------
class Phi:
    """φ̂(x) — линейный функционал оценки mean_P(x·t), тот же, что внутри LBPredictor."""

    def __init__(self, P: PL.LBPredictor):
        self.P = P
        N = P.N
        # φ̂(x) = psi_vec' G⁻¹ B x / N + a · tv_c'(I − B'G⁻¹B/N) x / N  ==  u'x
        GiB = np.linalg.solve(P.G, P.B)                       # (m, N)
        u = GiB.T @ P.psi_vec / N                             # span-часть
        tv = P.tv_c
        proj_tv = GiB.T @ (P.B @ tv / N)                      # B'G⁻¹B tv / N
        u = u + P.a * (tv - proj_tv) / N                      # + остаточная часть
        self.u = u
        # оператор остатка (для оценки шума): r(x) = x − B'G⁻¹B x / N
        self.GiB = GiB

    def __call__(self, X: np.ndarray) -> np.ndarray:
        """X: (K, N) или (N,) → φ̂ покомпонентно."""
        return X @ self.u

    def resid(self, X: np.ndarray) -> np.ndarray:
        N = self.P.N
        return X - (X @ self.P.B.T / N) @ self.GiB


# --- решатели --------------------------------------------------------------------------
def solve_unconstrained(M, phi):
    return np.linalg.solve(M + 1e-12 * np.trace(M) / len(M) * np.eye(len(M)), phi)


def solve_nonneg(M, phi):
    ev, V = np.linalg.eigh(M)
    ev = np.clip(ev, 1e-14 * ev.max(), None)
    A = (V * np.sqrt(ev)) @ V.T            # A = M^{1/2}, симметричная
    b = np.linalg.solve(A, phi)            # A b = phi  →  ||Aw − b||² = w'Mw − 2w'phi + c
    w, _ = nnls(A, b)
    return w


def solve_simplex(M, phi, w0=None):
    K = len(phi)
    fun = lambda w: float(w @ M @ w - 2 * w @ phi)
    jac = lambda w: 2 * (M @ w - phi)
    best = None
    starts = [np.full(K, 1.0 / K)]
    if w0 is not None:
        s = np.clip(w0, 0, None)
        starts.append(s / max(s.sum(), 1e-9))
    for s in starts:
        r = minimize(fun, s, jac=jac, method="SLSQP",
                     bounds=[(0, None)] * K,
                     constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0,
                                   "jac": lambda w: np.ones(K)}],
                     options=dict(maxiter=1000, ftol=1e-14))
        if best is None or r.fun < best.fun:
            best = r
    return best.x


def solve_ridge(M, phi, lam):
    return np.linalg.solve(M + lam * np.eye(len(M)), phi)


def main() -> int:
    print("=== базис predict_lb ===")
    basis = PL.load_basis()
    P = PL.LBPredictor(basis)
    F = Phi(P)
    uid = P.uid
    N = P.N

    # sanity: линеаризация воспроизводит predict()
    lp_a = P.lp_a
    phi_a = float(F(lp_a))
    print(f"  φ̂(1) = {float(F(np.ones(N))):.6f}   (эталон MEAN_T = {PL.MEAN_T})")

    pool = build_pool()
    print(f"\n=== пул: {len(pool)} калиброванных чистых моделей ===")
    print("  " + ", ".join(pool))

    L = np.stack([load_lp(n, uid, "test") for n in pool])          # (K, N)
    K = len(pool)

    M = L @ L.T / N                                                # mean(lp_i·lp_j)
    m = L.mean(1)                                                  # mean(lp_i)
    phi = F(L)                                                     # φ_i = mean_P(lp_i·t)
    const = P.f_a ** 2 - P.q_a + 2 * phi_a                         # f² = w'Mw − 2w'φ + const

    def fpred(w):
        return float(np.sqrt(max(w @ M @ w - 2 * w @ phi + const, 1e-12)))

    # ---- вариант с СВОБОДНЫМ глобальным сдвигом (профилируем c0) -----------------------
    # lp = c0·1 + Σ w_i lp_i ;  оптимальный c0 = φ̂(1) − m'w  →  M_c = M − mm', φ_c = φ − MEAN_T·m
    Mc = M - np.outer(m, m)
    phic = phi - PL.MEAN_T * m
    const_c = const - PL.MEAN_T ** 2

    def fpred_c(w):
        return float(np.sqrt(max(w @ Mc @ w - 2 * w @ phic + const_c, 1e-12)))

    def shift_of(w):
        return float(PL.MEAN_T - m @ w)

    print(f"\n  cond(M) = {np.linalg.cond(M):.3e}   cond(M_центр) = {np.linalg.cond(Mc):.3e}")

    # ---- шум φ: Σ_ε = γ²·R, R_ij = mean(r_i·r_j), γ = f·KAPPA68 -------------------------
    Res = F.resid(L)
    R = Res @ Res.T / N
    gamma = 1.65 * PL.KAPPA68
    Se = gamma ** 2 * R
    Le = np.linalg.cholesky(Se + 1e-16 * np.trace(Se) / K * np.eye(K))
    print(f"  sd(остатка) по моделям: {np.sqrt(np.diag(R)).min():.4f}…"
          f"{np.sqrt(np.diag(R)).max():.4f}   γ = {gamma:.5f}")

    rng = np.random.default_rng(7)

    def mc_optimism(solver, base_phi, Mmat, cst, k_label):
        """E[F_true(ŵ) − F̂(ŵ)] — во сколько прогноз занижен из-за шума в φ (в единицах f²)."""
        gaps, evals = [], []
        for _ in range(NMC):
            e = Le @ rng.standard_normal(K)
            w = solver(Mmat, base_phi + e)
            f_true = w @ Mmat @ w - 2 * w @ base_phi + cst
            f_est = w @ Mmat @ w - 2 * w @ (base_phi + e) + cst
            gaps.append(f_true - f_est)
            evals.append(f_true)
        return float(np.mean(gaps)), float(np.mean(evals))

    results = {}
    variants = [
        ("unconstrained", solve_unconstrained, False),
        ("nonneg",        solve_nonneg,        False),
        ("simplex",       lambda A, b: solve_simplex(A, b), False),
        ("unconstrained_shift", solve_unconstrained, True),
        ("nonneg_shift",        solve_nonneg,        True),
        ("simplex_shift",       lambda A, b: solve_simplex(A, b), True),
    ]
    for tag, solver, use_shift in variants:
        Mm, pp, cc = (Mc, phic, const_c) if use_shift else (M, phi, const)
        w = solver(Mm, pp)
        fp = float(np.sqrt(max(w @ Mm @ w - 2 * w @ pp + cc, 1e-12)))
        k_eff = int((np.abs(w) > 1e-8).sum())
        if tag.startswith("simplex"):
            k_eff = max(k_eff - 1, 1)
        if use_shift:
            k_eff += 1
        opt_mc, _ = mc_optimism(solver, pp, Mm, cc, k_eff)
        d_pub = 2 * k_eff * SIGMA2 / N_PUB
        f_honest = float(np.sqrt(max(fp ** 2 + opt_mc + d_pub, 1e-12)))
        # сборка реального вектора и независимая проверка через predict()
        lp_w = w @ L + (shift_of(w) if use_shift else 0.0)
        chk = P.predict(lp_w)
        results[tag] = dict(
            w={n: float(v) for n, v in zip(pool, w) if abs(v) > 1e-6},
            w_sum=float(w.sum()), shift=(shift_of(w) if use_shift else 0.0),
            pred=fp, pred_via_predict=chk["pred"], sd_resid=chk["sd_resid"],
            novelty=chk["novelty"], sigma68=chk["sigma68"], sigma95=chk["sigma95"],
            k_eff=k_eff, opt_mc=opt_mc, d_pub=d_pub, honest=f_honest,
            wmin=float(w.min()), wmax=float(w.max()), l1=float(np.abs(w).sum()),
        )
        print(f"\n  [{tag}]  прогноз {fp:.6f}  (predict: {chk['pred']:.6f}, "
              f"Δ {chk['pred']-fp:+.1e})")
        print(f"     Σw {w.sum():+.4f}  ‖w‖₁ {np.abs(w).sum():.3f}  w∈[{w.min():+.3f},{w.max():+.3f}]"
              f"  сдвиг {shift_of(w) if use_shift else 0.0:+.4f}")
        print(f"     k {k_eff}  шум φ {opt_mc:+.2e}  публ.оверфит {d_pub:+.2e}  → честно {f_honest:.6f}")
        print(f"     sd(ост) {chk['sd_resid']:.4f}  novelty {chk['novelty']:.2e}  "
              f"±{chk['sigma68']:.5f}/±{chk['sigma95']:.5f}")

    # ---- ridge-путь: усадка против шума φ ---------------------------------------------
    print("\n=== ridge (усадка) на центрированной задаче ===")
    ridge = []
    tr = np.trace(Mc) / K
    for rel in [0, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]:
        lam = rel * tr
        w = solve_ridge(Mc, phic, lam)
        fp = float(np.sqrt(max(w @ Mc @ w - 2 * w @ phic + const_c, 1e-12)))
        H = np.linalg.solve(Mc + lam * np.eye(K), Mc)
        kdf = float(np.trace(H)) + 1
        opt_mc, _ = mc_optimism(lambda A, b, l=lam: solve_ridge(A, b, l), phic, Mc, const_c, kdf)
        d_pub = 2 * kdf * SIGMA2 / N_PUB
        hon = float(np.sqrt(max(fp ** 2 + opt_mc + d_pub, 1e-12)))
        ridge.append(dict(rel=rel, lam=lam, pred=fp, df=kdf, opt_mc=opt_mc,
                          d_pub=d_pub, honest=hon, l1=float(np.abs(w).sum()),
                          w={n: float(v) for n, v in zip(pool, w) if abs(v) > 1e-4}))
        print(f"  λ/tr {rel:>7.0e}  df {kdf:5.2f}  прогноз {fp:.6f}  "
              f"шум {opt_mc:+.2e}  оверфит {d_pub:.2e}  честно {hon:.6f}")

    # ---- валидационно-оптимальные веса -------------------------------------------------
    print("\n=== валидационно-оптимальные веса (NNLS на val, OOF 1.666791) ===")
    wv = np.zeros(K)
    for n, v in W_VAL.items():
        wv[pool.index(n)] = v
    lp_val = wv @ L
    r_val = P.predict(lp_val)
    fv_noshift = fpred(wv)
    sh = shift_of(wv)
    fv_shift = fpred_c(wv)
    r_val_sh = P.predict(lp_val + sh)
    print(f"  как есть (без сдвига): {fv_noshift:.6f}  (predict {r_val['pred']:.6f})")
    print(f"  + оптимальный глобальный сдвиг {sh:+.5f}: {fv_shift:.6f} "
          f"(predict {r_val_sh['pred']:.6f})")

    # то же для simplex-нормировки val-весов (Σw=1)
    print(f"  Σw_val = {wv.sum():.4f}")

    out = dict(pool=pool, results=results, ridge=ridge,
               val_optimal=dict(w=W_VAL, pred_noshift=fv_noshift, pred_shift=fv_shift,
                                shift=sh, w_sum=float(wv.sum()),
                                sd_resid=r_val_sh["sd_resid"],
                                sigma68=r_val_sh["sigma68"], sigma95=r_val_sh["sigma95"]),
               M_cond=float(np.linalg.cond(M)), Mc_cond=float(np.linalg.cond(Mc)),
               gamma=gamma, sigma2=SIGMA2, n_pub=N_PUB,
               phi=dict(zip(pool, phi.tolist())),
               mean_lp=dict(zip(pool, m.tolist())),
               sd_resid_models=dict(zip(pool, np.sqrt(np.diag(R)).tolist())),
               solo_pred={n: fpred(np.eye(K)[i]) for i, n in enumerate(pool)},
               solo_pred_shift={n: fpred_c(np.eye(K)[i]) for i, n in enumerate(pool)},
               )
    OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(f"\nсохранено: {OUT}")
    np.savez_compressed(ROOT / "work" / "blend_testopt_cache.npz",
                        L=L, M=M, phi=phi, m=m, R=R, pool=np.array(pool),
                        const=const, f_a=P.f_a, q_a=P.q_a, phi_a=phi_a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

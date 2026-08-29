"""r2d_adversarial.py — независимая адверсариальная проверка рамки priv = pub + 3.29e-5*k.

НИЧЕГО не переписывает в репо: только читает базис и печатает числа.
Разделы:
  A. арифметика переноса 1.25 = 1/(1-f) и порога k* = 63;
  B. эмпирическая sigma_u по всем осям базиса: во сколько раз Var(e*d) больше
     гомоскедастичного приближения E[e^2]*E[d^2] (это K1-шный множитель g^2);
  C. обобщённый след tr[M*Sigma]/(<e^2>*tr[M*G]) на спане SHOW11 — прямая проверка
     того, что ковариация шума линейного члена пропорциональна G (главная
     предпосылка формулы k_eff);
  D. спектральная дисперсия оптимизма (насколько E[priv] — это «точка ± сколько»);
  E. синтетика: последовательная жадная подгонка p скоррелированных направлений —
     сколько на самом деле степеней свободы (проверка пункта об аддитивности).
"""
import sys, json
from pathlib import Path
import numpy as np

ROOT = Path("/Users/alexanderkondakov/ozon-cup")
sys.path.insert(0, str(ROOT / "work" / "scripts"))
from predict_lb import ANCHOR, MEAN_T, load_basis  # noqa

F0 = 1.646
N_PUB, N_PRIV = 50_000, 200_000
N_TOT = N_PUB + N_PRIV
f = N_PUB / N_TOT

print("=" * 78)
print("A. АРИФМЕТИКА РАМКИ")
print("=" * 78)
print(f"f = n_pub/N = {f}")
print(f"1/(1-f) = {1/(1-f):.6f}   <- заявленный множитель переноса 1.25")
print(f"f/(1-f) = {f/(1-f):.6f}   <- анти-корреляция паблик/приват (K1 замерил -0.2500)")
print(f"(1-f)*F0/n_pub          = {(1-f)*F0/N_PUB:.4e}   <- заявленный fake на направление 2.63e-5")
print(f"F0/n_pub                = {F0/N_PUB:.4e}   <- 1.25*fake = 3.29e-5 (множители 1.25 и 0.8 сокращаются)")
print(f"F0/(2*n_pub)            = {F0/(2*N_PUB):.4e}   <- ТЕОРЕТИЧЕСКИЙ видимый выигрыш на паблике (без FPC)")
print(f"(1-f)*F0/(2*n_pub)      = {(1-f)*F0/(2*N_PUB):.4e}   <- он же с FPC")
print(f"замер noise_floor.py                     = 2.2000e-05   <- видимый выигрыш, замеренная sigma_u, без FPC")
print(f"отношение замер/теория (видимый выигрыш)  = {2.2e-5/(F0/(2*N_PUB)):.4f}  = g^2 (K1: g=1.15 => g^2=1.32)")
print(f"коэффициент, пересчитанный на ЗАМЕРЕННУЮ sigma_u: 2*(1-f)/(1-f)*2.2e-5 = 2*2.2e-5 = {2*2.2e-5:.4e}")

pubF8, kF8 = 1.6458057389, 8.3
pubS11 = 1.6440063524
for coef in (3.29e-5, 2*2.2e-5):
    privF8 = pubF8 + coef * kF8
    kstar = (privF8 - pubS11) / coef
    print(f"  coef {coef:.3e}: priv(F8) = {privF8:.10f}; k* = {kstar:.3f}"
          f"  (тождество k* = k_F8 + dpub/coef = {kF8 + (pubF8-pubS11)/coef:.3f})")
print(f"  dpub(F8-SHOW11) = {pubF8-pubS11:.10f}")

print()
print("=" * 78)
print("B/C/D. ЭМПИРИКА ПО БАЗИСУ")
print("=" * 78)
basis = load_basis()
names = list(basis["names"])
L = basis["L"].astype(np.float64)
tval = basis["tval"].astype(np.float64)
fscores = np.asarray(basis["f"], float)
N = L.shape[1]
a = names.index(ANCHOR)
print(f"базис {len(names)} файлов, N={N}, якорь {ANCHOR}")

e = tval - L[a]                       # остаток якоря на валидации
Ee2 = float((e ** 2).mean())
print(f"<e^2> на валидации = {Ee2:.5f}  (RMSLE_val = {np.sqrt(Ee2):.5f}; на LB F0 = {F0})")

# ---- B: g^2 по каждой оси базиса
idx_all = [i for i in range(len(names)) if i != a]
g2 = np.empty(len(idx_all)); qs = np.empty(len(idx_all))
for t, j in enumerate(idx_all):
    d = L[j] - L[a]
    q = float((d * d).mean())
    if q < 1e-14:
        g2[t] = np.nan; qs[t] = q; continue
    ed = e * d
    var_ed = float((ed * ed).mean() - ed.mean() ** 2)
    g2[t] = var_ed / (Ee2 * q)
    qs[t] = q
ok = np.isfinite(g2)
print(f"g^2 = Var(e*d)/(<e^2>*<d^2>) по {ok.sum()} осям: "
      f"медиана {np.nanmedian(g2):.3f}, среднее {np.nanmean(g2):.3f}, "
      f"квантили 10/90 {np.nanpercentile(g2[ok],10):.3f}/{np.nanpercentile(g2[ok],90):.3f}, "
      f"min {np.nanmin(g2):.3f}, max {np.nanmax(g2):.3f}")
print(f"  => g = sqrt(median) = {np.sqrt(np.nanmedian(g2)):.3f} (K1 намерил 1.15)")

# ---- C: обобщённый след на спане SHOW11
def parts_G(idx):
    ix = np.asarray(idx)
    n = len(ix) + 1
    M = L @ L.T / N
    return None

# аккуратно: строим G через грам-матрицу
Mfull = L @ L.T / N
mL = L.mean(1)

def G_of(idx):
    ix = np.asarray(idx)
    n = len(ix) + 1
    G = np.empty((n, n))
    G[0, 0] = 1.0
    G[0, 1:] = G[1:, 0] = mL[ix] - mL[a]
    G[1:, 1:] = Mfull[np.ix_(ix, ix)] - Mfull[np.ix_(ix, [a])] - Mfull[np.ix_([a], ix)] + Mfull[a, a]
    return G

def Sigma_of(idx, chunk=25000):
    """Sigma_jk = Cov_i(e_i*d_ij, e_i*d_ik), где d_0 = 1 (уровень), d_j = lp_j - lp_a."""
    ix = np.asarray(idx)
    n = len(ix) + 1
    S = np.zeros((n, n)); mu = np.zeros(n)
    for s in range(0, N, chunk):
        sl = slice(s, min(s + chunk, N))
        D = np.empty((n, sl.stop - sl.start))
        D[0] = 1.0
        D[1:] = L[ix, sl] - L[a, sl]
        X = D * e[sl]
        S += X @ X.T
        mu += X.sum(1)
    S /= N; mu /= N
    return S - np.outer(mu, mu)

SPANS = {"span163_SHOW11": (163, 2.17e-04), "span123_SHOW9": (123, 7.41e-04),
         "span123_SHOW10": (123, 3.26e-04)}
res_c = {}
for tag, (p, lam) in SPANS.items():
    idx = list(range(p))
    G = G_of(idx)
    Sg = Sigma_of(idx)
    n = len(G)
    r = np.ones(n); r[0] = 1e-4
    Sc = 1.0 / np.sqrt(r)
    A = (G * Sc).T * Sc; A = (A + A.T) / 2
    mu_e, U = np.linalg.eigh(A)
    mu_e = np.clip(mu_e, 0, None)
    w = mu_e / (mu_e + lam)
    k_eff = float(w.sum() - 1)
    # tr[M G] = sum w ; tr[M Sigma] через ту же замену переменных
    St = (Sg * Sc).T * Sc                      # R^-1/2 Sigma R^-1/2
    B = U.T @ St @ U
    tr_MS = float((np.diag(B) / (mu_e + lam)).sum())
    tr_MG = float(w.sum())
    rho = tr_MS / (Ee2 * tr_MG)
    sw2 = float((w ** 2).sum())
    res_c[tag] = dict(p=p, lam=lam, k_eff=k_eff, tr_MG=tr_MG, tr_MS_over_Ee2=tr_MS / Ee2,
                      rho=rho, sum_w2=sw2)
    print(f"{tag}: lam {lam:.2e}  k_eff {k_eff:7.2f}  tr[MG] {tr_MG:7.2f}  "
          f"tr[M*Sigma]/<e^2> {tr_MS/Ee2:7.2f}  rho {rho:6.3f}  sum w^2 {sw2:7.2f}")
    # D: дисперсия оптимизма при v_T = 0 (чистый шум): gap = 2 eps^T M eps
    #    E = 2 tr[M C], Var = 8 tr[(M C)^2], C = Cov(eps) = (1-f)/n_pub * Sigma
    lamvec = mu_e + lam
    MC = (B / lamvec[:, None]) * ((1 - f) / N_PUB)
    Egap = 2 * np.trace(MC) / (2 * F0)
    sdgap = np.sqrt(8 * float((MC * MC.T).sum())) / (2 * F0)
    print(f"    оптимизм F_T-F_P: E {Egap:.3e}  sd {sdgap:.3e}  (sd/E {sdgap/Egap:.2f}); "
          f"в приватных единицах E {Egap/(1-f):.3e} sd {sdgap/(1-f):.3e}")
    res_c[tag].update(E_gap=Egap, sd_gap=sdgap, E_priv=Egap / (1 - f), sd_priv=sdgap / (1 - f))

print()
print("=" * 78)
print("E. СИНТЕТИКА: сколько степеней свободы у ПОСЛЕДОВАТЕЛЬНОЙ подгонки")
print("=" * 78)
rng = np.random.default_rng(7)
n, p, rho_c = 50000, 20, 0.98
Z = rng.standard_normal((n, 1))
Dm = rho_c * Z + np.sqrt(1 - rho_c ** 2) * rng.standard_normal((n, p))
for mode in ("greedy1pass", "greedy3pass", "jointLS"):
    gaps = []
    for rep in range(200):
        y = rng.standard_normal(n)          # чистый шум, истинный сигнал = 0
        if mode == "jointLS":
            beta = np.linalg.lstsq(Dm, y, rcond=None)[0]
            fit = Dm @ beta
        else:
            npass = 1 if mode == "greedy1pass" else 3
            fit = np.zeros(n); r_ = y.copy()
            for _ in range(npass):
                for j in range(p):
                    s = (r_ @ Dm[:, j]) / (Dm[:, j] @ Dm[:, j])
                    fit += s * Dm[:, j]; r_ -= s * Dm[:, j]
        train = float(((y - fit) ** 2).mean())
        # «популяционный» риск того же fit при истинном y=0: E||fit||^2/n + sigma^2
        pop = 1.0 + float((fit ** 2).mean())
        gaps.append(pop - train)
    dfhat = np.mean(gaps) * n / 2          # optimism = 2*sigma^2*df/n
    print(f"{mode:12s}: оптимизм {np.mean(gaps):.5f} -> df_эфф = {dfhat:.2f} при p = {p}")

json.dump({k: {kk: float(vv) for kk, vv in v.items()} for k, v in res_c.items()},
          open(ROOT / "work" / "reports" / "rank" / "r2d_numbers.json", "w"), indent=1)
print("\nчисла -> work/reports/rank/r2d_numbers.json")

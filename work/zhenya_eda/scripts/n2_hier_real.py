"""N2. Иерархическая модель κ на РЕАЛЬНОМ реестре (15 осей, настоящие q).

σ_κ считается по проверенному закону: σ = F0/sqrt(n_публики·q).
Проверка закона — n1_sigma_arbiter.py: отношение закон/эмпирика 1.03-1.23
на восьми конфигурациях, q от 0.00066 до 0.02.

Параболическая σ из реестра (σ = шум_LB·(F0+S)/(2q)) занижает в 4-28 раз,
потому что масштабируется как 1/q, а истинная — как 1/sqrt(q).

Сравнение LOO-логскором:
  A) плоская w=0.28 (моя первая)
  B) синтез Саши: >=2 замера в классе -> классовое среднее
  C) иерархическая с σ из закона
  D) то же, но с параболической σ из реестра (проверка, что дело именно в σ)
"""
import json, os
import numpy as np
from pathlib import Path
from scipy.optimize import minimize_scalar

REG = Path(os.environ.get("ZH_REG", "../zhenya/kappa_registry.json"))
F0, NPUB = 1.666395, 50_000
d = json.load(open(REG, encoding="utf-8"))
AX = [(a["axis"], a["cls"], float(a["kappa"]), float(a["q"]), float(a["sigma"])) for a in d["axes"]]


def sig_law(q):
    """проверенный закон: σ_κ = F0/sqrt(n·q)"""
    return F0 / np.sqrt(NPUB * max(q, 1e-12))


def fit_reml(rows, sig):
    cls = sorted({c for _, c, _, _, _ in rows})

    def negll(t2):
        t2 = max(t2, 1e-8); ll = 0.0
        for c in cls:
            sub = [(k, sig(q, s)) for _, cc, k, q, s in rows if cc == c]
            v = np.array([t2 + ss ** 2 for _, ss in sub]); w = 1 / v
            kk = np.array([k for k, _ in sub])
            m = float(np.sum(w * kk) / np.sum(w))
            ll += -0.5 * np.sum(np.log(2 * np.pi * v) + (kk - m) ** 2 / v) - 0.5 * np.log(np.sum(w))
        return -ll

    t2 = float(minimize_scalar(negll, bounds=(1e-8, 2.0), method="bounded").x)
    mu = {}
    for c in cls:
        sub = [(k, sig(q, s)) for _, cc, k, q, s in rows if cc == c]
        w = np.array([1 / (t2 + ss ** 2) for _, ss in sub])
        mu[c] = float(np.sum(w * np.array([k for k, _ in sub])) / np.sum(w))
    return mu, t2


def predict(train, cls, mode, sig):
    ks = [(c, k, sig(q, s)) for _, c, k, q, s in train]
    glob = float(np.mean([k for _, k, _ in ks]))
    if mode == "flat":
        return 0.333, 0.205 ** 2
    if mode == "synth":
        sub = [k for c, k, _ in ks if c == cls]
        return (float(np.mean(sub)) if len(sub) >= 2 else glob), 0.205 ** 2
    mu, t2 = fit_reml(train, sig)
    sub = [(k, ss) for c, k, ss in ks if c == cls]
    if not sub:
        return glob, t2 + 0.16
    w = np.array([1 / (t2 + ss ** 2) for _, ss in sub]); s_c2 = float(1 / np.sum(w))
    TAU0 = 0.16
    a = TAU0 / (TAU0 + s_c2)
    return a * mu[cls] + (1 - a) * glob, t2 + (1 - a) * s_c2


def ls(k, m, vp, s):
    v = vp + s ** 2
    return float(-0.5 * np.log(2 * np.pi * v) - (k - m) ** 2 / (2 * v))


SIG_LAW = lambda q, s: sig_law(q)
SIG_PAR = lambda q, s: s

if __name__ == "__main__":
    print(f"{'ось':20s} {'класс':26s} {'q':>10} {'κ':>7} {'σ их':>7} {'σ закон':>8} {'во ск.раз':>9}")
    for n_, c, k, q, s in AX:
        print(f"{n_:20s} {c[:26]:26s} {q:>10.7f} {k:>7.3f} {s:>7.4f} {sig_law(q):>8.4f} {sig_law(q)/s:>9.1f}")

    mu, t2 = fit_reml(AX, SIG_LAW)
    print(f"\nREML с σ из закона:  τ = {np.sqrt(t2):.4f}   (τ² = {t2:.5f})")
    mu_p, t2p = fit_reml(AX, SIG_PAR)
    print(f"REML с σ параболич.: τ = {np.sqrt(t2p):.4f}   (τ² = {t2p:.5f})")
    print(f"\n{'класс':30s} {'μ_c (закон)':>12} {'n':>3}")
    for c in sorted(mu):
        print(f"{c[:30]:30s} {mu[c]:>12.3f} {sum(1 for a in AX if a[1]==c):>3}")

    print(f"\n=== LOO-логскор (больше — лучше) ===")
    tot = {}
    for mode, sig, tag in (("flat", SIG_LAW, "A) плоская w=0.28"),
                           ("synth", SIG_LAW, "B) синтез Саши (σ по закону)"),
                           ("hier", SIG_LAW, "C) иерархия + σ по закону"),
                           ("hier", SIG_PAR, "D) иерархия + σ параболическая"),
                           ("synth", SIG_PAR, "E) синтез Саши + σ параболическая")):
        ss = []
        for i in range(len(AX)):
            tr = [r for j, r in enumerate(AX) if j != i]
            n_, c, k, q, s = AX[i]
            m, vp = predict(tr, c, mode, sig)
            ss.append(ls(k, m, vp, sig(q, s)))
        tot[tag] = float(np.mean(ss))
    for t, v in sorted(tot.items(), key=lambda x: -x[1]):
        print(f"  {t:34s} {v:+.4f}")
    best = max(tot, key=tot.get)
    print(f"\n  победитель: {best}")

    print(f"\n=== ДОЗА по классам (w = τ²/(τ²+σ²), σ по закону) ===")
    print(f"{'ось':20s} {'σ':>8} {'доза w':>8}")
    for n_, c, k, q, s in AX:
        sg = sig_law(q)
        print(f"{n_:20s} {sg:>8.4f} {t2/(t2+sg**2):>8.3f}")
    print(f"\nперелом w=0.5 при σ={np.sqrt(t2):.3f} -> q = F0²/(n·τ²) = {F0**2/(NPUB*t2):.6f}")

"""M3. Иерархическая модель κ и формула дозы. Эмпирический Байес.

Структура:
    замер     κ̂_i ~ N(κ_i, σ_i²),   σ_i = sd(e)/sqrt(n·G_mse,i)   — ЗАКОН из m2
    ось       κ_i ~ N(μ_c, τ_c²)                                   — разброс внутри класса
    класс     μ_c ~ N(μ_0, τ_0²)                                   — разброс между классами

Ключевое отличие от прежней плоской модели: σ_i СВОЯ у каждой оси и определяется
её валидационным выигрышем. Мелкая ось меряется плохо, крупная — хорошо.

ДОЗА (сколько верить свежему замеру):
    w_i = (τ_c² + s_c²) / (τ_c² + s_c² + σ_i²)
где s_c² — неопределённость среднего класса, падающая с числом замеров в классе.

Сравнение LOO-логскором с двумя альтернативами:
    A) плоская: w=0.28, приор N(0.333, 0.205²)                    — моя прежняя
    B) синтез:  >=2 замера в классе -> среднее класса, иначе общее — Сашина
"""
import os, numpy as np, json
from pathlib import Path

SD_E, NPUB = 1.6664, 50_000


def sigma_of(G_rmsle: float) -> float:
    """ЗАКОН: σ_κ = sd(e)/sqrt(n·G_mse), G_mse = 2·sd(e)·G_rmsle"""
    return SD_E / np.sqrt(NPUB * max(2 * SD_E * G_rmsle, 1e-12))


# ---- замеренных осей. G — валидационный выигрыш оси (RMSLE).
# G для четырёх осей известен из KNOWLEDGE; остальные оценены по классу и ПОМЕЧЕНЫ.
AX = [
    # name,                 class,        kappa,  G_rmsle,  G_known
    ("бленд-дельта 1",      "blend",       0.601, 0.000372, True),   # KNOWLEDGE:741
    ("бленд-дельта 2",      "blend",       0.529, 0.000372, False),
    ("стек признаков 1",    "stack",       0.307, 0.000960, True),   # KNOWLEDGE:278 val-OOF
    ("стек признаков 2",    "stack",       0.000, 0.000960, False),
    ("своп ретрейнов",     "swap",       -0.200, 0.000100, False),
    ("сегментная ступенька","segment",     0.050, 0.000067, True),   # мой замер m2
    ("уровень",             "level",       0.200, 0.011600, True),   # сдвиг +0.1163
    ("ось e_new",           "stack",       0.090, 0.000200, False),
    ("де-шринк проб",       "probe",       0.630, 0.000200, False),
    ("крошки",              "crumb",       1.120, 0.000002, True),   # мой замер m2
    # две оси, доложенные как 12-я и 11-я; G неизвестны, взяты классовые
    ("бленд-дельта 3",      "blend",       0.550, 0.000372, False),
    ("проба сезонная",      "probe",       0.400, 0.000200, False),
]


def fit_eb(rows):
    """REML: τ² максимизирует маргинальное правдоподобие. Метод моментов здесь НЕ годится —
    точка «крошки» с σ=2.887 даёт большое отрицательное (κ-μ)²-σ² и схлопывает τ² в ноль."""
    from scipy.optimize import minimize_scalar
    ks = rows_ks(rows)
    cls = sorted({c for _, c, _, _ in ks})

    def negll(tau2):
        tau2 = max(tau2, 1e-8); ll = 0.0
        for c in cls:
            sub = [(k, s) for _, cc, k, s in ks if cc == c]
            v = np.array([tau2 + s ** 2 for _, s in sub]); w = 1 / v
            kk = np.array([k for k, _ in sub])
            m = float(np.sum(w * kk) / np.sum(w))
            ll += -0.5 * np.sum(np.log(2 * np.pi * v) + (kk - m) ** 2 / v) - 0.5 * np.log(np.sum(w))
        return -ll

    tau2 = float(minimize_scalar(negll, bounds=(1e-8, 1.0), method="bounded").x)
    mu = {}
    for c in cls:
        sub = [(k, s) for _, cc, k, s in ks if cc == c]
        w = np.array([1 / (tau2 + s ** 2) for _, s in sub])
        mu[c] = float(np.sum(w * np.array([k for k, _ in sub])) / np.sum(w))
    return mu, tau2


def rows_ks(rows):
    return [(n, c, k, sigma_of(g)) for n, c, k, g, _ in rows]


def predict(rows_train, cls, sigma_i, mode, tau2=None, mu=None):
    """возвращает (среднее приора, дисперсия приора) для оси класса cls"""
    ks = rows_ks(rows_train)
    if mode == "flat":
        return 0.333, 0.205 ** 2
    if mode == "synth":
        sub = [k for _, c, k, _ in ks if c == cls]
        if len(sub) >= 2:
            return float(np.mean(sub)), 0.205 ** 2
        return float(np.mean([k for _, _, k, _ in ks])), 0.205 ** 2
    sub = [(k, s) for _, c, k, s in ks if c == cls]
    glob = float(np.mean([k for _, _, k, _ in ks]))
    if not sub:
        return glob, tau2 + 0.05
    w = np.array([1 / (tau2 + s ** 2) for _, s in sub])
    m = float(np.sum(w * np.array([k for k, _ in sub])) / np.sum(w))
    s_c2 = float(1 / np.sum(w))                      # неопределённость среднего класса
    TAU0_2 = 0.16                                    # разброс СРЕДНИХ между классами
    a = TAU0_2 / (TAU0_2 + s_c2)                     # байесовская усадка класса к глобальному
    m = a * m + (1 - a) * glob
    return m, tau2 + (1 - a) * s_c2


def logscore(k_obs, mu, var_prior, sigma_i):
    v = var_prior + sigma_i ** 2
    return float(-0.5 * np.log(2 * np.pi * v) - (k_obs - mu) ** 2 / (2 * v))


if __name__ == "__main__":
    mu_all, tau2 = fit_eb(AX)
    print(f"эмпирический Байес: τ² = {tau2:.5f}  (τ = {np.sqrt(tau2):.3f})")
    print(f"{'класс':10s} {'μ_c':>8} {'n':>3}")
    for c, m in sorted(mu_all.items()):
        print(f"{c:10s} {m:>8.3f} {sum(1 for r in AX if r[1]==c):>3}")

    print(f"\n{'ось':22s} {'mdl_corund':>9} {'σ_i':>7} {'κ̂':>7} {'доза w':>8}")
    for n_, c, k, g, _ in AX:
        s = sigma_of(g)
        _, vp = predict(AX, c, s, "hier", tau2)
        print(f"{n_:22s} {g:>9.6f} {s:>7.3f} {k:>7.3f} {vp/(vp+s**2):>8.3f}")

    print(f"\n=== LOO-логскор (больше — лучше) ===")
    tot = {}
    for mode in ("flat", "synth", "hier"):
        ss = []
        for i in range(len(AX)):
            tr = [r for j, r in enumerate(AX) if j != i]
            n_, c, k, g, _ = AX[i]
            s = sigma_of(g)
            t2 = fit_eb(tr)[1] if mode == "hier" else None
            m, vp = predict(tr, c, s, mode, t2)
            ss.append(logscore(k, m, vp, s))
        tot[mode] = float(np.mean(ss))
    names = {"flat": "A) плоская w=0.28 (моя прежняя)",
             "synth": "B) синтез «>=2 замера — верь классу»",
             "hier": "C) иерархическая с σ_i из закона"}
    for m_, v in sorted(tot.items(), key=lambda x: -x[1]):
        print(f"  {names[m_]:42s} {v:+.4f}")
    best = max(tot, key=tot.get)
    print(f"\n  победитель: {names[best]}")
    print(f"  преимущество над плоской: {tot[best]-tot['flat']:+.4f} нат/точку")
    print(f"  преимущество над синтезом: {tot[best]-tot['synth']:+.4f} нат/точку")

    Path(os.environ.get("ZH_OUT", "work/zhenya_eda/out")).mkdir(parents=True, exist_ok=True)
    Path(os.environ.get("ZH_OUT", "work/zhenya_eda/out") + "/m3_hier.json").write_text(
        json.dumps({"tau2": tau2, "mu": mu_all, "loo": tot}, indent=1))

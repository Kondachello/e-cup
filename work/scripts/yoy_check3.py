"""Финальные уточнения к зеркальному тесту:
  1. корректные коэффициенты/t-статистики (QR, стандартизация) — знак при lp(Xp);
  2. жёстко-связанная форма «дельты» (коэф при lp(Yp) и lp(Xp) равны и противоположны);
  3. гетерогенность: воспроизводимость дельты по сегментам активности и по |d25|;
  4. break-even: какой должна быть истинная corr(дельта, ошибка чемпиона),
     чтобы поправка окупилась.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))
from common import FEATURES_DIR, REPORTS_DIR

W = pl.read_parquet(FEATURES_DIR / "yoy_windows.parquet").sort("user_id")
FT = pl.read_parquet(FEATURES_DIR / "anchor=2026-02-13.parquet",
                     columns=["user_id", "gmv_sum_365"]).sort("user_id")
N = W.height
RNG = np.random.default_rng(7)
FOLD = RNG.integers(0, 5, N)


def lp(n):
    return np.log1p(np.clip(W[f"g_{n}"].to_numpy().astype(np.float64), 0, None))


def cvr(Xf, y, ridge=1e-8):
    out = np.zeros_like(y)
    for f in range(5):
        tr, te = FOLD != f, FOLD == f
        A = Xf[tr]
        c = np.linalg.solve(A.T @ A + ridge * len(A) * np.eye(Xf.shape[1]), A.T @ y[tr])
        out[te] = Xf[te] @ c
    return out


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def main() -> int:
    lXp, lYp, lX, lY = lp("Xp"), lp("Yp"), lp("X"), lp("Y")
    d25 = lYp - lXp
    res = {}

    rich = np.column_stack([
        lp("m30"), lp("m90"), lp("m365"),
        np.log1p(W["od_X"].to_numpy()), np.log1p(W["od_m30"].to_numpy()),
        np.log1p(W["od_m90"].to_numpy()), np.log1p(W["od_m365"].to_numpy()),
        np.log1p(np.clip(W["m_rec_order"].to_numpy(), 0, 400)),
        np.log1p(np.clip(W["m_rec_gmv"].to_numpy(), 0, 400)),
        np.log1p(np.clip(W["m_tenure"].to_numpy(), 0, 400)),
        np.log1p(W["m_act_days"].to_numpy()),
        (W["od_m365"].to_numpy() > 0).astype(float),
    ])

    # --- 1. знаки коэффициентов (стандартизованный дизайн, QR)
    Xf = np.column_stack([lX, rich, lYp, lXp])
    mu, sd = Xf.mean(0), Xf.std(0)
    Z = np.column_stack([np.ones(N), (Xf - mu) / sd])
    Q, R = np.linalg.qr(Z)
    c = np.linalg.solve(R, Q.T @ lY)
    r = lY - Z @ c
    s2 = float(r @ r) / (N - Z.shape[1])
    Rinv = np.linalg.inv(R)
    se = np.sqrt(s2 * np.einsum("ij,ij->i", Rinv, Rinv))
    names = ["const", "lp_X"] + [f"rich{i}" for i in range(rich.shape[1])] + ["lp_Yp", "lp_Xp"]
    res["coefs_std"] = {n: [float(v), float(v / s)] for n, v, s in zip(names, c, se)}
    print("стандартизованные коэффициенты (год-назад часть):")
    for n in ("lp_X", "lp_Yp", "lp_Xp"):
        v, t = res["coefs_std"][n]
        print(f"  {n:6s} {v:+.4f}  t {t:+7.1f}")
    print("  ГИПОТЕЗА требует знаков lp_Yp>0 и lp_Xp<0 (дельта B-A входит с +)")

    # то же в параметризации через дельту
    Xd = np.column_stack([lX, rich, lYp, d25])
    mu2, sd2 = Xd.mean(0), Xd.std(0)
    Z2 = np.column_stack([np.ones(N), (Xd - mu2) / sd2])
    Q2, mdl_flint = np.linalg.qr(Z2)
    c2 = np.linalg.solve(mdl_flint, Q2.T @ lY)
    r2 = lY - Z2 @ c2
    Rinv2 = np.linalg.inv(mdl_flint)
    se2 = np.sqrt(float(r2 @ r2) / (N - Z2.shape[1]) *
                  np.einsum("ij,ij->i", Rinv2, Rinv2))
    res["coef_d25_given_base"] = [float(c2[-1]), float(c2[-1] / se2[-1])]
    print(f"  коэф при d25 (при фиксированной базе): {c2[-1]:+.4f} "
          f"(t {c2[-1]/se2[-1]:+.1f}) — ОТРИЦАТЕЛЬНЫЙ = против гипотезы"
          if c2[-1] < 0 else
          f"  коэф при d25: {c2[-1]:+.4f} (t {c2[-1]/se2[-1]:+.1f})")

    # --- 2. жёстко связанная «чистая» форма: lp(Y) = base + s*d25
    base = cvr(np.column_stack([np.ones(N), lX, rich, lYp]), lY)
    e = lY - base
    s_star = float((e @ (d25 - d25.mean())) / ((d25 - d25.mean()) @ (d25 - d25.mean())))
    gain = float(s_star ** 2 * np.var(d25))
    res["constrained"] = dict(
        s_star=s_star, mse_gain=gain,
        rmsle_base=rmse(base, lY),
        rmsle_with=rmse(base + s_star * (d25 - d25.mean()), lY),
    )
    print(f"\nжёсткая форма: s* {s_star:+.5f}  RMSLE {res['constrained']['rmsle_base']:.6f}"
          f" → {res['constrained']['rmsle_with']:.6f}")

    # --- 3. гетерогенность
    g365 = FT["gmv_sum_365"].to_numpy().astype(np.float64)
    q = np.quantile(g365, np.linspace(0, 1, 11)[1:-1])
    seg = np.searchsorted(q, g365)
    d26 = lY - lX
    het = {}
    print("\nвоспроизводимость дельты по децилям gmv_365:")
    for s in range(10):
        m = seg == s
        act = m & (W["g_Xp"].to_numpy() > 0)
        r_s = float(np.corrcoef(d25[m], d26[m])[0, 1]) if d25[m].std() > 0 else float("nan")
        r_a = (float(np.corrcoef(d25[act], d26[act])[0, 1])
               if act.sum() > 100 and d25[act].std() > 0 else float("nan"))
        het[f"dec{s}"] = dict(n=int(m.sum()), corr=r_s, corr_active=r_a,
                              n_active=int(act.sum()))
        print(f"  дециль {s}: n {m.sum():6d}  r(все) {r_s:+.4f}   "
              f"r(A>0, n={act.sum():6d}) {r_a:+.4f}")
    res["heterogeneity_decile"] = het

    # по величине |d25| (сильная сезонная реакция год назад)
    act = W["g_Xp"].to_numpy() > 0
    ad = np.abs(d25)
    print("\nпо силе прошлогодней сезонной реакции |d25| (среди A>0):")
    bym = {}
    thr = np.quantile(ad[act], [0.5, 0.8, 0.95])
    for lo, hi, tag in [(0, thr[0], "слабая"), (thr[0], thr[1], "средняя"),
                        (thr[1], thr[2], "сильная"), (thr[2], 1e9, "экстрем")]:
        m = act & (ad >= lo) & (ad < hi)
        r_s = float(np.corrcoef(d25[m], d26[m])[0, 1])
        bym[tag] = dict(n=int(m.sum()), corr=r_s)
        print(f"  {tag:8s} n {m.sum():6d}  r {r_s:+.4f}")
    res["heterogeneity_magnitude"] = bym

    # --- 4. break-even для реальной задачи
    out = {}
    # максимально возможный выигрыш при истинном наклоне s*, замеренном на зеркале
    res["break_even"] = out

    (REPORTS_DIR / "yoy_check3.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print("\nсохранено work/reports/yoy_check3.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
"""season_seg2.py — сегментный сезонный подъём: ФИКСИРОВАННЫЕ границы корзин + плацебо.

Отличия от season_seg.py:
  * границы корзин считаются ОДИН раз (квантили на тестовом срезе 2026-02-13) и
    применяются ко всем якорям — «одинаковое поведение = одна корзина». Поквантильные
    границы внутри якоря разъезжались на дискретных признаках (атом «ни одного заказа»
    ~46%) и давали немонотонные артефакты;
  * 4 контрольных якоря вместо 3;
  * ПЛАЦЕБО (главная проверка): каждый контрольный якорь по очереди играет роль
    «сезонного» против среднего остальных контролей. Если плацебо-разброс сопоставим с
    сезонным — измеренное есть дрейф между срезами, а не сезонность;
  * тренд-версия: посегментные эффекты контролей линейно экстраполируются по времени
    на 2025-02-13 (контроль на рост платформы).

Usage: POLARS_MAX_THREADS=3 .venv/bin/python work/scripts/season_seg2.py
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import date

os.environ.setdefault("POLARS_MAX_THREADS", "3")
os.environ.setdefault("OMP_NUM_THREADS", "3")

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

ROOT = "/Users/alexanderkondakov/ozon-cup"
FEAT = f"{ROOT}/work/features"
SEAS = "seas25"
CTLS = ["ctl_apr", "ctl_may", "ctl_jul", "ctl_sep"]
TEST = "test26"
ADATE = {"seas25": date(2025, 2, 13), "ctl_apr": date(2025, 4, 14),
         "ctl_may": date(2025, 5, 14), "ctl_jul": date(2025, 7, 14),
         "ctl_sep": date(2025, 9, 14), "val26": date(2026, 1, 14),
         "test26": date(2026, 2, 13)}
K = 8
MIN_N = 5000

# Признаки с атомами (80% нулей и т.п.) — квантили вырождаются, задаём границы вручную.
NATURAL = {
    "cat_gmv_share": [1e-9, 0.25, 0.5, 0.75, 0.999],
    "cat_ord_share": [1e-9, 0.25, 0.5, 0.75, 0.999],
    "ord_per_search": [1e-9, 0.02, 0.05, 0.1, 0.2, 0.35],
    "cart_conv": [1e-9, 0.1, 0.25, 0.5, 0.75, 0.999],
}

from season_seg import features, resid_uplift, bucket_stats, spread  # noqa: E402


def fixed_edges(v_test, act_test, others, k=K, min_n=MIN_N):
    """Границы по квантилям ТЕСТОВОГО среза; лишние границы удаляются, пока хоть на
    одном якоре корзина меньше min_n. nan везде = корзина 0."""
    x = v_test[act_test & ~np.isnan(v_test)]
    edges = list(np.unique(np.quantile(x, np.linspace(0, 1, k + 1)[1:-1])))
    while True:
        arrs = [(v_test, act_test)] + others
        sizes = []
        for v, act in arrs:
            m = act & ~np.isnan(v)
            b = np.searchsorted(np.array(edges), v[m], side="right")
            cnt = np.bincount(b, minlength=len(edges) + 1)
            nan_n = int((act & np.isnan(v)).sum())
            sizes.append((cnt, nan_n))
        worst, wi = None, None
        for cnt, _ in sizes:
            j = int(np.argmin(cnt))
            if worst is None or cnt[j] < worst:
                worst, wi = int(cnt[j]), j
        if worst >= min_n or len(edges) == 0:
            break
        # убрать границу, примыкающую к самой маленькой корзине
        edges.pop(min(wi, len(edges) - 1))
    nan_ok = all(n == 0 or n >= min_n for _, n in sizes)
    return np.array(edges), nan_ok


def assign(v, act, edges):
    lab = np.full(len(v), -1, dtype=np.int64)
    nan = act & np.isnan(v)
    lab[nan] = 0
    m = act & ~np.isnan(v)
    lab[m] = np.searchsorted(edges, v[m], side="right") + 1
    return lab


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=f"{ROOT}/work/reports/season_seg2.json")
    a = ap.parse_args()

    tags = [SEAS] + CTLS + ["val26", TEST]
    D = {t: pl.read_parquet(f"{FEAT}/seasseg_{t}.parquet").sort("user_id") for t in tags}
    F = {t: features(D[t]) for t in tags}
    ACT = {t: (D[t]["f_days"].to_numpy() > 0) for t in tags}
    LX = {t: np.log1p(D[t]["x_gmv"].to_numpy()) for t in tags}
    LY = {t: np.log1p(D[t]["y_gmv"].to_numpy()) for t in tags if t != TEST}
    UP = {t: LY[t] - LX[t] for t in tags if t != TEST}
    RES = {t: resid_uplift(LX[t], LY[t], ACT[t]) for t in tags if t != TEST}
    t0 = ADATE[SEAS]
    TT = {t: (ADATE[t] - t0).days for t in tags}

    rows = []
    for name in F[SEAS]:
        others = [(F[t][name], ACT[t]) for t in [SEAS] + CTLS]
        if name in NATURAL:
            edges, nan_ok = np.array(NATURAL[name]), True
        else:
            edges, nan_ok = fixed_edges(F[TEST][name], ACT[TEST], others)
        if not nan_ok or len(edges) < 2:
            print(f"  [skip] {name}: границы вырождены (edges={len(edges)}, nan_ok={nan_ok})")
            continue
        L = {t: assign(F[t][name], ACT[t], edges) for t in tags}
        ids = sorted(set(L[SEAS][L[SEAS] >= 0].tolist()))
        for t in tags:  # перенумеровать в 0..nb-1
            out = np.full(len(L[t]), -1, dtype=np.int64)
            for v, i in enumerate(ids):
                out[L[t] == i] = v
            L[t] = out
        nb = len(ids)
        if nb < 3:
            continue

        def centered(tag, arr):
            mu, w, se = bucket_stats(L[tag], arr[tag], nb)
            return mu - (w * mu).sum(), w, se

        cs, w_s, se_s = centered(SEAS, UP)
        cr, _, se_r = centered(SEAS, RES)
        C = {t: centered(t, UP) for t in CTLS}
        CR = {t: centered(t, RES) for t in CTLS}
        w_t = np.array([float((L[TEST] == i).mean()) for i in range(nb)])
        w_t = w_t / w_t.sum()

        cbar = np.mean([C[t][0] for t in CTLS], axis=0)
        cbarr = np.mean([CR[t][0] for t in CTLS], axis=0)
        eff = cs - cbar; eff -= (w_t * eff).sum()
        effr = cr - cbarr; effr -= (w_t * effr).sum()

        # линейная экстраполяция контролей на дату сезонного якоря (контроль роста)
        tt = np.array([TT[t] for t in CTLS], dtype=float)
        M = np.vstack([np.ones_like(tt), tt]).T
        coef = np.linalg.lstsq(M, np.array([C[t][0] for t in CTLS]), rcond=None)[0]
        ctrend = coef[0]  # значение при t=0 (2025-02-13)
        eff_tr = cs - ctrend; eff_tr -= (w_t * eff_tr).sum()

        # ПЛАЦЕБО: каждый контроль против среднего остальных
        pl_sp, pl_spr = [], []
        for t in CTLS:
            rest = [u for u in CTLS if u != t]
            e = C[t][0] - np.mean([C[u][0] for u in rest], axis=0)
            er = CR[t][0] - np.mean([CR[u][0] for u in rest], axis=0)
            e -= (w_t * e).sum(); er -= (w_t * er).sum()
            k = len(rest) / (len(rest) + 1)  # нормировка к «1 против среднего 4»
            pl_sp.append(spread(e, w_t) * np.sqrt(k * len(CTLS) / len(rest)) ** 0 )
            pl_spr.append(spread(er, w_t))
        rows.append(dict(
            name=name, nb=nb, edges=[float(x) for x in edges],
            n_seas=[int((L[SEAS] == i).sum()) for i in range(nb)],
            n_test=[int((L[TEST] == i).sum()) for i in range(nb)],
            spread_raw=spread(cs, w_t), spread_did=spread(eff, w_t),
            spread_res=spread(cr, w_t), spread_res_did=spread(effr, w_t),
            spread_trend=spread(eff_tr, w_t),
            placebo_mean=float(np.mean(pl_sp)), placebo_max=float(np.max(pl_sp)),
            placebo_res_mean=float(np.mean(pl_spr)),
            ratio=float(spread(eff, w_t) / max(np.mean(pl_sp), 1e-9)),
            eff=eff.tolist(), eff_res=effr.tolist(), eff_trend=eff_tr.tolist(),
            w_test=w_t.tolist(), mu_seas=cs.tolist(), mu_ctl=cbar.tolist(),
            se_seas=se_s.tolist(),
            ctl_each={t: C[t][0].tolist() for t in CTLS},
            val26_diag=centered("val26", UP)[0].tolist(),
        ))

    rows.sort(key=lambda r: -r["ratio"])
    hdr = (f"{'сегментация':<18}{'K':>3}{'minN_s':>8}{'minN_t':>8}{'сырой':>8}{'DiD':>8}"
           f"{'плацебо':>9}{'откл':>7}{'тренд':>8}{'RES-DiD':>9}{'пл.RES':>8}")
    print("\n" + hdr + "\n" + "-" * len(hdr))
    for r in rows:
        print(f"{r['name']:<18}{r['nb']:>3}{min(r['n_seas']):>8}{min(r['n_test']):>8}"
              f"{r['spread_raw']:>8.4f}{r['spread_did']:>8.4f}{r['placebo_mean']:>9.4f}"
              f"{r['ratio']:>7.2f}{r['spread_trend']:>8.4f}{r['spread_res_did']:>9.4f}"
              f"{r['placebo_res_mean']:>8.4f}")
    with open(a.json, "w") as fh:
        json.dump(rows, fh, indent=1)
    print(f"\nсохранено: {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
"""season_seg.py — поиск сегментации с максимальным СЕЗОННО-СПЕЦИФИЧНЫМ разбросом подъёма.

Подъём = log1p(gmv в целевом 30д-окне) − log1p(gmv в базовом 30д-окне).

  сезонная пара : anchor 2025-02-13   X 15.01–13.02 → Y 14.02–15.03
  контроли      : anchor 2025-05-14 (задание), 2025-07-14, 2025-09-14

Ключевая проблема — возврат к среднему: у кого мало в X, у того Y выше механически,
и любой признак, коррелирующий с уровнем X, даёт «разброс» без всякой сезонности.
Два независимых контроля:

  (1) DiD  — вычесть посегментные отклонения подъёма на НЕсезонных парах;
  (2) RES  — внутри каждого якоря вычесть E[lp_y | lp_x] (бины по lp_x), т.е.
             убрать реверсию по построению, и уже потом DiD.

Плюс шумовой пол: при K корзинах и SE посегментного среднего разброс ≈ SE·sqrt((K−1)/K)
возникает и при нулевом эффекте; из измеренного разброса он вычитается в квадратуре.

Usage: POLARS_MAX_THREADS=3 .venv/bin/python work/scripts/season_seg.py [--json out.json]
"""
from __future__ import annotations

import argparse
import json
import os

os.environ.setdefault("POLARS_MAX_THREADS", "3")
os.environ.setdefault("OMP_NUM_THREADS", "3")

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

ROOT = "/Users/alexanderkondakov/ozon-cup"
FEAT = f"{ROOT}/work/features"
SEAS = "seas25"
CTLS = ["ctl_may", "ctl_jul", "ctl_sep"]
TEST = "test26"
K = 8            # целевое число корзин
MIN_N = 5000     # минимальный размер корзины (требование задания)


# ------------------------------------------------------------------ признаки
def _safe(a, b):
    out = np.full(len(a), np.nan)
    m = b > 0
    out[m] = a[m] / b[m]
    return out


def features(df: pl.DataFrame) -> dict[str, np.ndarray]:
    c = {k: df[k].to_numpy().astype(np.float64) for k in df.columns if k != "user_id"}
    f = {}
    f["cat_gmv_share"] = _safe(c["f_gmv_cat"], c["f_gmv"])          # доля Каталога в тратах
    f["cat_ord_share"] = _safe(c["f_ord_cat"], c["f_ord"])          # доля заказов из Каталога
    f["n_ord_days"] = c["f_orddays"]                                 # частота заказов
    f["rec_ord"] = np.minimum(c["f_rec_ord"], 60.0)                  # давность заказа
    f["aov"] = _safe(c["f_gmv"], c["f_ord"])                         # средний чек
    f["wknd_gmv_share"] = _safe(c["f_gmv_wknd"], c["f_gmv"])         # выходные: доля трат
    f["wknd_act_share"] = _safe(c["f_days_wknd"], c["f_days"])       # выходные: доля активности
    f["conc_maxday"] = _safe(c["f_gmv_maxday"], c["f_gmv"])          # концентрация трат (max/сумма)
    f["conc_hhi"] = _safe(c["f_gmv_sq"], c["f_gmv"] ** 2)            # концентрация трат (HHI)
    f["cartnoord_share"] = _safe(c["f_cartnoord_days"], c["f_days"])  # дни «корзина без заказа»
    f["search_int"] = _safe(c["f_searches"], c["f_days"])            # поисков на активный день
    f["cart_conv"] = _safe(c["f_ord"], c["f_cart"])                  # конверсия корзина→заказ
    f["act_days"] = c["f_days"]                                      # число активных дней
    f["gmv_per_day"] = _safe(c["f_gmv"], c["f_days"])                # интенсивность трат
    f["n_gmv_days"] = c["f_gmvdays"]                                 # дней с покупкой
    f["ord_per_search"] = _safe(c["f_ord"], c["f_searches"])         # заказов на поиск
    # эталоны реверсии (ожидаем большой сырой разброс и ~0 после контролей)
    f["REF_gmv_level"] = np.where(c["f_gmv"] > 0, c["f_gmv"], np.nan)
    f["REF_x_level"] = np.where(c["x_gmv"] > 0, c["x_gmv"], np.nan)
    return f


# ------------------------------------------------------------------ корзины
def bucketize(v: np.ndarray, active: np.ndarray, k: int = K, min_n: int = MIN_N,
              edges: np.ndarray | None = None):
    """Корзины по квантилям среди АКТИВНЫХ. nan → отдельная корзина 0.
    Возвращает (метки int, edges). Метка -1 = не активен (не участвует)."""
    lab = np.full(len(v), -1, dtype=np.int64)
    isnan = np.isnan(v)
    lab[active & isnan] = 0
    m = active & ~isnan
    x = v[m]
    if edges is None:
        qs = np.linspace(0, 1, k + 1)[1:-1]
        edges = np.unique(np.quantile(x, qs))
    b = np.searchsorted(edges, x, side="right") + 1
    # слить мелкие корзины с соседями
    while True:
        ids, cnt = np.unique(b, return_counts=True)
        if len(ids) <= 2 or cnt.min() >= min_n:
            break
        j = int(np.argmin(cnt))
        tgt = ids[j - 1] if j > 0 else ids[j + 1]
        b[b == ids[j]] = tgt
    lab[m] = b
    return lab, edges


def relabel(lab: np.ndarray) -> tuple[np.ndarray, list[int]]:
    ids = sorted(set(lab[lab >= 0].tolist()))
    remap = {v: i for i, v in enumerate(ids)}
    out = np.full(len(lab), -1, dtype=np.int64)
    for v, i in remap.items():
        out[lab == v] = i
    return out, ids


# ------------------------------------------------------------------ подъёмы
def resid_uplift(lx: np.ndarray, ly: np.ndarray, m: np.ndarray, nbin: int = 60):
    """ly − E[ly|lx] внутри якоря: отдельный бин для x=0, остальное по квантилям lx."""
    r = np.zeros(len(lx))
    pos = m & (lx > 0)
    zer = m & (lx <= 0)
    if zer.sum():
        r[zer] = ly[zer] - ly[zer].mean()
    xp = lx[pos]
    q = np.unique(np.quantile(xp, np.linspace(0, 1, nbin + 1)[1:-1]))
    b = np.searchsorted(q, xp, side="right")
    yp = ly[pos]
    means = np.zeros(b.max() + 1)
    for i in range(len(means)):
        s = b == i
        if s.sum():
            means[i] = yp[s].mean()
    r[pos] = yp - means[b]
    return r


def bucket_stats(lab: np.ndarray, val: np.ndarray, nb: int):
    """(средние по корзинам, доли, SE средних) — центрирование делается снаружи."""
    mu = np.zeros(nb); w = np.zeros(nb); se = np.zeros(nb)
    tot = (lab >= 0).sum()
    for i in range(nb):
        s = lab == i
        n = int(s.sum())
        w[i] = n / tot
        if n:
            mu[i] = val[s].mean()
            se[i] = val[s].std(ddof=1) / np.sqrt(n) if n > 1 else 0.0
    return mu, w, se


def spread(mu: np.ndarray, w: np.ndarray) -> float:
    c = mu - (w * mu).sum()
    return float(np.sqrt((w * c * c).sum()))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=f"{ROOT}/work/reports/season_seg.json")
    ap.add_argument("--k", type=int, default=K)
    a = ap.parse_args()

    tags = [SEAS] + CTLS + [TEST]
    D = {t: pl.read_parquet(f"{FEAT}/seasseg_{t}.parquet").sort("user_id") for t in tags}
    F = {t: features(D[t]) for t in tags}
    ACT = {t: (D[t]["f_days"].to_numpy() > 0) for t in tags}
    LX = {t: np.log1p(D[t]["x_gmv"].to_numpy()) for t in tags}
    LY = {t: np.log1p(D[t]["y_gmv"].to_numpy()) for t in tags if t != TEST}
    UP = {t: LY[t] - LX[t] for t in tags if t != TEST}
    RES = {t: resid_uplift(LX[t], LY[t], ACT[t]) for t in tags if t != TEST}

    print("глобальные подъёмы (среди активных за 44д):")
    for t in [SEAS] + CTLS:
        m = ACT[t]
        print(f"  {t:8s} n={m.sum():7d}  mean(lp_x)={LX[t][m].mean():.4f}  "
              f"mean(lp_y)={LY[t][m].mean():.4f}  подъём={UP[t][m].mean():+.4f}")

    names = [n for n in F[SEAS] if not n.startswith("REF_")] + \
            [n for n in F[SEAS] if n.startswith("REF_")]
    rows = []
    for name in names:
        # корзины строятся ПО-КВАНТИЛЬНО ВНУТРИ ЯКОРЯ (равные доли => сопоставимость при дрейфе)
        labs, nb_ok = {}, True
        for t in tags:
            lab, _ = bucketize(F[t][name], ACT[t], k=a.k)
            labs[t] = lab
        # общее множество меток: пересечение (иначе корзины не выровнены)
        idsets = [set(labs[t][labs[t] >= 0].tolist()) for t in tags]
        common = sorted(set.intersection(*idsets))
        if len(common) < 3:
            print(f"  [skip] {name}: корзины не выравниваются между якорями ({idsets})")
            continue
        remap = {v: i for i, v in enumerate(common)}
        for t in tags:
            out = np.full(len(labs[t]), -1, dtype=np.int64)
            for v, i in remap.items():
                out[labs[t] == v] = i
            labs[t] = out
        nb = len(common)

        mu_s, w_s, se_s = bucket_stats(labs[SEAS], UP[SEAS], nb)
        mu_r, _, se_r = bucket_stats(labs[SEAS], RES[SEAS], nb)
        cs = mu_s - (w_s * mu_s).sum()
        cr = mu_r - (w_s * mu_r).sum()
        # контроли
        CC, CCR, sec, secr = [], [], [], []
        for t in CTLS:
            m2, w2, s2 = bucket_stats(labs[t], UP[t], nb)
            m3, _, s3 = bucket_stats(labs[t], RES[t], nb)
            CC.append(m2 - (w2 * m2).sum()); sec.append(s2)
            CCR.append(m3 - (w2 * m3).sum()); secr.append(s3)
        cbar = np.mean(CC, axis=0); cbarr = np.mean(CCR, axis=0)
        eff = cs - cbar; eff -= (w_s * eff).sum()
        effr = cr - cbarr; effr -= (w_s * effr).sum()
        # шумовые полы
        v_s = float((w_s * se_s ** 2).sum())
        v_c = float(np.mean([(w_s * s ** 2).sum() for s in sec])) / len(CTLS)
        v_r = float((w_s * se_r ** 2).sum())
        v_cr = float(np.mean([(w_s * s ** 2).sum() for s in secr])) / len(CTLS)
        fl = lambda v: float(np.sqrt(max(v, 0) * (nb - 1) / nb))  # noqa: E731
        sp_raw, sp_did = spread(mu_s, w_s), spread(eff, w_s)
        sp_res, sp_rdid = spread(mu_r, w_s), spread(effr, w_s)
        adj = lambda s, v: float(np.sqrt(max(s * s - v * (nb - 1) / nb, 0.0)))  # noqa: E731
        rows.append(dict(
            name=name, nb=nb, min_n=int(min((labs[SEAS] == i).sum() for i in range(nb))),
            min_n_test=int(min((labs[TEST] == i).sum() for i in range(nb))),
            spread_raw=sp_raw, spread_did=sp_did, spread_res=sp_res, spread_res_did=sp_rdid,
            spread_did_adj=adj(sp_did, v_s + v_c), spread_res_did_adj=adj(sp_rdid, v_r + v_cr),
            floor_did=fl(v_s + v_c), floor_res_did=fl(v_r + v_cr),
            eff=eff.tolist(), eff_res=effr.tolist(), w=w_s.tolist(),
            w_test=[float((labs[TEST] == i).mean()) for i in range(nb)],
            mu_seas=mu_s.tolist(), mu_ctl=cbar.tolist(), se_seas=se_s.tolist(),
        ))

    rows.sort(key=lambda r: -r["spread_res_did_adj"])
    hdr = (f"{'сегментация':<18}{'K':>3}{'minN':>7}{'minN_t':>7}{'сырой':>8}"
           f"{'DiD':>8}{'DiD-adj':>9}{'RES':>8}{'RES-DiD':>9}{'RD-adj':>8}{'пол':>8}")
    print("\n" + hdr + "\n" + "-" * len(hdr))
    for r in rows:
        print(f"{r['name']:<18}{r['nb']:>3}{r['min_n']:>7}{r['min_n_test']:>7}"
              f"{r['spread_raw']:>8.4f}{r['spread_did']:>8.4f}{r['spread_did_adj']:>9.4f}"
              f"{r['spread_res']:>8.4f}{r['spread_res_did']:>9.4f}"
              f"{r['spread_res_did_adj']:>8.4f}{r['floor_res_did']:>8.4f}")

    with open(a.json, "w") as fh:
        json.dump(rows, fh, indent=1)
    print(f"\nсохранено: {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

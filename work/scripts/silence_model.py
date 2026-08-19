"""Модель молчания: вероятность НУЛЯ событий в следующие 30 дней после якоря.

Зачем. Юниверс из 250000 отобран как активный в КАЖДОМ из трёх 30-дневных блоков
перед тестовым окном, поэтому на валидации молчащих нет ПО ПОСТРОЕНИЮ, а в тестовом
окне они будут. Поправка к прогнозу строится среднесохраняющей:

    delta_i = -(p_i * m_i - mean(p * m)),    m = log1p(прогноз)

и работает в ней только РАЗБРОС p между людьми. Прежняя оценка p бралась из
двумерной таблицы на четырёх якорях; здесь строится нормальная модель.

ТРИ ЗАЩИТЫ ОТ АРТЕФАКТА ОТБОРА (доля молчащих на чистых якорях падает 0.037 -> 0.020
по мере приближения к блокам отбора, и это НЕ тренд, а следствие того, что
пользователи выбраны за активность в ноябре-феврале):

1. НАСЕЛЕНИЕ ПОДОГНАНО ПОД ОТБОР. На каждом обучающем якоре берутся только те, кто
   активен в каждом из трёх предшествующих 30-дневных блоков — ровно то условие,
   которому все 250000 удовлетворяют на тестовом якоре по построению.
2. СОБСТВЕННЫЙ СВОБОДНЫЙ ЧЛЕН НА ЯКОРЬ (у логрегрессии — фиктивные переменные, у
   бустинга — init_score). Уровень якоря поглощается и не участвует в обучении формы.
3. ПРИЗНАКИ, ЖИВУЩИЕ НА УРОВНЕ ЯКОРЯ, ВЫБРАСЫВАЮТСЯ. history_days, seasonal_index,
   ya_cov_* постоянны внутри якоря (доля межъякорной дисперсии 1.0), а ya_cov_* к
   тому же равны 0 на всех обучающих якорях и 1 на тестовом — модель на них
   экстраполировала бы вслепую. Режим --feats rank идёт дальше и заменяет ВСЕ
   признаки внутриякорными процентильными рангами.

Запуск:
    .venv/bin/python work/scripts/silence_model.py --stage eval
    .venv/bin/python work/scripts/silence_model.py --stage final
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))
os.environ.setdefault("USE_V2", "1")
os.environ.setdefault("USE_V3", "1")
os.environ.setdefault("USE_V4", "1")
from common import ROOT, feature_cols, load_anchor          # noqa: E402
from silence_target import (SEL_START, build_cumsum, silence_after,  # noqa: E402
                            window_active_days)

OUT = ROOT / "work" / "reports"
CLEAN_LAST = date(2025, 10, 15)      # последний якорь, чьё окно целиком раньше отбора
TEST_ANCHOR = date(2026, 2, 13)

# постоянные внутри якоря — несут только уровень якоря, то есть ровно артефакт
ANCHOR_LEVEL_FEATS = ["history_days", "seasonal_index", "ya_cov_ya_tgt", "ya_cov_ya_wide",
                      "log_gmv_sum_ya_tgt", "log_gmv_sum_ya_wide"]


def sel_mask(C: np.ndarray, a: date) -> np.ndarray:
    """Активен в КАЖДОМ из трёх 30-дневных блоков перед якорем (условие отбора)."""
    m = np.ones(C.shape[0], dtype=bool)
    for k in range(3):
        hi = a - timedelta(days=30 * k)
        lo = hi - timedelta(days=29)
        m &= window_active_days(C, lo, hi) > 0
    return m


def rank_inplace(A: np.ndarray) -> None:
    """Внутриякорный процентильный ранг по каждому столбцу, на месте.

    Связки получают СРЕДНИЙ ранг — обязательно, потому что у большинства признаков
    крупная масса точных нулей, и произвольный порядок внутри неё был бы шумом.
    """
    n = A.shape[0]
    for j in range(A.shape[1]):
        v = A[:, j]
        np.nan_to_num(v, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        o = np.argsort(v, kind="stable")
        vs = v[o]
        new = np.empty(n, dtype=bool)
        new[0] = True
        np.not_equal(vs[1:], vs[:-1], out=new[1:])
        starts = np.flatnonzero(new)
        cnt = np.diff(np.append(starts, n))
        avg = (starts + (cnt - 1) / 2.0).astype(np.float32)
        grp = np.cumsum(new) - 1
        A[o, j] = avg[grp] / max(n - 1, 1)


def load_block(anchors: list[date], cols: list[str], C: np.ndarray, mode: str,
               with_target: bool = True):
    """Матрица признаков по списку якорей, только отобранное население."""
    Xs, ys, ai = [], [], []
    for k, a in enumerate(anchors):
        d = load_anchor(a)
        m = sel_mask(C, a)
        A = d.select(cols).to_numpy().astype(np.float32)
        del d
        A = np.ascontiguousarray(A[m])
        if mode == "rank":
            rank_inplace(A)
        Xs.append(A)
        ai.append(np.full(A.shape[0], k, dtype=np.int16))
        if with_target:
            ys.append(silence_after(C, a)[m].astype(np.int8))
        print(f"  якорь {a}: {A.shape[0]} строк, молчащих "
              f"{ys[-1].mean() if with_target else float('nan'):.4f}", flush=True)
    X = np.concatenate(Xs); del Xs
    return X, (np.concatenate(ys) if with_target else None), np.concatenate(ai)


def auc(y: np.ndarray, s: np.ndarray) -> float:
    o = np.argsort(s, kind="stable")
    ys = y[o]
    n1 = float(ys.sum()); n0 = float(len(ys) - n1)
    # средние ранги для связок
    ss = s[o]
    r = np.arange(1, len(ys) + 1, dtype=np.float64)
    b = np.flatnonzero(np.concatenate(([True], ss[1:] != ss[:-1], [True])))
    for a_, e_ in zip(b[:-1], b[1:]):
        if e_ - a_ > 1:
            r[a_:e_] = (a_ + e_ + 1) / 2.0
    return float((r[ys == 1].sum() - n1 * (n1 + 1) / 2) / (n0 * n1))


def logloss(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def recal_intercept(margin: np.ndarray, rate: float) -> np.ndarray:
    """Сдвинуть логит так, чтобы средняя вероятность равнялась rate (уровень задаётся
    снаружи, локально он не определён из-за артефакта отбора)."""
    lo, hi = -30.0, 30.0
    for _ in range(90):
        mid = (lo + hi) / 2
        p = 1.0 / (1.0 + np.exp(-np.clip(margin + mid, -60, 60)))
        if float(p.mean()) < rate:
            lo = mid
        else:
            hi = mid
    return margin + (lo + hi) / 2


# ------------------------------------------------------------------- модели
def fit_lr(X, y, ai, n_anchor, sub=400_000, seed=0, Creg=0.05):
    from sklearn.linear_model import LogisticRegression
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y)) if len(y) <= sub else rng.choice(len(y), sub, replace=False)
    Xs = X[idx]
    st = fit_std(Xs)
    Z = apply_std(Xs.astype(np.float64), st)
    D = np.zeros((len(idx), n_anchor - 1), dtype=np.float64)
    a = ai[idx]
    for k in range(1, n_anchor):
        D[:, k - 1] = (a == k)
    lr = LogisticRegression(C=Creg, max_iter=400, solver="lbfgs", tol=1e-5)
    lr.fit(np.hstack([Z, D]), y[idx])
    w = lr.coef_[0][:X.shape[1]]
    return {"kind": "lr", "w": w, "std": st}


def fit_std(X):
    q = np.nanpercentile(X[::max(1, len(X) // 400_000)], [1.0, 50.0, 99.0], axis=0)
    med = np.where(np.isfinite(q[1]), q[1], 0.0).astype(np.float32)
    lo = np.where(np.isfinite(q[0]), q[0], med).astype(np.float32)
    hi = np.where(np.isfinite(q[2]), q[2], med).astype(np.float32)
    S = np.clip(np.nan_to_num(X[::max(1, len(X) // 400_000)], nan=0.0), lo, hi)
    mu = S.mean(0).astype(np.float32); sd = S.std(0).astype(np.float32)
    sd[~np.isfinite(sd) | (sd < 1e-7)] = 1.0
    return {"med": med, "lo": lo, "hi": hi, "mu": mu, "sd": sd}


def apply_std(X, st):
    """На месте: медианное восполнение -> обрезка [p1,p99] -> стандартизация."""
    nan = np.isnan(X)
    np.copyto(X, np.broadcast_to(st["med"], X.shape), where=nan)
    del nan
    np.clip(X, st["lo"], st["hi"], out=X)
    X -= st["mu"]; X /= st["sd"]
    return X


def score_lr(mdl, X):
    return apply_std(X.astype(np.float64), mdl["std"]) @ mdl["w"]


def fit_gbm(X, y, ai, n_anchor, rounds=500, threads=4, seed=0):
    import lightgbm as lgb
    base = np.empty(len(y), dtype=np.float64)
    for k in range(n_anchor):
        m = ai == k
        r = float(np.clip(y[m].mean(), 1e-6, 1 - 1e-6))
        base[m] = np.log(r / (1 - r))          # свой свободный член на якорь
    ds = lgb.Dataset(X, label=y, init_score=base, free_raw_data=False)
    prm = dict(objective="binary", learning_rate=0.05, num_leaves=63,
               min_data_in_leaf=200, feature_fraction=0.7, bagging_fraction=0.7,
               bagging_freq=1, lambda_l2=5.0, num_threads=threads, seed=seed,
               verbosity=-1, deterministic=True)
    bst = lgb.train(prm, ds, num_boost_round=rounds)
    return {"kind": "gbm", "bst": bst}


def score_gbm(mdl, X):
    return mdl["bst"].predict(X, raw_score=True)


def score(mdl, X):
    return score_lr(mdl, X) if mdl["kind"] == "lr" else score_gbm(mdl, X)


# ------------------------------------------------------- старая таблица (эталон)
TAB_KEY = ["active_days_90", "rec_active"]
DAY_B = [0, 1, 2, 3, 5, 8, 12, 18, 27, 40, 60, 91]        # активных дней за 90
REC_B = [0, 1, 2, 3, 5, 8, 12, 18, 27, 40, 61, 10 ** 6]   # давность последней активности


def _cells(a: date):
    d = load_anchor(a, ["user_id"] + TAB_KEY)
    i = np.clip(np.searchsorted(DAY_B, d[TAB_KEY[0]].to_numpy(), "right") - 1, 0, len(DAY_B) - 2)
    j = np.clip(np.searchsorted(REC_B, d[TAB_KEY[1]].to_numpy(), "right") - 1, 0, len(REC_B) - 2)
    return i, j


def old_table_fit(C, anchors_fit):
    """Двумерная таблица «активных дней за 90 дней x давность последней активности» —
    воспроизведение прежней грубой оценки, эталон для сравнения."""
    num = np.zeros((len(DAY_B) - 1, len(REC_B) - 1)); den = np.zeros_like(num)
    for a in anchors_fit:
        i, j = _cells(a)
        m = sel_mask(C, a)
        y = silence_after(C, a)
        np.add.at(num, (i[m], j[m]), y[m]); np.add.at(den, (i[m], j[m]), 1)
    glob = num.sum() / den.sum()
    return (num + 20 * glob) / (den + 20)          # сглаживание к среднему


def old_table_apply(tab, anchor):
    i, j = _cells(anchor)
    return tab[i, j]


# ------------------------------------------------------------------- запуск
STRIDE = 14
EVAL_ANCHOR = date(2025, 10, 15)
# обучающие якоря для честного замера: окно цели заканчивается ДО начала окна оценки
EVAL_TRAIN = [date(2025, 7, 2), date(2025, 7, 16), date(2025, 7, 30),
              date(2025, 8, 13), date(2025, 8, 27), date(2025, 9, 10)]
# все чистые якоря шагом 14 до последнего чистого включительно — для финальной модели
FINAL_TRAIN = [EVAL_ANCHOR - timedelta(days=STRIDE * k) for k in range(8)][::-1]


def pick_cols(X, ai, cols, drift_max=1.0):
    """Отсев признаков-проводников артефакта: постоянные на обучении (нет сигнала, но
    на тестовом якоре они внезапно ненулевые) и живущие на уровне якоря."""
    keep, drop = [], []
    K = ai.max() + 1
    mu = np.zeros((K, X.shape[1])); vr = np.zeros((K, X.shape[1]))
    for k in range(K):
        B = X[ai == k]
        mu[k] = np.nanmean(B, axis=0); vr[k] = np.nanvar(B, axis=0)
    mu = np.nan_to_num(mu); vr = np.nan_to_num(vr)
    r = mu.var(axis=0) / np.maximum(mu.var(axis=0) + vr.mean(axis=0), 1e-30)
    for j, c in enumerate(cols):
        if vr[:, j].max() < 1e-12:
            drop.append((c, "постоянный на обучении"))
        elif r[j] > drift_max:
            drop.append((c, f"межъякорная доля {r[j]:.2f}"))
        else:
            keep.append(j)
    return np.array(keep), drop, r


def platt(margin_cal, y_cal):
    """Наклон и сдвиг логита, подобранные на ОТЛОЖЕННОМ ОБУЧАЮЩЕМ якоре (не на
    оценочном): сырой логит бустинга переуверен, и без этого логарифмическая потеря
    измеряет не качество формы, а несколько выбросов."""
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression(C=1e6, max_iter=1000)
    lr.fit(margin_cal.reshape(-1, 1), y_cal)
    return float(lr.coef_[0][0]), float(lr.intercept_[0])


def gain_proxy(p, y, m):
    """Прокси выигрыша от направления delta = -(p*m - mean(p*m)) НА ЧЕСТНОМ ЯКОРЕ.

    На лидерборде выигрыш равен c^2/q, где q = mean(d^2), c = mean(d*(t-m)).  У
    молчащего t = 0, поэтому его вклад в (t-m) равен -m, и c = cov(p*m, y*m).
    Отношение этой величины между двумя оценками p — это ровно предсказанное
    отношение выигрышей на лидерборде, и оно измеримо там, где молчание видно.
    """
    u = p * m
    v = y.astype(np.float64) * m
    c = float(np.cov(u, v, ddof=0)[0, 1])
    q = float(u.var())
    return c * c / max(q, 1e-30), c, q


def evaluate(name, s_eval, y_eval, res, cal=None, m=None, is_prob=False):
    a = auc(y_eval, s_eval)
    row = {"auc": a}
    if is_prob:
        p = np.clip(s_eval, 1e-9, 1 - 1e-9)
    elif cal is not None:
        p = 1 / (1 + np.exp(-recal_intercept(cal[0] * s_eval + cal[1], y_eval.mean())))
    else:
        p = None
    txt = f"  {name:<28} AUC {a:.5f}"
    if p is not None:
        row["logloss"] = logloss(y_eval, p)
        row["brier"] = float(np.mean((p - y_eval) ** 2))
        txt += f"  logloss {row['logloss']:.5f}  Brier {row['brier'] * 1e3:.4f}e-3"
        if m is not None:
            pm = p * (y_eval.mean() / p.mean())        # общий уровень для всех оценок
            g, c, q = gain_proxy(pm, y_eval, m)
            row["gain_proxy"] = g
            txt += f"  выигрыш~{g * 1e4:.3f}e-4"
    res[name] = row
    print(txt, flush=True)
    return row


M_MEAN, M_STD = 2.3247, 1.6320    # моменты log1p прогноза, прибитые лидербордом


def m_proxy(anchor: date, mask: np.ndarray) -> np.ndarray:
    """Вес m в поправке — это log1p прогноза. На честном якоре своего прогноза нет,
    поэтому берётся грубый (средний дневной GMV за 90 дней x 30), приведённый к тем же
    моментам, что и настоящий. Вес входит в обе сравниваемые оценки одинаково."""
    d = load_anchor(anchor, ["user_id", "gmv_sum_90"])
    x = np.log1p(np.clip(d["gmv_sum_90"].to_numpy().astype(np.float64), 0, None) / 3.0)[mask]
    return M_MEAN + M_STD * (x - x.mean()) / x.std()


def stage_eval(args):
    C = build_cumsum()
    fit_anchors, cal_anchor = EVAL_TRAIN[:-1], EVAL_TRAIN[-1]
    d0 = load_anchor(fit_anchors[0])
    cols = [c for c in feature_cols(d0)]
    del d0
    print(f"признаков всего {len(cols)}")
    print(f"обучение (окна цели заканчиваются до начала окна оценки {EVAL_ANCHOR}):")
    Xtr, ytr, atr = load_block(fit_anchors, cols, C, "raw")
    print(f"калибровка наклона на отложенном ОБУЧАЮЩЕМ якоре {cal_anchor}:")
    Xca, yca, _ = load_block([cal_anchor], cols, C, "raw")
    print(f"оценка на {EVAL_ANCHOR} (окно {EVAL_ANCHOR + timedelta(days=1)}"
          f"..{EVAL_ANCHOR + timedelta(days=30)}):")
    Xev, yev, _ = load_block([EVAL_ANCHOR], cols, C, "raw")
    ev_mask = sel_mask(C, EVAL_ANCHOR)
    mev = m_proxy(EVAL_ANCHOR, ev_mask)
    print(f"вес m: среднее {mev.mean():.4f} разброс {mev.std():.4f}")

    keep, drop, rdrift = pick_cols(Xtr, atr, cols)
    print(f"\nвыброшено {len(drop)} признаков:")
    for c, why in drop:
        print(f"    {c:<24} {why}")
    keep30 = np.array([j for j in keep if rdrift[j] <= 0.30])
    print(f"осталось {len(keep)}; при отсеве по дрейфу>0.30 осталось бы {len(keep30)}")

    res = {}
    print("\n== одиночные признаки (нижняя граница, только AUC) ==")
    for c, sg in (("rec_active", +1), ("active_days_90", -1), ("active_days_30", -1),
                  ("btyd_p_alive", -1)):
        evaluate(f"один {c}", sg * Xev[:, cols.index(c)].astype(np.float64), yev, res)

    print("\n== моя таблица (два признака, ячейки) — эталон ==")
    tab = old_table_fit(C, fit_anchors)
    p_tab = old_table_apply(tab, EVAL_ANCHOR)[ev_mask]
    row_tab = evaluate("таблица 2D", p_tab, yev, res, m=mev, is_prob=True)

    variants = {}
    for mode in ("raw", "rank"):
        if mode == "rank":
            Xtr_m = Xtr.copy(); Xca_m = Xca.copy(); Xev_m = Xev.copy()
            for k in range(atr.max() + 1):
                sub = np.ascontiguousarray(Xtr_m[atr == k]); rank_inplace(sub)
                Xtr_m[atr == k] = sub; del sub
            rank_inplace(Xca_m); rank_inplace(Xev_m)
        else:
            Xtr_m, Xca_m, Xev_m = Xtr, Xca, Xev
        for ks, kname in ((keep, "все"), (keep30, "дрейф<0.3")):
            A = np.ascontiguousarray(Xtr_m[:, ks])
            Bc = np.ascontiguousarray(Xca_m[:, ks])
            B = np.ascontiguousarray(Xev_m[:, ks])
            print(f"\n== признаки {mode}/{kname} ({len(ks)} шт) ==")
            for mn in ("логрегрессия", "бустинг"):
                if mn == "бустинг":
                    mdl = fit_gbm(A, ytr, atr, atr.max() + 1, threads=args.threads)
                else:
                    mdl = fit_lr(A, ytr, atr, atr.max() + 1)
                cal = platt(score(mdl, Bc), yca)
                s = score(mdl, B)
                nm = f"{mn} {mode}/{kname}"
                evaluate(nm, s, yev, res, cal=cal, m=mev)
                variants[nm] = (s, cal)
            del A, B, Bc
        if mode == "rank":
            del Xtr_m, Xca_m, Xev_m

    print("\n== смесь логрегрессии и бустинга в вероятностях ==")
    def prob(nm):
        s, cal = variants[nm]
        return 1 / (1 + np.exp(-(cal[0] * s + cal[1])))
    best = max((k for k in variants if k.startswith("бустинг")), key=lambda k: res[k]["auc"])
    mate = best.replace("бустинг", "логрегрессия")
    print(f"  члены: {best} + {mate}")
    for w in (0.3, 0.5, 0.7):
        evaluate(f"смесь {w:.1f}*лог+{1 - w:.1f}*буст", w * prob(mate) + (1 - w) * prob(best),
                 yev, res, m=mev, is_prob=True)

    print("\n== дальний перенос: обучение только на ИЮЛЬСКИХ якорях ==")
    far = np.isin(atr, [0, 1])
    A = np.ascontiguousarray(Xtr[far][:, keep])
    mdl = fit_gbm(A, ytr[far], atr[far], 2, threads=args.threads)
    cal = platt(score(mdl, np.ascontiguousarray(Xca[:, keep])), yca)
    evaluate("бустинг raw/все только июль", score(mdl, np.ascontiguousarray(Xev[:, keep])),
             yev, res, cal=cal, m=mev)
    del A

    res["_meta"] = {"fit": [str(a) for a in fit_anchors], "cal": str(cal_anchor),
                    "eval": str(EVAL_ANCHOR),
                    "n_fit": int(len(ytr)), "n_eval": int(len(yev)),
                    "rate_fit": float(ytr.mean()), "rate_eval": float(yev.mean()),
                    "kept": len(keep), "kept30": len(keep30),
                    "dropped": [c for c, _ in drop],
                    "table": row_tab}
    (OUT / "silence_eval.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(f"\nзаписан {OUT / 'silence_eval.json'}")


# ------------------------------------------------------------- финальная сборка
P_LEVEL_OLD = 0.030843


def stage_final(args):
    from subs import MEASURED, lp, novelty, span_matrix
    C = build_cumsum()
    fit_anchors, cal_anchor = EVAL_TRAIN[:-1], EVAL_TRAIN[-1]
    cols = feature_cols(load_anchor(fit_anchors[0]))
    print(f"обучение на {[str(a) for a in fit_anchors]}")
    Xtr, ytr, atr = load_block(fit_anchors, cols, C, "raw")
    keep, drop, rdrift = pick_cols(Xtr, atr, cols)
    keep = np.array([j for j in keep if rdrift[j] <= 0.30])
    print(f"признаков в модели {len(keep)} из {len(cols)}")
    for k in range(atr.max() + 1):
        sub = np.ascontiguousarray(Xtr[atr == k]); rank_inplace(sub); Xtr[atr == k] = sub
    A = np.ascontiguousarray(Xtr[:, keep]); del Xtr

    print(f"калибровка наклона на {cal_anchor}")
    Xca, yca, _ = load_block([cal_anchor], cols, C, "rank")
    Bc = np.ascontiguousarray(Xca[:, keep]); del Xca
    print(f"честная проверка на {EVAL_ANCHOR}")
    Xev, yev, _ = load_block([EVAL_ANCHOR], cols, C, "rank")
    Be = np.ascontiguousarray(Xev[:, keep]); del Xev
    ev_mask = sel_mask(C, EVAL_ANCHOR)
    mev = m_proxy(EVAL_ANCHOR, ev_mask)

    print(f"тестовый якорь {TEST_ANCHOR}")
    # все 250000 удовлетворяют условию отбора по построению — то же население, что на
    # обучающих якорях после sel_mask; проверяем, а не полагаемся на слово
    assert sel_mask(C, TEST_ANCHOR).all(), "на тестовом якоре население НЕ совпадает с отбором"
    dte = load_anchor(TEST_ANCHOR)
    Bt = np.ascontiguousarray(dte.select(cols).to_numpy().astype(np.float32)[:, keep])
    del dte
    rank_inplace(Bt)

    pe, pt = {}, {}
    for mn in ("логрегрессия", "бустинг"):
        mdl = fit_gbm(A, ytr, atr, atr.max() + 1, threads=args.threads) if mn == "бустинг" \
            else fit_lr(A, ytr, atr, atr.max() + 1)
        cal = platt(score(mdl, Bc), yca)
        pe[mn] = 1 / (1 + np.exp(-(cal[0] * score(mdl, Be) + cal[1])))
        pt[mn] = 1 / (1 + np.exp(-(cal[0] * score(mdl, Bt) + cal[1])))
        print(f"  {mn}: наклон Платта {cal[0]:.4f}, сдвиг {cal[1]:.4f}, "
              f"среднее p на тесте {pt[mn].mean():.5f}")
    del A, Bc, Be, Bt

    p_ev = 0.5 * pe["логрегрессия"] + 0.5 * pe["бустинг"]
    p_te = 0.5 * pt["логрегрессия"] + 0.5 * pt["бустинг"]

    # ----- честная проверка на отложенном чистом якоре, против моей таблицы
    res = {}
    tab = old_table_fit(C, fit_anchors)
    p_tab = old_table_apply(tab, EVAL_ANCHOR)[ev_mask]
    print("\n== честный замер на " + str(EVAL_ANCHOR) + " ==")
    for nm in ("логрегрессия", "бустинг"):
        evaluate(nm, sig_level(pe[nm], yev.mean()), yev, res, m=mev, is_prob=True)

    # отношение выигрышей с интервалом (бутстрап по пользователям).
    # ВАЖНО: c^2/q не зависит от масштаба p, поэтому это чистая мера ВЫРАВНИВАНИЯ
    # формы, не зависящая от того, какой уровень p мы потом выберем.
    rng = np.random.default_rng(7)
    rr = []
    pm_a = p_ev * (yev.mean() / p_ev.mean())          # общий уровень, умножением
    pm_b = p_tab * (yev.mean() / p_tab.mean())        # — у обеих оценок одинаково
    ga0, ca0, qa0 = gain_proxy(pm_a, yev, mev)
    gb0, cb0, qb0 = gain_proxy(pm_b, yev, mev)
    for _ in range(200):
        i = rng.integers(0, len(yev), len(yev))
        rr.append(gain_proxy(pm_a[i], yev[i], mev[i])[0] / gain_proxy(pm_b[i], yev[i], mev[i])[0])
    rr = np.array(rr)
    ratio = ga0 / gb0
    lo_r, hi_r = float(np.percentile(rr, 5)), float(np.percentile(rr, 95))
    print(f"\nна честном якоре: c модель {ca0:.6f} / таблица {cb0:.6f}, "
          f"q модель {qa0:.6f} / таблица {qb0:.6f}")
    print(f"отношение выигрышей модель/таблица: {ratio:.3f}  бутстрап [{lo_r:.3f}, {hi_r:.3f}]")

    # ----- направление на тестовом якоре
    print(f"\nсырое p модели на тесте: среднее {p_te.mean():.5f} разброс {p_te.std():.5f}")
    p_use = sig_level(p_te, P_LEVEL_OLD)
    print(f"уровень приведён к старому {P_LEVEL_OLD}: среднее {p_use.mean():.5f} "
          f"разброс {p_use.std():.5f}")

    # НОРМИРОВКА НАПРАВЛЕНИЯ. Уровень p не определён ни одним локальным замером —
    # он лишь перепараметризует силу, — поэтому направление приводится к ТОМУ ЖЕ
    # РАЗМЕРУ q, что у старого. Тогда «сила 0.9» означает ровно то же физическое

    # Оптимальная сила при таком нормировании. Выигрыш = c^2/q не зависит от масштаба,
    # значит отношение выигрышей = (c_нов/c_стар)^2 при общем q, откуда
    # s* = s*_старое * sqrt(отношение выигрышей).
    s_opt = 0.929 * float(np.sqrt(ratio))
    s_lo, s_hi = 0.929 * float(np.sqrt(lo_r)), 0.929 * float(np.sqrt(hi_r))
    print(f"оптимальная сила при этом нормировании: {s_opt:.3f} [{s_lo:.3f}, {s_hi:.3f}] "
          f"(у старого направления было 0.929)")

    print(f"корреляция p с m: на тесте {np.corrcoef(p_use, m)[0, 1]:.4f}, "
          f"на честном якоре с прокси {np.corrcoef(p_ev, mev)[0, 1]:.4f} "
          f"(проверка пригодности прокси)")

    print(f"итог: среднее {new.mean():.4f} разброс {new.std():.4f}, "
          f"клипуется в ноль {(new <= 0).sum()} строк")

    (OUT / "silence_final.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(f"записан {OUT / 'silence_final.json'}")


def sig_level(p, rate):
    """Тот же логит, сдвинутый до среднего уровня rate (уровень задаётся снаружи)."""
    lg = np.log(np.clip(p, 1e-12, 1 - 1e-12) / (1 - np.clip(p, 1e-12, 1 - 1e-12)))
    return 1 / (1 + np.exp(-recal_intercept(lg, rate)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="eval", choices=["eval", "final"])
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--strength", type=float, default=0.9)
    args = ap.parse_args()
    if args.stage == "eval":
        stage_eval(args)
    else:
        stage_final(args)


if __name__ == "__main__":
    main()

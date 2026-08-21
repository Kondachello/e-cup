"""Ночной вал-онли скрининг: какие варианты ridge-корректора остатка бленда устойчивее.

Контекст: ridge-стек на 117 «чистых» колонках дал честный OOF +0.000595 (эпоха
1.665647), но на LB перенёсся с коэффициентом 0.307 (передоз 3.3x). Гипотеза:
непереносимость — от концентрации выигрыша в хвостах (киты/нули) и от вал-специфичной
части сигнала. Скринятся вал-онли варианты по трём осям: (а) честный OOF-выигрыш,
(б) стабильность по 5 фолд-сидам, (в) доля выигрыша из топ-1% юзеров (прокси
непереносимости) + средняя межсидовая корреляция OOF-поправок.

Варианты (протокол везде тот из exp_resid_ridge.py: OOF по юзерам, 5 фолдов,
альфа на внутреннем 80/20 сплите обучающего фолда, grid 1e0..1e5):
  base   — точное воспроизведение вчерашнего (сид 0 обязан дать +0.000595)
  a10    — альфа, выбранная внутренним сплитом, x10
  a100   — то же, x100
  cnt    — только счётчиковые/рекенси-признаки (без gmv*/hv*/денежных), тот же протокол
  win    — винзоризация остатка-таргета на 1%/99% внутри обучающего фолда
  hub    — Хубер вместо квадрата: IRLS 4 итерации, delta=1.345*sigma_MAD, альфа базовая
  rank   — ridge на РАНГАХ признаков (average-ранги по 250k, затем стандартизация)

НИКАКИХ сабмитов, никаких записей вне work/reports/night_*. Только numpy-алгебра.

Запуск: USE_V2=1 USE_V3=1 .venv/bin/python work/reports/night_corrector_variants.py
Артефакт: work/reports/night_corrector_variants.json
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
from scipy.stats import rankdata

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from common import REPORTS_DIR, ROOT, VAL_ANCHOR, feature_cols, load_anchor  # noqa: E402
from exp_resid_ridge import ALPHAS, FOLDS  # noqa: E402
from margin import score  # noqa: E402

SEEDS = [0, 1, 2, 3, 4]
HUBER_ITERS = 4
MONEY_TOKENS = ("gmv", "hv50", "hv200", "hv1000")  # денежные и денежно-определённые


def prep_matrix(val: pl.DataFrame, cols: list[str]) -> np.ndarray:
    X = val.select(cols).to_numpy().astype(np.float64)
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)


def standardize(Xr: np.ndarray) -> np.ndarray:
    mu, sd = Xr.mean(0), Xr.std(0)
    sd[sd == 0] = 1.0
    return np.column_stack([(Xr - mu) / sd, np.ones(len(Xr))])


def rank_transform(Xr: np.ndarray) -> np.ndarray:
    """Average-ранги (0..1): без ординального шума на связках нулей."""
    n = len(Xr)
    out = np.empty_like(Xr)
    for j in range(Xr.shape[1]):
        out[:, j] = (rankdata(Xr[:, j], method="average") - 1.0) / (n - 1.0)
    return out


def winsor(x: np.ndarray, ref: np.ndarray) -> np.ndarray:
    lo, hi = np.quantile(ref, [0.01, 0.99])
    return np.clip(x, lo, hi)


def family_oof(X: np.ndarray, r: np.ndarray, folds: np.ndarray,
               rng: np.random.Generator, variants: list[str]) -> tuple[dict, dict]:
    """Один проход по фолдам считает все варианты семейства на данной матрице X.

    Порядок потребления rng байт-в-байт как в exp_resid_ridge.ridge_oof:
    одна перестановка на inner-сплит каждого фолда, больше ничего — сид 0 базы
    обязан воспроизвести вчерашние +0.000595.
    """
    p = X.shape[1]
    oofs = {v: np.zeros_like(r) for v in variants}
    alphas: dict[str, list[float]] = {v: [] for v in variants}
    eye = np.eye(p)
    for f in range(FOLDS):
        tr, te = folds != f, folds == f
        Xt, rt = X[tr], r[tr]
        inner = rng.permutation(len(rt)) < int(0.8 * len(rt))
        Xin, rin = Xt[inner], rt[inner]
        Xev, rev = Xt[~inner], rt[~inner]
        G_in = Xin.T @ Xin
        b_in = Xin.T @ rin
        # выбор альфы: квадратичная задача, оценка на сыром остатке (протокол базы)
        best = (9e9, ALPHAS[0])
        for al in ALPHAS:
            w = np.linalg.solve(G_in + al * eye, b_in)
            best = min(best, (float(np.mean((Xev @ w - rev) ** 2)), al))
        al = best[1]

        G = Xt.T @ Xt
        b = Xt.T @ rt
        w_base = np.linalg.solve(G + al * eye, b)

        for v in variants:
            if v in ("base", "cnt", "rank", "cnt_rank"):
                w, al_used = w_base, al
            elif v == "a10":
                al_used = al * 10.0
                w = np.linalg.solve(G + al_used * eye, b)
            elif v == "a100":
                al_used = al * 100.0
                w = np.linalg.solve(G + al_used * eye, b)
            elif v == "win":
                # отдельный выбор альфы: fit на винзоризованном (границы по inner-train),
                # оценка на сыром — цель развёртывания сырая
                b_in_w = Xin.T @ winsor(rin, rin)
                bw = (9e9, ALPHAS[0])
                for a2 in ALPHAS:
                    ww = np.linalg.solve(G_in + a2 * eye, b_in_w)
                    bw = min(bw, (float(np.mean((Xev @ ww - rev) ** 2)), a2))
                al_used = bw[1]
                w = np.linalg.solve(G + al_used * eye, Xt.T @ winsor(rt, rt))
            elif v in ("hub", "hub95"):
                # hub: классика delta=1.345*sigma_MAD; hub95: мягкий, delta=q95(|e|) —
                # давит только экстремальные 5% остатков (адресно киты/нули)
                al_used = al
                w = w_base.copy()
                for _ in range(HUBER_ITERS):
                    e = Xt @ w - rt
                    if v == "hub":
                        med = np.median(e)
                        sigma = 1.4826 * np.median(np.abs(e - med))
                        delta = 1.345 * max(sigma, 1e-9)
                    else:
                        delta = max(float(np.quantile(np.abs(e), 0.95)), 1e-9)
                    wt = np.minimum(1.0, delta / np.maximum(np.abs(e), 1e-12))
                    Xw = Xt * wt[:, None]
                    w = np.linalg.solve(Xt.T @ Xw + al_used * eye, Xw.T @ rt)
            else:
                raise ValueError(v)
            oofs[v][te] = X[te] @ w
            alphas[v].append(float(al_used))
    return oofs, alphas


def main():
    assert os.environ.get("USE_V2") and os.environ.get("USE_V3"), "нужны USE_V2=1 USE_V3=1"
    t0 = time.time()
    pack = pl.read_parquet(ROOT / "work" / "preds_pack" / "val_preds.parquet").sort("user_id")
    uid = pack["user_id"].to_numpy()
    ly = np.log1p(np.clip(pack["target"].to_numpy().astype(np.float64), 0, None))
    lb = pack["blend"].to_numpy().astype(np.float64)
    sb = score(lb, ly)
    resid = ly - lb
    eb = lb - ly
    n = len(ly)
    assert abs(sb - 1.665647) < 1e-5, f"эпоха уехала: {sb:.6f}"

    bad = {x.split(" ")[0] for x in json.loads(
        (REPORTS_DIR / "maxback_affected_cols.json").read_text())}
    val = load_anchor(VAL_ANCHOR).sort("user_id")
    assert np.array_equal(val["user_id"].to_numpy(), uid)
    cols = [c for c in feature_cols(val) if c not in bad]
    cnt_cols = [c for c in cols if not any(t in c for t in MONEY_TOKENS)]
    print(f"эталон {sb:.6f}; колонок всего {len(cols)}, счётчиковых/рекенси {len(cnt_cols)}")

    k1, k01 = int(0.01 * n), int(0.001 * n)

    def conc(oof: np.ndarray) -> tuple[float, float]:
        d = eb ** 2 - (eb + oof) ** 2          # повользовательский вклад в MSE-выигрыш
        tot = float(d.sum())
        if tot <= 0:
            return float("nan"), float("nan")
        ds = np.sort(d)[::-1]
        return float(ds[:k1].sum() / tot), float(ds[:k01].sum() / tot)

    def evaluate(oof: np.ndarray) -> dict:
        t1, t01 = conc(oof)
        oc = oof - oof.mean()                  # как в деплое mdl_gypsum: поправка центрируется
        t1c, t01c = conc(oc)
        return {
            "gain": float(sb - score(lb + oof, ly)),
            "gain_c": float(sb - score(lb + oc, ly)),
            "top1_share": t1, "top01_share": t01,
            "top1_share_c": t1c, "top01_share_c": t01c,
            "oof_sd": float(oof.std()),
        }

    
    families = [
        ("std", lambda: standardize(prep_matrix(val, cols)),
         ["base", "a10", "a100", "win", "hub", "hub95"]),
        ("cnt", lambda: standardize(prep_matrix(val, cnt_cols)), ["cnt"]),
        ("rank", lambda: standardize(rank_transform(prep_matrix(val, cols))), ["rank"]),
        ("cntrank", lambda: standardize(rank_transform(prep_matrix(val, cnt_cols))),
         ["cnt_rank"]),
    ]

    res: dict[str, dict] = {}
    oof_store: dict[str, list[np.ndarray]] = {}
    for fam_name, build, variants in families:
        X = build()
        for seed in SEEDS:
            rng = np.random.default_rng(seed)
            folds = rng.permutation(n) % FOLDS
            oofs, alphas = family_oof(X, resid, folds, rng, variants)
            for v in variants:
                ev = evaluate(oofs[v])
                res.setdefault(v, {"seeds": {}, "alphas": {}})
                res[v]["seeds"][seed] = ev
                res[v]["alphas"][seed] = alphas[v]
                oof_store.setdefault(v, []).append(oofs[v])
            print(f"[{time.time() - t0:6.1f}s] {fam_name} seed {seed}: " +
                  "  ".join(f"{v} {res[v]['seeds'][seed]['gain']:+.6f}" for v in variants))
        del X

    for v, d in res.items():
        gains = np.array([d["seeds"][s]["gain"] for s in SEEDS])
        gains_c = np.array([d["seeds"][s]["gain_c"] for s in SEEDS])
        t1 = np.array([d["seeds"][s]["top1_share"] for s in SEEDS])
        t01 = np.array([d["seeds"][s]["top01_share"] for s in SEEDS])
        t1c = np.array([d["seeds"][s]["top1_share_c"] for s in SEEDS])
        sds = np.array([d["seeds"][s]["oof_sd"] for s in SEEDS])
        O = np.stack(oof_store[v])
        C = np.corrcoef(O)
        cross = float(C[np.triu_indices(len(SEEDS), 1)].mean())
        d["summary"] = {
            "gain_mean": float(gains.mean()), "gain_std": float(gains.std(ddof=1)),
            "gain_min": float(gains.min()), "gain_seed0": float(gains[0]),
            "gain_c_mean": float(gains_c.mean()), "gain_c_std": float(gains_c.std(ddof=1)),
            "top1_share_mean": float(np.nanmean(t1)), "top01_share_mean": float(np.nanmean(t01)),
            "top1_share_c_mean": float(np.nanmean(t1c)),
            "oof_sd_mean": float(sds.mean()), "cross_seed_corr": cross,
        }

    out = {
        "reference_blend": round(sb, 6), "n_users": n,
        "n_cols": len(cols), "n_cnt_cols": len(cnt_cols),
        "cnt_cols": cnt_cols,
        "money_tokens": list(MONEY_TOKENS),
        "seeds": SEEDS, "huber_iters": HUBER_ITERS,
        "protocol": "exp_resid_ridge: OOF по юзерам, 5 фолдов, альфа на внутреннем 80/20, grid 1e0..1e5",
        "lb_context": {"yesterday_oof": 0.000595, "lb_transfer_coef": 0.307},
        "variants": res,
        "runtime_sec": round(time.time() - t0, 1),
    }
    p = REPORTS_DIR / "night_corrector_variants.json"
    p.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(f"JSON: {p}  ({out['runtime_sec']}s)")

    hdr = (f"{'вариант':<9}{'OOF сид0':>10}{'mean±std (5 сидов)':>22}{'min':>10}"
           f"{'центр.':>10}{'top1%':>8}{'top0.1%':>9}{'xseed r':>9}{'sd(oof)':>9}")
    print(hdr)
    print("-" * len(hdr))
    for v in ["base", "a10", "a100", "cnt", "win", "hub", "hub95", "rank", "cnt_rank"]:
        s = res[v]["summary"]
        print(f"{v:<9}{s['gain_seed0']:+10.6f}"
              f"{s['gain_mean']:+13.6f}±{s['gain_std']:.6f}"
              f"{s['gain_min']:+10.6f}{s['gain_c_mean']:+10.6f}"
              f"{s['top1_share_mean']:8.3f}"
              f"{s['top01_share_mean']:9.3f}{s['cross_seed_corr']:9.4f}"
              f"{s['oof_sd_mean']:9.5f}")


if __name__ == "__main__":
    main()

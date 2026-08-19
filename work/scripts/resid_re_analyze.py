"""Анализ персонального случайного эффекта в остатках (см. resid_re.py).

Шаг 1 (главный): устойчив ли per-user остаток на НЕПЕРЕСЕКАЮЩИХСЯ группах срезов.
Ловушка: шаг срезов 7 дней при окне таргета 30 дней -> у соседних срезов целевые окна
пересекаются на 23 дня, и остатки коррелируют механически. Группы разносятся на >=30 дней
по якорю, тогда целевые окна не пересекаются вообще.

Второй контроль: сегментное смещение калибровочной кривой (модель систематически
переоценивает низкие прогнозы) тоже даст корреляцию, но это НЕ персональный эффект —
его снимает обычная калибровка. Поэтому всё считается дважды: на сырых остатках
и на остатках, из которых убрано E[r | бин прогноза] внутри каждого среза.

Запуск: POLARS_MAX_THREADS=3 .venv/bin/python work/scripts/resid_re_analyze.py
"""
from __future__ import annotations

import json
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))
from common import PREDS_DIR, REPORTS_DIR

RES = PREDS_DIR / "resid_re"
ANCHORS = [date(2025, 7, 2) + timedelta(days=7 * i) for i in range(24)]
N_BINS = 40


def load_residuals():
    """R[user, anchor] сырых остатков ly-lp и LP[user, anchor] прогнозов."""
    parts = [pl.read_parquet(p) for p in sorted(RES.glob("fold*.parquet"))]
    d = pl.concat(parts)
    uid = np.sort(d["user_id"].unique().to_numpy())
    idx = {u: i for i, u in enumerate(uid.tolist())}
    R = np.full((len(uid), len(ANCHORS)), np.nan, np.float32)
    LP = np.full((len(uid), len(ANCHORS)), np.nan, np.float32)
    for j, a in enumerate(ANCHORS):
        s = d.filter(pl.col("anchor") == a.isoformat()).sort("user_id")
        if s.height == 0:
            continue
        rows = np.fromiter((idx[u] for u in s["user_id"].to_list()), np.int64, s.height)
        lp = s["lp"].to_numpy().astype(np.float32)
        R[rows, j] = s["ly"].to_numpy().astype(np.float32) - lp
        LP[rows, j] = lp
    return uid, R, LP


def decalibrate(R: np.ndarray, LP: np.ndarray, bins: int = N_BINS) -> np.ndarray:
    """Убрать внутри каждого среза среднее остатка по бину прогноза (кривую калибровки)."""
    out = np.array(R, dtype=np.float64)
    for j in range(R.shape[1]):
        ok = ~np.isnan(R[:, j])
        if ok.sum() == 0:
            continue
        lp, r = LP[ok, j].astype(np.float64), out[ok, j]
        qs = np.quantile(lp, np.linspace(0, 1, bins + 1))
        qs[0] -= 1e-9
        qs[-1] += 1e-9
        b = np.clip(np.searchsorted(qs, lp, side="left") - 1, 0, bins - 1)
        means = np.zeros(bins)
        cnt = np.bincount(b, minlength=bins)
        sums = np.bincount(b, weights=r, minlength=bins)
        nz = cnt > 0
        means[nz] = sums[nz] / cnt[nz]
        out[ok, j] = r - means[b]
    return out


def gmean(R: np.ndarray, cols: list[int]) -> np.ndarray:
    return np.nanmean(R[:, cols], axis=1)


def corr(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    return float(np.corrcoef(a[m], b[m])[0, 1])


def spearman(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    ra = np.argsort(np.argsort(a[m])).astype(np.float64)
    rb = np.argsort(np.argsort(b[m])).astype(np.float64)
    return float(np.corrcoef(ra, rb)[0, 1])


def idx_of(dates):
    return [ANCHORS.index(d) for d in dates]


def rng_anchors(lo: str, hi: str):
    a, b = date.fromisoformat(lo), date.fromisoformat(hi)
    return [d for d in ANCHORS if a <= d <= b]


GROUPS = {
    "jul_aug": ("2025-07-02", "2025-08-27"),
    "oct_nov": ("2025-10-01", "2025-11-26"),
    "jul": ("2025-07-02", "2025-07-30"),
    "nov_dec": ("2025-11-05", "2025-12-10"),
    "f1a_sep_oct": ("2025-09-24", "2025-10-08"),
    "f1b_nov_dec": ("2025-11-12", "2025-12-10"),
    "f2a_jul": ("2025-07-02", "2025-07-30"),
    "f2b_sep": ("2025-09-03", "2025-09-17"),
    "ctl_ovl_a": ("2025-10-01", "2025-10-15"),
    "ctl_ovl_b": ("2025-10-22", "2025-11-05"),
}
PAIRS = [
    ("jul_aug", "oct_nov", 35, "ГЛАВНАЯ: июль-август (fold2) против октября-ноября (fold1)"),
    ("jul", "nov_dec", 98, "далёкая пара, разные модели"),
    ("f1a_sep_oct", "f1b_nov_dec", 35, "внутри fold1 (одна модель)"),
    ("f2a_jul", "f2b_sep", 35, "внутри fold2 (одна модель)"),
    ("ctl_ovl_a", "ctl_ovl_b", 7, "КОНТРОЛЬ: окна пересекаются -> механическая корреляция"),
]


def main():
    uid, R, LP = load_residuals()
    have = [j for j in range(len(ANCHORS)) if np.isfinite(R[:, j]).any()]
    print(f"юзеров {len(uid)}, срезов с остатками {len(have)}/{len(ANCHORS)}")
    Rc = R - np.nanmean(R, axis=0, keepdims=True)          # только центрирование среза
    Rd = decalibrate(R, LP)                                  # + снятие кривой калибровки
    print(f"sd остатка (сырой) {np.nanstd(R):.4f}, после декалибровки {np.nanstd(Rd):.4f}")

    out = {"n_users": int(len(uid)), "pairs": {}}
    print(f"\n{'пара':38s} {'дней':>5s} {'corr_raw':>9s} {'corr_decal':>11s} {'spearman':>9s}")
    for ga, gb, gap, note in PAIRS:
        ia, ib = idx_of(rng_anchors(*GROUPS[ga])), idx_of(rng_anchors(*GROUPS[gb]))
        if not (np.isfinite(R[:, ia]).any() and np.isfinite(R[:, ib]).any()):
            continue
        a_raw, b_raw = gmean(Rc, ia), gmean(Rc, ib)
        a_d, b_d = gmean(Rd, ia), gmean(Rd, ib)
        c_raw, c_d, sp = corr(a_raw, b_raw), corr(a_d, b_d), spearman(a_d, b_d)
        cov_d = float(np.nanmean((a_d - np.nanmean(a_d)) * (b_d - np.nanmean(b_d))))
        out["pairs"][f"{ga}|{gb}"] = dict(
            gap_days=gap, n_a=len(ia), n_b=len(ib), corr_raw=c_raw, corr_decal=c_d,
            spearman_decal=sp, cov_decal=cov_d,
            var_a=float(np.nanvar(a_d)), var_b=float(np.nanvar(b_d)), note=note)
        print(f"{ga+'|'+gb:38s} {gap:5d} {c_raw:9.4f} {c_d:11.4f} {sp:9.4f}   {note}")

    se = 1.0 / np.sqrt(len(uid))
    print(f"\nSE корреляции при n={len(uid)}: {se:.4f}")
    REPORTS_DIR.mkdir(exist_ok=True, parents=True)
    (REPORTS_DIR / "resid_re_corr.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    np.save(RES / "R_decal.npy", Rd.astype(np.float32))
    np.save(RES / "uid.npy", uid)
    print("сохранено: work/reports/resid_re_corr.json, work/preds/resid_re/R_decal.npy")


if __name__ == "__main__":
    main()

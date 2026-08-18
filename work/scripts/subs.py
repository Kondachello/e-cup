"""Устойчивая загрузка сабмитов: ищет файл сначала в submissions/, затем в
submissions/canonical/ (копии под исходными именами — при заливке файлы часто
переименовывают, и расчёты по каноническим именам ломались).

Использование:
    from subs import lp, uid_of, snapshot
"""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import polars as pl

SUB = Path("/Users/alexanderkondakov/ozon-cup/submissions")
CANON = SUB / "canonical"


def snapshot() -> int:
    """Сохранить копии всех сабмитов под текущими именами."""
    CANON.mkdir(exist_ok=True)
    n = 0
    for p in SUB.glob("*.csv"):
        t = CANON / p.name
        if not t.exists():
            shutil.copy2(p, t)
            n += 1
    return n


def _resolve(name: str) -> Path:
    p = SUB / name
    if p.exists():
        return p
    c = CANON / name
    if c.exists():
        return c
    raise FileNotFoundError(f"не найден {name} ни в submissions/, ни в canonical/")


def lp(name: str):
    """(user_id, log1p(predict)) с сортировкой по user_id."""
    d = pl.read_csv(_resolve(name), schema_overrides={"user_id": pl.Int64}).sort("user_id")
    col = "predict" if "predict" in d.columns else d.columns[1]
    return d["user_id"].to_numpy(), np.log1p(np.clip(d[col].to_numpy().astype(np.float64), 0, None))


def lp_pred(name: str):
    """log1p предсказаний модели из work/preds/NAME_test.parquet."""
    p = Path("/Users/alexanderkondakov/ozon-cup/work/preds") / f"{name}_test.parquet"
    d = pl.read_parquet(p).sort("user_id")
    return np.log1p(np.clip(d["pred"].to_numpy().astype(np.float64), 0, None))


def span_matrix(files: list[str], n: int):
    """Матрица измеренного базиса (константа + перечисленные файлы)."""
    return np.stack([np.ones(n)] + [lp(f)[1] for f in files])


def novelty(d: np.ndarray, Sp: np.ndarray):
    """Доля дисперсии направления, не объяснённая измеренным базисом, и сам остаток."""
    n = len(d)
    G = Sp @ Sp.T / n
    coef = np.linalg.solve(G + 1e-9 * np.eye(len(G)), Sp @ d / n)
    r = d - coef @ Sp
    return float((r ** 2).mean() / max((d ** 2).mean(), 1e-12)), r


MEASURED = ["A1_gram7_shift.csv", "A2_probe_s1_gmv.csv", "sub_blend_w1a.csv", "sub_twlog_probe.csv",
            "sub_c_cand.csv", "lbmix4_3way.csv"]

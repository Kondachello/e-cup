"""Experiment harness: standard data loading / scoring / saving for model agents.

Contract for every model experiment `NAME`:
  1. Train on anchors < VAL_ANCHOR, validate on VAL_ANCHOR (early stopping allowed).
  2. Save validation predictions -> work/preds/NAME_val.parquet  (user_id, pred)
  3. Retrain on train+val anchors (best_iter * 1.07 if early stopped), predict
     TEST_ANCHOR -> work/preds/NAME_test.parquet (user_id, pred)
  4. Append one line to work/reports/scores.tsv: NAME\tval_rmsle\tnotes
Predictions are raw GMV scale (>=0), NOT log.
"""
from __future__ import annotations

import os

try:
    import fcntl
except ImportError:      # Windows: нет flock, трек 5 упирался в это
    fcntl = None
import sys
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))
from common import (  # noqa: F401
    FEATURES_DIR, PREDS_DIR, REPORTS_DIR, TEST_ANCHOR, VAL_ANCHOR,
    feature_cols, load_anchor, rmsle, train_anchors,
)

PREDS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def available_train_anchors() -> list[date]:
    out = []
    for p in sorted(FEATURES_DIR.glob("anchor=*.parquet")):
        if "." in p.stem.split("=")[1]:
            continue
        a = date.fromisoformat(p.stem.split("=")[1])
        if a < VAL_ANCHOR:
            out.append(a)
    return out


def protocol_train_anchors(n: int = 14, stride: int = 14, source: str = "protocol") -> list[date]:
    """Обучающие якоря ПО ПРОТОКОЛУ, а не по содержимому каталога.

    `available_train_anchors()` сканирует каталог, и это трижды оказывалось решающей
    переменной невоспроизводимости: lag_tta подхватил 11 якорей вместо 9, когда рядом
    построили два лишних; wklin не сходился, пока не восстановили набор шага 7; в таблице
    ретрейна 25.08 дрейфовали РОВНО те два члена, чей набор берётся из каталога
    (`weak_an_d` +0.000093 и `weak_ft_recency` +0.000055), а всё, чей набор пинится
    протоколом, воспроизвелось побитово.

    Скверное свойство правила `[-N:]` по каталогу: добавление якоря не добавляет данных,
    а ВЫТЕСНЯЕТ старый новым и сжимает окно обучения — то есть посторонний файл молча
    меняет обучающую выборку каждой последующей модели.

    Здесь набор задаётся `train_anchors(n, stride)` — той же сеткой, что строит
    `build_features.py --preset all`. Недостающие файлы называются по имени, а не
    подменяются молча ближайшими. На каноническом каталоге результат совпадает с
    `available_train_anchors()`, поэтому поведение действующих прогонов не меняется.

    `source="disk"` возвращает историческое поведение — нужно, чтобы воспроизводить
    артефакты, собранные до этой правки.
    """
    if source == "disk":
        return available_train_anchors()
    want = train_anchors(n, stride)
    have = set(available_train_anchors())
    miss = [a for a in want if a not in have]
    if miss:
        print("ВНИМАНИЕ: протокол требует якорей, которых нет на диске: "
              + ", ".join(a.isoformat() for a in miss)
              + f"\n  собрать: build_features.py --anchors {','.join(a.isoformat() for a in miss)}",
              file=sys.stderr, flush=True)
    return [a for a in want if a in have]


def load_matrix(anchors: list[date], columns: list[str] | None = None) -> pl.DataFrame:
    dfs = [load_anchor(a, columns) for a in anchors]
    return pl.concat(dfs, how="vertical_relaxed")


def to_xy(df: pl.DataFrame, cols: list[str]):
    X = df.select(cols).to_numpy().astype(np.float32)
    y = df["target"].to_numpy().astype(np.float64)
    return X, y


def save_preds(name: str, split: str, user_ids: np.ndarray, preds: np.ndarray):
    preds = np.clip(np.asarray(preds, dtype=np.float64), 0, None)
    pl.DataFrame({"user_id": user_ids.astype(np.int64), "pred": preds}).write_parquet(
        PREDS_DIR / f"{name}_{split}.parquet"
    )
    # Происхождение рядом с прогнозом: набор якорей, включённые тиры, версии библиотек,
    # хеш train.parquet, коммит. Без этого артефакт не помнит, из чего собран, и вопрос
    # «почему не воспроизводится» неразрешим (см. work/scripts/provenance.py).
    # Стемпинг не имеет права уронить обучение, поэтому все ошибки глушатся внутри.
    try:
        from provenance import stamp
        stamp(name, split, PREDS_DIR)
    except Exception:
        pass


def note(**kv):
    """Сообщить провенансу ЭФФЕКТИВНЫЕ параметры прогона (см. provenance.note).

    Трейнеры зовут это, а не provenance напрямую: ошибка стемпинга не должна
    ронять обучение, и глушится она в одном месте.
    """
    try:
        from provenance import note as _n
        _n(**kv)
    except Exception:
        pass


def log_score(name: str, val_rmsle: float, notes: str = ""):
    path = REPORTS_DIR / "scores.tsv"
    with open(path, "a") as f:
        if fcntl:
            fcntl.flock(f, fcntl.LOCK_EX)
        f.write(f"{name}\t{val_rmsle:.6f}\t{notes}\n")
        # flush ДО снятия блокировки: f.write кладёт строку в буфер питона, а сброс на
        # диск раньше происходил при выходе из with, то есть уже ПОСЛЕ LOCK_UN — и
        # блокировка не защищала собственно запись.
        f.flush()
        os.fsync(f.fileno())
        if fcntl:
            fcntl.flock(f, fcntl.LOCK_UN)
    print(f"[SCORE] {name}: {val_rmsle:.6f} ({notes})", flush=True)

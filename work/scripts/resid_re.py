"""Персональный случайный эффект (random effect) в остатках модели.

Идея: одни и те же 250k юзеров есть во всех срезах, включая тест. Если у юзера
есть устойчивое личное смещение, которого не ловят признаки, оно должно быть видно
как корреляция средних остатков на НЕПЕРЕСЕКАЮЩИХСЯ по целевому окну группах срезов.

Здесь считаются только остатки (out-of-time, кросс-фит по половинам срезов):
  fold1: обучение на ранней половине (2025-07-02..09-17) -> предсказание поздней
  fold2: обучение на поздней половине (2025-09-24..12-10) -> предсказание ранней
Так у каждого среза остаток получен моделью, которая этот срез НЕ видела.

Рецепт чемпионский: lgb, objective tweedie (power 1.45) на log1p-таргете,
USE_V2/V3/V4. Ранняя остановка — по отложенным 10% ЮЗЕРОВ внутри обучающей половины.

Выход: work/preds/resid_re/{fold}.parquet со столбцами user_id, anchor, lp, ly.
Запуск: OMP_NUM_THREADS=2 POLARS_MAX_THREADS=3 USE_V2=1 USE_V3=1 USE_V4=1 \
        .venv/bin/python work/scripts/resid_re.py --threads 2
"""
from __future__ import annotations

import argparse
import gc
import os
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))
from common import PREDS_DIR, VAL_ANCHOR, feature_cols, load_anchor

# 24 недельных среза с полным покрытием тиров v2/v3/v4 (февральские 2025 без v2)
ANCHORS = [date(2025, 7, 2) + timedelta(days=7 * i) for i in range(24)]
EARLY, LATE = ANCHORS[:12], ANCHORS[12:]

CHAMP = dict(
    objective="tweedie", tweedie_variance_power=1.45, metric="rmse",
    learning_rate=0.04, num_leaves=255, min_data_in_leaf=300,
    feature_fraction=0.75, bagging_fraction=0.8, bagging_freq=1,
    lambda_l2=5.0, max_bin=127, seed=42, verbosity=-1,
)
OUT = PREDS_DIR / "resid_re"


def build_split(anchors, cols, es_uid: np.ndarray):
    """Поанкорная загрузка сразу в две преаллоцированные матрицы (fit / early-stop).

    Пик памяти = итоговые матрицы + один срез, без промежуточного vstack.
    """
    head = load_anchor(anchors[0], columns=["user_id"])
    n_per, n_f = head.height, len(cols)
    es_mask0 = np.isin(head["user_id"].to_numpy(), es_uid)
    n_es, n_tr = int(es_mask0.sum()), n_per - int(es_mask0.sum())
    Xtr = np.empty((n_tr * len(anchors), n_f), np.float32)
    Xes = np.empty((n_es * len(anchors), n_f), np.float32)
    ytr = np.empty(n_tr * len(anchors), np.float64)
    yes = np.empty(n_es * len(anchors), np.float64)
    for i, a in enumerate(anchors):
        df = load_anchor(a, columns=["user_id", "target"] + cols)
        assert df.height == n_per, f"{a}: {df.height} != {n_per}"
        m = np.isin(df["user_id"].to_numpy(), es_uid)
        Xa = df.select(cols).to_numpy().astype(np.float32)
        ya = np.log1p(df["target"].to_numpy().astype(np.float64))
        Xtr[i * n_tr:(i + 1) * n_tr] = Xa[~m]
        Xes[i * n_es:(i + 1) * n_es] = Xa[m]
        ytr[i * n_tr:(i + 1) * n_tr] = ya[~m]
        yes[i * n_es:(i + 1) * n_es] = ya[m]
        del df, Xa, ya
        gc.collect()
    return Xtr, ytr, Xes, yes


def run_fold(tag: str, tr_anchors, pr_anchors, cols, n_est: int, threads: int, seed: int):
    import lightgbm as lgb
    t0 = time.time()
    uni = load_anchor(tr_anchors[0], columns=["user_id"])["user_id"].to_numpy()
    rng = np.random.default_rng(777)
    es_uid = np.sort(rng.choice(uni, size=len(uni) // 10, replace=False))

    Xtr, ytr, Xes, yes = build_split(tr_anchors, cols, es_uid)
    print(f"[{tag}] fit {Xtr.shape}, es {Xes.shape}, load {time.time()-t0:.0f}s", flush=True)
    p = dict(CHAMP, num_threads=threads, seed=seed)
    dtr = lgb.Dataset(Xtr, ytr, free_raw_data=True)
    dva = lgb.Dataset(Xes, yes, reference=dtr, free_raw_data=True)
    m = lgb.train(p, dtr, num_boost_round=n_est, valid_sets=[dva],
                  callbacks=[lgb.early_stopping(200, verbose=False), lgb.log_evaluation(500)])
    best = m.best_iteration or n_est
    print(f"[{tag}] fitted, best_iter={best}, {time.time()-t0:.0f}s", flush=True)
    del Xtr, ytr, Xes, yes, dtr, dva
    gc.collect()

    rows = []
    for a in pr_anchors:
        df = load_anchor(a, columns=["user_id", "target"] + cols)
        Xa = df.select(cols).to_numpy().astype(np.float32)
        lp = np.clip(m.predict(Xa, num_iteration=best), 0, None)
        ly = np.log1p(df["target"].to_numpy().astype(np.float64))
        rows.append(pl.DataFrame({
            "user_id": df["user_id"].to_numpy().astype(np.int64),
            "anchor": np.full(len(lp), a.isoformat()),
            "lp": lp.astype(np.float32), "ly": ly.astype(np.float32),
        }))
        r = ly - lp
        print(f"[{tag}] {a} rmsle={float(np.sqrt((r**2).mean())):.4f} mean_r={float(r.mean()):+.4f}",
              flush=True)
        del df, Xa
        gc.collect()
    OUT.mkdir(parents=True, exist_ok=True)
    pl.concat(rows).write_parquet(OUT / f"{tag}.parquet")
    del m
    gc.collect()
    print(f"[{tag}] done in {time.time()-t0:.0f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--n-est", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--folds", type=str, default="1,2")
    args = ap.parse_args()
    os.environ["OMP_NUM_THREADS"] = str(args.threads)

    cols = feature_cols(load_anchor(ANCHORS[0]))
    print(f"{len(cols)} features; early {EARLY[0]}..{EARLY[-1]}, late {LATE[0]}..{LATE[-1]}",
          flush=True)
    folds = set(args.folds.split(","))
    if "1" in folds:
        run_fold("fold1_trainEARLY_predLATE", EARLY, LATE, cols, args.n_est, args.threads, args.seed)
    if "2" in folds:
        run_fold("fold2_trainLATE_predEARLY", LATE, EARLY, cols, args.n_est, args.threads, args.seed)


if __name__ == "__main__":
    main()

"""Mirror-window experiment: does config ranking change between "normal" and "March" windows?

Design (see work/reports/mirror_window.md):
  * One SHORT-FEATURE training pool (all feature windows <= 42d, recency clipped at 42),
    anchors 2025-07-23..2025-09-10 (7d stride, 8 anchors). Training targets span
    2025-07-24..2025-10-10.
  * Two families of held-out evaluation anchors, both >= 30d away from every training
    target window:
      MARCH  (backward transfer): 2025-02-06 / 2025-02-13 / 2025-02-20
              -> targets 2025-02-07..2025-03-22, i.e. the pre-March-8 gifting regime.
              2025-02-13 is the EXACT mirror of the test window (2026-02-14..03-15).
      NORMAL (forward transfer) : 2025-11-12 / 2025-12-31 / 2026-01-14
              -> targets Nov-Dec, Jan (NY), and the real competition val window.
  * Same rows, same features, same #rounds for every config -> only the config differs.

Metrics per (config, anchor, rounds, population):
  rmsle  = sqrt(mean(r^2)),  r = log1p(y) - log1p(pred)
  bias   = mean(r)                    (global level error, log space)
  shape  = std(r)                     (= RMSLE after optimal global log-shift)
  rmsle^2 = bias^2 + shape^2

Populations: "act42" = users active within 42d of the anchor (the test anchor is 100%
act42, the mirror anchor only 79%), and "full" = whole 250K universe.

Results -> work/reports/mirror_window_results.jsonl (one row per config, appended
immediately after each fit so partial runs survive).
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import date
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))
from common import REPORTS_DIR, user_universe  # noqa: E402
from train_feb_specialist import FEATS, prep, to_X  # noqa: E402

NTHREAD = int(os.environ["OMP_NUM_THREADS"])
OUT = REPORTS_DIR / "mirror_window_results.jsonl"

TRAIN_ANCHORS_8 = [date(2025, 7, 23), date(2025, 7, 30), date(2025, 8, 6), date(2025, 8, 13),
                   date(2025, 8, 20), date(2025, 8, 27), date(2025, 9, 3), date(2025, 9, 10)]
TRAIN_ANCHORS_4 = TRAIN_ANCHORS_8[-4:]
REF_ANCHOR = TRAIN_ANCHORS_8[-1]

MARCH_EVAL = [date(2025, 2, 6), date(2025, 2, 13), date(2025, 2, 20)]
NORMAL_EVAL = [date(2025, 11, 12), date(2025, 12, 31), date(2026, 1, 14)]
EVAL_ANCHORS = MARCH_EVAL + NORMAL_EVAL

N_USERS = 100_000          # subsample -> 8 anchors x 100k = 800k rows (budget <= 1M)
N_EVAL_USERS = 150_000     # eval subsample (paired comparisons -> plenty precise)
ROUND_GRID = [200, 400, 700, 1000]
REC_CLIP = 42.0

CONFIGS = {
    #  name                 objective   vp     n_anch  tau    seed
    "base_tw13_n8":       dict(obj="tweedie", vp=1.30, n=8, tau=0.0,   seed=42),
    "tw12_n8":            dict(obj="tweedie", vp=1.20, n=8, tau=0.0,   seed=42),
    "tw145_n8":           dict(obj="tweedie", vp=1.45, n=8, tau=0.0,   seed=42),
    "mse_n8":             dict(obj="regression", vp=None, n=8, tau=0.0, seed=42),
    "tw13_n8_tau150":     dict(obj="tweedie", vp=1.30, n=8, tau=150.0, seed=42),
    "tw13_n4":            dict(obj="tweedie", vp=1.30, n=4, tau=0.0,   seed=42),
    "base_tw13_n8_s1337": dict(obj="tweedie", vp=1.30, n=8, tau=0.0,   seed=1337),
}


def sub_users(n: int, seed: int) -> pl.Series:
    u = user_universe()["user_id"]
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(u), size=min(n, len(u)), replace=False)
    return u.gather(np.sort(idx))


def load_train(anchors: list[date], users: pl.Series):
    Xs, ys, ws_anchor = [], [], []
    for a in anchors:
        df = prep(a, need_target=True).filter(pl.col("user_id").is_in(users))
        Xs.append(to_X(df))
        ys.append(np.log1p(df["target"].to_numpy().astype(np.float64)))
        ws_anchor.append(np.full(len(df), (REF_ANCHOR - a).days, dtype=np.float64))
        del df
    return (np.concatenate(Xs), np.concatenate(ys), np.concatenate(ws_anchor))


def load_eval(users: pl.Series):
    """cache: anchor -> (X, y, act42_mask)"""
    cache = {}
    for a in EVAL_ANCHORS:
        df = prep(a, need_target=True).filter(pl.col("user_id").is_in(users))
        X = to_X(df)
        y = df["target"].to_numpy().astype(np.float64)
        # rec_active was already clipped/filled at REC_CLIP by prep(); "< 42" == active in window
        m = (df["rec_active"].to_numpy() < REC_CLIP)
        cache[a] = (X, y, m)
        print(f"  eval {a}: n={len(y)} act42={m.mean():.3f} tgt_mean={y.mean():.1f}", flush=True)
        del df
    return cache


def metric_rows(y, p, mask):
    out = {}
    r_all = np.log1p(np.clip(y, 0, None)) - np.log1p(np.clip(p, 0, None))
    for pop, r in (("full", r_all), ("act42", r_all[mask])):
        out[pop] = dict(rmsle=float(np.sqrt((r ** 2).mean())),
                        bias=float(r.mean()), shape=float(r.std()))
    return out


def staged_raw(model, X, grid):
    """raw scores at each cumulative round count, one pass over the trees."""
    acc = np.zeros(len(X), dtype=np.float64)
    prev = 0
    out = {}
    for k in grid:
        acc = acc + model.predict(X, start_iteration=prev, num_iteration=k - prev, raw_score=True)
        out[k] = acc.copy()
        prev = k
    return out


def run(name: str, cfg: dict, Xtr8, ytr8, gap8, evalcache):
    import lightgbm as lgb
    t0 = time.time()
    if cfg["n"] == 8:
        X, y, gap = Xtr8, ytr8, gap8
    else:
        keep = gap8 <= (REF_ANCHOR - TRAIN_ANCHORS_4[0]).days
        X, y, gap = Xtr8[keep], ytr8[keep], gap8[keep]
    n_rows = int(len(y))
    w = np.exp(-gap / cfg["tau"]) if cfg["tau"] else None

    params = dict(objective=cfg["obj"], metric="rmse", learning_rate=0.05,
                  num_leaves=127, min_data_in_leaf=300, feature_fraction=0.75,
                  bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0, max_bin=127,
                  num_threads=NTHREAD, seed=cfg["seed"], verbosity=-1)
    if cfg["obj"] == "tweedie":
        params["tweedie_variance_power"] = cfg["vp"]
    ds = lgb.Dataset(X, y, weight=w, free_raw_data=True)
    m = lgb.train(params, ds, num_boost_round=max(ROUND_GRID))
    ttrain = time.time() - t0
    print(f"[{name}] trained rows={n_rows} in {ttrain:.0f}s", flush=True)
    del ds
    if cfg["n"] != 8:
        del X, y

    res = {"config": name, "params": {k: v for k, v in cfg.items()},
           "n_rows": n_rows, "train_sec": round(ttrain, 1), "scores": {}}
    for a, (Xe, ye, mask) in evalcache.items():
        raws = staged_raw(m, Xe, ROUND_GRID)
        for k, raw in raws.items():
            z = np.exp(raw) if cfg["obj"] == "tweedie" else raw     # value on log1p scale
            p = np.expm1(np.clip(z, 0, None))
            res["scores"].setdefault(a.isoformat(), {})[str(k)] = metric_rows(ye, p, mask)
        del raws
    print(f"[{name}] done in {time.time()-t0:.0f}s", flush=True)
    with open(OUT, "a") as f:
        f.write(json.dumps(res) + "\n")
    del m
    return res


def main():
    only = sys.argv[1].split(",") if len(sys.argv) > 1 else list(CONFIGS)
    t0 = time.time()
    users = sub_users(N_USERS, 20250213)
    print(f"subsample users={len(users)}", flush=True)
    Xtr, ytr, gap = load_train(TRAIN_ANCHORS_8, users)
    print(f"train matrix {Xtr.shape} ({Xtr.nbytes/1e6:.0f}MB) in {time.time()-t0:.0f}s", flush=True)
    ev = load_eval(sub_users(N_EVAL_USERS, 987654))
    print(f"eval cached in {time.time()-t0:.0f}s", flush=True)
    for name in only:
        run(name, CONFIGS[name], Xtr, ytr, gap, ev)
    print(f"ALL DONE {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

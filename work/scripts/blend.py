"""Blend model predictions: greedy hill-climb over log1p-space weighted average.

Usage:
  blend.py                       # auto-discover all *_val.parquet preds
  blend.py --include lgblog_final,cblog_final --name blend_v1
  blend.py --scale-grid          # also fit global log-space scale on val
Outputs: work/preds/NAME_{val,test}.parquet + prints weights and scores.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))
from common import PREDS_DIR, VAL_ANCHOR, load_anchor, rmsle
from exp_lib import save_preds, log_score


def discover(include: list[str] | None, exclude: list[str]):
    names = []
    for p in sorted(PREDS_DIR.glob("*_val.parquet")):
        n = p.name[: -len("_val.parquet")]
        if include and n not in include:
            continue
        if n in exclude or n.startswith(("blend", "smoke")):
            continue
        if not (PREDS_DIR / f"{n}_test.parquet").exists():
            print(f"  [skip] {n}: no test preds")
            continue
        names.append(n)
    return names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--include", type=str, default="")
    ap.add_argument("--exclude", type=str, default="")
    ap.add_argument("--name", type=str, default="blend_v1")
    ap.add_argument("--iters", type=int, default=60)
    ap.add_argument("--scale-grid", action="store_true")
    args = ap.parse_args()

    include = [s for s in args.include.split(",") if s] or None
    exclude = [s for s in args.exclude.split(",") if s]
    names = discover(include, exclude)
    print(f"models: {names}")

    val = load_anchor(VAL_ANCHOR, columns=["user_id", "target"]).sort("user_id")
    y = val["target"].to_numpy().astype(np.float64)
    uid = val["user_id"].to_numpy()

    P = {}   # name -> log1p val preds aligned to uid
    T = {}   # name -> log1p test preds
    uid_t = None
    for n in names:
        dv = pl.read_parquet(PREDS_DIR / f"{n}_val.parquet").sort("user_id")
        assert (dv["user_id"].to_numpy() == uid).all(), f"uid mismatch {n}"
        P[n] = np.log1p(np.clip(dv["pred"].to_numpy(), 0, None))
        dt = pl.read_parquet(PREDS_DIR / f"{n}_test.parquet").sort("user_id")
        if uid_t is None:
            uid_t = dt["user_id"].to_numpy()
        else:
            assert (dt["user_id"].to_numpy() == uid_t).all()
        T[n] = np.log1p(np.clip(dt["pred"].to_numpy(), 0, None))
        print(f"  {n}: solo val {rmsle(y, np.expm1(P[n])):.6f}")

    ly = np.log1p(y)

    # greedy hill-climb with replacement (weights = pick counts / total)
    picks: list[str] = []
    cur = np.zeros_like(ly)
    best_score = np.sqrt(np.mean((ly - cur) ** 2))
    for it in range(args.iters):
        cand_best, cand_name = None, None
        k = len(picks)
        for n in names:
            trial = (cur * k + P[n]) / (k + 1)
            s = np.sqrt(np.mean((ly - trial) ** 2))
            if cand_best is None or s < cand_best:
                cand_best, cand_name = s, n
        if cand_best >= best_score - 1e-7 and picks:
            break
        picks.append(cand_name)
        cur = (cur * k + P[cand_name]) / (k + 1)
        best_score = cand_best
    w = {n: picks.count(n) / len(picks) for n in sorted(set(picks))}
    print(f"picks={len(picks)} weights={w}")
    print(f"blend val RMSLE: {best_score:.6f}")

    lv = sum(P[n] * wi for n, wi in w.items())
    lt = sum(T[n] * wi for n, wi in w.items())

    scale = 1.0
    if args.scale_grid:
        best_s = best_score
        for s in np.arange(0.90, 1.101, 0.005):
            sc = np.sqrt(np.mean((ly - lv * s) ** 2))
            if sc < best_s:
                best_s, scale = sc, s
        print(f"log-space scale={scale:.3f} -> val {best_s:.6f}")
        lv = lv * scale
        lt = lt * scale
        best_score = best_s

    pv = np.expm1(np.clip(lv, 0, None))
    pt = np.expm1(np.clip(lt, 0, None))
    save_preds(args.name, "val", uid, pv)
    save_preds(args.name, "test", uid_t, pt)
    log_score(args.name, best_score, f"blend of {len(names)}: {w} scale={scale:.3f}")


if __name__ == "__main__":
    main()

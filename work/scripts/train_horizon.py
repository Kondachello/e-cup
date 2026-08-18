"""Weekly-horizon decomposition trainer (TEAM_PLAN person 2):
gmv_30d = gmv_w1 + gmv_w2 + gmv_w3 + gmv_w4.

Four champion-config LGBM models (tweedie vp1.45 ON log1p of the weekly target,
nl255 mdl300 lr0.05 ff0.75, early stop per week on VAL):
  model w1 predicts log1p(sum gmv over (A,    A+7 ])
  model w2 predicts log1p(sum gmv over (A+7,  A+14])
  model w3 predicts log1p(sum gmv over (A+14, A+21])
  model w4 predicts log1p(sum gmv over (A+21, A+30])
Final prediction (raw GMV scale) = sum_w expm1(pw); metric applies log1p on top.
Errors are structured differently from the direct 30d model (per-week horizon
difficulty + Jensen in linear space).

Follows the exp_lib contract: gap-30 protocol, val preds -> NAME_val.parquet,
retrain (train + gap anchors + val, iters scaled by row ratio) -> NAME_test.parquet,
one line in scores.tsv. Logs val RMSLE of the total AND of each week + delta
vs the direct champion (1.6927).

Needs anchor=DATE.hztgt.parquet (build_horizon_targets.py) for train/gap/VAL anchors.

Full champion run:
  USE_V2=1 USE_V3=1 USE_V4=1 OMP_NUM_THREADS=6 \
    train_horizon.py --name horizon4 --threads 6
Smoke (adds a same-protocol direct 30d baseline for the accept criterion):
  USE_V2=1 USE_V3=1 USE_V4=1 train_horizon.py --name horizon4_smoke \
    --n-anchors 2 --threads 2 --params '{"n_estimators":300}' --no-test \
    --direct-baseline
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))
from common import FEATURES_DIR, TEST_ANCHOR, VAL_ANCHOR, feature_cols, load_anchor, rmsle
from exp_lib import log_score, save_preds
from train_gbdt import fit_lgb

DIRECT_CHAMPION = 1.6927  # twl_repair_ab (lgb tweedie1.45 on log1p, gap30, 14 anchors)

CHAMPION_PARAMS = dict(
    objective="tweedie", tweedie_variance_power=1.45,
    num_leaves=255, min_data_in_leaf=300, learning_rate=0.05,
    feature_fraction=0.75, n_estimators=6000,
)

WEEKS = ("w1", "w2", "w3", "w4")


def hztgt_path(a: date) -> Path:
    return FEATURES_DIR / f"anchor={a.isoformat()}.hztgt.parquet"


def hztgt_anchors() -> list[date]:
    out = []
    for p in sorted(FEATURES_DIR.glob("anchor=*.hztgt.parquet")):
        a = date.fromisoformat(p.stem.split("=")[1].split(".")[0])
        if a < VAL_ANCHOR:
            out.append(a)
    return out


def load_with_weeks(anchors: list[date], cols: list[str]) -> pl.DataFrame:
    dfs = []
    for a in anchors:
        df = load_anchor(a, columns=["user_id", "anchor_date", "target"] + cols)
        hz = pl.read_parquet(hztgt_path(a))
        j = df.join(hz, on="user_id", how="left")
        assert j["tgt_w1"].null_count() == 0, f"hztgt misses users at {a}"
        dfs.append(j)
    return pl.concat(dfs, how="vertical_relaxed")


def to_arrays(df: pl.DataFrame, cols: list[str]):
    X = df.select(cols).to_numpy().astype(np.float32)
    y = {w: df[f"tgt_{w}"].to_numpy().astype(np.float64) for w in WEEKS}
    y_tot = df["target"].to_numpy().astype(np.float64)
    return X, y, y_tot


def combine(pred_log: dict[str, np.ndarray]) -> np.ndarray:
    """log-week predictions -> raw total: sum_w expm1(pw)."""
    return sum(np.expm1(np.clip(pred_log[w], 0, None)) for w in WEEKS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="horizon4")
    ap.add_argument("--params", type=str, default="{}",
                    help="JSON overrides on top of the champion params (all weeks)")
    ap.add_argument("--n-anchors", type=int, default=14)
    ap.add_argument("--gap-days", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--no-test", action="store_true")
    ap.add_argument("--direct-baseline", action="store_true",
                    help="also train a direct 30d model (same params/protocol) "
                         "and report it in RESULT; smoke accept criterion")
    ap.add_argument("--notes", type=str, default="")
    args = ap.parse_args()
    if args.threads:
        os.environ["OMP_NUM_THREADS"] = str(args.threads)
    params = dict(CHAMPION_PARAMS)
    params.update(json.loads(args.params))

    t0 = time.time()
    avail = hztgt_anchors()
    assert avail, "no .hztgt.parquet files; run build_horizon_targets.py first"
    assert hztgt_path(VAL_ANCHOR).exists(), "VAL anchor has no horizon targets"
    cutoff = VAL_ANCHOR - timedelta(days=args.gap_days)
    tr_anchors = [a for a in avail if a <= cutoff][-args.n_anchors:]
    gap_anchors = [a for a in avail if cutoff < a < VAL_ANCHOR]
    print(f"train anchors ({len(tr_anchors)}): {[a.isoformat() for a in tr_anchors]}",
          flush=True)
    print(f"gap anchors for retrain ({len(gap_anchors)}): "
          f"{[a.isoformat() for a in gap_anchors]}", flush=True)

    cols = feature_cols(load_anchor(VAL_ANCHOR))  # BEFORE the hztgt join
    cols = [c for c in cols if c not in [f"tgt_{w}" for w in WEEKS]]
    print(f"{len(cols)} features", flush=True)

    val = load_with_weeks([VAL_ANCHOR], cols)
    Xv, yv, yv_tot = to_arrays(val, cols)
    uid_val = val["user_id"].to_numpy()
    ident = float(np.abs(sum(yv[w] for w in WEEKS) - yv_tot).max())
    print(f"val weekly identity max|sum_w-target|={ident:.5f}", flush=True)
    assert ident < 1.0

    tr = load_with_weeks(tr_anchors, cols)
    X, y, y_tot_tr = to_arrays(tr, cols)
    del tr, val
    print(f"X {X.shape}, Xv {Xv.shape}, load {time.time()-t0:.0f}s", flush=True)

    # --- four weekly models, early stop per week on VAL ---
    best_it, pred_val_log = {}, {}
    for i, w in enumerate(WEEKS):
        print(f"--- week {w}: nz_rate={float((y[w] > 0).mean()):.4f} "
              f"val_nz={float((yv[w] > 0).mean()):.4f}", flush=True)
        m, it = fit_lgb(X, np.log1p(y[w]), None, Xv, np.log1p(yv[w]),
                        dict(params), "log_mse", args.seed + i)
        best_it[w] = it
        pred_val_log[w] = m.predict(Xv)
        del m

    pv_tot = combine(pred_val_log)
    score = rmsle(yv_tot, pv_tot)
    wk_score = {w: rmsle(yv[w], np.expm1(np.clip(pred_val_log[w], 0, None)))
                for w in WEEKS}

    direct = None
    if args.direct_baseline:
        print("--- direct 30d baseline (same protocol)", flush=True)
        md, itd = fit_lgb(X, np.log1p(y_tot_tr), None, Xv, np.log1p(yv_tot),
                          dict(params), "log_mse", args.seed)
        pd_ = np.expm1(np.clip(md.predict(Xv), 0, None))
        direct = {"score": rmsle(yv_tot, pd_), "it": itd}
        del md
        print(f"direct baseline: {direct['score']:.6f} it={itd}", flush=True)

    notes = (args.notes or
             f"4wk sum-expm1; tw1.45-on-log nl{params['num_leaves']} "
             f"lr{params['learning_rate']} gap{args.gap_days} n{len(tr_anchors)}") + (
             "; " + " ".join(f"{w}={wk_score[w]:.4f}" for w in WEEKS) +
             f" it={'/'.join(str(best_it[w]) for w in WEEKS)}; "
             f"direct_champ={DIRECT_CHAMPION} d={score - DIRECT_CHAMPION:+.4f}" +
             (f"; smoke_direct={direct['score']:.6f}" if direct else ""))
    save_preds(args.name, "val", uid_val, pv_tot)
    log_score(args.name, score, notes)
    print("RESULT " + json.dumps({
        "name": args.name, "total": round(score, 6),
        "weekly": {w: round(wk_score[w], 6) for w in WEEKS},
        "delta_vs_champion": round(score - DIRECT_CHAMPION, 6),
        "direct_baseline": None if direct is None else round(direct["score"], 6),
        "best_it": best_it, "n_anchors": len(tr_anchors),
    }), flush=True)

    if args.no_test:
        return

    # --- retrain on train + gap + val, predict test (exp_lib contract) ---
    parts = [X]
    y_parts = {w: [np.log1p(y[w])] for w in WEEKS}
    if gap_anchors:
        g = load_with_weeks(gap_anchors, cols)
        Xg, yg, _ = to_arrays(g, cols)
        del g
        parts.append(Xg)
        for w in WEEKS:
            y_parts[w].append(np.log1p(yg[w]))
        print(f"retrain adds gap anchors: +{Xg.shape[0]} rows", flush=True)
    parts.append(Xv)
    for w in WEEKS:
        y_parts[w].append(np.log1p(yv[w]))
    Xall = np.vstack(parts)
    row_ratio = Xall.shape[0] / max(X.shape[0], 1)
    iter_mult = 1.0 + 0.7 * max(row_ratio - 1.0, 0.0)
    print(f"retrain: row_ratio={row_ratio:.3f} iter_mult={iter_mult:.3f}", flush=True)
    del X, Xv, parts

    test = load_anchor(TEST_ANCHOR)
    Xt = test.select(cols).to_numpy().astype(np.float32)
    uid_t = test["user_id"].to_numpy()
    del test

    pred_test_log = {}
    for i, w in enumerate(WEEKS):
        p = dict(params)
        p["n_estimators"] = max(50, int(best_it[w] * iter_mult))
        print(f"--- retrain {w}: {p['n_estimators']} iters", flush=True)
        mf, _ = fit_lgb(Xall, np.concatenate(y_parts[w]), None, None, None,
                        p, "log_mse", args.seed + i)
        pred_test_log[w] = mf.predict(Xt)
        del mf

    save_preds(args.name, "test", uid_t, combine(pred_test_log))
    print(f"[DONE] {args.name} val_rmsle={score:.6f} total {time.time()-t0:.0f}s",
          flush=True)


if __name__ == "__main__":
    main()

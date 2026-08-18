"""Channel-decomposition trainer (TEAM_PLAN person 2): gmv = gmv_search + gmv_cat.

Two champion-config LGBM models (tweedie vp1.45 ON log1p of the channel target,
nl255 mdl300 lr0.05 ff0.75, n_estimators 6000, early stop per channel on VAL):
  model S predicts log1p(sum gmv_search over (A, A+30])
  model C predicts log1p(sum gmv_cat    over (A, A+30])
Final prediction (raw GMV scale) = expm1(ps) + expm1(pc); the errors of this
construction are structured differently from the direct total model (Jensen:
sum in linear space, not in log space).

Follows the exp_lib contract: gap-30 protocol, val preds -> NAME_val.parquet,
retrain (train + gap anchors + val, iters scaled by row ratio) -> NAME_test.parquet,
one line in scores.tsv. Logs val RMSLE of the total AND of each channel + delta
vs the direct champion (1.6927).

Needs anchor=DATE.chtgt.parquet (build_channel_targets.py) for train/gap/VAL anchors.

Full champion run:
  USE_V2=1 USE_V3=1 USE_V4=1 OMP_NUM_THREADS=6 \
    train_channel.py --name channel2 --threads 6
Smoke:
  USE_V2=1 USE_V3=1 USE_V4=1 train_channel.py --name channel2_smoke \
    --n-anchors 2 --threads 2 --params '{"n_estimators":200}' --no-test
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

CHANNELS = ("search", "cat")


def chtgt_path(a: date) -> Path:
    return FEATURES_DIR / f"anchor={a.isoformat()}.chtgt.parquet"


def chtgt_anchors() -> list[date]:
    out = []
    for p in sorted(FEATURES_DIR.glob("anchor=*.chtgt.parquet")):
        a = date.fromisoformat(p.stem.split("=")[1].split(".")[0])
        if a < VAL_ANCHOR:
            out.append(a)
    return out


def load_with_channels(anchors: list[date], cols: list[str]) -> pl.DataFrame:
    dfs = []
    for a in anchors:
        df = load_anchor(a, columns=["user_id", "anchor_date", "target"] + cols)
        ch = pl.read_parquet(chtgt_path(a))
        j = df.join(ch, on="user_id", how="left")
        assert j["tgt_search"].null_count() == 0, f"chtgt misses users at {a}"
        dfs.append(j)
    return pl.concat(dfs, how="vertical_relaxed")


def to_arrays(df: pl.DataFrame, cols: list[str]):
    X = df.select(cols).to_numpy().astype(np.float32)
    y = {c: df[f"tgt_{c}"].to_numpy().astype(np.float64) for c in CHANNELS}
    y_tot = df["target"].to_numpy().astype(np.float64)
    return X, y, y_tot


def combine(pred_log: dict[str, np.ndarray]) -> np.ndarray:
    """log-channel predictions -> raw total: expm1(ps) + expm1(pc)."""
    return sum(np.expm1(np.clip(pred_log[c], 0, None)) for c in CHANNELS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="channel2")
    ap.add_argument("--params", type=str, default="{}",
                    help="JSON overrides on top of the champion channel params")
    ap.add_argument("--n-anchors", type=int, default=14)
    ap.add_argument("--gap-days", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--no-test", action="store_true")
    ap.add_argument("--notes", type=str, default="")
    args = ap.parse_args()
    if args.threads:
        os.environ["OMP_NUM_THREADS"] = str(args.threads)
    params = dict(CHAMPION_PARAMS)
    params.update(json.loads(args.params))

    t0 = time.time()
    avail = chtgt_anchors()
    assert avail, "no .chtgt.parquet files; run build_channel_targets.py first"
    assert chtgt_path(VAL_ANCHOR).exists(), "VAL anchor has no channel targets"
    cutoff = VAL_ANCHOR - timedelta(days=args.gap_days)
    tr_anchors = [a for a in avail if a <= cutoff][-args.n_anchors:]
    gap_anchors = [a for a in avail if cutoff < a < VAL_ANCHOR]
    print(f"train anchors ({len(tr_anchors)}): {[a.isoformat() for a in tr_anchors]}",
          flush=True)
    print(f"gap anchors for retrain ({len(gap_anchors)}): "
          f"{[a.isoformat() for a in gap_anchors]}", flush=True)

    cols = feature_cols(load_anchor(VAL_ANCHOR))  # BEFORE the chtgt join
    cols = [c for c in cols if c not in ("tgt_search", "tgt_cat")]
    print(f"{len(cols)} features", flush=True)

    val = load_with_channels([VAL_ANCHOR], cols)
    Xv, yv, yv_tot = to_arrays(val, cols)
    uid_val = val["user_id"].to_numpy()
    ident = float(np.abs(yv["search"] + yv["cat"] - yv_tot).max())
    print(f"val channel identity max|s+c-target|={ident:.5f}", flush=True)
    assert ident < 1.0

    tr = load_with_channels(tr_anchors, cols)
    X, y, _ = to_arrays(tr, cols)
    del tr, val
    print(f"X {X.shape}, Xv {Xv.shape}, load {time.time()-t0:.0f}s", flush=True)

    # --- two channel models, early stop per channel on VAL ---
    best_it, pred_val_log = {}, {}
    for i, c in enumerate(CHANNELS):
        print(f"--- channel {c}: nz_rate={float((y[c] > 0).mean()):.4f} "
              f"val_nz={float((yv[c] > 0).mean()):.4f}", flush=True)
        m, it = fit_lgb(X, np.log1p(y[c]), None, Xv, np.log1p(yv[c]),
                        dict(params), "log_mse", args.seed + i)
        best_it[c] = it
        pred_val_log[c] = m.predict(Xv)
        del m

    pv_tot = combine(pred_val_log)
    score = rmsle(yv_tot, pv_tot)
    ch_score = {c: rmsle(yv[c], np.expm1(np.clip(pred_val_log[c], 0, None)))
                for c in CHANNELS}
    notes = (args.notes or
             f"2ch sum-expm1; tw1.45-on-log nl255 lr0.05 gap{args.gap_days} "
             f"n{len(tr_anchors)}") + (
             f"; search={ch_score['search']:.4f} cat={ch_score['cat']:.4f} "
             f"it={best_it['search']}/{best_it['cat']}; "
             f"direct_champ={DIRECT_CHAMPION} d={score - DIRECT_CHAMPION:+.4f}")
    save_preds(args.name, "val", uid_val, pv_tot)
    log_score(args.name, score, notes)
    print("RESULT " + json.dumps({
        "name": args.name, "total": round(score, 6),
        "search": round(ch_score["search"], 6), "cat": round(ch_score["cat"], 6),
        "delta_vs_champion": round(score - DIRECT_CHAMPION, 6),
        "best_it": best_it, "n_anchors": len(tr_anchors),
    }), flush=True)

    if args.no_test:
        return

    # --- retrain on train + gap + val, predict test (exp_lib contract) ---
    parts = [X]
    y_parts = {c: [np.log1p(y[c])] for c in CHANNELS}
    if gap_anchors:
        g = load_with_channels(gap_anchors, cols)
        Xg, yg, _ = to_arrays(g, cols)
        del g
        parts.append(Xg)
        for c in CHANNELS:
            y_parts[c].append(np.log1p(yg[c]))
        print(f"retrain adds gap anchors: +{Xg.shape[0]} rows", flush=True)
    parts.append(Xv)
    for c in CHANNELS:
        y_parts[c].append(np.log1p(yv[c]))
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
    for i, c in enumerate(CHANNELS):
        p = dict(params)
        p["n_estimators"] = max(50, int(best_it[c] * iter_mult))
        print(f"--- retrain {c}: {p['n_estimators']} iters", flush=True)
        mf, _ = fit_lgb(Xall, np.concatenate(y_parts[c]), None, None, None,
                        p, "log_mse", args.seed + i)
        pred_test_log[c] = mf.predict(Xt)
        del mf

    save_preds(args.name, "test", uid_t, combine(pred_test_log))
    print(f"[DONE] {args.name} val_rmsle={score:.6f} total {time.time()-t0:.0f}s",
          flush=True)


if __name__ == "__main__":
    main()

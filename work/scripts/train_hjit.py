"""Snapshot-jitter ("hjit") trainer: champion tweedie-on-log at horizon H in {37, 44}.

Idea: the test forecast rests on the single 2026-02-13 feature snapshot; any
single-day anomaly of that snapshot propagates into the prediction. A model
trained to horizon H predicts the SAME test window (2026-02-14..2026-03-15)
from an EARLIER snapshot; averaging (done later by the coordinator, in log1p
space) h30@2026-02-13 + h37@2026-02-06 + h44@2026-01-30 damps snapshot noise.

One horizon-H model per run:
  target      log1p(sum gmv over (A, A+H])  from anchor=DATE.hjit.parquet
              (build_hjit_targets.py)
  val anchor  DATA_END - H  (window fully observed, ends exactly at DATA_END):
              h37 -> 2026-01-07, h44 -> 2025-12-31
  config      exact twl_repair_ab champion: lgb tweedie vp1.45 ON the log1p
              target (fit_lgb defaults lr0.04 nl255 mdl300 ff0.75), gap-30 to
              the model's own val anchor, early stop on that val anchor
  retrain     train + gap + val anchors, iters scaled by row ratio (as
              train_gbdt), then predict TEST from the SHIFTED snapshot
              TEST_ANCHOR - (H-30): h37 -> 2026-02-06, h44 -> 2026-01-30

CAVEAT: NAME_val.parquet lives on the model's OWN val window, not the standard
VAL (2026-01-14, 30d) — never mix it into standard val-window blends. There is
no exact validation of the 3-way average itself (the val windows differ);
per-horizon val quality + error correlation vs the h30 champion is the
acceptance evidence.

Full champion runs (queued):
  USE_V2=1 USE_V3=1 train_hjit.py --name hjit37 --horizon 37 --threads 6
  USE_V2=1 USE_V3=1 train_hjit.py --name hjit44 --horizon 44 --threads 6
Smoke:
  USE_V2=1 USE_V3=1 train_hjit.py --name hjit37_smoke --horizon 37 \
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
from common import (
    DATA_END, FEATURES_DIR, HORIZON, PREDS_DIR, TEST_ANCHOR, VAL_ANCHOR,
    feature_cols, load_anchor, rmsle,
)
from exp_lib import log_score, save_preds
from train_gbdt import fit_lgb

DIRECT_CHAMPION = 1.6927  # twl_repair_ab @ standard VAL 2026-01-14 (30d window)

# Exact twl_repair_ab config: fit_lgb defaults + tweedie vp1.45 ON log1p target.
CHAMPION_PARAMS = dict(objective="tweedie", tweedie_variance_power=1.45,
                       n_estimators=6000)

# feature tiers that must exist for every anchor we load (env flag -> suffix)
TIER_SUFFIX = {"USE_V2": "extra", "USE_V3": "v3", "USE_V4": "v4",
               "USE_V6": "v6", "USE_SEQOOF": "seqoof"}


def hjit_path(a: date) -> Path:
    return FEATURES_DIR / f"anchor={a.isoformat()}.hjit.parquet"


def hjit_anchors(before: date) -> list[date]:
    out = []
    for p in sorted(FEATURES_DIR.glob("anchor=*.hjit.parquet")):
        a = date.fromisoformat(p.stem.split("=")[1].split(".")[0])
        if a < before:
            out.append(a)
    return out


def assert_anchor_files(anchors: list[date]):
    missing = []
    for a in anchors:
        if not (FEATURES_DIR / f"anchor={a.isoformat()}.parquet").exists():
            missing.append(f"{a} base")
        for env, suf in TIER_SUFFIX.items():
            if os.environ.get(env) and not (
                    FEATURES_DIR / f"anchor={a.isoformat()}.{suf}.parquet").exists():
                missing.append(f"{a} {suf}")
    assert not missing, f"missing feature files: {missing}"


def load_with_h(anchors: list[date], cols: list[str], h: int) -> pl.DataFrame:
    dfs = []
    for a in anchors:
        df = load_anchor(a, columns=["user_id", "anchor_date", "target"] + cols)
        hz = pl.read_parquet(hjit_path(a), columns=["user_id", f"tgt_h{h}"])
        j = df.join(hz, on="user_id", how="left")
        assert j[f"tgt_h{h}"].null_count() == 0, \
            f"tgt_h{h} missing or unobserved at {a}"
        dfs.append(j)
    return pl.concat(dfs, how="vertical_relaxed")


def err_corr_vs_ref(ref_name: str, uid: np.ndarray, e_h: np.ndarray):
    """corr of this model's val errors (its own window) with the h30 champion's
    val errors (standard VAL window), matched by user_id. Windows differ ->
    approximate ensemble evidence, not an exact blend validation."""
    p = PREDS_DIR / f"{ref_name}_val.parquet"
    if not p.exists():
        return None
    ref = pl.read_parquet(p).sort("user_id")
    v30 = load_anchor(VAL_ANCHOR, columns=["user_id", "target"]).sort("user_id")
    assert np.array_equal(ref["user_id"].to_numpy(), v30["user_id"].to_numpy())
    assert np.array_equal(ref["user_id"].to_numpy(), uid), "user universe mismatch"
    e_ref = (np.log1p(np.clip(ref["pred"].to_numpy().astype(np.float64), 0, None))
             - np.log1p(v30["target"].to_numpy().astype(np.float64)))
    return float(np.corrcoef(e_h, e_ref)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--horizon", type=int, required=True, choices=[37, 44])
    ap.add_argument("--params", type=str, default="{}",
                    help="JSON overrides on top of the champion params")
    ap.add_argument("--n-anchors", type=int, default=14)
    ap.add_argument("--gap-days", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--no-test", action="store_true")
    ap.add_argument("--direct-baseline", action="store_true",
                    help="also train a same-protocol h30 model at the same val "
                         "anchor (base 30d target); isolates the horizon cost")
    ap.add_argument("--ref-preds", type=str, default="twl_repair_ab",
                    help="h30 champion val preds for error correlation")
    ap.add_argument("--notes", type=str, default="")
    args = ap.parse_args()
    if args.threads:
        os.environ["OMP_NUM_THREADS"] = str(args.threads)
    params = dict(CHAMPION_PARAMS)
    params.update(json.loads(args.params))

    H = args.horizon
    val_h = DATA_END - timedelta(days=H)          # h37 -> 2026-01-07, h44 -> 2025-12-31
    test_snap = TEST_ANCHOR - timedelta(days=H - HORIZON)  # h37 -> 2026-02-06, h44 -> 2026-01-30
    assert val_h + timedelta(days=H) == DATA_END

    t0 = time.time()
    avail = hjit_anchors(before=val_h)
    assert avail, "no .hjit.parquet files; run build_hjit_targets.py first"
    assert hjit_path(val_h).exists(), f"val anchor {val_h} has no hjit targets"
    cutoff = val_h - timedelta(days=args.gap_days)
    tr_anchors = [a for a in avail if a <= cutoff][-args.n_anchors:]
    gap_anchors = [a for a in avail if cutoff < a < val_h]
    print(f"H={H} val_anchor={val_h} test_snapshot={test_snap}", flush=True)
    print(f"train anchors ({len(tr_anchors)}): {[a.isoformat() for a in tr_anchors]}",
          flush=True)
    print(f"gap anchors for retrain ({len(gap_anchors)}): "
          f"{[a.isoformat() for a in gap_anchors]}", flush=True)
    assert_anchor_files(tr_anchors + gap_anchors + [val_h]
                        + ([] if args.no_test else [test_snap]))

    cols = feature_cols(load_anchor(val_h))
    assert f"tgt_h{H}" not in cols
    print(f"{len(cols)} features", flush=True)

    val = load_with_h([val_h], cols, H)
    Xv = val.select(cols).to_numpy().astype(np.float32)
    yv_raw = val[f"tgt_h{H}"].to_numpy().astype(np.float64)
    yv30_raw = val["target"].to_numpy().astype(np.float64)
    uid_val = val["user_id"].to_numpy()
    del val

    tr = load_with_h(tr_anchors, cols, H)
    X = tr.select(cols).to_numpy().astype(np.float32)
    y_raw = tr[f"tgt_h{H}"].to_numpy().astype(np.float64)
    y30_raw = tr["target"].to_numpy().astype(np.float64)
    del tr
    print(f"X {X.shape}, Xv {Xv.shape}, nz_rate tr={float((y_raw > 0).mean()):.4f} "
          f"val={float((yv_raw > 0).mean()):.4f}, load {time.time()-t0:.0f}s", flush=True)

    m, best_it = fit_lgb(X, np.log1p(y_raw), None, Xv, np.log1p(yv_raw),
                         dict(params), "log_mse", args.seed)
    pv = np.expm1(np.clip(m.predict(Xv), 0, None))
    del m
    score = rmsle(yv_raw, pv)

    direct = None
    if args.direct_baseline:
        print("--- direct h30 baseline at the same val anchor (same protocol)", flush=True)
        md, itd = fit_lgb(X, np.log1p(y30_raw), None, Xv, np.log1p(yv30_raw),
                          dict(params), "log_mse", args.seed)
        direct = {"score": rmsle(yv30_raw, np.expm1(np.clip(md.predict(Xv), 0, None))),
                  "it": itd}
        del md
        print(f"direct h30 baseline: {direct['score']:.6f} it={itd}", flush=True)

    e_h = np.log1p(np.clip(pv, 0, None)) - np.log1p(yv_raw)
    corr = err_corr_vs_ref(args.ref_preds, uid_val, e_h)

    horizon_cost = None if direct is None else score - direct["score"]
    notes = (args.notes or
             f"hjit h{H} val@{val_h} (window!=standard VAL); tw1.45-on-log "
             f"champion cfg gap{args.gap_days} n{len(tr_anchors)}") + (
             f"; it={best_it}; direct_champ={DIRECT_CHAMPION} "
             f"d={score - DIRECT_CHAMPION:+.4f}"
             + (f"; direct_h30@same_anchor={direct['score']:.6f} "
                f"horizon_cost={horizon_cost:+.4f}" if direct else "")
             + (f"; err_corr_vs_{args.ref_preds}={corr:.4f}" if corr is not None else ""))
    save_preds(args.name, "val", uid_val, pv)
    log_score(args.name, score, notes)
    print("RESULT " + json.dumps({
        "name": args.name, "horizon": H, "val_anchor": val_h.isoformat(),
        "val_rmsle": round(score, 6),
        "delta_vs_champion": round(score - DIRECT_CHAMPION, 6),
        "direct_h30_same_anchor": None if direct is None else round(direct["score"], 6),
        "horizon_cost": None if horizon_cost is None else round(horizon_cost, 6),
        "err_corr_vs_ref": None if corr is None else round(corr, 6),
        "best_it": best_it, "n_anchors": len(tr_anchors),
        "test_snapshot": test_snap.isoformat(),
    }), flush=True)

    if args.no_test:
        return

    # --- retrain on train + gap + val, predict TEST from the shifted snapshot ---
    parts = [X]
    y_parts = [np.log1p(y_raw)]
    if gap_anchors:
        g = load_with_h(gap_anchors, cols, H)
        Xg = g.select(cols).to_numpy().astype(np.float32)
        y_parts.append(np.log1p(g[f"tgt_h{H}"].to_numpy().astype(np.float64)))
        del g
        parts.append(Xg)
        print(f"retrain adds gap anchors: +{Xg.shape[0]} rows", flush=True)
    parts.append(Xv)
    y_parts.append(np.log1p(yv_raw))
    Xall = np.vstack(parts)
    row_ratio = Xall.shape[0] / max(X.shape[0], 1)
    iter_mult = 1.0 + 0.7 * max(row_ratio - 1.0, 0.0)
    del X, Xv, parts
    p_final = dict(params)
    p_final["n_estimators"] = max(50, int(best_it * iter_mult))
    print(f"retrain: row_ratio={row_ratio:.3f} iter_mult={iter_mult:.3f} "
          f"iters={p_final['n_estimators']}", flush=True)

    test = load_anchor(test_snap)
    Xt = test.select(cols).to_numpy().astype(np.float32)
    uid_t = test["user_id"].to_numpy()
    assert np.array_equal(uid_t, uid_val), "universe mismatch between anchors"
    del test

    mf, _ = fit_lgb(Xall, np.concatenate(y_parts), None, None, None,
                    p_final, "log_mse", args.seed)
    pt = np.expm1(np.clip(mf.predict(Xt), 0, None))
    save_preds(args.name, "test", uid_t, pt)
    print(f"[DONE] {args.name} val_rmsle={score:.6f} test_snap={test_snap} "
          f"total {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

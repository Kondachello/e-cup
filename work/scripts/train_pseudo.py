"""Transductive pseudo-label (soft-target) training: real rows + TEST-slice rows.

Motivation
----------
Adversarial AUC train-anchors vs test-anchor is 1.0000 (KNOWLEDGE F23/N6): the
covariate shift is structural and cannot be cured by dropping features.  Here we
let the model SEE the test feature distribution during training: the 250k rows of
the test slice (anchor=2026-02-13) are appended with SOFT targets taken from our
best submission, at a reduced sample weight.  The value is regularisation towards
the test covariate distribution, not new label information -- a student that only
copies the teacher cannot beat it.

Three guards against the obvious failure modes:
     global correction +0.1163 (log) baked in; we subtract it back out so the
     student does not cement the seasonal level and we can still correct it
     separately downstream.  (`--teacher blendcal` uses the uncalibrated-for-test
     blend instead, which never had the shift.)
  2. PSEUDO ROWS ARE DOWN-WEIGHTED (0.2-0.6) so real labels dominate.
  3. INTERLEAVING: trees are grown in alternating chunks -- `--chunk-real` trees
     on real rows only (pseudo weight 0), then `--chunk-mix` trees on the mixture.
     Implemented by flipping the weight vector of one Dataset between continuation
     calls, so binning/bagging stay identical across phases.

Protocol follows exp_lib: train on anchors < VAL_ANCHOR (gap-30 clean protocol),
validate on VAL_ANCHOR, then retrain including VAL and predict TEST.
Scores are reported RAW and after honest binned-log-shift calibration on a
half-user holdout -- the pipeline calibrates every model before blending, so the
calibrated number is the decision-relevant one (KNOWLEDGE: raw comparisons lie).

Examples
--------
  # smoke, 3 slices, 400 trees, pseudo weight 0.4
  train_pseudo.py --name pseudo_w04 --n-anchors 3 --n-trees 400 --pseudo-weight 0.4 \
      --chunk-real 50 --chunk-mix 50 --threads 2
  # identical control without pseudo rows
  train_pseudo.py --name pseudo_base --n-anchors 3 --n-trees 400 --no-pseudo --threads 2
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import timedelta
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))
from common import PREDS_DIR, REPORTS_DIR, TEST_ANCHOR, VAL_ANCHOR, feature_cols, load_anchor, rmsle
from exp_lib import available_train_anchors, log_score, save_preds
from calibrate import apply_shifts, fit_shifts


SHIFT_G5 = 0.1163

CHAMPION = dict(objective="tweedie", tweedie_variance_power=1.45, learning_rate=0.04,
                num_leaves=255, min_data_in_leaf=300, feature_fraction=0.75,
                bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0, max_bin=127,
                verbosity=-1)


# --------------------------------------------------------------------------- teacher
def teacher_logs(kind: str, uid_ref: np.ndarray) -> tuple[np.ndarray, str]:
    """log1p soft targets on the test slice, WITHOUT the global seasonal shift."""
    if kind == "g5deshift":
        from subs import lp as sub_lp
        l = l - SHIFT_G5
    elif kind == "blendcal":
        d = pl.read_parquet(PREDS_DIR / "blend_cal_test.parquet").sort("user_id")
        uid, l = d["user_id"].to_numpy(), np.log1p(np.clip(d["pred"].to_numpy(), 0, None))
        note = "blend_cal_test (never had the shift)"
    elif kind.startswith("file:"):
        name = kind.split(":", 1)[1]
        d = pl.read_parquet(PREDS_DIR / f"{name}_test.parquet").sort("user_id")
        uid, l = d["user_id"].to_numpy(), np.log1p(np.clip(d["pred"].to_numpy(), 0, None))
        note = f"work/preds/{name}_test.parquet"
    else:
        raise SystemExit(f"unknown teacher {kind}")
    if not np.array_equal(uid, uid_ref):
        raise SystemExit("teacher user_id order does not match the test slice")
    lo = float((l < 0).mean())
    l = np.clip(l, 0, None)
    print(f"teacher: {note}  mean {l.mean():.4f} sd {l.std():.4f} clipped_at_0 {lo:.4%}", flush=True)
    return l, note


# ------------------------------------------------------------------------- schedule
def _even(n_trees: int, step: int, phase: str) -> list[tuple[str, int]]:
    step = step if step > 0 else n_trees
    out, left = [], n_trees
    while left > 0:
        r = min(step, left)
        out.append((phase, r))
        left -= r
    return out


def schedule(n_trees: int, chunk_real: int, chunk_mix: int) -> list[tuple[str, int]]:
    """Alternating [real-only, mixture] chunks summing to n_trees.

    Chunk size doubles as the validation cadence, so the control arm (chunk_mix=0)
    is evaluated at exactly the same tree counts as the pseudo arm.
    """
    if chunk_mix <= 0:
        return _even(n_trees, chunk_real, "real")
    if chunk_real <= 0:
        return _even(n_trees, chunk_mix, "mix")
    out, left = [], n_trees
    while left > 0:
        r = min(chunk_real, left)
        out.append(("real", r))
        left -= r
        if left <= 0:
            break
        m = min(chunk_mix, left)
        out.append(("mix", m))
        left -= m
    return out


def weight_vectors(n_real: int, n_pseudo: int, wp: float):
    """(off, on) weight vectors.  LightGBM silently drops an all-ones weight array
    (Dataset.set_weight: `if np.all(weight == 1): weight = None`) and then keeps
    the PREVIOUS weights -- verified.  Never emit all-ones."""
    if n_pseudo == 0:
        return None, None
    if wp == 1.0:
        wp = 0.999999
    off = np.concatenate([np.ones(n_real), np.zeros(n_pseudo)])
    on = np.concatenate([np.ones(n_real), np.full(n_pseudo, wp)])
    return off, on


def run_schedule(ds, params, sched, w_off, w_on, seed, eval_cb=None):
    """Grow trees chunk by chunk, flipping the pseudo weights between chunks."""
    import lightgbm as lgb
    p = dict(params)
    p["seed"] = seed
    booster, done, phase_now = None, 0, None
    for phase, rounds in sched:
        if w_off is not None and phase != phase_now:
            ds.set_weight(w_off if phase == "real" else w_on)
            phase_now = phase
        booster = lgb.train(p, ds, num_boost_round=rounds, init_model=booster,
                            keep_training_booster=True)
        done += rounds
        if eval_cb is not None:
            eval_cb(done, phase, booster)
    return booster


# ------------------------------------------------------------------------ evaluation
def cal_holdout(lp: np.ndarray, ly: np.ndarray, y_raw: np.ndarray, half: np.ndarray, bins: int = 24):
    """Honest binned log-shift calibration: fit on half the users, score the rest."""
    lp = np.clip(lp, 0, None)
    c, s = fit_shifts(lp[half], ly[half], bins)
    return (float(rmsle(y_raw[~half], np.expm1(lp[~half]))),
            float(rmsle(y_raw[~half], np.expm1(apply_shifts(lp[~half], c, s)))))


# ------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--n-anchors", type=int, default=3, help="last N gap-eligible train slices")
    ap.add_argument("--gap-days", type=int, default=30)
    ap.add_argument("--n-trees", type=int, default=400)
    ap.add_argument("--params", type=str, default="{}", help="overrides on the champion tw1.45-on-log recipe")
    ap.add_argument("--teacher", type=str, default="g5deshift",
                    help="g5deshift | blendcal | file:NAME")
    ap.add_argument("--pseudo-weight", type=float, default=0.4)
    ap.add_argument("--chunk-real", type=int, default=50)
    ap.add_argument("--chunk-mix", type=int, default=50)
    ap.add_argument("--no-pseudo", action="store_true", help="control arm: no pseudo rows at all")
    ap.add_argument("--zero-pseudo", action="store_true",
                    help="binning-matched control: pseudo rows present, weight 0 throughout")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--no-test", action="store_true")
    ap.add_argument("--retrain-gap", action="store_true",
                    help="test retrain also uses the gap anchors (full-run setting)")
    ap.add_argument("--notes", type=str, default="")
    args = ap.parse_args()

    if args.threads:
        os.environ["OMP_NUM_THREADS"] = str(args.threads)
    params = dict(CHAMPION)
    params["num_threads"] = args.threads or int(os.environ.get("OMP_NUM_THREADS", "2"))
    params.update(json.loads(args.params))
    t0 = time.time()

    tr_anchors = [a for a in available_train_anchors()
                  if a <= VAL_ANCHOR - timedelta(days=args.gap_days)]
    if args.n_anchors:
        tr_anchors = tr_anchors[-args.n_anchors:]
    print(f"train anchors: {[a.isoformat() for a in tr_anchors]}", flush=True)

    val = load_anchor(VAL_ANCHOR)
    cols = feature_cols(val)
    n_users = val.height
    Xv = val.select(cols).to_numpy().astype(np.float32)
    yv_raw = val["target"].to_numpy().astype(np.float64)
    uid_val = val["user_id"].to_numpy()
    del val
    yv_log = np.log1p(yv_raw)
    half = np.random.default_rng(0).permutation(n_users) < n_users // 2  # same split as calibrate.py
    print(f"{len(cols)} features, {n_users} users", flush=True)

    # ---- real rows
    X = np.empty((len(tr_anchors) * n_users, len(cols)), np.float32)
    y = np.empty(len(tr_anchors) * n_users, np.float64)
    for i, a in enumerate(tr_anchors):
        df = load_anchor(a, ["user_id", "target"] + cols)
        assert df.height == n_users, f"{a}: {df.height} rows"
        X[i * n_users:(i + 1) * n_users] = df.select(cols).to_numpy().astype(np.float32)
        y[i * n_users:(i + 1) * n_users] = np.log1p(df["target"].to_numpy().astype(np.float64))
        del df
    n_real = X.shape[0]

    # ---- pseudo rows (test slice features + soft targets)
    test = load_anchor(TEST_ANCHOR)
    Xt = test.select(cols).to_numpy().astype(np.float32)
    uid_t = test["user_id"].to_numpy()
    del test
    t_note = "none"
    if args.no_pseudo:
        n_pseudo, wp = 0, 0.0
    else:
        soft, t_note = teacher_logs(args.teacher, uid_t)
        wp = 0.0 if args.zero_pseudo else args.pseudo_weight
        X = np.vstack([X, Xt])
        y = np.concatenate([y, soft])
        n_pseudo = n_users
    print(f"rows: real {n_real} + pseudo {n_pseudo} (weight {wp}); "
          f"mean log-target real {y[:n_real].mean():.4f}"
          + (f" pseudo {y[n_real:].mean():.4f}" if n_pseudo else "")
          + f" | val {yv_log.mean():.4f} [{time.time()-t0:.0f}s]", flush=True)

    sched = schedule(args.n_trees, args.chunk_real, args.chunk_mix if n_pseudo else 0)
    print(f"schedule: {sched}", flush=True)

    import lightgbm as lgb
    w_off, w_on = weight_vectors(n_real, n_pseudo, wp)
    ds = lgb.Dataset(X, y, weight=w_off, free_raw_data=False)

    curve = []

    def ev(done, phase, booster):
        lp = np.clip(booster.predict(Xv), 0, None)
        raw, cal = cal_holdout(lp, yv_log, yv_raw, half)
        full = rmsle(yv_raw, np.expm1(lp))
        curve.append(dict(trees=done, phase=phase, val_raw_full=float(full),
                          holdout_raw=raw, holdout_cal=cal,
                          mean_lp=float(lp.mean()), sd_lp=float(lp.std())))
        print(f"  [{done:4d} trees after {phase:4s}] val_raw={full:.6f} "
              f"holdout raw={raw:.6f} cal={cal:.6f} mean={lp.mean():.4f} sd={lp.std():.4f} "
              f"[{time.time()-t0:.0f}s]", flush=True)

    booster = run_schedule(ds, params, sched, w_off, w_on, args.seed, ev)
    lp_val = np.clip(booster.predict(Xv), 0, None)
    score = rmsle(yv_raw, np.expm1(lp_val))
    best = min(curve, key=lambda c: c["holdout_cal"])
    print(f"[VAL] {args.name} raw={score:.6f} cal_holdout={curve[-1]['holdout_cal']:.6f} "
          f"| best cal {best['holdout_cal']:.6f} @ {best['trees']} trees", flush=True)
    save_preds(args.name, "val", uid_val, np.expm1(lp_val))
    note = (args.notes or
            f"pseudo teacher={t_note} w={wp} sched(real{args.chunk_real}/mix{args.chunk_mix}) "
            f"{len(tr_anchors)}sl {args.n_trees}t gap{args.gap_days}; "
            f"cal_holdout={curve[-1]['holdout_cal']:.6f}")
    log_score(args.name, score, note)

    rep = dict(name=args.name, teacher=t_note, pseudo_weight=wp, n_pseudo=n_pseudo,
               n_real=n_real, anchors=[a.isoformat() for a in tr_anchors],
               n_trees=args.n_trees, chunk_real=args.chunk_real, chunk_mix=args.chunk_mix,
               params=params, seed=args.seed, val_raw=float(score),
               cal_holdout_final=curve[-1]["holdout_cal"], curve=curve,
               best=best)
    out = REPORTS_DIR / f"{args.name}.json"
    out.write_text(json.dumps(rep, indent=1, default=str))
    print(f"[JSON] {out}", flush=True)

    if args.no_test:
        print(f"[DONE] {args.name} {time.time()-t0:.0f}s", flush=True)
        return

    # ---------------------------------------------------------------- retrain -> test
    del ds, booster
    best_it = best["trees"]
    extra = []
    if args.retrain_gap:
        extra = [a for a in available_train_anchors()
                 if VAL_ANCHOR - timedelta(days=args.gap_days) < a < VAL_ANCHOR]
    add = extra + [VAL_ANCHOR]
    print(f"[retrain] adding {[a.isoformat() for a in add]}", flush=True)
    Xa = np.empty((len(add) * n_users, len(cols)), np.float32)
    ya = np.empty(len(add) * n_users, np.float64)
    for i, a in enumerate(add):
        df = load_anchor(a, ["user_id", "target"] + cols)
        Xa[i * n_users:(i + 1) * n_users] = df.select(cols).to_numpy().astype(np.float32)
        ya[i * n_users:(i + 1) * n_users] = np.log1p(df["target"].to_numpy().astype(np.float64))
        del df
    if n_pseudo:  # keep pseudo rows last so the weight vectors stay valid
        X = np.vstack([X[:n_real], Xa, X[n_real:]])
        y = np.concatenate([y[:n_real], ya, y[n_real:]])
    else:
        X = np.vstack([X, Xa])
        y = np.concatenate([y, ya])
    n_real2 = n_real + Xa.shape[0]
    del Xa, ya, Xv
    row_ratio = n_real2 / n_real
    iter_mult = 1.0 + 0.7 * max(row_ratio - 1.0, 0.0)
    n_trees2 = max(50, int(best_it * iter_mult))
    print(f"[retrain] rows real {n_real2} + pseudo {n_pseudo}; row_ratio={row_ratio:.3f} "
          f"iter_mult={iter_mult:.3f} trees {best_it}->{n_trees2}", flush=True)
    w_off2, w_on2 = weight_vectors(n_real2, n_pseudo, wp)
    ds2 = lgb.Dataset(X, y, weight=w_off2, free_raw_data=False)
    sched2 = schedule(n_trees2, args.chunk_real, args.chunk_mix if n_pseudo else 0)
    booster = run_schedule(ds2, params, sched2, w_off2, w_on2, args.seed)
    pt = np.expm1(np.clip(booster.predict(Xt), 0, None))
    save_preds(args.name, "test", uid_t, pt)
    print(f"[DONE] {args.name} val_rmsle={score:.6f} test mean_lp={np.log1p(pt).mean():.4f} "
          f"total {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

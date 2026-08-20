"""Direct staleness-conditioned model: one LGB trained on (features@A-k, target of A).

Why: lagd28 is a REPURPOSED model. It was trained to answer "GMV in the 30 days right
after the feature date" and is then asked "GMV in the window starting 28 days after the
feature date". All of its margin (+0.0031) comes from solving the wrong task. The direct
model trains exactly the mapping that is used - the M5 winners' direct strategy, with
staleness k as a numeric input (Azure AutoML's "horizon feature").

Row construction. For a target anchor A (its target column = GMV in (A, A+30]) and lag k,
features come from the anchor file at A-k. Both files exist on the 14-day grid; the join
is by user_id. k enters as a feature, so one model serves every staleness and, at test
time, interpolates to k=30 (the grid forces the test vintage to 2026-01-14, 30 days
before the test anchor - inside the trained range, no extrapolation).

Leakage guard is inherited from the lag-TTA runs and states the same invariant twice:
  - target anchors A obey the 30-day gap to the validation window;
  - no (A-k) feature date may equal or pass the val anchor (it cannot: A <= val-30).
The model DOES train on rows whose feature dates are also val-side vintage dates - that
is fine and intended: what must stay unseen is the val TARGET window, not the calendar.

Emits, per exp_lib contract:
  NAME0_val / NAME0_test    k=0  (fresh member, reference)
  NAME28_val / NAME28_test  k=28 on val, k=30 on test (the decorrelated member)

Usage:
  USE_V2=1 USE_V3=1 python work/scripts/train_lagdirect.py --name lagdir
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))
from common import TEST_ANCHOR, VAL_ANCHOR, feature_cols, load_anchor, rmsle
from exp_lib import available_train_anchors, log_score, save_preds

LAGS = (0, 14, 28, 42)


def build_rows(target_anchors, lags, cols, feat_dates, n_users):
    """(features@A-k, k) -> X float32, log1p target -> y. Filled block by block.

    Concatenating the blocks as polars frames and calling to_numpy at the end peaks at
    17-25 GB for 42 blocks x 250k users x 197 columns: the frame and its numpy copy are
    alive at the same time. The OOM killer took the first run at 23 GB RSS. Writing each
    block straight into a preallocated array holds one block (~200 MB) above the 8.3 GB
    result, so the peak is the result itself.
    """
    pairs = [(A, k) for A in target_anchors for k in lags
             if (A - timedelta(days=k)) in feat_dates]
    n = len(pairs) * n_users
    print(f"  блоков {len(pairs)}, строк {n/1e6:.1f}М, X={n*(len(cols)+1)*4/1e9:.1f} ГБ",
          flush=True)
    X = np.empty((n, len(cols) + 1), dtype=np.float32)
    y = np.empty(n, dtype=np.float64)
    at = 0
    for A, k in pairs:
        tgt = load_anchor(A, columns=["user_id", "target"]).sort("user_id")
        f = load_anchor(A - timedelta(days=k), columns=["user_id"] + cols).sort("user_id")
        assert np.array_equal(f["user_id"].to_numpy(), tgt["user_id"].to_numpy())
        m = f.height
        X[at:at + m, :len(cols)] = f.select(cols).to_numpy().astype(np.float32)
        X[at:at + m, len(cols)] = np.float32(k)
        y[at:at + m] = np.log1p(np.clip(tgt["target"].to_numpy().astype(np.float64), 0, None))
        at += m
        del f, tgt
    assert at == n, f"заполнено {at} из {n}"
    return X, y


def fit(X, y, Xv, yv, seed, rounds):
    import lightgbm as lgb
    params = dict(objective="tweedie", tweedie_variance_power=1.45, metric="rmse",
                  learning_rate=0.05, num_leaves=255, min_data_in_leaf=300,
                  feature_fraction=0.75, bagging_fraction=0.8, bagging_freq=1,
                  lambda_l2=5.0, max_bin=127, num_threads=7, seed=seed, verbosity=-1)
    dtr = lgb.Dataset(X, y, free_raw_data=True)
    if Xv is None:
        return lgb.train(params, dtr, num_boost_round=rounds), rounds
    dv = lgb.Dataset(Xv, yv, reference=dtr, free_raw_data=True)
    m = lgb.train(params, dtr, num_boost_round=rounds, valid_sets=[dv],
                  callbacks=[lgb.early_stopping(200, verbose=False), lgb.log_evaluation(400)])
    return m, m.best_iteration


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="lagdir")
    ap.add_argument("--rounds", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-test", action="store_true")
    args = ap.parse_args()
    assert os.environ.get("USE_V2") and os.environ.get("USE_V3")
    t0 = time.time()

    feat_dates = set(available_train_anchors()) | {VAL_ANCHOR, TEST_ANCHOR}
    val = load_anchor(VAL_ANCHOR).sort("user_id")
    cols = feature_cols(val)
    uid = val["user_id"].to_numpy()
    yv_raw = np.clip(val["target"].to_numpy().astype(np.float64), 0, None)
    yv = np.log1p(yv_raw)

    # -------- validation-side model: target anchors respect the 30-day gap --------
    gap_cut = VAL_ANCHOR - timedelta(days=30)
    tgt_anchors = [a for a in available_train_anchors() if a <= gap_cut]
    print(f"val-модель: таргет-якоря {tgt_anchors[0]}..{tgt_anchors[-1]} ({len(tgt_anchors)})",
          flush=True)
    X, y = build_rows(tgt_anchors, LAGS, cols, feat_dates, len(uid))
    # early stopping on the STALE task (k=28): that member is the point of the exercise;
    # stopping on fresh rows would tune the wrong task again.
    v28 = load_anchor(VAL_ANCHOR - timedelta(days=28), columns=["user_id"] + cols).sort("user_id")
    Xv28 = np.column_stack([v28.select(cols).to_numpy().astype(np.float32),
                            np.full(len(uid), 28.0, dtype=np.float32)])
    del v28
    print(f"X {X.shape}, load {time.time()-t0:.0f}s", flush=True)
    rows_val = X.shape[0]
    m, it = fit(X, y, Xv28, yv, args.seed, args.rounds)
    print(f"best_iteration={it}, train {time.time()-t0:.0f}s", flush=True)
    del X

    def emit(split, feat_anchor, k, suffix, model, iters_note):
        f = load_anchor(feat_anchor, columns=["user_id"] + cols).sort("user_id")
        assert np.array_equal(f["user_id"].to_numpy(), uid)
        Z = np.column_stack([f.select(cols).to_numpy().astype(np.float32),
                             np.full(len(uid), float(k), dtype=np.float32)])
        pv = np.expm1(np.clip(model.predict(Z), 0, None))
        save_preds(f"{args.name}{suffix}", split, uid, pv)
        if split == "val":
            log_score(f"{args.name}{suffix}", rmsle(yv_raw, pv),
                      f"direct stale-conditioned tweedie, k={k} as feature; "
                      f"lags {LAGS}, {iters_note}")
        return pv

    emit("val", VAL_ANCHOR, 0, "0", m, f"it={it}")
    emit("val", VAL_ANCHOR - timedelta(days=28), 28, "28", m, f"it={it}")

    if args.no_test:
        print(f"[DONE] {time.time()-t0:.0f}s", flush=True)
        return

    # -------- test-side model: all anchors with a fully observed target --------
    tgt_all = [a for a in available_train_anchors() if a <= VAL_ANCHOR] + [VAL_ANCHOR]
    tgt_all = sorted(set(tgt_all))
    print(f"test-модель: таргет-якоря до {tgt_all[-1]} ({len(tgt_all)})", flush=True)
    Xa, ya = build_rows(tgt_all, LAGS, cols, feat_dates, len(uid))
    # same convention as train_gbdt retrain: iterations grow with the data, damped 0.7
    mult = 1.0 + 0.7 * max(Xa.shape[0] / rows_val - 1.0, 0.0)
    n_iter = max(50, int(it * mult))
    print(f"test-модель: {Xa.shape[0]} строк, {n_iter} итераций", flush=True)
    mf, _ = fit(Xa, ya, None, None, args.seed, n_iter)
    del Xa
    emit("test", TEST_ANCHOR, 0, "0", mf, "")
    # тестовое окно начинается через 30 дней после якоря 2026-01-14 -> k=30, внутри диапазона
    emit("test", date(2026, 1, 14), 30, "28", mf, "")
    print(f"[DONE] {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

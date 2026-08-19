"""febspec3 -- Feb-2025 SEASONAL SPECIALIST on the short-feature tier.

The lesson febspec2 taught (and this script encodes)
---------------------------------------------------
febspec2 took the short-feature tier and trained it on the whole weekly grid
(39 anchors, 1.94M rows). It became a good short model -- calibrated val 1.7707
against 1.8266 for the old febspec -- and therefore USELESS in the blend: optimal
weight -0.005, contribution +0.000002. The old specialist was never valuable
because it was accurate; it was valuable because it was the only model that had
seen the February->March transition, the single window in the data where the
March-8 ramp is observable. Training it well destroyed exactly that.

So febspec3 keeps the improvements that are about REPRESENTATION (short <=42d
features incl. the per-channel funnel; the two-head architecture that won on both
windows of the mirror experiment) and throws away the one that was about COVERAGE
(the wide anchor grid). Training anchors: weekly 2025-01-30..2025-02-27, five of
them, against three for the old febspec.

Protocol
  * ONE fit; it predicts both the val anchor (2026-01-14) and the test anchor
    (2026-02-13). No retrain is needed or wanted -- Feb-2025 target windows are
    ~11 months from the val window and ~12 from the test window, so neither
    evaluation overlaps training. This is how the original febspec worked too.
  * Iterations/epochs stop on a held-out anchor INSIDE the February grid
    (default 2025-01-30: shortest history, furthest from the March-8 ramp), so
    the anchors that actually carry the seasonal signal all stay in training.
  * History note: the tier uses windows <= 42d, but the data start 2025-01-01
    truncates them on the two earliest anchors (2025-01-30 has 30 days,
    2025-02-06 has 37). The original febspec accepted the same truncation
    (37-51 days). Volume features are correspondingly smaller there.

Acceptance is NOT error correlation (febspec2 had 0.9423 and still failed) but
the err_corr.py gain: positive optimal weight and gain > 0.0005.

Usage:
  POLARS_MAX_THREADS=3 .venv/bin/python work/scripts/train_febspec3.py \
      --name febspec3_mlp --config mlp2 --seeds 42,1337
  POLARS_MAX_THREADS=3 .venv/bin/python work/scripts/train_febspec3.py \
      --name febspec3_tw  --config lgb_tw145_nofun
"""
from __future__ import annotations

import os

_T = os.environ.get("THREADS", "3")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "POLARS_MAX_THREADS"):
    os.environ.setdefault(_v, _T)

import argparse  # noqa: E402
import gc  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from datetime import date, timedelta  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from common import DATA_START, rmsle  # noqa: E402
from exp_lib import log_score, save_preds  # noqa: E402
from build_features_short import (FEATS, JAN_ANCHOR, TEST_ANCHOR_, funnel_cols)  # noqa: E402
from short_family import CONFIGS, RUNNERS, build_eval, build_train, log  # noqa: E402
import mirror_val  # noqa: E402

FEB_FROM, FEB_TO = date(2025, 1, 30), date(2025, 2, 27)
DEFAULT_ES = [date(2025, 1, 30)]


def grid(a: date, b: date, step: int) -> list[date]:
    out = []
    while a <= b:
        out.append(a)
        a += timedelta(days=step)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--config", default="mlp2", choices=sorted(CONFIGS))
    ap.add_argument("--grid-from", type=str, default=FEB_FROM.isoformat())
    ap.add_argument("--grid-to", type=str, default=FEB_TO.isoformat())
    ap.add_argument("--step", type=int, default=7)
    ap.add_argument("--es-anchors", type=str,
                    default=",".join(d.isoformat() for d in DEFAULT_ES),
                    help="anchors held out of training and used only to stop")
    ap.add_argument("--cohort", type=float, default=1.0)
    ap.add_argument("--seeds", type=str, default="42")
    ap.add_argument("--threads", type=int, default=int(_T))
    ap.add_argument("--max-rounds", type=int, default=3000)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--patience", type=int, default=4)
    args = ap.parse_args()

    cfg = dict(CONFIGS[args.config])
    cfg["params"] = dict(cfg["params"])
    if cfg["kind"] == "mlp":
        cfg["params"].update(epochs=args.epochs, patience=args.patience)
    seeds = [int(s) for s in args.seeds.split(",")]
    t0 = time.time()

    g = grid(date.fromisoformat(args.grid_from), date.fromisoformat(args.grid_to), args.step)
    es_a = [date.fromisoformat(s) for s in args.es_anchors.split(",") if s]
    tr_a = [a for a in g if a not in es_a]
    assert tr_a and es_a, "need at least one training and one stopping anchor"
    short = [a for a in g if a - timedelta(days=41) < DATA_START]
    log(f"{args.name}: config={args.config} grid {g[0]}..{g[-1]} ({len(g)} anchors) "
        f"-> {len(tr_a)} train / {len(es_a)} stop; cohort {args.cohort:.0%}; seeds {seeds}")
    log(f"  truncated-history anchors (data start {DATA_START}): "
        f"{[a.isoformat() for a in short]}")

    keep = np.arange(len(FEATS)) if cfg["funnel"] else \
        np.array([i for i, c in enumerate(FEATS) if c not in set(funnel_cols())])
    X, y, _ = build_train(tr_a, args.cohort, len(FEATS))
    Xes, yes, _ = build_train(es_a, args.cohort, len(FEATS))
    uid_v, Xv, yv = build_eval(JAN_ANCHOR)
    uid_t, Xt, _ = build_eval(TEST_ANCHOR_)
    if len(keep) != len(FEATS):
        X, Xes, Xv, Xt = X[:, keep], Xes[:, keep], Xv[:, keep], Xt[:, keep]
    log(f"  train {X.shape} ({X.nbytes/1e9:.2f} GB) stop {Xes.shape} "
        f"pos_rate {(y > 0).mean():.4f} | load {time.time()-t0:.0f}s")

    # ONE fit predicts both evaluation anchors: neither window overlaps Feb-2025.
    pv_all, pt_all, infos = [], [], []
    for s in seeds:
        preds, info = RUNNERS[cfg["kind"]](X, y, Xes, yes,
                                           {"val": (Xv,), "test": (Xt,)},
                                           cfg, s, args.threads, args.max_rounds)
        pv_all.append(preds["val"])
        pt_all.append(preds["test"])
        infos.append(info)
        log(f"  seed {s}: {info} val_rmsle={rmsle(yv, preds['val']):.6f}")
        gc.collect()
    # seeds averaged in log1p -- the space the metric lives in
    pv = np.expm1(np.mean([np.log1p(p) for p in pv_all], axis=0))
    pt = np.expm1(np.mean([np.log1p(p) for p in pt_all], axis=0))

    score = rmsle(yv, pv)
    sj = mirror_val.score(uid_v, pv, "jan")
    log(f"VAL {JAN_ANCHOR} rmsle={score:.6f} honest-cal {sj['rmsle_cal']:.6f} "
        f"bias {sj['mean_log_err']:+.4f} | test mean_log1p {np.log1p(pt).mean():.4f}")
    save_preds(args.name, "val", uid_v, pv)
    save_preds(args.name, "test", uid_t, pt)
    log_score(args.name, score,
              f"Feb2025 specialist v3: {args.config} on {len(tr_a)} anchors "
              f"{tr_a[0]}..{tr_a[-1]} (stop on {es_a[0]}), short<=42d tier "
              f"{len(keep)} feats funnel={cfg['funnel']}, cohort {args.cohort:.0%}, "
              f"seeds={args.seeds}")
    print(json.dumps({"name": args.name, "config": args.config,
                      "val_rmsle": round(score, 6),
                      "val_rmsle_cal_honest": round(sj["rmsle_cal"], 6),
                      "train_anchors": [a.isoformat() for a in tr_a],
                      "train_rows": int(X.shape[0]), "info": infos,
                      "seconds": round(time.time() - t0)}), flush=True)


if __name__ == "__main__":
    main()

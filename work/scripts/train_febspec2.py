"""febspec2 -- rebuilt Feb->March seasonal specialist on the short-feature tier.

Why a rebuild. The original febspec (train_feb_specialist.py) trains on THREE
Feb-2025 anchors, 750k rows, 62 hand-picked short columns, a fixed 1500 tweedie-1.3
iterations. It scores 1.8266 on val -- by far the weakest model in the pool -- but
its error correlation with the champion blend is 0.921, the lowest in the project,
so it earns blend weight anyway. Two things were leaving value on the table:
  * only 3 anchors, all from one month;
  * no per-channel conversion funnel (search->cart->order, catalog->cart->order),
    which build_features_short.py now supplies.

What this trains
  features   build_features_short.py, every window <= 42d, so the semantics are
             identical on the val anchor, on the test anchor and on 2025-02-13
  val        anchor 2026-01-14, trained on the weekly grid up to 2025-12-15
             (gap 30 = no train target window overlaps the val window)
  test       anchor 2026-02-13, retrained on the weekly grid up to 2026-01-14
  iterations chosen by early stopping on ANCHORS HELD OUT INSIDE the training
             grid, never on the evaluated window, so the reported val RMSLE is
             not the number the stopping rule optimised
The training grid reaches back to 2025-02-17, so unlike every long-window model
in the pool this one has actually SEEN Feb->March target windows, March-8 included.

Note it cannot be scored on the mirror window: its training grid overlaps it.

Usage:
  POLARS_MAX_THREADS=2 .venv/bin/python work/scripts/train_febspec2.py \
      --name febspec2 --cohort 0.20 --threads 2
"""
from __future__ import annotations

import os

_T = os.environ.get("THREADS", "2")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "POLARS_MAX_THREADS"):
    os.environ.setdefault(_v, _T)

import argparse  # noqa: E402
import gc  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from common import rmsle  # noqa: E402
from exp_lib import log_score, save_preds  # noqa: E402
from build_features_short import (FEATS, JAN_ANCHOR, TEST_ANCHOR_, funnel_cols,  # noqa: E402
                                  jan_grid, test_grid)
from short_family import (CONFIGS, ES_EVERY, build_eval, build_train, fit_lgb,  # noqa: E402
                          log, predict_lgb)
import mirror_val  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="febspec2")
    ap.add_argument("--config", default="auto",
                    choices=sorted(CONFIGS) + ["auto"],
                    help="'auto' = best boosting config by CALIBRATED score on the "
                         "MARCH mirror window in work/reports/short_family.json "
                         "(the test-like window is the selection criterion; falls "
                         "back to lgb_tw145 when the file is absent)")
    ap.add_argument("--cohort", type=float, default=0.20)
    ap.add_argument("--step", type=int, default=7)
    ap.add_argument("--threads", type=int, default=int(_T))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-rounds", type=int, default=2000)
    ap.add_argument("--no-test", action="store_true")
    args = ap.parse_args()

    if args.config == "auto":
        from common import REPORTS_DIR
        p = REPORTS_DIR / "short_family.json"
        args.config = "lgb_tw145"
        if p.exists():
            rows = [r for r in json.loads(p.read_text()).get("configs", [])
                    if CONFIGS.get(r["name"], {}).get("kind") in ("lgb", "two_stage")]
            if rows:
                args.config = min(rows, key=lambda r: r["mar_cal"])["name"]
        log(f"auto-selected config {args.config} (best mar_cal in {p.name})")
    cfg = CONFIGS[args.config]
    assert cfg["kind"] in ("lgb", "two_stage"), "only the boosting arms are wired here"
    t0 = time.time()

    vg = jan_grid(args.step)
    es_idx = set(range(0, len(vg), ES_EVERY))
    tr_a = [a for i, a in enumerate(vg) if i not in es_idx]
    es_a = [a for i, a in enumerate(vg) if i in es_idx]
    log(f"val grid {vg[0]}..{vg[-1]}: {len(vg)} anchors "
        f"({len(tr_a)} train / {len(es_a)} early-stop), cohort {args.cohort:.0%}, "
        f"config {args.config}")

    keep = np.arange(len(FEATS)) if cfg["funnel"] else \
        np.array([i for i, c in enumerate(FEATS) if c not in set(funnel_cols())])
    X, y, _ = build_train(tr_a, args.cohort, len(FEATS))
    Xes, yes, _ = build_train(es_a, args.cohort, len(FEATS))
    if len(keep) != len(FEATS):
        X, Xes = X[:, keep], Xes[:, keep]
    uid_v, Xv, yv = build_eval(JAN_ANCHOR)
    Xv = Xv[:, keep]
    log(f"train {X.shape} ({X.nbytes/1e9:.2f} GB) es {Xes.shape} val {Xv.shape} "
        f"| {time.time()-t0:.0f}s")

    m, best_it = fit_lgb(X, y, Xes, yes, cfg["params"], args.seed, args.threads,
                         args.max_rounds)
    pv = np.expm1(np.clip(predict_lgb(m, Xv), 0, None))
    score = rmsle(yv, pv)
    sj = mirror_val.score(uid_v, pv, "jan")
    n_rows = X.shape[0]
    log(f"VAL {JAN_ANCHOR} rmsle={score:.6f} (honest-cal {sj['rmsle_cal']:.6f}, "
        f"bias {sj['mean_log_err']:+.4f}) best_iter={best_it}")
    imp = sorted(zip([FEATS[i] for i in keep], m.feature_importance("gain")),
                 key=lambda t: -t[1])[:12]
    log("top gain: " + ", ".join(f"{f}={g:.0f}" for f, g in imp))
    save_preds(args.name, "val", uid_v, pv)
    del X, Xes, Xv, m
    gc.collect()

    out = {"name": args.name, "config": args.config, "val_rmsle": round(score, 6),
           "val_rmsle_cal": round(sj["rmsle_cal"], 6), "best_iter": int(best_it),
           "val_anchors": len(tr_a), "train_rows": int(n_rows),
           "cohort": args.cohort, "seconds": round(time.time() - t0)}
    if args.no_test:
        print(json.dumps(out), flush=True)
        return

    # ---- test: same recipe, grid extended to 2026-01-14 (gap 30 to the test window)
    tg = test_grid(args.step)
    log(f"test grid {tg[0]}..{tg[-1]}: {len(tg)} anchors")
    Xa, ya, _ = build_train(tg, args.cohort, len(FEATS))
    if len(keep) != len(FEATS):
        Xa = Xa[:, keep]
    ratio = Xa.shape[0] / max(n_rows, 1)
    n_it = max(50, int(best_it * (1.0 + 0.7 * max(ratio - 1.0, 0.0))))
    log(f"retrain rows {Xa.shape[0]} (ratio {ratio:.3f}) -> {n_it} iters")

    import lightgbm as lgb
    from short_family import LGB_BASE
    p = dict(LGB_BASE, num_threads=args.threads, seed=args.seed)
    p.update(cfg["params"])
    mf = lgb.train(p, lgb.Dataset(Xa, ya, free_raw_data=True), num_boost_round=n_it)
    del Xa
    gc.collect()
    uid_t, Xt, _ = build_eval(TEST_ANCHOR_)
    pt = np.expm1(np.clip(predict_lgb(mf, Xt[:, keep]), 0, None))
    save_preds(args.name, "test", uid_t, pt)
    log(f"test pred mean_log1p={np.log1p(pt).mean():.4f} share>1={(pt > 1).mean():.4f} "
        f"| val pred mean_log1p={np.log1p(pv).mean():.4f}")

    log_score(args.name, score,
              f"short<=42d tier ({len(keep)} feats, funnel={cfg['funnel']}) "
              f"{args.config}; {len(tr_a)} weekly anchors from {vg[0]} cohort "
              f"{args.cohort:.0%}; gap30; iters from held-out anchors ({best_it})")
    out.update(test_anchors=len(tg), test_iters=n_it, seconds=round(time.time() - t0))
    print(json.dumps(out), flush=True)


if __name__ == "__main__":
    main()

"""Verdict on the early-stopping CRITERION of the BOOSTING trainer: raw vs calibrated.

Same experiment as es_verdict.py (fusion) and es_verdict_mlp.py (the three MLPs),
moved to train_gbdt.py.  The defect (KNOWLEDGE.md, last section): early stopping was
driven by the trainer's raw validation metric, but every prediction file goes through
calibrate.py's binned log-shift before it reaches a blend, and that calibration
REWRITES THE LEVEL of the forecast.  So the raw criterion spends its stopping decision
on a level that is about to be overwritten for free, and pays for it in RANKING, which
calibration preserves.  Measured on fusion_v3, three seeds of three: calibrated val
1.670330 -> 1.668676, 1.672695 -> 1.668725, 1.671173 -> 1.668446, mean -0.0028.

Boosting is half the pool by weight (c_ts2 0.25 + twl_v7 0.055 in the current blend,
plus countaov/behavonly/channel2 which have their own trainers), so the same defect
there is worth as much again.

TWO DIFFERENCES FROM THE NEURAL TRAINERS.
 1. Evaluation count.  LightGBM evaluates EVERY boosting iteration, thousands of times
    per run, against ~10 for an MLP epoch loop.  At 0.06 s per calibration that would
    double the wall time, so the calibrated metric is recomputed every --es-period
    iterations (default 10) and cached in between.  Patience is still counted in
    ITERATIONS — a repeated value is never an improvement — so only the grid of
    candidate stopping points gets coarser.  A coarser grid can only HURT the cal arm,
    which makes any measured gain conservative.
 2. Two stages.  c_ts2 is P(y>0) x E[log1p|y>0] and each stage early-stops on its own.
    The calibrated criterion is meaningful for the FINAL forecast only, so it is
    applied to stage 2 and scores p_1 * mu (the whole forecast), with stage 1 frozen.
    Stage 1 keeps AUC: when it trains there is no stage 2 yet, and AUC is a pure
    ranking metric, hence immune to the level-vs-ranking defect being fixed here.

Arms: --es-metric cal|raw, everything else identical (same seed, same anchors, same
gap, same params).  The flag never touches training — same trees, same rows — it only
decides which iteration is kept as best_iteration.  The paired runs are --no-test on
purpose: a name without a matching _test.parquet is skipped by blend_reopt.discover(),
so the arms cannot enter a pool as "same val, different test" duplicates (KNOWLEDGE.md:
that silently breaks the weight/prediction match).

Comparison is done AFTER calibration only (raw comparisons misled eight times in one
night).  Two honest estimators are reported: the 2-fold user split of
work/preds_pack/README.md (the number the trainer itself prints) and the 5-fold
cross-fitted one used by the other two verdict scripts.

Output: work/reports/es_verdict_gbdt.json (+ stdout summary)
Run: POLARS_MAX_THREADS=3 .venv/bin/python work/scripts/es_verdict_gbdt.py
"""
from __future__ import annotations

import os

os.environ.setdefault("POLARS_MAX_THREADS", "3")

import argparse  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from calibrate import apply_shifts, fit_shifts  # noqa: E402
from common import FEATURES_DIR, PREDS_DIR, REPORTS_DIR, VAL_ANCHOR, rmsle  # noqa: E402

FAMILIES = {"twl": "lgb tweedie-on-log1p, 8 anchors (twl_v7 config)",
            "cts2": "lgb two_stage, 14 anchors (c_ts2 config)"}
SEEDS = [42]
NAME_FMT = "es_{fam}_{arm}{seed}"
KFOLD, BINS, SEED = 5, 24, 0
NOISE = 0.0003            # порог приёмки; уровень шума одного замера 0.000022


def xfit_calibrated(lp: np.ndarray, ly: np.ndarray) -> np.ndarray:
    """Cross-fitted binned log-shift calibration, K folds by user (no in-sample optimism)."""
    rng = np.random.default_rng(SEED)
    fold = rng.integers(0, KFOLD, len(lp))
    out = np.empty_like(lp)
    for f in range(KFOLD):
        tr, te = fold != f, fold == f
        c, s = fit_shifts(lp[tr], ly[tr], BINS)
        out[te] = apply_shifts(lp[te], c, s)
    return out


def half_calibrated(lp: np.ndarray, ly: np.ndarray) -> np.ndarray:
    """The 2-fold honest split of work/preds_pack/README.md, i.e. what the trainer prints."""
    rng = np.random.default_rng(SEED)
    half = rng.permutation(len(ly)) < len(ly) // 2
    out = np.empty_like(lp)
    for m in (half, ~half):
        c, s = fit_shifts(lp[m], ly[m], BINS)
        out[~m] = apply_shifts(lp[~m], c, s)
    return out


def load_pred(name: str, uid: np.ndarray) -> np.ndarray | None:
    p = PREDS_DIR / f"{name}_val.parquet"
    if not p.exists():
        return None
    d = pl.read_parquet(p).sort("user_id")
    assert np.array_equal(d["user_id"].to_numpy(), uid), f"user_id mismatch in {p}"
    return d["pred"].to_numpy().astype(np.float64)


def log_facts(name: str) -> dict:
    """Stopping iteration and calibration cost, read out of the job log."""
    p = REPORTS_DIR / f"job_{name}.log"
    if not p.exists():
        return {}
    txt = p.read_text(errors="replace")
    stop = re.findall(r"stop_iter ([^;]+);", txt)
    cost = re.findall(r"(\d+) calibrations, ([\d.]+)s total, ([\d.]+)s each", txt)
    out = {"stop_iter": stop[-1].strip() if stop else None}
    if cost:
        out.update(n_cal=int(cost[-1][0]), cal_total_s=float(cost[-1][1]),
                   cal_per_eval_s=float(cost[-1][2]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", default=",".join(FAMILIES))
    args = ap.parse_args()
    fams = [f for f in args.families.split(",") if f]
    assert all(f in FAMILIES for f in fams), f"unknown family in {fams}"

    val = pl.read_parquet(FEATURES_DIR / f"anchor={VAL_ANCHOR.isoformat()}.parquet",
                          columns=["user_id", "target"]).sort("user_id")
    uid = val["user_id"].to_numpy()
    y = val["target"].to_numpy().astype(np.float64)
    ly = np.log1p(y)

    pairs, lcal, missing = [], {}, []
    for fam in fams:
        for sd in SEEDS:
            names = {a: NAME_FMT.format(fam=fam, arm=a, seed=sd) for a in ("raw", "cal")}
            preds = {a: load_pred(n, uid) for a, n in names.items()}
            if any(v is None for v in preds.values()):
                missing += [names[a] for a, v in preds.items() if v is None]
                continue
            row = {"family": fam, "config": FAMILIES[fam], "seed": sd}
            for arm, p_raw in preds.items():
                lp = np.log1p(np.clip(p_raw, 0, None))
                lc = xfit_calibrated(lp, ly)
                lcal[(fam, sd, arm)] = lc
                row[f"{arm}_raw"] = round(rmsle(y, np.expm1(lp)), 6)
                row[f"{arm}_cal"] = round(rmsle(y, np.expm1(lc)), 6)
                row[f"{arm}_cal_2fold"] = round(
                    rmsle(y, np.expm1(half_calibrated(lp, ly))), 6)
                row[f"{arm}_log"] = log_facts(names[arm])
            row["delta"] = round(row["cal_cal"] - row["raw_cal"], 6)
            row["delta_2fold"] = round(row["cal_cal_2fold"] - row["raw_cal_2fold"], 6)
            row["delta_raw"] = round(row["cal_raw"] - row["raw_raw"], 6)
            row["same_checkpoint"] = bool(np.array_equal(preds["raw"], preds["cal"]))
            pairs.append(row)
            print(f"{fam} s{sd}: raw-ES raw {row['raw_raw']:.6f} cal {row['raw_cal']:.6f} "
                  f"(stop {row['raw_log'].get('stop_iter')}) | cal-ES raw {row['cal_raw']:.6f} "
                  f"cal {row['cal_cal']:.6f} (stop {row['cal_log'].get('stop_iter')}) | "
                  f"delta_cal {row['delta']:+.6f} (2fold {row['delta_2fold']:+.6f})",
                  flush=True)

    if missing:
        print(f"MISSING preds ({len(missing)}), queue jobs not finished yet: {missing}",
              flush=True)
    if not pairs:
        return

    per_family = {}
    for fam in sorted({p["family"] for p in pairs}):
        rows = [p for p in pairs if p["family"] == fam]
        d_mean = float(np.mean([r["delta"] for r in rows]))
        e_raw = np.concatenate([(lcal[(fam, r["seed"], "raw")] - ly) ** 2 for r in rows])
        e_cal = np.concatenate([(lcal[(fam, r["seed"], "cal")] - ly) ** 2 for r in rows])
        rng, n, boot = np.random.default_rng(1), len(ly), []
        for _ in range(400):
            i = rng.integers(0, n, n)
            j = np.concatenate([i + k * n for k in range(len(rows))])
            boot.append(np.sqrt(e_cal[j].mean()) - np.sqrt(e_raw[j].mean()))
        lo, hi = np.percentile(boot, [2.5, 97.5])
        per_family[fam] = {
            "n_seeds": len(rows),
            "delta_cal_mean": round(d_mean, 6),
            "delta_cal_ci95": [round(float(lo), 6), round(float(hi), 6)],
            "verdict": ("cal-ES HELPS" if d_mean < -NOISE else
                        "cal-ES HURTS" if d_mean > NOISE else
                        f"NOISE (|delta| <= {NOISE})"),
        }
        print(f"[{fam}] delta_cal mean {d_mean:+.6f} ci95 [{lo:+.6f}, {hi:+.6f}] over "
              f"{len(rows)} seed(s) -> {per_family[fam]['verdict']}", flush=True)

    d_all = float(np.mean([p["delta"] for p in pairs]))
    out = {"pairs": pairs, "per_family": per_family,
           "delta_cal_mean_all": round(d_all, 6), "missing": missing,
           "two_stage_handling": "stage2 stops on the honest calibrated RMSLE of the "
                                 "FINAL forecast p1*mu (stage 1 frozen); stage 1 keeps "
                                 "AUC, which is level-free and therefore immune to the "
                                 "defect, and has no final forecast to score anyway",
           "verdict": ("cal-ES HELPS" if d_all < -NOISE else
                       "cal-ES HURTS" if d_all > NOISE else
                       f"NOISE (|delta| <= {NOISE})")}
    (REPORTS_DIR / "es_verdict_gbdt.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1))
    print("\n=== RAW JSON ===")
    print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()

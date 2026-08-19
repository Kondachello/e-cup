"""Verdict on the early-stopping CRITERION of fusion_v3: raw val RMSLE vs calibrated.

The defect being tested (KNOWLEDGE.md, last section): early stopping was driven by the
raw val RMSLE, but every prediction file is passed through calibrate.py's binned
log-shift before it reaches a blend, and that calibration rewrites the LEVEL of the
forecast.  So the raw criterion spends its checkpoint choice on a level that is about to
be overwritten for free, and pays for it in RANKING, which calibration preserves.
Measured on the 5-seed average: finer evaluation moved raw 1.681560 -> 1.677117 while
the calibrated score got WORSE, 1.668594 -> 1.669033.

Arms (train_fusion3.py --es-metric raw|cal), everything else identical — same seed, same
anchors, same tensors, --eval-every 246 --n-ch 12 --final.  The flag never touches
training: same batches, same steps, same RNG streams.  It only decides WHICH phase-1
checkpoint is kept, so:
  * the raw arm must reproduce the existing fusion_v3_fine{seed} runs BIT FOR BIT
    (this is the default-path regression test at full scale), and
  * the --final phase-2 test preds, retrained with no early stopping at all, must be
    bit-identical BETWEEN the arms.
Both are asserted below; a mismatch invalidates the comparison rather than proving a win.
It stays legal for an arm's VAL preds to coincide too — that just means both criteria
picked the same checkpoint (it already happened to seed 1337 on the eval-frequency grid).

Comparison is done AFTER calibration only (raw comparisons misled eight times in one
night).  Calibration here is the cross-fitted binned log-shift, K folds by user, which
is unbiased; calibrate.py's own in-sample number is reported alongside for continuity
with the numbers quoted in KNOWLEDGE.md.

Output: work/reports/es_verdict.json (+ stdout summary)
Run: POLARS_MAX_THREADS=3 .venv/bin/python work/scripts/es_verdict.py [--archive-identical]
"""
from __future__ import annotations

import os

os.environ.setdefault("POLARS_MAX_THREADS", "3")

import argparse  # noqa: E402
import json  # noqa: E402
import shutil  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from calibrate import apply_shifts, fit_shifts  # noqa: E402
from common import FEATURES_DIR, PREDS_DIR, REPORTS_DIR, VAL_ANCHOR, rmsle  # noqa: E402

SEEDS = [555, 7]
RAW_FMT, CAL_FMT = "fusion_v3_esraw{}", "fusion_v3_escal{}"
BASE_FMT = "fusion_v3_fine{}"        # the pre-existing --es-metric raw run of that seed
KFOLD, BINS, SEED = 5, 24, 0
NOISE = 0.0003                       # порог приёмки; уровень шума одного замера 0.000022


def xfit_calibrated(lp: np.ndarray, ly: np.ndarray) -> np.ndarray:
    """Cross-fitted binned log-shift calibration (no in-sample optimism)."""
    rng = np.random.default_rng(SEED)
    fold = rng.integers(0, KFOLD, len(lp))
    out = np.empty_like(lp)
    for f in range(KFOLD):
        tr, te = fold != f, fold == f
        c, s = fit_shifts(lp[tr], ly[tr], BINS)
        out[te] = apply_shifts(lp[te], c, s)
    return out


def insample_calibrated(lp: np.ndarray, ly: np.ndarray) -> np.ndarray:
    """Exactly what calibrate.py writes into NAME_cal (fit and apply on all of val)."""
    c, s = fit_shifts(lp, ly, BINS)
    return apply_shifts(lp, c, s)


def load_pred(name: str, split: str, uid: np.ndarray) -> np.ndarray | None:
    p = PREDS_DIR / f"{name}_{split}.parquet"
    if not p.exists():
        return None
    d = pl.read_parquet(p).sort("user_id")
    assert np.array_equal(d["user_id"].to_numpy(), uid), f"user_id mismatch in {p}"
    return d["pred"].to_numpy().astype(np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive-identical", action="store_true",
                    help="move the raw-arm files out of work/preds once they are proven "
                         "bit-identical to fusion_v3_fine{seed}: they are exact duplicates "
                         "and a duplicate column is a collinear nuisance for blend_reopt")
    args = ap.parse_args()

    val = pl.read_parquet(FEATURES_DIR / f"anchor={VAL_ANCHOR.isoformat()}.parquet",
                          columns=["user_id", "target"]).sort("user_id")
    uid = val["user_id"].to_numpy()
    y = val["target"].to_numpy().astype(np.float64)
    ly = np.log1p(y)
    uid_t = pl.read_parquet(PREDS_DIR / f"{BASE_FMT.format(SEEDS[0])}_test.parquet") \
        .sort("user_id")["user_id"].to_numpy()

    pairs, identity, lcal, missing = [], {}, {}, []
    for sd in SEEDS:
        names = {"raw": RAW_FMT.format(sd), "cal": CAL_FMT.format(sd)}
        preds = {a: load_pred(n, "val", uid) for a, n in names.items()}
        if any(v is None for v in preds.values()):
            missing += [names[a] for a, v in preds.items() if v is None]
            continue

        row = {"seed": sd}
        for arm, lp_raw in preds.items():
            lp = np.log1p(np.clip(lp_raw, 0, None))
            lc = xfit_calibrated(lp, ly)
            lcal[(sd, arm)] = lc
            row[f"{arm}_raw"] = round(rmsle(y, np.expm1(lp)), 6)
            row[f"{arm}_cal"] = round(rmsle(y, np.expm1(lc)), 6)
            row[f"{arm}_cal_insample"] = round(
                rmsle(y, np.expm1(insample_calibrated(lp, ly))), 6)
        row["delta"] = round(row["cal_cal"] - row["raw_cal"], 6)
        row["delta_raw"] = round(row["cal_raw"] - row["raw_raw"], 6)
        row["same_checkpoint"] = bool(np.array_equal(preds["raw"], preds["cal"]))
        pairs.append(row)

        # --- regression tests, see module docstring ---
        base_v = load_pred(BASE_FMT.format(sd), "val", uid)
        t_raw = load_pred(names["raw"], "test", uid_t)
        t_cal = load_pred(names["cal"], "test", uid_t)
        base_t = load_pred(BASE_FMT.format(sd), "test", uid_t)
        identity[str(sd)] = {
            "raw_arm_reproduces_fine_val": (None if base_v is None
                                            else bool(np.array_equal(preds["raw"], base_v))),
            "raw_arm_reproduces_fine_test": (None if base_t is None or t_raw is None
                                             else bool(np.array_equal(t_raw, base_t))),
            "test_preds_equal_across_arms": (None if t_raw is None or t_cal is None
                                             else bool(np.array_equal(t_raw, t_cal))),
        }
        print(f"seed {sd}: raw-ES raw {row['raw_raw']:.6f} cal {row['raw_cal']:.6f} | "
              f"cal-ES raw {row['cal_raw']:.6f} cal {row['cal_cal']:.6f} | "
              f"delta_cal {row['delta']:+.6f}", flush=True)
        print(f"          identity {identity[str(sd)]}", flush=True)

    if missing:
        print(f"MISSING preds, run the queue jobs first: {missing}", flush=True)
        return

    # --- two-seed average, log1p space, the way merge_seeds.py builds pool members ---
    avg = {}
    for arm in ("raw", "cal"):
        lp = np.mean([np.log1p(np.clip(load_pred(
            (RAW_FMT if arm == "raw" else CAL_FMT).format(sd), "val", uid), 0, None))
            for sd in SEEDS], axis=0)
        avg[arm] = {"raw": round(rmsle(y, np.expm1(lp)), 6),
                    "cal": round(rmsle(y, np.expm1(xfit_calibrated(lp, ly))), 6),
                    "cal_insample": round(rmsle(y, np.expm1(insample_calibrated(lp, ly))), 6)}
        print(f"avg({len(SEEDS)} seeds) es={arm}: raw {avg[arm]['raw']:.6f} "
              f"cal(xfit) {avg[arm]['cal']:.6f} cal(in-sample) {avg[arm]['cal_insample']:.6f}",
              flush=True)

    # paired bootstrap over users on the pooled calibrated squared errors of both seeds
    e_raw = np.concatenate([(lcal[(sd, "raw")] - ly) ** 2 for sd in SEEDS])
    e_cal = np.concatenate([(lcal[(sd, "cal")] - ly) ** 2 for sd in SEEDS])
    rng = np.random.default_rng(1)
    n = len(ly)
    boot = []
    for _ in range(400):
        i = rng.integers(0, n, n)
        j = np.concatenate([i + k * n for k in range(len(SEEDS))])
        boot.append(np.sqrt(e_cal[j].mean()) - np.sqrt(e_raw[j].mean()))
    lo, hi = np.percentile(boot, [2.5, 97.5])
    d_mean = float(np.mean([p["delta"] for p in pairs]))

    all_ok = all(v is not False for d in identity.values() for v in d.values())
    out = {
        "pairs": pairs,
        "avg": avg,
        "delta_cal_mean": round(d_mean, 6),
        "delta_cal_ci95": [round(float(lo), 6), round(float(hi), 6)],
        "identity": identity,
        "default_path_identical": all_ok,
        "verdict": ("BROKEN: identity checks failed, comparison invalid" if not all_ok else
                    "cal-ES HELPS" if d_mean < -NOISE else
                    "cal-ES HURTS" if d_mean > NOISE else
                    f"NOISE (|delta| <= {NOISE})"),
    }

    if args.archive_identical:
        dst = PREDS_DIR / "es_arm_dupes"
        moved = []
        for sd in SEEDS:
            idn = identity[str(sd)]
            if idn["raw_arm_reproduces_fine_val"] and idn["raw_arm_reproduces_fine_test"]:
                dst.mkdir(exist_ok=True)
                for split in ("val", "test"):
                    for nm in (RAW_FMT.format(sd), RAW_FMT.format(sd) + "_cal"):
                        p = PREDS_DIR / f"{nm}_{split}.parquet"
                        if p.exists():
                            shutil.move(str(p), dst / p.name)
                            moved.append(p.name)
        out["archived"] = moved
        print(f"archived {len(moved)} duplicate files -> {dst}", flush=True)

    (REPORTS_DIR / "es_verdict.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print("\n=== RAW JSON ===")
    print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()

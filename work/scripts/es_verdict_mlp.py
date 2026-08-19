"""Verdict on the early-stopping CRITERION of the three MLP trainers: raw vs calibrated.

Same experiment as es_verdict.py, moved from train_fusion3.py to train_mlpziln.py /
train_mlpbin.py / train_mlp2.py.  The defect (KNOWLEDGE.md, last section): early
stopping was driven by the raw val RMSLE, but every prediction file goes through
calibrate.py's binned log-shift before it reaches a blend, and that calibration
rewrites the LEVEL of the forecast.  So the raw criterion spends its checkpoint
choice on a level that is about to be overwritten for free, and pays for it in
RANKING, which calibration preserves.  Measured on fusion_v3: seed 555 calibrated
1.670330 -> 1.668676, seed 42 1.672695 -> 1.668725, mean -0.0028.

ONE IMPORTANT DIFFERENCE from fusion.  These trainers early-stop on EPOCHS, not
steps, and they have few evaluation points to begin with (historically the best
epoch is 1-10 out of 30, patience 4).  If the calibrated optimum lies later, as it
did for fusion (step 738 -> 2706), the epoch budget itself can become the binding
constraint.  train_one() therefore prints "БЮДЖЕТ ЭПОХ ИСЧЕРПАН" when the loop ends
because it ran out of epochs rather than out of patience; this script reads that
marker out of the job logs and reports it as `epochs_binding`.

Arms: --es-metric raw|cal, everything else identical (same seed, same 14 anchors,
same gap, same batch, same epochs).  The flag never touches training — same batches,
same steps, same RNG streams — it only decides WHICH epoch's checkpoint is kept and
when patience runs out.  The paired runs are --no-test on purpose: this is a VAL
measurement, and a name without a matching _test.parquet is skipped by
blend_reopt.discover(), so the arms cannot enter a pool as "same val, different
test" duplicates (KNOWLEDGE.md: that silently breaks the weight/prediction match).

Comparison is done AFTER calibration only (raw comparisons misled eight times in one
night).  Calibration here is the cross-fitted binned log-shift, K folds by user,
which is unbiased; the in-sample number calibrate.py itself would print is reported
alongside for continuity with the numbers quoted in KNOWLEDGE.md.

Output: work/reports/es_verdict_mlp.json (+ stdout summary)
Run: POLARS_MAX_THREADS=3 .venv/bin/python work/scripts/es_verdict_mlp.py [--families ziln,bin,mlp2]
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

FAMILIES = {"ziln": "train_mlpziln.py", "bin": "train_mlpbin.py",
            "mlp2": "train_mlp2.py"}
SEEDS = [42, 1337]
NAME_FMT = "es_{fam}_{arm}{seed}"
KFOLD, BINS, SEED = 5, 24, 0
NOISE = 0.0003            # порог приёмки; уровень шума одного замера 0.000022
BUDGET_MARK = "БЮДЖЕТ ЭПОХ ИСЧЕРПАН"


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


def load_pred(name: str, uid: np.ndarray) -> np.ndarray | None:
    p = PREDS_DIR / f"{name}_val.parquet"
    if not p.exists():
        return None
    d = pl.read_parquet(p).sort("user_id")
    assert np.array_equal(d["user_id"].to_numpy(), uid), f"user_id mismatch in {p}"
    return d["pred"].to_numpy().astype(np.float64)


def log_facts(name: str) -> dict:
    """best_epoch, epochs run and the epoch-budget marker, read from the job log."""
    p = REPORTS_DIR / f"job_{name}.log"
    if not p.exists():
        return {}
    txt = p.read_text(errors="replace")
    be = re.findall(r"best_epoch=(\d+)", txt)
    eps = re.findall(r"^\[s\d+\] ep (\d+) ", txt, flags=re.M)
    return {"best_epoch": int(be[-1]) if be else None,
            "epochs_run": int(eps[-1]) if eps else None,
            "budget_exhausted": BUDGET_MARK in txt}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", default=",".join(FAMILIES),
                    help="comma-separated subset of " + ",".join(FAMILIES))
    args = ap.parse_args()
    fams = [f for f in args.families.split(",") if f]
    assert all(f in FAMILIES for f in fams), f"unknown family in {fams}"

    val = pl.read_parquet(FEATURES_DIR / f"anchor={VAL_ANCHOR.isoformat()}.parquet",
                          columns=["user_id", "target"]).sort("user_id")
    uid = val["user_id"].to_numpy()
    y = val["target"].to_numpy().astype(np.float64)
    ly = np.log1p(y)

    pairs, lcal, missing, budget = [], {}, [], {}
    for fam in fams:
        for sd in SEEDS:
            names = {a: NAME_FMT.format(fam=fam, arm=a, seed=sd)
                     for a in ("raw", "cal")}
            preds = {a: load_pred(n, uid) for a, n in names.items()}
            if any(v is None for v in preds.values()):
                missing += [names[a] for a, v in preds.items() if v is None]
                continue
            row = {"trainer": FAMILIES[fam], "family": fam, "seed": sd}
            for arm, p_raw in preds.items():
                lp = np.log1p(np.clip(p_raw, 0, None))
                lc = xfit_calibrated(lp, ly)
                lcal[(fam, sd, arm)] = lc
                row[f"{arm}_raw"] = round(rmsle(y, np.expm1(lp)), 6)
                row[f"{arm}_cal"] = round(rmsle(y, np.expm1(lc)), 6)
                row[f"{arm}_cal_insample"] = round(
                    rmsle(y, np.expm1(insample_calibrated(lp, ly))), 6)
                row[f"{arm}_log"] = log_facts(names[arm])
            row["delta"] = round(row["cal_cal"] - row["raw_cal"], 6)
            row["delta_raw"] = round(row["cal_raw"] - row["raw_raw"], 6)
            row["same_checkpoint"] = bool(np.array_equal(preds["raw"], preds["cal"]))
            budget[f"{fam}_s{sd}"] = bool(row["cal_log"].get("budget_exhausted"))
            pairs.append(row)
            print(f"{fam} s{sd}: raw-ES raw {row['raw_raw']:.6f} cal {row['raw_cal']:.6f} "
                  f"(ep {row['raw_log'].get('best_epoch')}/{row['raw_log'].get('epochs_run')}) | "
                  f"cal-ES raw {row['cal_raw']:.6f} cal {row['cal_cal']:.6f} "
                  f"(ep {row['cal_log'].get('best_epoch')}/{row['cal_log'].get('epochs_run')}"
                  f"{', БЮДЖЕТ ИСЧЕРПАН' if budget[f'{fam}_s{sd}'] else ''}) | "
                  f"delta_cal {row['delta']:+.6f}", flush=True)

    if missing:
        print(f"MISSING preds ({len(missing)}), queue jobs not finished yet: "
              f"{missing}", flush=True)
    if not pairs:
        return

    # per-family means + paired bootstrap over users on the pooled calibrated errors
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
            "epochs_binding": any(budget.get(f"{fam}_s{r['seed']}") for r in rows),
            "verdict": ("cal-ES HELPS" if d_mean < -NOISE else
                        "cal-ES HURTS" if d_mean > NOISE else
                        f"NOISE (|delta| <= {NOISE})"),
        }
        print(f"[{fam}] delta_cal mean {d_mean:+.6f} "
              f"ci95 [{lo:+.6f}, {hi:+.6f}] over {len(rows)} seeds -> "
              f"{per_family[fam]['verdict']}"
              f"{'  | ЭПОХИ ОГРАНИЧИВАЮТ' if per_family[fam]['epochs_binding'] else ''}",
              flush=True)

    d_all = float(np.mean([p["delta"] for p in pairs]))
    out = {
        "pairs": pairs,
        "per_family": per_family,
        "delta_cal_mean_all": round(d_all, 6),
        "epochs_binding_any": any(budget.values()),
        "missing": missing,
        "verdict": ("cal-ES HELPS" if d_all < -NOISE else
                    "cal-ES HURTS" if d_all > NOISE else
                    f"NOISE (|delta| <= {NOISE})"),
    }
    (REPORTS_DIR / "es_verdict_mlp.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1))
    print("\n=== RAW JSON ===")
    print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()

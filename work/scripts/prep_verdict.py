"""Verdict on two never-swept defaults of the tabular MLP trainers.

Both were written once, in the first version, and never questioned — the same
kind of default as the raw early-stopping criterion that turned out to cost
0.0028 per model.

  1. FEATURE PREPROCESSING (--feat-prep, featprep.py, shared by mlp2 / mlpbin /
     mlpziln).  The historical path winsorises every feature at the train p1/p99
     before standardising.  On money features with a heavy right tail that
     throws away exactly the information that separates big spenders, while the
     target is itself heavy-tailed and the metric lives in logs.  Arms:
     clip99 (historical) / clip999 / noclip / signlog / rank.
  2. OUTPUT PARAMETRISATION of mlpbin (--k-bins).  31 positive quantile bins,
     chosen once.  More bins = finer decode grid but fewer rows per class.

The third suspect, the Gauss-Hermite node count of the mlpziln decode, needs no
training run at all and is settled in the header of that trainer: softplus is
analytic, the reachable head box is sigma <= 2.6 / mu 2.8..6.3, and 20 nodes sit
within 7e-9 of exact there.  Measured on a real trained model, gh in
{4,8,...,128} give the SAME honest calibrated val score to nine decimals
(max per-row |gh20 - gh128| = 1.4e-6 log1p, zero rows above 1e-5).

PROTOCOL.  Every arm is paired: same seed, same 14 anchors, same gap, same
epochs/batch/lr, --es-metric cal on both sides (without it the measurement would
be dominated by the known early-stopping defect instead of the change under
test), --no-test so the arms cannot enter a blend pool as "same val, different
test" duplicates.  The baseline arm is the already-existing es_{fam}_cal{seed}
run, which differs from a variant arm in exactly one flag.

COMPARISON IS CALIBRATED ONLY.  Raw val scores misled this project eight times.
The headline number is the honest 2-fold split from work/preds_pack/README.md
(half the users fit the shifts, the other half is scored, and vice versa), using
calibrate.py's own fit_shifts/apply_shifts.  Two robustness columns are printed
next to it: the same estimator with a different split seed, and the 5-fold
cross-fit, so a verdict that only survives one particular split is visible.

Output: work/reports/prep_verdict.json (+ stdout summary)
Run: POLARS_MAX_THREADS=3 .venv/bin/python work/scripts/prep_verdict.py
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

FAMILIES = ("mlp2", "bin", "ziln")
SEEDS = (42, 1337)
PREP_VARIANTS = ("signlog", "clip999", "rank", "noclip")
KBINS_VARIANTS = (15, 63, 127)
BASELINE = "es_{fam}_cal{seed}"
PREP_NAME = "fp_{fam}_{var}_{seed}"
KB_NAME = "kb_bin_k{k}_{seed}"
BINS = 24
THRESH = 0.0003           # порог осмысленности; шум одного замера 0.000022
BUDGET_MARK = "БЮДЖЕТ ЭПОХ ИСЧЕРПАН"


def honest_2fold(lp: np.ndarray, ly: np.ndarray, bins: int = BINS, seed: int = 0):
    """work/preds_pack/README.md calibrate_honest, on calibrate.py's primitives."""
    rng = np.random.default_rng(seed)
    half = rng.permutation(len(ly)) < len(ly) // 2
    out = np.empty_like(lp)
    for m in (half, ~half):
        c, s = fit_shifts(lp[m], ly[m], bins)
        out[~m] = apply_shifts(lp[~m], c, s)
    return out


def xfit_5(lp: np.ndarray, ly: np.ndarray, bins: int = BINS, seed: int = 0):
    rng = np.random.default_rng(seed)
    fold = rng.integers(0, 5, len(lp))
    out = np.empty_like(lp)
    for f in range(5):
        c, s = fit_shifts(lp[fold != f], ly[fold != f], bins)
        out[fold == f] = apply_shifts(lp[fold == f], c, s)
    return out


def load_lp(name: str, uid: np.ndarray):
    p = PREDS_DIR / f"{name}_val.parquet"
    if not p.exists():
        return None
    d = pl.read_parquet(p).sort("user_id")
    assert np.array_equal(d["user_id"].to_numpy(), uid), f"user_id mismatch in {p}"
    return np.log1p(np.clip(d["pred"].to_numpy().astype(np.float64), 0, None))


def log_facts(name: str) -> dict:
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
    ap.add_argument("--boot", type=int, default=400)
    args = ap.parse_args()

    val = pl.read_parquet(FEATURES_DIR / f"anchor={VAL_ANCHOR.isoformat()}.parquet",
                          columns=["user_id", "target"]).sort("user_id")
    uid = val["user_id"].to_numpy()
    y = val["target"].to_numpy().astype(np.float64)
    ly = np.log1p(y)

    def score(lp):
        return {"cal": rmsle(y, np.expm1(honest_2fold(lp, ly))),
                "cal_split7": rmsle(y, np.expm1(honest_2fold(lp, ly, seed=7))),
                "cal_xfit5": rmsle(y, np.expm1(xfit_5(lp, ly))),
                "raw": rmsle(y, np.expm1(lp))}

    base_lp, base_sc = {}, {}
    for fam in FAMILIES:
        for sd in SEEDS:
            lp = load_lp(BASELINE.format(fam=fam, seed=sd), uid)
            if lp is not None:
                base_lp[(fam, sd)] = lp
                base_sc[(fam, sd)] = score(lp)

    rows, missing, resid = [], [], {}
    arms = ([("prep", f, v, s) for f in FAMILIES for v in PREP_VARIANTS for s in SEEDS]
            + [("kbins", "bin", k, s) for k in KBINS_VARIANTS for s in SEEDS])
    for kind, fam, var, sd in arms:
        name = (PREP_NAME.format(fam=fam, var=var, seed=sd) if kind == "prep"
                else KB_NAME.format(k=var, seed=sd))
        lp = load_lp(name, uid)
        if lp is None:
            missing.append(name)
            continue
        if (fam, sd) not in base_lp:
            missing.append(BASELINE.format(fam=fam, seed=sd) + " (baseline)")
            continue
        sc, b = score(lp), base_sc[(fam, sd)]
        resid[(kind, fam, var, sd)] = (honest_2fold(lp, ly) - ly) ** 2
        resid[("base", fam, "clip99", sd)] = (honest_2fold(base_lp[(fam, sd)], ly) - ly) ** 2
        rows.append({
            "kind": kind, "family": fam, "variant": str(var), "seed": sd,
            "name": name, "baseline": BASELINE.format(fam=fam, seed=sd),
            "base_cal": round(b["cal"], 6), "arm_cal": round(sc["cal"], 6),
            "delta": round(sc["cal"] - b["cal"], 6),
            "delta_split7": round(sc["cal_split7"] - b["cal_split7"], 6),
            "delta_xfit5": round(sc["cal_xfit5"] - b["cal_xfit5"], 6),
            "delta_raw": round(sc["raw"] - b["raw"], 6),
            "identical_to_baseline": bool(np.array_equal(lp, base_lp[(fam, sd)])),
            "log": log_facts(name),
        })
        r = rows[-1]
        print(f"{fam:5s} {str(var):8s} s{sd:<5d} base_cal {r['base_cal']:.6f} -> "
              f"{r['arm_cal']:.6f}  delta {r['delta']:+.6f} "
              f"(split7 {r['delta_split7']:+.6f}, xfit5 {r['delta_xfit5']:+.6f}, "
              f"raw {r['delta_raw']:+.6f})"
              f"{'  ДУБЛИКАТ БАЗЫ!' if r['identical_to_baseline'] else ''}", flush=True)

    summary = {}
    for kind, fam, var in sorted({(r["kind"], r["family"], r["variant"]) for r in rows}):
        rs = [r for r in rows if (r["kind"], r["family"], r["variant"]) == (kind, fam, var)]
        d = float(np.mean([r["delta"] for r in rs]))
        ea = np.concatenate([resid[(kind, fam, int(var) if kind == "kbins" else var,
                                    r["seed"])] for r in rs])
        eb = np.concatenate([resid[("base", fam, "clip99", r["seed"])] for r in rs])
        rng, n, boot = np.random.default_rng(1), len(ly), []
        for _ in range(args.boot):
            i = rng.integers(0, n, n)
            j = np.concatenate([i + k * n for k in range(len(rs))])
            boot.append(np.sqrt(ea[j].mean()) - np.sqrt(eb[j].mean()))
        lo, hi = (float(x) for x in np.percentile(boot, [2.5, 97.5]))
        summary[f"{fam}:{var}"] = {
            "kind": kind, "n_seeds": len(rs), "delta_mean": round(d, 6),
            "delta_ci95": [round(lo, 6), round(hi, 6)],
            "per_seed": {r["seed"]: r["delta"] for r in rs},
            "epochs_binding": any(r["log"].get("budget_exhausted") for r in rs),
            "verdict": ("HELPS" if d < -THRESH else "HURTS" if d > THRESH
                        else f"NOISE (|delta| <= {THRESH})"),
        }
        print(f"[{fam}:{var}] delta {d:+.6f} ci95 [{lo:+.6f}, {hi:+.6f}] "
              f"over {len(rs)} seeds -> {summary[f'{fam}:{var}']['verdict']}"
              f"{'  | ЭПОХИ ОГРАНИЧИВАЮТ' if summary[f'{fam}:{var}']['epochs_binding'] else ''}",
              flush=True)

    prep = {k: v for k, v in summary.items() if v["kind"] == "prep"}
    kb = {k: v for k, v in summary.items() if v["kind"] == "kbins"}
    # preprocessing is shared by all three families, so the decision-relevant
    # number is the mean over families, not the best single family
    by_variant = {}
    for v in PREP_VARIANTS:
        ds = [prep[k]["delta_mean"] for k in prep if k.endswith(":" + v)]
        if ds:
            by_variant[v] = {"n_families": len(ds), "delta_mean": round(float(np.mean(ds)), 6),
                             "delta_worst_family": round(float(np.max(ds)), 6)}
            print(f"[ALL FAMILIES] {v:8s} mean delta {np.mean(ds):+.6f} "
                  f"over {len(ds)} families (worst {np.max(ds):+.6f})", flush=True)
    best_prep = (min(by_variant, key=lambda v: by_variant[v]["delta_mean"])
                 if by_variant else None)
    if best_prep and by_variant[best_prep]["delta_mean"] >= -THRESH:
        best_prep = "clip99"          # nothing beat the historical default
    out = {
        "rows": rows, "summary": summary, "by_variant": by_variant, "missing": missing,
        "best_prep": best_prep,
        "best_prep_delta": (by_variant[best_prep]["delta_mean"]
                            if best_prep in by_variant else 0.0),
        "best_kbins": (min(kb, key=lambda k: kb[k]["delta_mean"]) if kb else None),
        "gh_nodes": {"tested": [4, 8, 12, 16, 20, 24, 32, 64, 128], "best": 20,
                     "delta_max_abs": 0.0,
                     "note": "квадратура сошлась: |gh20-gh128| <= 1.4e-6 на строку, "
                             "калиброванный скор совпадает до девятого знака"},
        "threshold": THRESH,
    }
    (REPORTS_DIR / "prep_verdict.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1, default=str))
    if missing:
        print(f"\nMISSING ({len(missing)}), очередь ещё не доработала: {missing}",
              flush=True)
    print("\n=== RAW JSON ===")
    print(json.dumps({k: out[k] for k in
                      ("summary", "by_variant", "best_prep", "best_prep_delta",
                       "best_kbins", "gh_nodes", "missing")},
                     ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()

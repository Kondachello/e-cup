"""Correlation of a model's validation errors with the current blend's errors.

The project's bottleneck is model uniformity, not model quality: anything with
error correlation below ~0.97 is valuable even if its own RMSLE is much worse.
This is the acceptance metric for every new model.

Usage:
  python work/scripts/err_corr.py NAME [NAME2 ...]
  python work/scripts/err_corr.py --file /path/to/preds.parquet [--file ...]

Errors are measured in log1p space (the space RMSLE lives in):
  err = log1p(pred) - log1p(y_true)
The reference blend is the current champion mix, blended in log1p space.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))
from common import PREDS_DIR, ROOT, VAL_ANCHOR, load_anchor, rmsle

# Действующий честный бленд (val OOF 1.666718). Прежний эталон содержал gru_final
# с весом 0.145 — модель, обученную до введения зазора 30 дней, её валидационный скор
# завышен. Из-за этого критерий систематически занижал вес новых моделей: febspec2
# получал -0.005 против эталона и +0.0156 против настоящего пула.
BLEND = {"fusion_f_cal": 0.32, "c_ts2_s42_cal": 0.25, "mlpziln_cal": 0.12,
         "behavonly_cal": 0.08, "countaov_cal": 0.07, "seq2tr_f_cal": 0.07,
         "twl_v7_cal": 0.055, "hmmsim_cal": 0.028, "channel2_cal": 0.012}


def load_lp(path: Path, uid_ref: np.ndarray) -> np.ndarray:
    df = pl.read_parquet(path).sort("user_id")
    assert np.array_equal(df["user_id"].to_numpy(), uid_ref), f"user_id mismatch in {path}"
    return np.log1p(np.clip(df["pred"].to_numpy().astype(np.float64), 0, None))


def blend_lp(uid_ref: np.ndarray) -> np.ndarray:
    """Reference blend, preferring the pack column over the hardcoded weights above.

    Bug found by track 5: rebuilding the blend from BLEND scores 1.666718 while the live
    champion (`blend` column of work/preds_pack/val_preds.parquet) scores 1.666395. Since
    margin = blend_score/model_score - corr, a weaker reference inflates the margin of
    EVERY candidate by about +0.00019 - 8 noise units, enough to accept a dead model.
    The pack column tracks the live blend; the hardcoded dict goes stale by construction.
    """
    pack = ROOT / "work" / "preds_pack" / "val_preds.parquet"
    if pack.exists():
        df = pl.read_parquet(pack).sort("user_id")
        if "blend" in df.columns and np.array_equal(df["user_id"].to_numpy(), uid_ref):
            return df["blend"].to_numpy().astype(np.float64)      # already log1p
        print("ВНИМАНИЕ: пакет есть, но колонка blend не подошла — беру старые веса")
    else:
        print("ВНИМАНИЕ: пакета нет, эталон из захардкоженных весов (устаревает)")
    return sum(w * load_lp(PREDS_DIR / f"{n}_val.parquet", uid_ref) for n, w in BLEND.items())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="*", help="model names -> work/preds/NAME_val.parquet")
    ap.add_argument("--file", action="append", default=[], help="explicit parquet path(s)")
    ap.add_argument("--calibrate", action="store_true",
                    help="calibrate inside (cross-fit binned log-shifts) instead of "
                         "requiring a pre-calibrated _cal input")
    ap.add_argument("--json", type=str, default="",
                    help="also write the table as JSON (blend score + one row per model)")
    args = ap.parse_args()
    out = {"models": {}}

    val = load_anchor(VAL_ANCHOR, columns=["user_id", "target"]).sort("user_id")
    uid = val["user_id"].to_numpy()
    y = val["target"].to_numpy().astype(np.float64)
    ly = np.log1p(np.clip(y, 0, None))

    lb = blend_lp(uid)
    eb = lb - ly
    blend_rmsle = float(np.sqrt(np.mean(eb ** 2)))
    out["blend"] = {"rmsle": blend_rmsle, "n": int(len(y)), "weights": BLEND}
    print(f"blend val_rmsle={blend_rmsle:.6f}  n={len(y)}", flush=True)

    targets = [(n, PREDS_DIR / f"{n}_val.parquet") for n in args.names]
    targets += [(Path(f).stem, Path(f)) for f in args.file]
    for name, path in targets:
        if not path.exists():
            print(f"{name}: MISSING {path}")
            continue
        lp = load_lp(path, uid)
        if args.calibrate:
            from margin import calibrate_honest
            lp = calibrate_honest(lp, ly, 24, 0)
        elif not name.endswith("_cal"):
            # Rule 1 of the team protocol: models may only be compared AFTER calibration -
            # the raw ordering misled us eight times. This tool does not calibrate, so a raw
            # name silently produces the wrong number: kostya46 reads 1.7024 raw against
            # 1.6699 calibrated, and its margin +0.00083 against +0.00133. Warn, do not guess.
            print(f"  ВНИМАНИЕ: {name} не похож на калиброванный (_cal). Числа ниже — по сырым "
                  f"предсказаниям, сравнивать их с калиброванными НЕЛЬЗЯ. "
                  f"Либо calibrate.py, либо флаг --calibrate.", flush=True)
        e = lp - ly
        sm = float(np.sqrt(np.mean(e ** 2)))
        # UNcentered correlation: the identity margin = sb/sm - c holds for E[e*eb]/(sm*sb).
        # np.corrcoef centres both errors, which distorted 6 of 30 models - every one with a
        # non-zero mean error, i.e. every uncalibrated model (Zhenya, zhenya_report.md).
        c = float(np.mean(e * eb) / max(sm * blend_rmsle, 1e-12))
        # optimal 2-way weight in log space: minimise ||(1-w) eb + w e||
        d = e - eb
        w = float(-np.dot(eb, d) / max(np.dot(d, d), 1e-12))
        best = float(np.sqrt(np.mean(((1 - w) * lb + w * lp - ly) ** 2)))
        # Корреляция ошибок сама по себе НИЧЕГО не значит: для любой модели внутри
        # линейной оболочки бленда она тождественно равна sb/sm (остаток бленда
        # ортогонален оболочке). Работает только ЗАПАС — доля модели вне оболочки.
        margin = blend_rmsle / max(sm, 1e-12) - c
        out["models"][name] = {"val_rmsle": rmsle(y, np.expm1(lp)), "err_corr": c,
                               "corr_expected": blend_rmsle / max(sm, 1e-12),
                               "margin": margin,
                               "w_opt": w, "blend_rmsle": best,
                               "gain": blend_rmsle - best}
        print(f"{name}: val_rmsle={rmsle(y, np.expm1(lp)):.6f}  err_corr={c:.4f} "
              f"(тождество {blend_rmsle / max(sm, 1e-12):.4f}, ЗАПАС {margin:+.5f})  "
              f"w*={w:.3f} -> {best:.6f} (gain {blend_rmsle - best:+.6f})",
              flush=True)
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=1))
        print(f"wrote {args.json}", flush=True)


if __name__ == "__main__":
    main()

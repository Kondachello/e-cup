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
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))
from common import PREDS_DIR, VAL_ANCHOR, load_anchor, rmsle

BLEND = {"mlpziln_cal": 0.536, "mlpbin_cal": 0.283, "gru_final": 0.145, "c_xtw_s42": 0.036}


def load_lp(path: Path, uid_ref: np.ndarray) -> np.ndarray:
    df = pl.read_parquet(path).sort("user_id")
    assert np.array_equal(df["user_id"].to_numpy(), uid_ref), f"user_id mismatch in {path}"
    return np.log1p(np.clip(df["pred"].to_numpy().astype(np.float64), 0, None))


def blend_lp(uid_ref: np.ndarray) -> np.ndarray:
    return sum(w * load_lp(PREDS_DIR / f"{n}_val.parquet", uid_ref) for n, w in BLEND.items())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="*", help="model names -> work/preds/NAME_val.parquet")
    ap.add_argument("--file", action="append", default=[], help="explicit parquet path(s)")
    args = ap.parse_args()

    val = load_anchor(VAL_ANCHOR, columns=["user_id", "target"]).sort("user_id")
    uid = val["user_id"].to_numpy()
    y = val["target"].to_numpy().astype(np.float64)
    ly = np.log1p(np.clip(y, 0, None))

    lb = blend_lp(uid)
    eb = lb - ly
    print(f"blend val_rmsle={float(np.sqrt(np.mean(eb ** 2))):.6f}  n={len(y)}", flush=True)

    targets = [(n, PREDS_DIR / f"{n}_val.parquet") for n in args.names]
    targets += [(Path(f).stem, Path(f)) for f in args.file]
    for name, path in targets:
        if not path.exists():
            print(f"{name}: MISSING {path}")
            continue
        lp = load_lp(path, uid)
        e = lp - ly
        c = float(np.corrcoef(e, eb)[0, 1])
        # optimal 2-way weight in log space: minimise ||(1-w) eb + w e||
        d = e - eb
        w = float(-np.dot(eb, d) / max(np.dot(d, d), 1e-12))
        best = float(np.sqrt(np.mean(((1 - w) * lb + w * lp - ly) ** 2)))
        print(f"{name}: val_rmsle={rmsle(y, np.expm1(lp)):.6f}  err_corr={c:.4f}  "
              f"w*={w:.3f} -> {best:.6f} (gain {float(np.sqrt(np.mean(eb ** 2))) - best:+.6f})",
              flush=True)


if __name__ == "__main__":
    main()

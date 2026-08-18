"""Make a submission CSV from test preds parquet.

Usage: make_submission.py --pred NAME [--mult 1.0] [--out FILE]
Writes /Users/alexanderkondakov/ozon-cup/submissions/NAME[_mMULT].csv
in exact sample_submit.csv format/order.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))
from common import PREDS_DIR, ROOT, SAMPLE_SUBMIT

SUB_DIR = ROOT / "submissions"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--mult", type=float, default=1.0)
    ap.add_argument("--logshift", type=float, default=0.0,
                    help="pred := expm1(max(log1p(pred)+s, 0)); use s=ln(M) for seasonal mult M")
    ap.add_argument("--out", type=str, default="")
    args = ap.parse_args()

    SUB_DIR.mkdir(exist_ok=True)
    sample = pl.read_csv(SAMPLE_SUBMIT, schema_overrides={"user_id": pl.Int64})
    preds = pl.read_parquet(PREDS_DIR / f"{args.pred}_test.parquet")
    out = sample.select("user_id").join(preds, on="user_id", how="left")
    assert out["pred"].null_count() == 0, "missing users in preds!"
    vals = np.clip(out["pred"].to_numpy() * args.mult, 0, None)
    if args.logshift:
        vals = np.expm1(np.clip(np.log1p(vals) + args.logshift, 0, None))
    out = out.with_columns(pl.Series("predict", vals)).select(["user_id", "predict"])

    name = args.out or (args.pred
                        + (f"_m{args.mult:g}" if args.mult != 1.0 else "")
                        + (f"_s{args.logshift:g}" if args.logshift else ""))
    path = SUB_DIR / f"{name}.csv"
    out.write_csv(path)
    print(f"wrote {path} rows={len(out)} mean={vals.mean():.3f} zeros={(vals==0).mean():.3%}")


if __name__ == "__main__":
    main()

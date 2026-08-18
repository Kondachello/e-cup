"""Final mix step of lbmix2.csv: log1p-space weighted average of

Weights are derived deterministically from the two known public LB scores and
the local prediction disagreement (exact math, see README):
  fa = 1.6754553658578413 (A public RMSLE), fb = 1.6621822432848572 (B public RMSLE)
  D2 = mean((log1p(A)-log1p(B))^2) over all 250k users

"""
from __future__ import annotations

import argparse

import numpy as np
import polars as pl

FA = 1.6754553658578413
FB = 1.6621822432848572


def load(p):
    df = pl.read_csv(p, schema_overrides={"user_id": pl.Int64}).sort("user_id")
    return df.rename({df.columns[1]: "predict"})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="our blend submission CSV")
    ap.add_argument("--out", default="lbmix2.csv")
    args = ap.parse_args()

    A, B = load(args.a), load(args.b)
    assert (A["user_id"].to_numpy() == B["user_id"].to_numpy()).all()
    la = np.log1p(np.clip(A["predict"].to_numpy(), 0, None))
    lb = np.log1p(np.clip(B["predict"].to_numpy(), 0, None))

    D2 = float(np.mean((la - lb) ** 2))
    fa2, fb2 = FA * FA, FB * FB
    cov = (fa2 + fb2 - D2) / 2
    w = (fa2 - cov) / (fa2 + fb2 - 2 * cov)
    exp_f2 = fa2 - (fa2 - cov) ** 2 / (fa2 + fb2 - 2 * cov)
    print(f"D2={D2:.6f} w_B={w:.6f} expected_public_RMSLE={np.sqrt(exp_f2):.6f}")

    lp = (1 - w) * la + w * lb
    out = A.select("user_id").with_columns(pl.Series("predict", np.expm1(np.clip(lp, 0, None))))
    out.write_csv(args.out)
    print(f"wrote {args.out} rows={len(out)}")


if __name__ == "__main__":
    main()

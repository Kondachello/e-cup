"""Optimal 2-submission blend using only their public LB scores + local pred files.

Math (log1p space, per-user errors a=lpA-ly, b=lpB-ly on the public subset):
  fA^2=E[a^2], fB^2=E[b^2] known from LB; D^2=E[(a-b)^2]=E[(lpA-lpB)^2] computed
  locally (identical on public subset up to sampling of users, n=50k).
  cov=E[ab]=(fA^2+fB^2-D^2)/2;  w*=(fA^2-cov)/(fA^2+fB^2-2cov)
  expected f^2 = fA^2 - (fA^2-cov)^2/(fA^2+fB^2-2cov)

"""
from __future__ import annotations

import os
import argparse
from pathlib import Path

import numpy as np
import polars as pl

# Корень репозитория: OZON_ROOT, иначе поднимаемся от этого файла.
# Захардкоженный путь одной машины делал скрипт неработающим у всех
# остальных членов команды и на чистом клоне.
ROOT = Path(os.environ.get("OZON_ROOT", str(Path(__file__).resolve().parents[2])))


def load(p):
    df = pl.read_csv(p, schema_overrides={"user_id": pl.Int64}).sort("user_id")
    cols = df.columns
    return df.rename({cols[1]: "predict"})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--fa", type=float, required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--fb", type=float, required=True)
    ap.add_argument("--w", type=float, default=None, help="override weight on B")
    ap.add_argument("--out", type=str, default="lb_blend")
    args = ap.parse_args()

    A, B = load(args.a), load(args.b)
    assert (A["user_id"].to_numpy() == B["user_id"].to_numpy()).all()
    la = np.log1p(np.clip(A["predict"].to_numpy(), 0, None))
    lb = np.log1p(np.clip(B["predict"].to_numpy(), 0, None))

    D2 = float(np.mean((la - lb) ** 2))
    fa2, fb2 = args.fa ** 2, args.fb ** 2
    cov = (fa2 + fb2 - D2) / 2
    denom = fa2 + fb2 - 2 * cov
    w = (fa2 - cov) / denom if denom > 1e-12 else 0.5
    exp_f2 = fa2 - (fa2 - cov) ** 2 / denom if denom > 1e-12 else fa2
    print(f"D^2={D2:.6f} cov={cov:.6f} corr_of_errors={cov/np.sqrt(fa2*fb2):.4f}")
    print(f"optimal w_B={w:.3f}, expected public RMSLE={np.sqrt(max(exp_f2,0)):.6f} "
          f"(A alone {args.fa:.6f}, B alone {args.fb:.6f})")

    if args.w is not None:
        w = args.w
        exp = fa2 * (1 - w) ** 2 + fb2 * w ** 2 + 2 * w * (1 - w) * cov
        print(f"using w_B={w:.3f}, expected public RMSLE={np.sqrt(exp):.6f}")

    lp = (1 - w) * la + w * lb
    out = A.select("user_id").with_columns(pl.Series("predict", np.expm1(np.clip(lp, 0, None))))
    path = ROOT / "submissions" / f"{args.out}.csv"
    out.write_csv(path)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()

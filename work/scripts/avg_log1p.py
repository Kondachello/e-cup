"""Average prediction files in log1p space (the space RMSLE lives in).

Inputs are model NAMES following the exp_lib contract; for every split present
for all of them it writes work/preds/OUT_<split>.parquet.

Average the CALIBRATED files, not the raw ones: raw models sit ~0.25 too high in
log and mixing calibrated with raw masks real model strength (project rule
"calibrate every model BEFORE blending").

Usage:
  avg_log1p.py --out febspec3_avg --preds febspec3_mlp_cal,febspec3_tw_cal
  avg_log1p.py --out x --preds a,b --weights 0.6,0.4 --splits val,test
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))
from common import PREDS_DIR, VAL_ANCHOR, load_anchor, rmsle  # noqa: E402
from exp_lib import save_preds  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--preds", required=True, help="comma-separated model names")
    ap.add_argument("--weights", type=str, default="")
    ap.add_argument("--splits", type=str, default="val,test")
    args = ap.parse_args()

    names = [n for n in args.preds.split(",") if n]
    w = np.array([float(x) for x in args.weights.split(",")]) if args.weights \
        else np.full(len(names), 1.0 / len(names))
    assert len(w) == len(names), "weights/preds length mismatch"
    w = w / w.sum()
    out = {"out": args.out, "parts": dict(zip(names, np.round(w, 4).tolist()))}

    for split in args.splits.split(","):
        paths = [PREDS_DIR / f"{n}_{split}.parquet" for n in names]
        missing = [str(p) for p in paths if not p.exists()]
        if missing:
            print(f"skip split {split}: missing {missing}", flush=True)
            continue
        uid, acc = None, None
        for p, wi in zip(paths, w):
            d = pl.read_parquet(p).sort("user_id")
            if uid is None:
                uid = d["user_id"].to_numpy()
            else:
                assert np.array_equal(uid, d["user_id"].to_numpy()), f"user_id mismatch {p}"
            lp = np.log1p(np.clip(d["pred"].to_numpy().astype(np.float64), 0, None))
            acc = wi * lp if acc is None else acc + wi * lp
        pred = np.expm1(acc)
        save_preds(args.out, split, uid, pred)
        out[f"{split}_mean_log1p"] = round(float(acc.mean()), 6)
        if split == "val":
            y = load_anchor(VAL_ANCHOR, columns=["user_id", "target"]).sort("user_id")
            assert np.array_equal(y["user_id"].to_numpy(), uid)
            out["val_rmsle"] = round(rmsle(y["target"].to_numpy().astype(np.float64), pred), 6)
        print(f"wrote {args.out}_{split}.parquet ({len(uid)} rows)", flush=True)
    print(json.dumps(out), flush=True)


if __name__ == "__main__":
    main()

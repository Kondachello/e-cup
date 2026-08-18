"""Acceptance test for a new blend member: beta, not error correlation.

The project's bottleneck is model uniformity, not model quality -- but "low error
correlation" is NOT the right acceptance rule: correlation is trivially bought by
making a model worse, and a weak-but-uncorrelated model still gets zero weight.

The exact rule. Let e_b be the blend's log-space error and e_m the candidate's.
The derivative of the blended MSE at weight zero is

    d/dw MSE[(1-w) e_b + w e_m] |_{w=0} = 2 (<e_b, e_m> - <e_b, e_b>) = 2 s_b^2 (beta - 1)

so the candidate earns a NON-NEGATIVE weight if and only if

    beta = <e_b, e_m> / <e_b, e_b>  <  1        (beta = corr * rmsle_m / rmsle_b)

Write e_m = beta * e_b + eps. Then the best achievable blend is

    s_blend = s_b / sqrt(1 + SNR),   SNR = s_b^2 (1 - beta)^2 / s_eps^2

i.e. the gain is QUADRATIC in the margin (1 - beta): a model that sits just below
the frontier is worth nothing. Practical thresholds (see --help for the verdicts):
beta <= 0.96 is worth ~0.004-0.015 depending on s_eps; beta >= 0.99 is worth <0.0005.

Every model in the zoo as of 2026-08-18 sits at beta = 1.00-1.016 -- the signature of
"blend + independent noise", i.e. no information the blend does not already have.

NOTE on negative weights: an unconstrained optimum with w* < 0 is scale extrapolation
along a near-collinear direction. On validation it is an artefact (the team measured
all 54 such contrasts together at 0.00002 on the leaderboard), so the gain reported
here clips w to [0, 1]. Directions worth exploiting with a negative coefficient are
the business of the LB-probe machinery, where the coefficient is measured on the test
set itself -- not of this tool.

Errors are measured in log1p space (the space RMSLE lives in): err = log1p(pred) - log1p(y).

Usage:
  python work/scripts/err_corr.py NAME [NAME2 ...]
  python work/scripts/err_corr.py --file /path/to/preds.parquet [--file ...]
  # second window (gate for survivors; needs preds saved under that suffix):
  python work/scripts/err_corr.py NAME --anchor 2025-12-31 --suffix dec31 \
      --blend twdeep:0.6,c_xtw_s42:0.4
"""
from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))
from common import PREDS_DIR, VAL_ANCHOR, load_anchor, rmsle

BLEND = {"mlpziln_cal": 0.536, "mlpbin_cal": 0.283, "gru_final": 0.145, "c_xtw_s42": 0.036}

# beta thresholds -> verdict
TAKE, SMALL = 0.96, 0.99


def load_lp(path: Path, uid_ref: np.ndarray) -> np.ndarray:
    df = pl.read_parquet(path).sort("user_id")
    assert np.array_equal(df["user_id"].to_numpy(), uid_ref), f"user_id mismatch in {path}"
    return np.log1p(np.clip(df["pred"].to_numpy().astype(np.float64), 0, None))


def blend_lp(uid_ref: np.ndarray, spec: dict[str, float], suffix: str) -> np.ndarray:
    return sum(w * load_lp(PREDS_DIR / f"{n}_{suffix}.parquet", uid_ref) for n, w in spec.items())


def parse_blend(s: str) -> dict[str, float]:
    spec = {}
    for part in s.split(","):
        name, _, w = part.partition(":")
        spec[name.strip()] = float(w)
    return spec


def verdict(beta: float) -> str:
    if beta <= TAKE:
        return "TAKE"
    if beta < SMALL:
        return "small w"
    return "DROP"


def main():
    ap = argparse.ArgumentParser(
        description="Acceptance: beta < 1 is necessary and sufficient for a non-zero blend weight. "
                    f"beta <= {TAKE}: take it. {TAKE} < beta < {SMALL}: small weight. "
                    f"beta >= {SMALL}: drop it, whatever the solo RMSLE and correlation say.")
    ap.add_argument("names", nargs="*", help="model names -> work/preds/NAME_<suffix>.parquet")
    ap.add_argument("--file", action="append", default=[], help="explicit parquet path(s)")
    ap.add_argument("--anchor", type=str, default="", help="alternative window, e.g. 2025-12-31")
    ap.add_argument("--suffix", type=str, default="val", help="preds file suffix (default: val)")
    ap.add_argument("--blend", type=str, default="", help="reference blend 'name:w,name:w'")
    args = ap.parse_args()

    anchor = date.fromisoformat(args.anchor) if args.anchor else VAL_ANCHOR
    spec = parse_blend(args.blend) if args.blend else BLEND
    tot = sum(spec.values())
    if abs(tot - 1.0) > 1e-6:
        print(f"warning: blend weights sum to {tot:.4f}, not 1 -- beta is measured against a "
              f"rescaled reference", flush=True)

    val = load_anchor(anchor, columns=["user_id", "target"]).sort("user_id")
    uid = val["user_id"].to_numpy()
    y = val["target"].to_numpy().astype(np.float64)
    ly = np.log1p(np.clip(y, 0, None))

    lb = blend_lp(uid, spec, args.suffix)
    eb = lb - ly
    sb2 = float(np.mean(eb ** 2))
    sb = float(np.sqrt(sb2))
    print(f"anchor={anchor} n={len(y)} blend={'+'.join(f'{n}*{w:g}' for n, w in spec.items())}")
    print(f"blend val_rmsle={sb:.6f}\n", flush=True)

    hdr = (f"{'model':<22}{'rmsle':>9}{'corr':>8}{'beta':>8}{'s_eps':>8}"
           f"{'w_raw':>8}{'gain':>10}  verdict")
    print(hdr)
    print("-" * len(hdr))

    targets = [(n, PREDS_DIR / f"{n}_{args.suffix}.parquet") for n in args.names]
    targets += [(Path(f).stem, Path(f)) for f in args.file]
    for name, path in targets:
        if not path.exists():
            print(f"{name:<22}MISSING {path}")
            continue
        lp = load_lp(path, uid)
        e = lp - ly
        beta = float(np.dot(eb, e) / np.dot(eb, eb))          # <e_b,e_m>/<e_b,e_b>
        s_eps = float(np.sqrt(np.mean((e - beta * eb) ** 2)))  # part orthogonal to e_b
        corr = float(np.corrcoef(e, eb)[0, 1])
        d = e - eb
        w_raw = float(-np.dot(eb, d) / max(np.dot(d, d), 1e-12))
        w = min(max(w_raw, 0.0), 1.0)                          # no negative-weight extrapolation
        best = float(np.sqrt(np.mean(((1 - w) * lb + w * lp - ly) ** 2)))
        print(f"{name:<22}{rmsle(y, np.expm1(lp)):9.4f}{corr:8.4f}{beta:8.4f}{s_eps:8.3f}"
              f"{w_raw:+8.3f}{sb - best:+10.5f}  {verdict(beta)}", flush=True)


if __name__ == "__main__":
    main()

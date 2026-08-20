"""Is the blend residual predictable by a model from OUTSIDE our zoo?

The framing rule of the project ("a model whose prediction is a function of our 203
features lies in the blend hull and contributes exactly zero") rests on evidence
gathered with OUR OWN model families: LGB stacks on the residual gave OOF R^2 = -0.019
(H1). That is evidence about gradient boosting, not about tabular space as such.

TabPFN is a transformer pretrained on synthetic tabular tasks: a genuinely different
prior, in-context rather than fitted. So it is the right instrument to cross-examine
the theorem:

    R^2 ~ 0  -> the theorem is confirmed by a tool from outside our zoo; the tabular
               frame is closed for good and the team stops re-testing it.
    R^2 > 0  -> the residual IS learnable, and the whole tabular space reopens.

Design (three numbers, not one):
  real      TabPFN fits on a context of rows from half A, predicts residuals in half B
  placebo   identical run with the target shuffled -> calibrates "what R^2 does zero
            look like at this n" (the project rule: every gain is compared to placebo)
  ridge     a linear reference on the same split

Contexts are subsampled (TabPFN caps at ~10k rows / 100 cols) and bagged, per the
recipe in arXiv:2502.17361. Users are split by hash, so no user appears in both the
context and the evaluation set.

Run (CPU, hours - queue it):
  USE_V2=1 USE_V3=1 .venv/bin/python work/scripts/tabpfn_probe.py --contexts 4
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))
from common import ROOT, VAL_ANCHOR, feature_cols, load_anchor


def r2(y, p):
    return float(1.0 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--contexts", type=int, default=4, help="bagged TabPFN contexts")
    ap.add_argument("--ctx-rows", type=int, default=8000, help="rows per context (cap ~10k)")
    ap.add_argument("--n-feats", type=int, default=90, help="features per context (cap 100)")
    ap.add_argument("--eval-rows", type=int, default=12000)
    ap.add_argument("--pack", type=Path, default=ROOT / "work" / "preds_pack")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    t0 = time.time()

    pack = pl.read_parquet(args.pack / "val_preds.parquet").sort("user_id")
    ly = np.log1p(np.clip(pack["target"].to_numpy().astype(np.float64), 0, None))
    resid = pack["blend"].to_numpy().astype(np.float64) - ly       # what we try to predict

    val = load_anchor(VAL_ANCHOR).sort("user_id")
    assert np.array_equal(val["user_id"].to_numpy(), pack["user_id"].to_numpy()), "user_id mismatch"
    cols = feature_cols(val)
    X = val.select(cols).to_numpy().astype(np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    print(f"X {X.shape}, residual sd {resid.std():.4f}, load {time.time()-t0:.0f}s", flush=True)

    rng = np.random.default_rng(args.seed)
    half = rng.permutation(len(resid)) < len(resid) // 2       # context pool vs eval pool
    ev_idx = rng.choice(np.flatnonzero(~half), size=min(args.eval_rows, int((~half).sum())),
                        replace=False)
    y_ev = resid[ev_idx]

    from tabpfn import TabPFNRegressor
    n_jobs = int(os.environ.get("OMP_NUM_THREADS", "7"))

    preds = {"real": [], "placebo": []}
    for c in range(args.contexts):
        crng = np.random.default_rng(1000 + c)
        ctx = crng.choice(np.flatnonzero(half), size=args.ctx_rows, replace=False)
        feats = crng.choice(len(cols), size=min(args.n_feats, len(cols)), replace=False)
        for tag in ("real", "placebo"):
            y_ctx = resid[ctx] if tag == "real" else crng.permutation(resid[ctx])
            m = TabPFNRegressor(n_estimators=1, device="cpu", random_state=c,
                                n_jobs=n_jobs, ignore_pretraining_limits=True)
            m.fit(X[np.ix_(ctx, feats)], y_ctx)
            preds[tag].append(m.predict(X[np.ix_(ev_idx, feats)]))
            print(f"ctx {c} {tag:8} mdl_flint={r2(y_ev, preds[tag][-1]):+.5f} "
                  f"bag={r2(y_ev, np.mean(preds[tag], axis=0)):+.5f} "
                  f"[{time.time()-t0:.0f}s]", flush=True)

    from sklearn.linear_model import Ridge
    ctx = np.random.default_rng(7).choice(np.flatnonzero(half), size=args.ctx_rows, replace=False)
    mu, sd = X[ctx].mean(0), X[ctx].std(0) + 1e-9
    ridge = Ridge(alpha=10.0).fit((X[ctx] - mu) / sd, resid[ctx])
    r2_ridge = r2(y_ev, ridge.predict((X[ev_idx] - mu) / sd))

    real = r2(y_ev, np.mean(preds["real"], axis=0))
    plac = r2(y_ev, np.mean(preds["placebo"], axis=0))
    print(f"\n{'':10}{'mdl_flint':>10}")
    for tag, v in (("real", real), ("placebo", plac), ("ridge", r2_ridge)):
        print(f"{tag:10}{v:+10.5f}")
    print(f"\nreal - placebo = {real - plac:+.5f}  (n_eval={len(ev_idx)})")
    print("ВЫВОД: остаток предсказуем — табличное пространство ОТКРЫТО заново"
          if real - plac > 0.01 else
          "ВЫВОД: остаток непредсказуем и снаружи нашего зоопарка — теорема оболочки подтверждена")
    print(f"[DONE] {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

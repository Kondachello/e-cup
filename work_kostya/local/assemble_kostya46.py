"""Assemble kostya46_s{KSEED} from the four head pairs (NEW FILE — the original
assembly step was not committed; formula from work_kostya/README.md "Состав kostya46"):

    log-mix = 0.25*direct_1 + 0.25*(p*size)_1 + 0.2*direct_2 + 0.3*(p*size)_2
    pred    = expm1(max(log-mix, 0))          # clip-at-0 matches rmsle_log convention
                                              # and the exact 0.0 minima in kostya46_*.parquet

val heads : val_pz / val_two (config 1, train_model.py), m2_val_pz / m2_val_two (config 2).
test heads: mt1_test_pz / mt1_test_two (config 1, train_test_m1.py), mt_test_pz / mt_test_two (config 2).

Output contract (matches existing kostya46_val.parquet): 250000 rows, columns
user_id (i64, sorted) + pred (f64, raw GMV), written to work_kostya/preds/.
Must run right after the train scripts of the SAME seed (head .npy names are fixed and
overwritten by the next seed run) — the queue job chains them with &&.
"""
import os
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
W = ROOT / "work_kostya" / "work"
P = ROOT / "work_kostya" / "preds"
P.mkdir(parents=True, exist_ok=True)
K = os.environ.get("KSEED", "").strip() or "1"

users = pl.read_parquet(W / "users_order.parquet")["user_id"]
assert len(users) == 250000


def logmix(d1, t1, d2, t2):
    z = (0.25 * np.load(W / d1).astype(np.float64) + 0.25 * np.load(W / t1).astype(np.float64)
         + 0.2 * np.load(W / d2).astype(np.float64) + 0.3 * np.load(W / t2).astype(np.float64))
    assert z.shape == (250000,), z.shape
    return z


zv = logmix("val_pz.npy", "val_two.npy", "m2_val_pz.npy", "m2_val_two.npy")
zt = logmix("mt1_test_pz.npy", "mt1_test_two.npy", "mt_test_pz.npy", "mt_test_two.npy")

for tag, z in [("val", zv), ("test", zt)]:
    pred = np.expm1(np.maximum(z, 0.0))
    assert np.isfinite(pred).all() and (pred >= 0).all()
    out = P / f"kostya46_s{K}_{tag}.parquet"
    pl.DataFrame({"user_id": users, "pred": pred}).write_parquet(out)
    print(f"saved {out}  rows=250000  mean={pred.mean():.4f}  zeros={(pred == 0).sum()}", flush=True)

# --- sanity vs existing seed-1 kostya46 and vs val target (read-only) ---
ref = P / "kostya46_val.parquet"
if ref.exists():
    r = np.log1p(pl.read_parquet(ref)["pred"].to_numpy())
    cv = float(np.corrcoef(np.maximum(zv, 0.0), r)[0, 1])
    print(f"corr(log) with existing kostya46_val (s1): {cv:.5f}", flush=True)
ref_t = P / "kostya46_test.parquet"
if ref_t.exists():
    r = np.log1p(pl.read_parquet(ref_t)["pred"].to_numpy())
    ct = float(np.corrcoef(np.maximum(zt, 0.0), r)[0, 1])
    print(f"corr(log) with existing kostya46_test (s1): {ct:.5f}", flush=True)
pack = ROOT / "work" / "preds_pack" / "val_preds.parquet"
if pack.exists():
    tgt = pl.read_parquet(pack, columns=["user_id", "target"])
    assert (tgt["user_id"].to_numpy() == users.to_numpy()).all()
    ztrue = np.log1p(tgt["target"].to_numpy())
    rm = float(np.sqrt(np.mean((np.maximum(zv, 0.0) - ztrue) ** 2)))
    print(f"raw (uncalibrated) val RMSLE of kostya46_s{K}: {rm:.6f}", flush=True)
print("assemble done", flush=True)

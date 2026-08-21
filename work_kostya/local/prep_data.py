"""Prep artifacts for the kostya46 pipeline (NEW FILE — original prep ran on Kostya's
machine under /root/work and is not in the repo; reconstructed from script usage).

Derivations, each verified against team artifacts on 2026-08-21 (see local/README.md):
- act.parquet        = train.parquet sorted by (user_id, event_date); train.parquet is
                       already sorted, so the file is byte-copied (features.interval_feats
                       relies on day-sorted-within-user).
- users_order        = sorted unique user_id (250000); matches preds_pack user order exactly.
- gmv_mat[:, j]      = sum(gmv) over days [a, a+30) for anchor a; at a=379 reproduces
                       preds_pack target to 7e-12 (float64).
- buy_mat            = gmv_mat > 0 (bool; must be bool — used as boolean row mask).
                       In the val window (gmv>0) == (to_ord>0) for every user; zero share
                       0.45934 matches REPORT (0.4593). train_test_model itself uses G>0.
- anchor_days        = 36..379 step 7 (50 anchors; train_clf.py comment "36..379 step 7").
- testgrid_days      = 311..374 step 7 (10 anchors; train_test_model.py docstring).
- app_*              = "natural anchors 248..283" (6 anchors); app_mat_natural[:, j] =
                       any activity row in [a, a+30). Reconstructed guess — feeds ONLY the
                       auxiliary P(appear) head (kostya46shade), NOT kostya46 itself.
"""
import json
import shutil
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
W = ROOT / "work_kostya" / "work"
W.mkdir(parents=True, exist_ok=True)
DAY0 = date(2025, 1, 1)
EPOCH_OFF = (DAY0 - date(1970, 1, 1)).days

# ---------- act.parquet + users_order.parquet ----------
ue = pl.scan_parquet(ROOT / "train.parquet").select("user_id", "event_date").collect()
u = ue["user_id"].to_numpy()
d_epoch = ue["event_date"].cast(pl.Int32).to_numpy()
sorted_ok = bool((u[1:] >= u[:-1]).all())
if sorted_ok:
    same = u[1:] == u[:-1]
    sorted_ok = bool((d_epoch[1:][same] >= d_epoch[:-1][same]).all())
if sorted_ok:
    shutil.copyfile(ROOT / "train.parquet", W / "act.parquet")
    print("act.parquet = byte copy of train.parquet (already sorted by user_id, event_date)", flush=True)
else:  # safety fallback, not expected
    pl.read_parquet(ROOT / "train.parquet").sort(["user_id", "event_date"]).write_parquet(W / "act.parquet")
    print("act.parquet written sorted", flush=True)

users = np.sort(np.unique(u))
N = len(users)
assert N == 250000, N
pl.DataFrame({"user_id": users}).write_parquet(W / "users_order.parquet")

day = (d_epoch.astype(np.int64) - EPOCH_OFF)
assert day.min() >= 0, day.min()
MAXD = int(day.max()) + 1
print("N =", N, "| day range 0..", MAXD - 1, flush=True)
assert MAXD == 409, MAXD  # data ends 2026-02-13; test anchor 409 = first day past data

uidx = np.searchsorted(users, u).astype(np.int64)
flat = uidx * MAXD + day
gmv = pl.scan_parquet(ROOT / "train.parquet").select("gmv").collect()["gmv"].to_numpy()

# daily per-user gmv -> prefix sums -> 30-day window sums
gd = np.bincount(flat, weights=gmv, minlength=N * MAXD).reshape(N, MAXD)
C = np.zeros((N, MAXD + 1), dtype=np.float64)
np.cumsum(gd, axis=1, out=C[:, 1:])
del gd


def win_gmv(a: int) -> np.ndarray:  # float64, sum gmv over [a, a+30)
    hi = min(a + 30, MAXD)
    return C[:, hi] - C[:, a]


anchor_days = np.arange(36, 380, 7)  # 36..379
G64 = np.stack([win_gmv(int(a)) for a in anchor_days], axis=1)
np.save(W / "gmv_mat.npy", G64.astype(np.float32))
np.save(W / "buy_mat.npy", G64 > 0)  # bool
np.save(W / "anchor_days.npy", anchor_days)
print("gmv_mat/buy_mat:", G64.shape, "anchors", int(anchor_days[0]), "..", int(anchor_days[-1]), flush=True)

testgrid = list(range(311, 375, 7))  # 311..374, 10 anchors
Gt = np.stack([win_gmv(a) for a in testgrid], axis=1)
np.save(W / "gmv_mat_testgrid.npy", Gt.astype(np.float32))
json.dump(testgrid, open(W / "testgrid_days.json", "w"))
print("gmv_mat_testgrid:", Gt.shape, testgrid, flush=True)

# ---------- verification against team artifacts ----------
j379 = int(np.where(anchor_days == 379)[0][0])
zero_share = float((G64[:, j379] == 0).mean())
print("val (anchor 379) zero share:", round(zero_share, 5), flush=True)
assert abs(zero_share - 0.45934) < 0.0005, zero_share
pack = ROOT / "work" / "preds_pack" / "val_preds.parquet"
if pack.exists():
    t = pl.read_parquet(pack, columns=["user_id", "target"])
    assert (t["user_id"].to_numpy() == users).all(), "user order mismatch vs preds_pack"
    md = float(np.abs(t["target"].to_numpy() - G64[:, j379]).max())
    print("max |preds_pack.target - gmv_mat[:,379]| =", md, flush=True)
    assert md < 1e-6, md
del Gt, G64

# ---------- appearance matrices (auxiliary P(appear) head only) ----------
ad = np.bincount(flat, minlength=N * MAXD).reshape(N, MAXD) > 0
A = np.zeros((N, MAXD + 1), dtype=np.int32)
np.cumsum(ad, axis=1, out=A[:, 1:])
del ad
app_days = list(range(248, 284, 7))  # 248..283, 6 anchors
app_mat = np.stack([(A[:, a + 30] - A[:, a]) > 0 for a in app_days], axis=1)
np.save(W / "app_mat_natural.npy", app_mat)
json.dump(app_days, open(W / "app_anchor_days.json", "w"))
print("app_mat_natural:", app_mat.shape, "anchors", app_days,
      "| appearance rate per anchor:", [round(float(m), 4) for m in app_mat.mean(0)], flush=True)
print("prep done", flush=True)

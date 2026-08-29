"""Day-level per-user event cube for the neural TPP (Program 1, research week).
Channels per (user, day): active, n_orders, log1p(gmv), searches, to_cart, s2c, cat.
Grid: day 0 = 2025-01-01 .. day 408 = 2026-02-13 (T=409). No week binning.
"""
import polars as pl, numpy as np
from datetime import date

DAY0 = date(2025, 1, 1)
T = 409

act = pl.read_parquet("/root/work/act.parquet")
users = pl.read_parquet("/root/work/users_order.parquet")["user_id"]
N = len(users)
uid_to_idx = pl.DataFrame({"user_id": users, "uix": np.arange(N, dtype=np.int32)})
act = act.join(uid_to_idx, on="user_id")
day = act.select((pl.col("event_date") - pl.lit(DAY0)).dt.total_days())["event_date"].to_numpy().astype(np.int32)
assert day.min() >= 0 and day.max() < T, (day.min(), day.max())
uix = act["uix"].to_numpy().astype(np.int64)
flat = uix * T + day

def scat_u8(vals):
    a = np.zeros(N * T, dtype=np.int32)
    np.add.at(a, flat, vals.astype(np.int32))
    return np.clip(a, 0, 255).astype(np.uint8).reshape(N, T)

def scat_f(vals):
    a = np.zeros(N * T, dtype=np.float64)
    np.add.at(a, flat, vals)
    return a.reshape(N, T)

active = scat_u8(np.ones(len(day), dtype=np.int32))
nord   = scat_u8(act["to_ord"].to_numpy())
srch   = scat_u8(act["searches"].to_numpy())
cart   = scat_u8(act["to_cart"].to_numpy())
s2c    = scat_u8(act["search_to_cart"].to_numpy())
catf   = scat_u8(act["cat"].to_numpy())
gmv    = scat_f(act["gmv"].to_numpy())
lgmv   = np.log1p(np.clip(gmv, 0, None)).astype(np.float16)

np.savez("/root/work/tpp/day_cube.npz", active=active, nord=nord, srch=srch,
         cart=cart, s2c=s2c, catf=catf, lgmv=lgmv)
buy = nord > 0
print("cube", (N, T), "buy user-days:", int(buy.sum()),
      "buyers ever:", int(buy.any(1).sum()),
      "active user-days:", int((active > 0).sum()))
print("gmv day totals head:", np.asarray(gmv.sum(0)[:5], dtype=np.int64))

"""Weekly per-user metric cube aligned to a boundary day.
Weeks: week w = days [B-7*(NW-w), B-7*(NW-w-1)) for w=0..NW-1 (w increasing in time).
Any anchor day A = B - 7k is a week boundary: history = weeks [0, NW-k).
Rows with day >= B or day < B-7*NW are dropped (only history matters).
"""
import polars as pl, numpy as np
from paths import wp
from datetime import date
import sys

DAY0 = date(2025, 1, 1)

METRICS = ["active", "searches", "to_cart", "to_ord", "gmv", "buyday",
           "s_day", "c_day", "cartday", "s2c"]

def build_cube(boundary_day, out):
    NW = int(np.ceil(boundary_day / 7))
    act = pl.read_parquet(wp("act.parquet"))
    users = pl.read_parquet(wp("users_order.parquet"))["user_id"]
    uid_to_idx = pl.DataFrame({"user_id": users, "uix": np.arange(len(users), dtype=np.int32)})
    act = act.join(uid_to_idx, on="user_id")
    day = act.select((pl.col("event_date") - pl.lit(DAY0)).dt.total_days())["event_date"].to_numpy().astype(np.int32)
    keep = day < boundary_day
    act = act.filter(pl.Series(keep)); day = day[keep]
    week = (day - (boundary_day - 7 * NW)) // 7
    assert week.min() >= 0 and week.max() < NW, (week.min(), week.max(), NW)
    uix = act["uix"].to_numpy()
    N = len(users)
    cube = np.zeros((N, NW, len(METRICS)), dtype=np.float32)
    flat = uix.astype(np.int64) * NW + week
    to_ord = act["to_ord"].to_numpy()
    vals = {
        "active": np.ones(len(day), dtype=np.float32),
        "searches": act["searches"].to_numpy().astype(np.float32),
        "to_cart": act["to_cart"].to_numpy().astype(np.float32),
        "to_ord": to_ord.astype(np.float32),
        "gmv": act["gmv"].to_numpy().astype(np.float32),
        "buyday": (to_ord > 0).astype(np.float32),
        "s_day": act["search"].to_numpy().astype(np.float32),
        "c_day": act["cat"].to_numpy().astype(np.float32),
        "cartday": (act["to_cart"].to_numpy() > 0).astype(np.float32),
        "s2c": act["search_to_cart"].to_numpy().astype(np.float32),
    }
    for mi, m in enumerate(METRICS):
        np.add.at(cube[:, :, mi].reshape(-1), flat, vals[m])
    np.save(out, cube)
    print("cube saved", out, cube.shape, round(cube.nbytes / 1e9, 2), "GB, NW =", NW)

if __name__ == "__main__":
    b = int(sys.argv[1]); out = sys.argv[2]
    build_cube(b, out)

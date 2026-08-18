"""Seasonality EDA - step 2: compute seasonal ratios, YoY growth, M estimate,
weekday pattern, per-user log-scale seasonal shift. Prints everything as text."""
import os

os.environ.setdefault("POLARS_MAX_THREADS", "3")
os.environ.setdefault("OMP_NUM_THREADS", "3")

from datetime import date

import numpy as np
import polars as pl

DATA = "/Users/alexanderkondakov/ozon-cup/work/data"
daily = daily.with_columns(
    pl.col("event_date").dt.strftime("%Y-%m").alias("ym"),
    pl.col("event_date").dt.weekday().alias("wd"),  # 1=Mon..7=Sun
)

D = pl.col("event_date")


def win(a, b):
    return daily.filter((D >= a) & (D <= b))


def wsum(col, a, b):
    return float(win(a, b)[col].sum())


# ---------------------------------------------------------------- monthly table
print("=" * 100)
print("MONTHLY AGGREGATES")
m = (
    daily.group_by("ym")
    .agg(
        pl.col("gmv").sum().alias("gmv"),
        pl.col("orders").sum().alias("orders"),
        pl.col("active_users").mean().alias("dau_avg"),
        pl.col("buyers").sum().alias("buyer_days"),
        pl.len().alias("ndays"),
        (pl.col("gmv").sum() / pl.len()).alias("gmv_per_day"),
        (pl.col("gmv").sum() / pl.col("orders").sum()).alias("aov"),
    )
    .sort("ym")
)
with pl.Config(tbl_rows=30, float_precision=1, tbl_width_chars=140):
    print(m)

# ---------------------------------------------------------------- top spike days
print("=" * 100)
print("TOP 15 DAYS BY GMV")
with pl.Config(tbl_rows=15, float_precision=0):
    print(daily.sort("gmv", descending=True).head(15).select("event_date", "gmv", "orders", "active_users"))

print("BOTTOM 10 DAYS BY GMV")
with pl.Config(tbl_rows=10, float_precision=0):
    print(daily.sort("gmv").head(10).select("event_date", "gmv", "orders", "active_users"))

# key periods detail
for lbl, a, b in [
    ("Feb-Mar 2025 (Feb23 + Mar8)", date(2025, 2, 18), date(2025, 3, 12)),
    ("May holidays 2025", date(2025, 4, 25), date(2025, 5, 12)),
    ("November 2025 (11.11 / BF)", date(2025, 11, 5), date(2025, 12, 2)),
    ("NY peak + Jan slump", date(2025, 12, 15), date(2026, 1, 12)),
]:
    print("-" * 60, lbl)
    with pl.Config(tbl_rows=40, float_precision=0):
        print(win(a, b).select("event_date", "wd", "gmv", "orders", "active_users"))

# ---------------------------------------------------------------- window sums 2025 + 2026
W1_25 = (date(2025, 1, 15), date(2025, 2, 13))
W2_25 = (date(2025, 2, 14), date(2025, 3, 15))
W1_26 = (date(2026, 1, 15), date(2026, 2, 13))

print("=" * 100)
print("WINDOW SUMS")
rows = []
for lbl, (a, b) in [("W1_25 (val-like 2025)", W1_25), ("W2_25 (test-like 2025)", W2_25), ("W1_26 (actual val window)", W1_26)]:
    g, o = wsum("gmv", a, b), wsum("orders", a, b)
    nd = win(a, b).height
    rows.append((lbl, nd, g, o))
    print(f"{lbl}: days={nd} gmv={g:,.0f} orders={o:,.0f} gmv/day={g/nd:,.0f}")

g_w1_25, g_w2_25 = rows[0][2], rows[1][2]
o_w1_25, o_w2_25 = rows[0][3], rows[1][3]
g_w1_26 = rows[2][2]

R_season = g_w2_25 / g_w1_25
R_season_ord = o_w2_25 / o_w1_25
print(f"\nR_season (gmv W2_25/W1_25)    = {R_season:.4f}")
print(f"R_season_orders               = {R_season_ord:.4f}")

# distinct actives per window from user file
user = pl.read_parquet(f"{DATA}/seas_user_windows_2025.parquet")
n_w1 = int(user["act_w1"].sum())
n_w2 = int(user["act_w2"].sum())
R_gmv_per_active = (g_w2_25 / n_w2) / (g_w1_25 / n_w1)
print(f"window actives 2025: W1={n_w1:,} mdl_onyx={n_w2:,} ratio={n_w2/n_w1:.4f}")
print(f"R_season gmv-per-active-user  = {R_gmv_per_active:.4f}")

# ---------------------------------------------------------------- YoY growth
print("=" * 100)
print("YOY GROWTH (2026 vs 2025, same calendar spans)")
spans = [
    ("Jan 1-14", date(2025, 1, 1), date(2025, 1, 14)),
    ("Jan 15-31", date(2025, 1, 15), date(2025, 1, 31)),
    ("Jan full", date(2025, 1, 1), date(2025, 1, 31)),
    ("Feb 1-13", date(2025, 2, 1), date(2025, 2, 13)),
    ("Jan15-Feb13 (=)", date(2025, 1, 15), date(2025, 2, 13)),
]
yoy = {}
for lbl, a, b in spans:
    a2, b2 = date(a.year + 1, a.month, a.day), date(b.year + 1, b.month, b.day)
    r_g = wsum("gmv", a2, b2) / wsum("gmv", a, b)
    r_o = wsum("orders", a2, b2) / wsum("orders", a, b)
    r_u = win(a2, b2)["active_users"].mean() / win(a, b)["active_users"].mean()
    yoy[lbl] = r_g
    print(f"{lbl:>18}: gmv x{r_g:.3f}  orders x{r_o:.3f}  DAU x{r_u:.3f}")

# also month-over-month recent trend in 2026 and Dec 2025 for context
print("\nDec 2025 gmv:", f"{wsum('gmv', date(2025,12,1), date(2025,12,31)):,.0f}")

# ---------------------------------------------------------------- M estimate
print("=" * 100)
print("M ESTIMATE (expected test-window / val-window global gmv, 2026)")
# YoY factor drift: yoy over sequential spans tells whether growth accelerates.
y1, y2, y3 = yoy["Jan 1-14"], yoy["Jan 15-31"], yoy["Feb 1-13"]
yoy_w1 = yoy["Jan15-Feb13 (=)"]
# mid-dates (days from Jan 8 approx): Jan 7.5, Jan 23, Feb 7 -> slope per 30d
xs = np.array([7.0, 23.0, 38.0])  # day-of-year midpoints approx
ys = np.log(np.array([y1, y2, y3]))
slope = np.polyfit(xs, ys, 1)[0]  # d log(yoy) / d day

drift = float(np.exp(slope * 30.5))
yoy_w2_est = yoy_w1 * drift
M_seasonal_only = R_season
M_drift = R_season * drift

M_direct = g_w2_25 * yoy_w2_est / g_w1_26
print(f"yoy drift per 30.5d = x{drift:.4f} (slope from Jan1-14/Jan15-31/Feb1-13 log-yoy fit)")
print(f"yoy_w1 (observed)   = {yoy_w1:.4f}; yoy_w2 extrapolated = {yoy_w2_est:.4f}")
print(f"M (pure 2025 seasonal carry)  = {M_seasonal_only:.4f}")
print(f"M (seasonal x yoy-drift)      = {M_drift:.4f}  [= M_direct {M_direct:.4f}]")
lo, hi = sorted([M_seasonal_only, M_drift])
lo, hi = lo * 0.97, hi * 1.03  # +-3% for estimation noise
print(f"M point = {M_drift:.3f}, plausible range = [{lo:.3f}, {hi:.3f}]")

# ---------------------------------------------------------------- weekday pattern
print("=" * 100)
print("WEEKDAY PATTERN (gmv / centered 7d MA, excludes NY + big-sale distortion automatically)")
dd = daily.with_columns(pl.col("gmv").rolling_mean(window_size=7, center=True).alias("ma7"))
dd = dd.with_columns((pl.col("gmv") / pl.col("ma7")).alias("f"))
wdt = dd.drop_nulls("f").group_by("wd").agg(pl.col("f").mean().alias("factor"), pl.len()).sort("wd")
names = {1: "Mon", 2: "Tue", 3: "Wed", 4: "Thu", 5: "Fri", 6: "Sat", 7: "Sun"}
for r in wdt.iter_rows(named=True):
    print(f"  {names[r['wd']]}: {r['factor']:.4f}")
fmax, fmin = wdt["factor"].max(), wdt["factor"].min()
print(f"weekday amplitude max/min = {fmax/fmin:.4f}")

# weekday composition of the two 2026 windows
for lbl, a, b in [("val W1_26", *W1_26), ("test W2_26", date(2026, 2, 14), date(2026, 3, 15))]:
    dr = pl.date_range(a, b, interval="1d", eager=True)
    wds = dr.dt.weekday().to_list()
    cnt = {names[k]: wds.count(k) for k in range(1, 8)}
    # expected weekday effect on window sum
    fac = {r["wd"]: r["factor"] for r in wdt.iter_rows(named=True)}
    eff = sum(fac[k] * wds.count(k) for k in range(1, 8)) / len(wds)
    print(f"{lbl}: {cnt}  mean weekday factor = {eff:.4f}")

# ---------------------------------------------------------------- per-user log shift
print("=" * 100)
print("PER-USER LOG-SCALE SEASONAL SHIFT (RMSLE view)")
for cohort, flag in [("active 2025-01-01..14 (anchor-like)", "act_jan14"), ("active any day Jan 2025", "act_jan")]:
    c = user.filter(pl.col(flag))
    l1 = c.select(pl.col("gmv_w1").log1p().mean()).item()
    l2 = c.select(pl.col("gmv_w2").log1p().mean()).item()
    sh = l2 - l1
    buy1 = c.select((pl.col("gmv_w1") > 0).mean()).item()
    buy2 = c.select((pl.col("gmv_w2") > 0).mean()).item()
    m1 = c["gmv_w1"].mean()
    m2 = c["gmv_w2"].mean()
    # log shift among users with gmv>0 in both windows
    both = c.filter((pl.col("gmv_w1") > 0) & (pl.col("gmv_w2") > 0))
    shb = both.select((pl.col("gmv_w2").log1p() - pl.col("gmv_w1").log1p()).mean()).item()
    print(f"cohort [{cohort}]: n={c.height:,}")
    print(f"  mean log1p(gmv): W1={l1:.4f} mdl_onyx={l2:.4f}  shift={sh:+.4f}  ratio={l2/l1:.4f}")
    print(f"  buyer share: W1={buy1:.4f} mdl_onyx={buy2:.4f}   mean gmv: W1={m1:.1f} mdl_onyx={m2:.1f} (x{m2/m1:.3f})")
    print(f"  among buyers in both windows (n={both.height:,}): mean dlog1p = {shb:+.4f}")

# ---------------------------------------------------------------- weekly series for report chart
print("=" * 100)
wk = (
    daily.group_by(pl.col("event_date").dt.truncate("1w").alias("week"))
    .agg(pl.col("gmv").sum(), pl.len().alias("nd"))
    .sort("week")
    .filter(pl.col("nd") == 7)
)
gm = wk["gmv"].max()
for r in wk.iter_rows(named=True):
    bar = "#" * int(round(60 * r["gmv"] / gm))
    print(f"{r['week']}  {r['gmv']/1e6:7.2f}M {bar}")

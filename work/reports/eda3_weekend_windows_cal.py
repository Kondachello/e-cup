"""Lens weekend/weekday: exact RF production-calendar composition of windows + daily off-day factors."""
import datetime as dt
import numpy as np
import pandas as pd

D = dt.date

# --- RF production calendar (official decrees) ---
# 2025 (post. 1335 of 04.10.2024): holidays Jan1-8; transfers: Jan4(Sat)->May2, Jan5(Sun)->Dec31,
#   Feb23(Sun)->May8, Mar8(Sat)->Jun13, Nov1(Sat working)->Nov3. Extra off: May1,2,8,9; Jun12,13; Nov3,4; Dec31.
# 2026 (post. of Sep 2025): holidays Jan1-8; transfers: Jan3(Sat)->Jan9, Jan4(Sun)->Dec31.
#   Feb23=Mon holiday itself. Mar8(Sun)->Mar9(Mon) auto per TK art.112. May9(Sat)->May11.
EXTRA_OFF = set(
    [D(2025, 1, d) for d in range(1, 9)]
    + [D(2025, 5, 1), D(2025, 5, 2), D(2025, 5, 8), D(2025, 5, 9),
       D(2025, 6, 12), D(2025, 6, 13), D(2025, 11, 3), D(2025, 11, 4), D(2025, 12, 31)]
    + [D(2026, 1, d) for d in range(1, 10)]  # Jan1-8 + Jan9 transferred
    + [D(2026, 2, 23), D(2026, 3, 9)]
)
WORKING_SAT = {D(2025, 11, 1)}

def is_off(d):
    if d in WORKING_SAT:
        return False
    return d.weekday() >= 5 or d in EXTRA_OFF

def win(d0, d1):
    days = [d0 + dt.timedelta(i) for i in range((d1 - d0).days + 1)]
    return days

WINDOWS = {
    "val26   15.01-13.02.26": (D(2026, 1, 15), D(2026, 2, 13)),
    "test26  14.02-15.03.26": (D(2026, 2, 14), D(2026, 3, 15)),
    "valmir25 15.01-13.02.25": (D(2025, 1, 15), D(2025, 2, 13)),
    "testmir25 14.02-15.03.25": (D(2025, 2, 14), D(2025, 3, 15)),
    "apr25   01.04-30.04.25": (D(2025, 4, 1), D(2025, 4, 30)),
    "may25   01.05-30.05.25": (D(2025, 5, 1), D(2025, 5, 30)),
    "sep25   01.09-30.09.25": (D(2025, 9, 1), D(2025, 9, 30)),
    "oct25   01.10-30.10.25": (D(2025, 10, 1), D(2025, 10, 30)),
    "nov25   01.11-30.11.25": (D(2025, 11, 1), D(2025, 11, 30)),
    "jun25   01.06-30.06.25": (D(2025, 6, 1), D(2025, 6, 30)),
}
dow_names = "Mon Tue Wed Thu Fri Sat Sun".split()
print("=== window composition (RF production calendar) ===")
for name, (a, b) in WINDOWS.items():
    days = win(a, b)
    off = [d for d in days if is_off(d)]
    dowc = np.bincount([d.weekday() for d in days], minlength=7)
    hol = [d for d in off if d.weekday() < 5]
    print(f"{name}: n={len(days)} off={len(off)} work={len(days)-len(off)} "
          f"dow={dict(zip(dow_names, dowc.tolist()))} weekday-holidays={[str(d) for d in hol]}")

df["date"] = df["event_date"].dt.date
df["lg"] = np.log(df["gmv_sum"])
df["dow"] = df["event_date"].dt.weekday
df["off"] = df["date"].map(is_off)
# trend base: centered 29-day rolling median of log gmv
df["base"] = df["lg"].rolling(29, center=True, min_periods=15).median()
df["res"] = df["lg"] - df["base"]
# exclude NY distortion zones from factor estimation
mask_ny = (df["date"] <= D(2025, 1, 14)) | ((df["date"] >= D(2025, 12, 20)) & (df["date"] <= D(2026, 1, 11)))
est = df[~mask_ny].copy()

hol_weekday = est[(~est["dow"].ge(5)) & est["off"]]  # weekday holidays
norm = est[~est["off"] | est["dow"].ge(5)]
print("\n=== dow residual factors (log daily GMV vs 29d rolling median), excl NY zones ===")
for d in range(7):
    sub = est[(est["dow"] == d) & (~est["date"].isin(set(hol_weekday["date"])))]
    print(f"{dow_names[d]}: mean res {sub['res'].mean():+.4f}  (n={len(sub)})")
print("\n=== weekday-holiday days individually ===")
for _, r in hol_weekday.iterrows():
    same_dow_norm = est[(est["dow"] == r["dow"]) & (~est["date"].isin(set(hol_weekday["date"])))]["res"].mean()
    print(f"{r['date']} ({dow_names[r['dow']]}): res {r['res']:+.4f}  vs same-dow norm {same_dow_norm:+.4f}  delta {r['res']-same_dow_norm:+.4f}")
hd = hol_weekday["res"].mean()
mon_norm = est[(est["dow"] < 5) & (~est["date"].isin(set(hol_weekday["date"])))]["res"].mean()
print(f"\nmean weekday-holiday res {hd:+.4f} vs normal-weekday res {mon_norm:+.4f} -> holiday delta {hd-mon_norm:+.4f} (log daily)")
# level effect of 2 extra off-Mondays in test26 window on window SUM
delta_frac = 2.0 * (np.exp(hd) - np.exp(mon_norm)) / 30.0
print(f"level effect on 30d window GMV sum of 2 holiday-Mondays: {delta_frac:+.4%} of window GMV")

# sanity: 2026 pre-anchor promo days 12-13.02
for d in [D(2026, 2, 10), D(2026, 2, 11), D(2026, 2, 12), D(2026, 2, 13)]:
    r = df[df["date"] == d]
    print(f"{d}: res {float(r['res'].iloc[0]):+.4f}")

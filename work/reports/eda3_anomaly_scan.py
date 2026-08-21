"""eda3 step 2: robust per-series anomaly scan of the daily calendar."""
import numpy as np
import polars as pl

dates = d["event_date"].to_list()
n = d.height

series_cols = [c for c in d.columns if c.endswith("_sum") or c.endswith("_nnz") or c == "dau"]
series_cols = [c for c in series_cols if not c.startswith("has_")]  # flags: sums only meaningful, keep has_*_sum
series_cols += [c for c in d.columns if c.startswith("has_") and c.endswith("_sum")]
series_cols = list(dict.fromkeys(series_cols))

d = d.with_columns((pl.col("ghost_rows") / pl.col("dau")).alias("ghost_share"))
extra = ["ghost_share"]
# per-active-user rates: remove DAU (behavioral) component -> pure instrumentation signal
rate_defs = {}
for c in series_cols:
    if c != "dau":
        rc = f"rate_{c}"
        d = d.with_columns((pl.col(c) / pl.col("dau")).alias(rc))
        rate_defs[rc] = c

def robust_z(y):
    """same-weekday local baseline: median of +-4 same weekdays, excl. self."""
    base = np.full(n, np.nan)
    for t in range(n):
        idx = [t + k * 7 for k in (-4, -3, -2, -1, 1, 2, 3, 4) if 0 <= t + k * 7 < n]
        base[t] = np.median(y[idx])
    r = y - base
    mad = np.median(np.abs(r - np.median(r)))
    sd = 1.4826 * mad if mad > 0 else (np.std(r) + 1e-12)
    return r / sd, r

rows = []
zmat = {}
for c in series_cols + extra + list(rate_defs):
    x = d[c].to_numpy().astype(float)
    y = np.log1p(x) if not (c.startswith("rate_") or c == "ghost_share") else np.log(x + 1e-4)
    z, r = robust_z(y)
    zmat[c] = z
    for t in np.where(np.abs(z) >= 4.5)[0]:
        rows.append((str(dates[t]), c, round(float(z[t]), 2), round(float(r[t]), 4), float(x[t])))

out = pl.DataFrame(rows, schema=["date", "series", "z", "log_resid", "value"], orient="row").sort(["date", "series"])

# summary: days with anomalies in >=2 series, and max |z| per day
agg = out.group_by("date").agg(pl.len().alias("n_series"),
                               pl.col("z").abs().max().alias("max_abs_z"),
                               pl.col("series").str.concat(",").alias("which")).sort("date")
pl.Config.set_tbl_rows(200); pl.Config.set_fmt_str_lengths(220)
print(agg)

# step-change scan: 28d-vs-28d rolling median shift on the RATE series (instrumentation regime changes)
print("\n=== step scan (rate series, |shift| of 28d medians, top) ===")
steps = []
for c in list(rate_defs) + ["ghost_share"]:
    y = np.log(d[c].to_numpy().astype(float) + 1e-4)
    for t in range(28, n - 28):
        a = np.median(y[t - 28:t]); b = np.median(y[t:t + 28])
        steps.append((str(dates[t]), c, round(float(b - a), 4)))
sdf = pl.DataFrame(steps, schema=["date", "series", "shift"], orient="row")
# top absolute step per series
top = (sdf.with_columns(pl.col("shift").abs().alias("a"))
          .sort("a", descending=True).group_by("series", maintain_order=True).head(1))
print(top.sort("a", descending=True).head(30))

"""Per-user off/work-day aggregates for the weekend-composition lens. One streaming pass."""
import datetime as dt
import polars as pl

D = dt.date
EXTRA_OFF = set(
    [D(2025, 1, d) for d in range(1, 9)]
    + [D(2025, 5, 1), D(2025, 5, 2), D(2025, 5, 8), D(2025, 5, 9),
       D(2025, 6, 12), D(2025, 6, 13), D(2025, 11, 3), D(2025, 11, 4), D(2025, 12, 31)]
    + [D(2026, 1, d) for d in range(1, 10)]
    + [D(2026, 2, 23), D(2026, 3, 9)]
)
WORKING_SAT = {D(2025, 11, 1)}
HOL8 = {D(2025, 5, 1), D(2025, 5, 2), D(2025, 5, 8), D(2025, 5, 9),
        D(2025, 6, 12), D(2025, 6, 13), D(2025, 11, 3), D(2025, 11, 4)}

d0, d1 = D(2025, 1, 1), D(2026, 2, 13)
rows = []
base_mon = D(2025, 1, 6)
for i in range((d1 - d0).days + 1):
    d = d0 + dt.timedelta(i)
    off = (d.weekday() >= 5 or d in EXTRA_OFF) and d not in WORKING_SAT
    par = ((d - base_mon).days // 7) % 2
    rows.append({
        "event_date": d, "off": off, "par": par,
        "p1": D(2025, 1, 12) <= d <= D(2025, 3, 31),
        "p2": D(2025, 7, 1) <= d <= D(2025, 10, 31),
        "p3": D(2025, 1, 12) <= d <= D(2025, 8, 31),
        "full": (D(2025, 1, 12) <= d <= D(2026, 2, 13)) and not (D(2025, 12, 20) <= d <= D(2026, 1, 11)),
        "m_apr": D(2025, 4, 1) <= d <= D(2025, 4, 30),
        "m_may": D(2025, 5, 1) <= d <= D(2025, 5, 30),
        "m_sep": D(2025, 9, 1) <= d <= D(2025, 9, 30),
        "m_oct": D(2025, 10, 1) <= d <= D(2025, 10, 30),
        "m_nov": D(2025, 11, 1) <= d <= D(2025, 11, 30),
        "hol8": d in HOL8,
    })
dim = pl.DataFrame(rows).with_columns(pl.col("event_date").cast(pl.Date))
# day counts per flag combo for later rate denominators
cnt = dim.group_by(["off"]).agg(
    pl.col("p1").sum(), pl.col("p2").sum(), pl.col("p3").sum(), pl.col("full").sum(),
    pl.col("m_apr").sum(), pl.col("m_may").sum(), pl.col("m_sep").sum(), pl.col("m_oct").sum(), pl.col("m_nov").sum())
cnt2 = dim.filter(pl.col("full")).group_by(["off", "par"]).agg(pl.len().alias("n"))

lf = pl.scan_parquet("/Users/alexanderkondakov/ozon-cup/train.parquet").select(
    "user_id", "event_date", "to_ord", "gmv").join(dim.lazy(), on="event_date", how="left")

def s(cond, col, name):
    return pl.when(cond).then(pl.col(col)).otherwise(0).sum().alias(name)

def sd(cond, name):  # active-day count
    return cond.cast(pl.Int32).sum().alias(name)

aggs = []
for per in ["p1", "p2", "p3", "full"]:
    for offv, tag in [(True, "off"), (False, "wk")]:
        c = pl.col(per) & (pl.col("off") == offv)
        aggs += [s(c, "to_ord", f"{per}_ord_{tag}"), s(c, "gmv", f"{per}_gmv_{tag}"), sd(c, f"{per}_ad_{tag}")]
# parity split within full
for pv in [0, 1]:
    for offv, tag in [(True, "off"), (False, "wk")]:
        c = pl.col("full") & (pl.col("par") == pv) & (pl.col("off") == offv)
        aggs += [s(c, "to_ord", f"fp{pv}_ord_{tag}"), s(c, "gmv", f"fp{pv}_gmv_{tag}"), sd(c, f"fp{pv}_ad_{tag}")]
for m in ["m_apr", "m_may", "m_sep", "m_oct", "m_nov"]:
    aggs += [s(pl.col(m), "gmv", f"{m}_gmv"), s(pl.col(m), "to_ord", f"{m}_ord")]
aggs += [s(pl.col("hol8"), "to_ord", "hol8_ord"), s(pl.col("hol8"), "gmv", "hol8_gmv"), sd(pl.col("hol8"), "hol8_ad")]

out = lf.group_by("user_id").agg(aggs).sort("user_id")
res = out.collect(engine="streaming")
print("users:", res.height)
res.write_parquet("/private/tmp/claude-501/-Users-alexanderkondakov-ozon-cup/b994bee8-2354-4524-9a2b-530595cd5feb/scratchpad/user_offday.parquet")
print(res.select(pl.col("full_ord_off").sum(), pl.col("full_ord_wk").sum(), pl.col("m_may_gmv").sum(), pl.col("m_apr_gmv").sum()))

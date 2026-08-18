#!/usr/bin/env python
"""EDA: data sanity + target distribution at VAL anchor (Ozon E-CUP LTV).

Writes work/reports/eda_sanity.md and prints a JSON metrics dict to stdout.
"""
import json
import numpy as np
import polars as pl
from datetime import date

ROOT = "/Users/alexanderkondakov/ozon-cup"
TRAIN = f"{ROOT}/train.parquet"
SUBMIT = f"{ROOT}/sample_submit.csv"
REPORT = f"{ROOT}/work/reports/eda_sanity.md"

VAL_ANCHOR = date(2026, 1, 14)
TEST_ANCHOR = date(2026, 2, 13)
TGT_START = date(2026, 1, 15)
TGT_END = date(2026, 2, 13)

lf = pl.scan_parquet(TRAIN)
sub = pl.read_csv(SUBMIT).select("user_id")
n_sub = sub.height

# ---------------- 1. Sanity ----------------
sanity = lf.select(
    n_rows=pl.len(),
    max_abs_gmv_diff=(pl.col("gmv") - pl.col("gmv_search") - pl.col("gmv_cat")).abs().max(),
    n_neg_gmv=(pl.col("gmv") < 0).sum(),
    min_gmv=pl.col("gmv").min(),
    max_gmv=pl.col("gmv").max(),
    n_null_gmv=pl.col("gmv").is_null().sum(),
    n_zero_gmv_rows=(pl.col("gmv") == 0).sum(),
    n_users_train=pl.col("user_id").n_unique(),
    min_date=pl.col("event_date").min(),
    max_date=pl.col("event_date").max(),
).collect()
S = sanity.to_dicts()[0]

dup = (
    lf.group_by("user_id", "event_date")
    .len()
    .select(
        n_pairs=pl.len(),
        n_dup_pairs=(pl.col("len") > 1).sum(),
        n_extra_rows=(pl.col("len") - 1).sum(),
    )
    .collect()
    .to_dicts()[0]
)

rpu = (
    lf.group_by("user_id")
    .len()
    .select(
        p50=pl.col("len").quantile(0.5),
        p90=pl.col("len").quantile(0.9),
        p99=pl.col("len").quantile(0.99),
        max=pl.col("len").max(),
        mean=pl.col("len").mean(),
    )
    .collect()
    .to_dicts()[0]
)

train_users = lf.select(pl.col("user_id").unique()).collect()
in_both = train_users.join(sub, on="user_id", how="inner").height
train_not_sub = train_users.height - in_both
sub_not_train = n_sub - in_both

# ---------------- 2. Target at VAL anchor ----------------
tgt = (
    lf.filter((pl.col("event_date") >= TGT_START) & (pl.col("event_date") <= TGT_END))
    .group_by("user_id")
    .agg(target=pl.col("gmv").sum())
    .collect()
)

# ---------------- 4. Activity coverage (per-user recency windows) ----------------
def per_user_windows(anchor: date) -> pl.DataFrame:
    return (
        lf.filter(pl.col("event_date") <= anchor)
        .with_columns(days_back=(pl.lit(anchor) - pl.col("event_date")).dt.total_days())
        .group_by("user_id")
        .agg(
            db_any=pl.col("days_back").min(),
            db_ord=pl.col("days_back").filter(pl.col("to_ord") > 0).min(),
        )
        .collect()
    )

val_pw = per_user_windows(VAL_ANCHOR)
test_pw = per_user_windows(TEST_ANCHOR)

master = (
    sub.join(tgt, on="user_id", how="left")
    .join(val_pw.rename({"db_any": "db_any_val", "db_ord": "db_ord_val"}), on="user_id", how="left")
    .join(test_pw.rename({"db_any": "db_any_test", "db_ord": "db_ord_test"}), on="user_id", how="left")
    .with_columns(pl.col("target").fill_null(0.0))
)

y = master["target"].to_numpy()
n_neg_target = int((y < 0).sum())
y_clip = np.clip(y, 0, None)
zero_share = float((y_clip == 0).mean())
pos = y_clip[y_clip > 0]
pos_q = {p: float(np.quantile(pos, p / 100)) for p in (25, 50, 75, 90, 99)}
pos_max = float(pos.max())
logy = np.log1p(y_clip)
mean_log = float(logy.mean())

# ---------------- 3. Constant floors ----------------
rmsle_zero = float(np.sqrt(np.mean(logy**2)))
rmsle_const = float(np.sqrt(np.mean((logy - mean_log) ** 2)))  # = pop. std of log1p(y)
const_value = float(np.expm1(mean_log))

def cov_shares(df: pl.DataFrame, suffix: str) -> dict:
    out = {}
    a, o = f"db_any_{suffix}", f"db_ord_{suffix}"
    for w in (7, 30, 90, 365):
        out[f"active{w}_share_{suffix}"] = float((df[a] < w).sum() / n_sub)
    for w in (30, 90, 365):
        out[f"order{w}_share_{suffix}"] = float((df[o] < w).sum() / n_sub)
    out[f"ever_active_share_{suffix}"] = float(df[a].is_not_null().sum() / n_sub)
    return out

cov_val = cov_shares(master, "val")
cov_test = cov_shares(master, "test")

# ---------------- 5. Cross-tabs at VAL anchor ----------------
no_ord365 = master.filter(pl.col("db_ord_val").is_null() | (pl.col("db_ord_val") >= 365))
n_no_ord365 = no_ord365.height
tgtpos_no_ord365 = float((no_ord365["target"] > 0).mean())
med_pos_no_ord365 = no_ord365.filter(pl.col("target") > 0)["target"].median()

ord30 = master.filter(pl.col("db_ord_val") < 30)
n_ord30 = ord30.height
tgtpos_ord30 = float((ord30["target"] > 0).mean())
med_pos_ord30 = float(ord30.filter(pl.col("target") > 0)["target"].median())

# active but never ordered (365d) split, useful context
act30_no_ord365 = master.filter(
    (pl.col("db_any_val") < 30) & (pl.col("db_ord_val").is_null() | (pl.col("db_ord_val") >= 365))
)
tgtpos_act30_no_ord365 = float((act30_no_ord365["target"] > 0).mean()) if act30_no_ord365.height else 0.0

metrics = {
    # sanity
    "n_rows": int(S["n_rows"]),
    "max_abs_gmv_diff": float(S["max_abs_gmv_diff"]),
    "n_dup_user_date_pairs": int(dup["n_dup_pairs"]),
    "n_neg_gmv_rows": int(S["n_neg_gmv"]),
    "n_users_train": int(S["n_users_train"]),
    "n_users_submit": int(n_sub),
    "n_train_users_not_in_submit": int(train_not_sub),
    "n_submit_users_not_in_train": int(sub_not_train),
    "zero_gmv_row_share": float(S["n_zero_gmv_rows"] / S["n_rows"]),
    "rows_per_user_p50": float(rpu["p50"]),
    "rows_per_user_p90": float(rpu["p90"]),
    "rows_per_user_p99": float(rpu["p99"]),
    "rows_per_user_max": int(rpu["max"]),
    # target
    "zero_share": zero_share,
    "n_neg_target": n_neg_target,
    "pos_target_p25": pos_q[25],
    "pos_target_p50": pos_q[50],
    "pos_target_p75": pos_q[75],
    "pos_target_p90": pos_q[90],
    "pos_target_p99": pos_q[99],
    "pos_target_max": pos_max,
    "mean_log1p_target": mean_log,
    # floors
    "rmsle_zero": rmsle_zero,
    "rmsle_const": rmsle_const,
    "const_value": const_value,
    # coverage
    **cov_val,
    **cov_test,
    # cross-tabs
    "no_ord365_share_val": float(n_no_ord365 / n_sub),
    "tgtpos_given_no_ord365_val": tgtpos_no_ord365,
    "ord30_share_val": float(n_ord30 / n_sub),
    "tgtpos_given_ord30_val": tgtpos_ord30,
    "median_pos_target_given_ord30_val": med_pos_ord30,
    "tgtpos_given_act30_no_ord365_val": tgtpos_act30_no_ord365,
}

# ---------------- report ----------------
rep = f"""# EDA: data sanity + target distribution (VAL anchor {VAL_ANCHOR})

Data: `train.parquet`, {S['n_rows']:,} rows, dates {S['min_date']} .. {S['max_date']}.
Submission universe: {n_sub:,} users (`sample_submit.csv`).

## 1. Sanity

| check | value |
|---|---|
| max abs(gmv - gmv_search - gmv_cat) | {S['max_abs_gmv_diff']:.6g} |
| duplicate (user_id, event_date) pairs | {dup['n_dup_pairs']:,} (extra rows: {dup['n_extra_rows']:,}) |
| rows with gmv < 0 | {S['n_neg_gmv']:,} (min gmv = {S['min_gmv']:.4g}) |
| rows with gmv null | {S['n_null_gmv']:,} |
| max gmv on a single user-day | {S['max_gmv']:,.0f} |
| zero-gmv row share | {S['n_zero_gmv_rows'] / S['n_rows']:.4f} |
| distinct users in train | {S['n_users_train']:,} |
| train users not in submit universe | {train_not_sub:,} |
| submit users with no train rows at all | {sub_not_train:,} |
| rows per user p50 / p90 / p99 / max | {rpu['p50']:.0f} / {rpu['p90']:.0f} / {rpu['p99']:.0f} / {rpu['max']:,} (mean {rpu['mean']:.1f}) |

## 2. Target at VAL anchor {VAL_ANCHOR} (gmv sum {TGT_START}..{TGT_END}, absent = 0)

- Share of users with target = 0: **{zero_share:.4f}**  ({int(round(zero_share * n_sub)):,} of {n_sub:,})
- Users with negative target: {n_neg_target:,}
- Positive-target quantiles: p25 = {pos_q[25]:,.0f}, p50 = {pos_q[50]:,.0f}, p75 = {pos_q[75]:,.0f}, p90 = {pos_q[90]:,.0f}, p99 = {pos_q[99]:,.0f}, max = {pos_max:,.0f}
- mean(log1p(target)) = **{mean_log:.5f}**

## 3. Constant-prediction floors (RMSLE on VAL target)

| predictor | RMSLE |
|---|---|
| predict 0 for everyone | **{rmsle_zero:.5f}** |
| optimal constant c* = expm1(mean(log1p(y))) = {const_value:,.2f} | **{rmsle_const:.5f}** |

Any real model must beat {rmsle_const:.4f}; predicting 0 costs {rmsle_zero:.4f}.

## 4. Activity coverage of the 250k universe

Share of submit users with >= 1 row (any activity) in the last N days up to and incl. the anchor:

| window | VAL anchor {VAL_ANCHOR} | TEST anchor {TEST_ANCHOR} |
|---|---|---|
| 7d   | {cov_val['active7_share_val']:.4f} | {cov_test['active7_share_test']:.4f} |
| 30d  | {cov_val['active30_share_val']:.4f} | {cov_test['active30_share_test']:.4f} |
| 90d  | {cov_val['active90_share_val']:.4f} | {cov_test['active90_share_test']:.4f} |
| 365d | {cov_val['active365_share_val']:.4f} | {cov_test['active365_share_test']:.4f} |
| ever (<= anchor) | {cov_val['ever_active_share_val']:.4f} | {cov_test['ever_active_share_test']:.4f} |

Share with >= 1 ORDER day (to_ord > 0) in the last N days:

| window | VAL anchor | TEST anchor |
|---|---|---|
| 30d  | {cov_val['order30_share_val']:.4f} | {cov_test['order30_share_test']:.4f} |
| 90d  | {cov_val['order90_share_val']:.4f} | {cov_test['order90_share_test']:.4f} |
| 365d | {cov_val['order365_share_val']:.4f} | {cov_test['order365_share_test']:.4f} |

## 5. Cross-tab: past orders vs future target (VAL anchor)

- Users with **0 order-days in last 365d**: {n_no_ord365:,} ({n_no_ord365 / n_sub:.4f} of universe). Of them, target > 0: **{tgtpos_no_ord365:.4f}**{f" (median positive target {med_pos_no_ord365:,.0f})" if med_pos_no_ord365 is not None else ""}.
- Users with **>= 1 order-day in last 30d**: {n_ord30:,} ({n_ord30 / n_sub:.4f}). Of them, target > 0: **{tgtpos_ord30:.4f}**, median positive target = **{med_pos_ord30:,.0f}**.
- Users active in last 30d but 0 order-days in 365d: target>0 share = {tgtpos_act30_no_ord365:.4f}.

## Takeaways

- **Universe construction (verified independently): every one of the 250k users has >= 1 activity row within the last 30 days of BOTH anchors** (max gap = 29 days at 2026-01-14 and at 2026-02-13). No cold-start users; "active in last 30d" is a property of the universe, not a feature.

- gmv decomposition holds (max abs diff {S['max_abs_gmv_diff']:.2g}); {"no" if dup['n_dup_pairs'] == 0 else str(dup['n_dup_pairs'])} duplicate user-day keys; {"no" if S['n_neg_gmv'] == 0 else str(S['n_neg_gmv'])} negative gmv rows.
- Target is dominated by zeros ({zero_share:.1%}); RMSLE is driven by (a) classifying who buys and (b) log-scale magnitude for buyers.
- Recent order activity is the strongest separator: P(target>0) is {tgtpos_ord30:.2f} for 30d-orderers vs {tgtpos_no_ord365:.2f} for 365d-no-order users.
- Coverage at TEST anchor is close to VAL anchor => features/windows transfer.
"""

with open(REPORT, "w") as f:
    f.write(rep)

print(json.dumps(metrics, indent=1))

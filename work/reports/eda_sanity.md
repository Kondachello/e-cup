# EDA: data sanity + target distribution (VAL anchor 2026-01-14)

Data: `train.parquet`, 30,631,006 rows, dates 2025-01-01 .. 2026-02-13.
Submission universe: 250,000 users (`sample_submit.csv`).

## 1. Sanity

| check | value |
|---|---|
| max abs(gmv - gmv_search - gmv_cat) | 5.27507e-11 |
| duplicate (user_id, event_date) pairs | 0 (extra rows: 0) |
| rows with gmv < 0 | 0 (min gmv = 0) |
| rows with gmv null | 0 |
| max gmv on a single user-day | 73,830 |
| zero-gmv row share | 0.8454 |
| distinct users in train | 250,000 |
| train users not in submit universe | 0 |
| submit users with no train rows at all | 0 |
| rows per user p50 / p90 / p99 / max | 102 / 254 / 356 / 409 (mean 122.5) |

## 2. Target at VAL anchor 2026-01-14 (gmv sum 2026-01-15..2026-02-13, absent = 0)

- Share of users with target = 0: **0.4593**  (114,835 of 250,000)
- Users with negative target: 0
- Positive-target quantiles: p25 = 23, p50 = 61, p75 = 160, p90 = 362, p99 = 1,396, max = 53,747
- mean(log1p(target)) = **2.24210**

## 3. Constant-prediction floors (RMSLE on VAL target)

| predictor | RMSLE |
|---|---|
| predict 0 for everyone | **3.20364** |
| optimal constant c* = expm1(mean(log1p(y))) = 8.41 | **2.28829** |

Any real model must beat 2.2883; predicting 0 costs 3.2036.

## 4. Activity coverage of the 250k universe

Share of submit users with >= 1 row (any activity) in the last N days up to and incl. the anchor:

| window | VAL anchor 2026-01-14 | TEST anchor 2026-02-13 |
|---|---|---|
| 7d   | 0.8028 | 0.8049 |
| 30d  | 1.0000 | 1.0000 |
| 90d  | 1.0000 | 1.0000 |
| 365d | 1.0000 | 1.0000 |
| ever (<= anchor) | 1.0000 | 1.0000 |

Share with >= 1 ORDER day (to_ord > 0) in the last N days:

| window | VAL anchor | TEST anchor |
|---|---|---|
| 30d  | 0.5631 | 0.5407 |
| 90d  | 0.7500 | 0.7533 |
| 365d | 0.8626 | 0.8721 |

## 5. Cross-tab: past orders vs future target (VAL anchor)

- Users with **0 order-days in last 365d**: 34,343 (0.1374 of universe). Of them, target > 0: **0.0926** (median positive target 24).
- Users with **>= 1 order-day in last 30d**: 140,770 (0.5631). Of them, target > 0: **0.7441**, median positive target = **75**.
- Users active in last 30d but 0 order-days in 365d: target>0 share = 0.0926.

## Takeaways

- **Universe construction (verified independently): every one of the 250k users has >= 1 activity row within the last 30 days of BOTH anchors** (max gap = 29 days at 2026-01-14 and at 2026-02-13). No cold-start users; "active in last 30d" is a property of the universe, not a feature.

- gmv decomposition holds (max abs diff 5.3e-11); no duplicate user-day keys; no negative gmv rows.
- Target is dominated by zeros (45.9%); RMSLE is driven by (a) classifying who buys and (b) log-scale magnitude for buyers.
- Recent order activity is the strongest separator: P(target>0) is 0.74 for 30d-orderers vs 0.09 for 365d-no-order users.
- Coverage at TEST anchor is close to VAL anchor => features/windows transfer.

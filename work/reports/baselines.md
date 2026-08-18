# Heuristic baselines (VAL anchor 2026-01-14)

Protocol: free params tuned on anchor **2025-12-31** (features from data <= anchor,
target = per-user gmv over 2026-01-01..2026-01-30), then applied frozen to the
VAL anchor **2026-01-14** (target 2026-01-15..2026-02-13). User universe: the
250,000 users of `sample_submit.csv` (absent users have target 0). Metric: RMSLE.

- Share of users with target>0: tune 0.5355, val 0.5407
- Windows end at the anchor day inclusive; `gmv_ya` = gmv over [anchor-364, anchor-335]
  (for TEST this is exactly one year before the prediction window).
- `dec_h` = sum(gmv * 0.5^(days_ago/h)); grids: alpha 0.10..1.50/0.05, c 0.10..1.20/0.05.

## Ranked results (by VAL RMSLE)

| rank | method | frozen params | tune RMSLE | VAL RMSLE |
|---|---|---|---|---|
| 1 | blend_log | w=[0.195, 0.166, 0.403] | 1.78041 | 1.76140 |
| 2 | decay | halflife=30, c=0.65 | 1.82842 | 1.81070 |
| 3 | log_ar30 | c=0.70 | 1.99879 | 2.01407 |
| 4 | alpha_ar30 | alpha=0.25 | 2.01541 | 2.03213 |
| 5 | ar30 | - | 2.22673 | 2.19506 |
| 6 | const | const=8.26 | 2.29227 | 2.28835 |
| 7 | yearago | c=0.80 | 2.45579 | 2.44459 |
| 8 | zero | - | 3.19496 | 3.20364 |

Method notes:
- `zero`: predict 0 for everyone.
- `const`: best constant = expm1(mean(log1p(y_tune))) (closed form).
- `ar30`: last-30d gmv sum as-is (naive AR).
- `alpha_ar30`: alpha * gmv_30.
- `log_ar30`: expm1(c * log1p(gmv_30)).
- `decay`: expm1(c * log1p(sum gmv * 0.5^(days_ago/halflife))), 2D grid over (c, halflife in {7,14,30,60,120}).
- `blend_log`: expm1(w1*log1p(gmv_30) + w2*log1p(gmv_90/3) + w3*log1p(gmv_365/12.17)), w>=0 via NNLS (exact for the squared-log objective).
- `yearago`: expm1(c * log1p(gmv_ya)).

## Best method: `blend_log` (w=[0.195, 0.166, 0.403])

- VAL RMSLE = **1.76140** (tune 1.78041).
- Predictions saved: `work/preds/base_best_val.parquet`, `work/preds/base_best_test.parquet`
  (TEST anchor 2026-02-13, same frozen params); row order verified to match `sample_submit.csv`.
- TEST preds: mean=20.59, median=8.68, share>0=0.8721.

Caveats: the tuning anchor's target window (Jan 1-30) covers the post-New-Year
period while features end in the December peak, so tuned shrinkage may be slightly
biased low for later anchors; VAL numbers above are the honest frozen-param scores.

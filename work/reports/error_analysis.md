# Error analysis — val anchor 2026-01-14 (best blend `c_cand`)

Rows: 250,000. Overall RMSLE: blend **1.67172**, mlp2_cal 1.67265, ts2 1.69314. Zero-target share 0.459.

Global mean signed log-error (pred−true): blend +0.04011, mlp2 -0.00024, ts2 +0.27617 (positive = overprediction).

Model correlation: preds(log) r=0.9948, errors r=0.9950. In-sample optimal global blend weight w(mlp2)=0.831 (rest ts2).


## Segment table — rec_order (blend)

| segment | n | share | mean_err | std_err | seg_rmsle | mse_share | gain_if_zeroed |
|---|---|---|---|---|---|---|---|
| rec0-3 | 44113 | 0.1765 | +0.0074 | 1.5781 | 1.5781 | 0.1572 | +0.0000 |
| rec4-7 | 25189 | 0.1008 | +0.0605 | 1.7346 | 1.7356 | 0.1086 | +0.0001 |
| rec8-14 | 26304 | 0.1052 | +0.0620 | 1.8076 | 1.8087 | 0.1232 | +0.0001 |
| rec15-30 | 47099 | 0.1884 | +0.0653 | 1.8958 | 1.8970 | 0.2426 | +0.0002 |
| rec31-90 | 45181 | 0.1807 | +0.0396 | 1.8545 | 1.8550 | 0.2225 | +0.0001 |
| rec>90 | 28172 | 0.1127 | +0.0416 | 1.5341 | 1.5347 | 0.0950 | +0.0001 |
| rec_never | 33942 | 0.1358 | +0.0151 | 1.0236 | 1.0237 | 0.0509 | +0.0000 |

In-sample gain if every segment's mean error were zeroed: **0.00063** RMSLE.

## Segment table — ord_days_90 (blend)

| segment | n | share | mean_err | std_err | seg_rmsle | mse_share | gain_if_zeroed |
|---|---|---|---|---|---|---|---|
| ord90=0 | 62512 | 0.2500 | +0.0278 | 1.2834 | 1.2837 | 0.1474 | +0.0001 |
| ord90=1 | 32456 | 0.1298 | +0.0494 | 1.7616 | 1.7623 | 0.1443 | +0.0001 |
| ord90=2 | 24561 | 0.0982 | +0.0855 | 1.9012 | 1.9031 | 0.1273 | +0.0002 |
| ord90=3-5 | 48531 | 0.1941 | +0.0572 | 1.9727 | 1.9736 | 0.2706 | +0.0002 |
| ord90=6-10 | 41796 | 0.1672 | +0.0388 | 1.8402 | 1.8406 | 0.2027 | +0.0001 |
| ord90>10 | 40144 | 0.1606 | +0.0046 | 1.3692 | 1.3692 | 0.1077 | +0.0000 |

In-sample gain if every segment's mean error were zeroed: **0.00063** RMSLE.

## Segment table — gmv365_decile (blend)

| segment | n | share | mean_err | std_err | seg_rmsle | mse_share | gain_if_zeroed |
|---|---|---|---|---|---|---|---|
| gmv_zero | 34343 | 0.1374 | +0.0122 | 1.0284 | 1.0285 | 0.0520 | +0.0000 |
| gmv_p1 | 23962 | 0.0958 | +0.0149 | 1.4578 | 1.4579 | 0.0729 | +0.0000 |
| gmv_p2 | 23962 | 0.0958 | +0.0077 | 1.6536 | 1.6536 | 0.0938 | +0.0000 |
| gmv_p3 | 23962 | 0.0958 | +0.0058 | 1.7952 | 1.7952 | 0.1105 | +0.0000 |
| gmv_p4 | 23962 | 0.0958 | +0.0157 | 1.8646 | 1.8647 | 0.1192 | +0.0000 |
| gmv_p5 | 23961 | 0.0958 | +0.0444 | 1.8881 | 1.8886 | 0.1223 | +0.0001 |
| gmv_p6 | 23962 | 0.0958 | +0.0601 | 1.8870 | 1.8879 | 0.1222 | +0.0001 |
| gmv_p7 | 23962 | 0.0958 | +0.0536 | 1.8481 | 1.8488 | 0.1172 | +0.0001 |
| gmv_p8 | 23962 | 0.0958 | +0.0995 | 1.7566 | 1.7594 | 0.1062 | +0.0003 |
| gmv_p9 | 23962 | 0.0958 | +0.0994 | 1.5578 | 1.5610 | 0.0836 | +0.0003 |

In-sample gain if every segment's mean error were zeroed: **0.00083** RMSLE.

(positive-gmv decile edges: [49.1, 120.4, 225.1, 375.8, 590.7, 909.9, 1442.2, 2574.0])

## Segment table — pred_decile (blend)

| segment | n | share | mean_err | std_err | seg_rmsle | mse_share | gain_if_zeroed |
|---|---|---|---|---|---|---|---|
| pred_d1 | 25000 | 0.1000 | +0.0160 | 0.9457 | 0.9458 | 0.0320 | +0.0000 |
| pred_d2 | 25000 | 0.1000 | +0.0127 | 1.2807 | 1.2808 | 0.0587 | +0.0000 |
| pred_d3 | 25000 | 0.1000 | +0.0232 | 1.5966 | 1.5968 | 0.0912 | +0.0000 |
| pred_d4 | 25000 | 0.1000 | +0.0423 | 1.7882 | 1.7887 | 0.1145 | +0.0001 |
| pred_d5 | 25000 | 0.1000 | +0.0494 | 1.9330 | 1.9336 | 0.1338 | +0.0001 |
| pred_d6 | 25000 | 0.1000 | +0.0489 | 1.9996 | 2.0002 | 0.1432 | +0.0001 |
| pred_d7 | 25000 | 0.1000 | +0.0619 | 1.9824 | 1.9833 | 0.1408 | +0.0001 |
| pred_d8 | 25000 | 0.1000 | +0.0580 | 1.8788 | 1.8797 | 0.1264 | +0.0001 |
| pred_d9 | 25000 | 0.1000 | +0.0460 | 1.6669 | 1.6675 | 0.0995 | +0.0001 |
| pred_d10 | 25000 | 0.1000 | +0.0428 | 1.2937 | 1.2944 | 0.0599 | +0.0001 |

In-sample gain if every segment's mean error were zeroed: **0.00056** RMSLE.

(blend log-pred decile edges: [0.365, 0.719, 1.115, 1.545, 2.038, 2.551, 3.156, 3.777, 4.559])

## Segment table — zero_target (blend)

| segment | n | share | mean_err | std_err | seg_rmsle | mse_share | gain_if_zeroed |
|---|---|---|---|---|---|---|---|
| target=0 | 114835 | 0.4593 | +1.2755 | 1.0070 | 1.6251 | 0.4341 | +0.2409 |
| target>0 | 135165 | 0.5407 | -1.0095 | 1.3806 | 1.7103 | 0.5659 | +0.1738 |

In-sample gain if every segment's mean error were zeroed: **0.44847** RMSLE.

## Segment table — tenure (blend)

| segment | n | share | mean_err | std_err | seg_rmsle | mse_share | gain_if_zeroed |
|---|---|---|---|---|---|---|---|
| ten<=90 | 9434 | 0.0377 | +0.0160 | 1.4314 | 1.4315 | 0.0277 | +0.0000 |
| ten91-180 | 11529 | 0.0461 | -0.0344 | 1.5187 | 1.5191 | 0.0381 | +0.0000 |
| ten181-300 | 19078 | 0.0763 | +0.0054 | 1.5722 | 1.5722 | 0.0675 | +0.0000 |
| ten301-365 | 47282 | 0.1891 | +0.0416 | 1.6885 | 1.6891 | 0.1931 | +0.0001 |
| ten>365 | 162677 | 0.6507 | +0.0504 | 1.7002 | 1.7010 | 0.6737 | +0.0005 |

In-sample gain if every segment's mean error were zeroed: **0.00061** RMSLE.

## Segment table — search_vs_cat (blend)

| segment | n | share | mean_err | std_err | seg_rmsle | mse_share | gain_if_zeroed |
|---|---|---|---|---|---|---|---|
| search_dom | 207794 | 0.8312 | +0.0478 | 1.7499 | 1.7506 | 0.9115 | +0.0006 |
| cat_dom | 7863 | 0.0315 | -0.0399 | 1.8015 | 1.8019 | 0.0365 | +0.0000 |
| no_gmv365 | 34343 | 0.1374 | +0.0122 | 1.0284 | 1.0285 | 0.0520 | +0.0000 |

In-sample gain if every segment's mean error were zeroed: **0.00059** RMSLE.

## Honest (50/50 cross-fit) per-segment mean-error calibration of blend

| scheme | rmsle plain | gain plain | rmsle shrunk(k=300) | gain shrunk |
|---|---|---|---|---|
| rec_order | 1.67112 | +0.00060 | 1.67112 | +0.00060 |
| ord_days_90 | 1.67113 | +0.00059 | 1.67113 | +0.00059 |
| gmv365_decile | 1.67106 | +0.00066 | 1.67106 | +0.00066 |
| pred_decile | 1.67126 | +0.00046 | 1.67126 | +0.00046 |
| tenure | 1.67112 | +0.00060 | 1.67112 | +0.00060 |
| search_vs_cat | 1.67115 | +0.00057 | 1.67115 | +0.00057 |
| rec_x_ord90 | 1.67119 | +0.00053 | 1.67115 | +0.00057 |
| rec_x_preddec | 1.67114 | +0.00058 | 1.67117 | +0.00055 |
| preddec_x_ord90 | 1.67066 | +0.00106 | 1.67071 | +0.00101 |

## mlp2 vs ts2 disagreement (per-segment)


### by rec_order

| segment | n | mean_diff_mlp2_minus_ts2 | mlp2_mean_err | ts2_mean_err | mlp2_rmsle | ts2_rmsle | blend_rmsle |
|---|---|---|---|---|---|---|---|
| rec0-3 | 44113 | -0.3083 | -0.0449 | +0.2635 | 1.5800 | 1.5993 | 1.5781 |
| rec4-7 | 25189 | -0.3486 | +0.0066 | +0.3552 | 1.7357 | 1.7690 | 1.7356 |
| rec8-14 | 26304 | -0.3366 | +0.0112 | +0.3477 | 1.8088 | 1.8371 | 1.8087 |
| rec15-30 | 47099 | -0.3078 | +0.0187 | +0.3265 | 1.8972 | 1.9226 | 1.8970 |
| rec31-90 | 45181 | -0.2787 | -0.0004 | +0.2783 | 1.8565 | 1.8724 | 1.8550 |
| rec>90 | 28172 | -0.2317 | +0.0192 | +0.2509 | 1.5363 | 1.5503 | 1.5347 |
| rec_never | 33942 | -0.1251 | +0.0017 | +0.1268 | 1.0246 | 1.0306 | 1.0237 |

### by pred_decile

| segment | n | mean_diff_mlp2_minus_ts2 | mlp2_mean_err | ts2_mean_err | mlp2_rmsle | ts2_rmsle | blend_rmsle |
|---|---|---|---|---|---|---|---|
| pred_d1 | 25000 | -0.1172 | +0.0050 | +0.1222 | 0.9465 | 0.9519 | 0.9458 |
| pred_d2 | 25000 | -0.1834 | -0.0042 | +0.1792 | 1.2815 | 1.2931 | 1.2808 |
| pred_d3 | 25000 | -0.2327 | +0.0005 | +0.2332 | 1.5973 | 1.6129 | 1.5968 |
| pred_d4 | 25000 | -0.2941 | +0.0062 | +0.3003 | 1.7897 | 1.8124 | 1.7887 |
| pred_d5 | 25000 | -0.3332 | +0.0009 | +0.3341 | 1.9343 | 1.9599 | 1.9336 |
| pred_d6 | 25000 | -0.3487 | -0.0045 | +0.3442 | 2.0015 | 2.0247 | 2.0002 |
| pred_d7 | 25000 | -0.3452 | +0.0078 | +0.3530 | 1.9848 | 2.0087 | 1.9833 |
| pred_d8 | 25000 | -0.3711 | -0.0017 | +0.3694 | 1.8804 | 1.9107 | 1.8797 |
| pred_d9 | 25000 | -0.3175 | -0.0076 | +0.3099 | 1.6685 | 1.6918 | 1.6675 |
| pred_d10 | 25000 | -0.2208 | -0.0047 | +0.2162 | 1.2955 | 1.3092 | 1.2944 |

### by ord_days_90

| segment | n | mean_diff_mlp2_minus_ts2 | mlp2_mean_err | ts2_mean_err | mlp2_rmsle | ts2_rmsle | blend_rmsle |
|---|---|---|---|---|---|---|---|
| ord90=0 | 62512 | -0.1742 | +0.0102 | +0.1845 | 1.2849 | 1.2955 | 1.2837 |
| ord90=1 | 32456 | -0.2427 | +0.0200 | +0.2627 | 1.7628 | 1.7800 | 1.7623 |
| ord90=2 | 24561 | -0.3021 | +0.0428 | +0.3449 | 1.9031 | 1.9299 | 1.9031 |
| ord90=3-5 | 48531 | -0.3439 | +0.0050 | +0.3488 | 1.9740 | 2.0013 | 1.9736 |
| ord90=6-10 | 41796 | -0.3531 | -0.0172 | +0.3359 | 1.8416 | 1.8684 | 1.8406 |
| ord90>10 | 40144 | -0.2856 | -0.0479 | +0.2377 | 1.3719 | 1.3865 | 1.3692 |

### by zero_target

| segment | n | mean_diff_mlp2_minus_ts2 | mlp2_mean_err | ts2_mean_err | mlp2_rmsle | ts2_rmsle | blend_rmsle |
|---|---|---|---|---|---|---|---|
| target=0 | 114835 | -0.2331 | +1.2464 | +1.4794 | 1.5939 | 1.8213 | 1.6251 |
| target>0 | 135165 | -0.3132 | -1.0593 | -0.7461 | 1.7367 | 1.5761 | 1.7103 |

## Honest (50/50 cross-fit) per-segment blending of mlp2+ts2

| scheme | global-w rmsle | seg-w rmsle | gain(w) | seg-select rmsle | gain(select) |
|---|---|---|---|---|---|
| rec_order | 1.67177 | 1.67164 | +0.00013 | 1.67265 | -0.00088 |
| ord_days_90 | 1.67177 | 1.67163 | +0.00013 | 1.67265 | -0.00088 |
| gmv365_decile | 1.67177 | 1.67187 | -0.00010 | 1.67265 | -0.00088 |
| pred_decile | 1.67177 | 1.67186 | -0.00010 | 1.67265 | -0.00088 |
| tenure | 1.67177 | 1.67173 | +0.00004 | 1.67265 | -0.00088 |
| search_vs_cat | 1.67177 | 1.67172 | +0.00005 | 1.67265 | -0.00088 |
| rec_x_ord90 | 1.67177 | 1.67165 | +0.00012 | 1.67276 | -0.00099 |
| rec_x_preddec | 1.67177 | 1.67182 | -0.00005 | 1.67256 | -0.00079 |
| preddec_x_ord90 | 1.67177 | 1.67165 | +0.00011 | 1.67291 | -0.00115 |

## Global log-space calibration of blend (honest 50/50 cross-fit)

| transform | fitted params (per half) | rmsle | gain |
|---|---|---|---|
| p_log − c, clip≥0 | c = 0.0378 / 0.0424 | 1.67124 | +0.00048 |
| s·p_log | s = 0.9864 / 0.9856 | 1.67127 | +0.00045 |
| a·p_log + b, clip≥0 | a≈0.993, b≈−0.02/−0.03 | 1.67121 | +0.00051 |

So about half of the best segment-calibration gain (+0.00106) is a single global downscale; the rest is quantile/activity-conditional structure captured by preddec_x_ord90.

## Takeaways

1. **No large exploitable feature-based bias exists.** Biggest per-segment |mean signed log-error| among feature-based segments is ~0.10 (gmv_p8/p9 +0.099, ord90=2 +0.086, rec15-30 +0.065, pred_d7 +0.062) against within-segment std ~1.6–2.0, so even perfect per-segment mean removal moves RMSLE by ≤0.0003 per segment.
2. **The blend is mildly biased upward everywhere** (global mean log-error +0.040; every rec/ord90/pred-decile segment except none is positive). Source: ts2 carries a +0.276 global bias and enters the blend; mlp2 is nearly unbiased (−0.0002). Overprediction is concentrated in mid/high-activity users, not the inactive tail.
3. **Honest calibration ceiling ≈ +0.0011 RMSLE** (preddec_x_ord90 cross-fitted 1.67066 vs 1.67172). Global affine alone gives +0.0005. In-sample per-scheme estimates agree (0.0006–0.0008), so this is real but small. Corrections fitted on val may transfer imperfectly to the test anchor (30-day shift, February seasonality) — treat +0.0005–0.0010 as the realistic range.
4. **Where quality matters most (MSE share):** rec15-30 (24.3%) + rec31-90 (22.2%) dominate; ord90=3-5 (27.1%); pred deciles d5–d8 hold ~54% of squared error; target>0 users hold 56.6%; search-dominant users hold 91.2%. Model improvements should aim at moderately-active recent-ish buyers, not the never/inactive tail (rec_never only 5.1% of MSE at rmsle 1.02).
5. **The zero/positive tension is the whole game:** target=0 (45.9% of users) overpredicted by +1.276 in log space; target>0 underpredicted by −1.010. Oracle knowledge of zero-status would be worth 0.448 RMSLE — but it is unobservable; feature proxies of it (rec/ord90 buckets) are already exploited by the models.
6. **mlp2 vs ts2: disagreement is a level shift, not a specialization.** ts2 predicts 0.13–0.37 higher (in log) than mlp2 in every segment; error correlation r=0.995. mlp2 has lower RMSLE in *every* feature-based segment; ts2 wins only on the unobservable target>0 side (1.576 vs 1.737) — its upward bias acts as a hedge, which is exactly what the global blend weight (w_mlp2≈0.83) already prices. Consequently per-segment blend weights give at most +0.00013 honest gain and hard per-segment model selection is *negative* (−0.0009). **Per-segment blending is not a worthwhile direction.**
7. zero_target segmentation is diagnostic only (target unknown at test time); all actionable numbers above use feature-based segments and honest 50/50 cross-fitting.

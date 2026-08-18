# Adversarial validation: 8 clean train anchors vs TEST anchor

AUC = 1.0000

Top-15 drifting features (gain):
- tenure: 2468827
- rk_log_gmv_sum_90: 462031
- rk_ord_days_365: 231415
- rk_ord_days_90: 151895
- ord_days_365: 111080
- btyd_T: 63559
- ord_days_90: 43516
- btyd_p_alive: 19617
- rk_rec_order: 17222
- gmv_sum_90: 16894
- rk_log_gmv_sum_30: 14518
- btyd_freq: 8733
- btyd_exp_purch30: 6196
- s2o_days_365: 4870
- gmv_conc_90: 3841

## Round 2 (без механических time-фичей)
AUC = 1.0000
- active_days_ya_tgt: 1687923
- rk_ord_days_90: 480021
- rk_log_gmv_sum_90: 460585
- rk_log_gmv_sum_30: 235326
- active_days_b180_364: 158954
- ord_days_90: 153620
- rk_ord_days_365: 147362
- gmv_sum_30: 78186
- ord_days_365: 47757
- rk_rec_order: 44478
- btyd_freq: 37520
- gmv_sum_90: 29943

## Final (interrupted run, adv_val.py, 2026-08-18)

Setup: label 1 = TEST 2026-02-13 (250k rows), label 0 = 50k x 8 clean gap30 anchors
2025-10-22..2025-12-10 (400k rows); 199 features (USE_V2+V3+V4 minus user_id,
anchor_date, target, seasonal_index, history_days); LGB binary 5-fold stratified CV,
lr 0.1, 400 trees, nl 63, min_data_in_leaf 500, 3 threads. Log: work/reports/adv_val.log.

- Round 0 (199 feats): pooled OOF AUC **1.00000** (all folds 1.00000).
- Round 1 (dropped 5: tenure, rk_log_gmv_sum_90, rk_ord_days_365, rk_ord_days_90,
  ord_days_365): AUC **1.00000**.
- Round 2 (dropped 5 more: active_days_ya_tgt, rk_log_gmv_sum_30, rk_log_gmv_sum_365,
  rk_rec_order, btyd_T; 10 total): interrupted at 4/5 folds, fold AUCs
  0.99993 / 0.99996 / 0.99996 / 0.99996 (~**0.99995**).

Conclusion: drift is structural, not concentrated — separation survives dropping the
top gainers because many features encode history length: tenure/btyd_T caps,
365d-window truncation (ord_days_365 et al.), rank tie-plateau values (rk_* — the
zero-activity tie mass shifts per anchor), and ya-features flipping from all-null
(2025 anchors) to populated (TEST). Run stopped by coordinator before the
weight-computation and stable-subset stages; av_weights.parquet and
av_stable_features.txt were NOT produced.

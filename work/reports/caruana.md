# Bagged Caruana ensemble selection (caruana_v1)

Script: `work/scripts/caruana.py` (numpy/polars only, 2 threads, seed 42, deterministic — rerun reproduces bit-identical weights).

## Setup

- Objective: RMSLE in log1p space vs anchor 2026-01-14 target (`common.load_anchor`), 250k users.
- Library: every `work/preds/*_val.parquet` with a matching `*_test.parquet`, excluding prefixes
  `smoke/blend/base_best/cand/c_cand/A/caruana` (`caruana` added so the script's own output can't
  re-enter the library on rerun); `mlp2_final_cal` force-included. 23 candidates discovered.
- Guard (<1.60 solo val): dropped **twlog_probe** (1.5945, contaminated-protocol probe — 14 recent
  anchors, no gap). Library after guard: **22 models**.
- Algorithm: B=20 bags; each bag samples 50% of models (11, w/o replacement) + 80% of val users
  (200k rows, w/o replacement); greedy forward selection **with replacement**, 40 fixed steps,
  minimizing bag-RMSLE of the running mean of log1p preds (Gram-matrix algebra, float64 — exact).
  Final weights = average of per-bag pick frequencies.

## Solo scores (library)

| model | solo val | | model | solo val |
|---|---|---|---|---|
| xgblog_final | 1.6304 | | c_twlog_s42/s1337 | 1.6941 |
| lgblog_final | 1.6334 | | twd_b2/b5 | 1.6946 |
| mlp_final | 1.6636 | | c_dirlgb_s42/s1337 | 1.6949/1.6951 |
| mlp2_final_cal | 1.6726 | | c_xtw_s42 | 1.6979 |
| mlp2_big | 1.6821 | | gru_final | 1.6988 |
| mlp2_final | 1.6848 | | seq2tr_f | 1.7102 |
| twdeep | 1.6925 | | febspec | 1.8266 |
| c_ts2_s42..twd_b4 | 1.6931–1.6940 | | *(twlog_probe)* | *(1.5945, guard-dropped)* |

## Results

| method | full-val RMSLE |
|---|---|
| **bagged Caruana (B=20, saved as caruana_v1)** | **1.623983** |
| plain hill-climb, blend.py logic (2 picks: 0.5 lgb + 0.5 xgb) | 1.623281 |
| manual mix 0.85·mlp2_final_cal + 0.15·c_xtw_s42 (log-space) | 1.671720 (raw-space 1.671844) |

Bagged weights (nonzero): `xgblog_final 0.5475, lgblog_final 0.3875, mlp_final 0.03875,
mlp2_big 0.015, mlp2_final_cal 0.00625, gru_final 0.005`. Mean per-bag rmsle 1.6291.
Bags containing both lgblog_final and xgblog_final converge to ~18/22 splits; bags with only one
of them lean on mlp_final/mlp2_big/gru_final as diversifiers (~2–5 of 40 picks).

Caruana beat the manual mix on full val → saved `work/preds/caruana_v1_{val,test}.parquet`
(test = same weights over `*_test.parquet`, log1p space, expm1 back) + scores.tsv line.
Test preds sane: 250k rows, min ~0.003, no NaN; log1p-corr 0.9998 vs blend_w2_test.

## Honest caveat — where the "win" comes from

The 0.048 gain over the manual mix is carried almost entirely (94%) by **lgblog_final +
xgblog_final**, which are pre-CLEAN-era models (trained without the gap30 protocol; anchors
adjacent to the val window). CLEAN gap30 retrains of the same architectures score ~1.69, not
~1.63, so their val advantage is likely optimistic and may not transfer to the leaderboard —
this is exactly why the team's current best manual mix is built on mlp2_final_cal despite its
worse val. Note hill-climb lands on the same pair (identical to old blend_w2 = 1.6233).

**Clean-only sensitivity** (drop lgblog_final/xgblog_final/mlp_final/gru_final, 18 models,
seed 43): bagged Caruana val **1.675029** — does *not* beat the manual mix (1.6717). Weights
concentrate on mlp2_final_cal 0.41 + mlp2_big 0.39 + mlp2_final 0.04 + small tails. So within
the trusted protocol, Caruana selection currently offers no gain over the hand mix; the
mlp2_big+mlp2_final_cal pairing (~1.675) is its best idea.

**Recommendation:** treat caruana_v1's 1.6240 val as protocol-tainted; do not switch the
submission to it on val evidence alone. A leaderboard probe would settle whether the old-era
models' edge is real.

## JSON

```json
{"n_models": 22, "caruana_val": 1.623983, "hillclimb_val": 1.623281, "manual_val": 1.67172,
 "saved": true,
 "weights": {"xgblog_final": 0.5475, "lgblog_final": 0.3875, "mlp_final": 0.03875,
             "mlp2_big": 0.015, "mlp2_final_cal": 0.00625, "gru_final": 0.005}}
```

# Residual-correction probe on c_cand (2026-08-18)

**Hypothesis:** residuals of the best blend are still predictable from features -> a
residual-correction stack adds gain. **Verdict: REFUTED — residuals are noise; correction not applied.**

## Setup
- Val: anchor 2026-01-14 (USE_V2=1 USE_V3=1 -> 194 numeric features), preds `work/preds/c_cand_val.parquet`
  (250k users, val RMSLE **1.671720**).
- Residual `r = log1p(y) - log1p(pred)`; mean **-0.0401** (preds slightly high in log space), std 1.671.
- 2-fold user split (seed 42; one row per user). honest eval on held-out fold with
  `pred' = expm1(clip(log1p(pred) + r_hat, 0, None))`.

## Results

| metric | fold 0 | fold 1 | mean |
|---|---|---|---|
| held-out RMSLE orig | 1.670214 | 1.673224 | 1.671720 (all) |
| held-out RMSLE corrected | 1.685849 | 1.688834 | 1.687342 (all) |
| **honest gain** | -0.015636 | -0.015610 | **-0.015623** |
| shift-only gain (global mean-r shift) | +0.000532 | +0.000422 | **+0.000477** |
| residual-model R² out-of-fold | -0.0196 | -0.0193 | **-0.0195** |
| residual-model R² in-sample | 0.2759 | 0.2759 | 0.2759 |

- **Honest gain -0.0156** — the correction actively hurts. Threshold (+0.0015) not met -> **not applied**,
  no `c_cand_rescor` preds saved, nothing logged to scores.tsv.
- **R² decomposition (the caveat check):** in-sample the LGB "explains" 27.6% of residual variance, but
  out-of-fold R² is **negative** — on held-out users of the *same* val window the fitted structure explains
  less than the constant mean does. What the model finds in residuals is per-user noise memorization, not
  feature-driven signal. Since it doesn't even transfer across users within the window, transfer to the
  test window is a non-question.
- The only real, transferable component is a **global calibration shift**: applying mean(r) ≈ -0.040 as a
  constant log-shift gives +0.0005 honest gain. That is the capacity→0 limit of this stack, so the best
  achievable residual correction of any capacity is bounded near +0.0005 « 0.0015 threshold. It is also
  plain recalibration (c_cand already carries scale=0.94; see mlp2_final_cal), not a stacking gain.

## Feature drivers (gain importance of the residual model, top-10)
`burstiness` (2.0%), `rec_cat` (1.9%), `gmv_concentration` (1.6%), `search_trend_30` (1.5%),
`rec_over_gap` (1.5%), `gmv_conc_90` (1.5%), `gmv_daystd_90` (1.4%), `ord_gap_max` (1.4%),
`ord_gap_std` (1.4%), `gmv_wknd_365` (1.3%).

Interpretation: drivers are *structural* (recency, dispersion, concentration, trend shares), not level-like
gmv sums — normally the transfer-friendly kind. But importance is diffuse (no feature >2%, ~flat tail),
the signature of a model spreading capacity over noise rather than locking onto real signal. Combined with
negative out-of-fold R², these importances describe how the model overfits, not exploitable structure.

## Conclusion
c_cand's log-residuals carry no feature-predictable component beyond a ~-0.04 global bias worth ~+0.0005
RMSLE. The blend is effectively residual-clean w.r.t. the current feature set; further gains must come from
new information (features/models/windows), not from stacking on residuals. Not applied (`applied=false`).

Artifacts: this report; raw metrics JSON in session scratchpad (`residual_probe_result.json`). No preds written.

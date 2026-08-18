# External tricks — winning-solution scan (2026-08-18)

Scope: techniques we have NOT tried, plausibly worth >=0.002 RMSLE on zero-inflated 30d-GMV
(250k users, daily aggregated counts/sums only, RMSLE on log1p, 20/80 public/private).
Already in stack (excluded): multi-window RFM, LGB/XGB tweedie-on-log, hurdle 2-head MLP,
196d seq transformer, hill-climb log blending, binned calibration, Gram-matrix LB mixing.

Sources: GStore (Google Analytics Customer Revenue Prediction — same task shape: log of
future-window revenue per user, ~99% zeros), Elo 1st place, AMEX Default top solutions,
M5 Accuracy 1st, Ventilator 1st, Recruit Restaurant golds, Google ZILN paper (arXiv:1912.07753),
BTYD/CLV literature. Santander Value = leak-driven (skip), H&M/Instacart = item-level two-stage
retrieval (not transferable — we have no item/category data), Ozon E-CUP 2024/2025 public
writeups cover matching/moderation/recs/logistics — nothing on GMV regression.

## Ranked list

1. **Multi-snapshot training (sliding as-of cutoffs) + recency weighting** — [GStore winners, M5, H&M golds]
   Build training rows at 4–8 cutoff dates (stride 15–30d): features from history before cutoff,
   target = GMV in next 30d after cutoff; stack all snapshots (user_id, cutoff) as rows, weight
   recent cutoffs higher; validate ONLY on the latest cutoff (mimics LB gap). GStore top solutions
   trained exactly this way for a future 62d window. Fixes both data volume (x4–8 rows) and
   train/test seasonal mismatch. If we already stack >=4 cutoffs — do the recency-weight sweep
   (w = decay^months_back, tune decay on last-cutoff val). Effort: medium (pipeline re-run per
   cutoff; 250k x 8 = 2M rows fits 16GB with float32).

2. **LightGBM DART mode** — [AMEX 1st place + many AMEX golds]
   boosting=dart (tune drop_rate 0.05–0.2, skip_drop) on our exact tweedie-on-log champion recipe.
   On AMEX-shaped aggregated-behavior data DART reliably beat gbdt by ~0.001–0.002 and adds a
   decorrelated member even when solo-equal.

3. **Whale/outlier specialist blend (Elo 1st place recipe)** — [Elo Merchant 1st]
   (a) binary classifier for top ~0.5–1% log-GMV users ("whales"); (b) main regressor retrained
   with whales removed (cleaner for the 99%); (c) specialist regressor on whale-heavy sample;
   final = p_whale-gated mix of (b) and (c). Elo 1st reported large gains + "+0.015 CV from linear
   stacking" of these heads. Squared-log error is still dominated by big-log residuals — same
   mechanics as Elo's -33 outliers. Effort: medium (3 trainings, reuse features).

4. **Regression-as-classification over log1p bins** — [Ventilator 1st; recurring gold trick]
   Bin log1p(target) into ~20–40 bins, zero = its own class; LGB multiclass or softmax head on the
   MLP; decode prediction = sum p_k * bin_center (this IS the RMSLE-optimal E[log1p]). Natively
   zero-inflated, learns the full conditional distribution, and is a genuinely new model family for
   the blend (our biggest ensemble gap: all members are point regressors). Effort: medium.

5. **BTYD probabilistic features (BG/NBD + Gamma-Gamma, P(alive))** — [CLV literature; ZILN paper baselines; lifetimes package]
   Fit BG/NBD on (frequency, recency, T) from our daily counts; features: P(alive),
   E[#purchases next 30d], Gamma-Gamma expected order value, and the ratio current_gap/mean_gap
   ("overdueness"). Gives trees a calibrated long-horizon prior they cannot assemble from raw RFM
   windows; classic CLV models are strong exactly on sparse/zero-inflated users. Effort: light
   (lifetimes fits 250k users in minutes, CPU-only).

6. **Stacking meta-learner with segment features (adaptive blend weights)** — [Elo 1st linear stacking; standard gold pattern]
   Replace/augment global hill-climb weights: ridge or shallow LGB on OOF predictions of all
   members PLUS segment features (activity decile, recency bucket, p_zero, history length) —
   i.e., blend weights that depend on user segment. Requires all members' OOF on shared folds.
   Effort: light-medium; typical +0.001–0.003 over flat weighted average when members disagree
   by segment (ours do: MLP vs LGB on dormant users).

7. **Zero-floor + global log-shrink post-processing** — [M5 "magic multipliers"; GStore ~99%-zero structure]
   Two 1-parameter sweeps on OOF (confirm on LB): (a) preds_log *= c, c in [0.90, 1.00] — blends
   systematically overshoot in log space; (b) floor: preds_log < eps -> 0, tune eps — RMSLE
   rewards exact zeros on the huge never-buy mass. Complements (does not duplicate) our binned
   calibration; add isotonic-on-OOF in log space as the continuous upgrade of the bins.
   Effort: light (an evening).

8. **Purchase-gap / periodicity / SVD latent features** — [Web Traffic 1st (autocorr), Avito/TalkingData (SVD of count matrices), Recruit golds (EWM with tuned alpha)]
   From daily vectors: inter-purchase gap stats (mean/median/std/max, last_gap/mean_gap),
   weekly autocorrelation strength, day-of-week entropy, slope+curvature of weekly GMV fit,
   EWM aggregates with 3–4 tuned half-lives (not fixed windows); plus TruncatedSVD (k=16–32) of
   the user x week GMV/count matrix as dense factors for LGB. Orthogonal to window-RFM sums.
   Effort: light-medium, CPU-friendly.

9. **Adversarial validation -> drift pruning + importance weighting** — [AMEX/Elo/Santander golds, standard]
   Classifier train-cutoff rows vs test-cutoff rows; drop/neutralize top-drift features (calendar
   artifacts of the cutoff position), optionally weight training rows by p(test)/(1-p(test)).
   Cheap insurance that our 0.012 gap to top-1 is not drift-driven. Effort: light.

10. **Knowledge distillation + pseudo-labeling for the NN family** — [AMEX 15th (Deotte): Transformer + LGB distillation; widely used in AMEX golds]
    Retrain our MLP/transformer on soft targets = OOF blend predictions (and test rows with
    pseudo-labels from the full ensemble). NNs distilled from GBDT ensembles consistently close
    the family gap and lift the blend. Effort: medium (one heavy retrain — queue it).

11. **ZILN sigma-head upgrade of the hurdle MLP** — [Google ZILN, arXiv:1912.07753]
    Add a lognormal sigma output to the two-head hurdle (p, mu, sigma; NLL loss). Better fit on
    the heavy tail, and sigma becomes a stacking feature (uncertainty-aware blending in #6).
    Prediction for RMSLE stays p*mu in log space. Effort: light (loss swap on existing net).

12. **Promo/seasonal alignment features + YoY multiplier** — [M5/Favorita golds (calendar events); retail practice]
    Infer platform-wide promo days from spikes in the aggregate daily totals (sum over all users);
    features: user's spend share in promo weeks, response to last big event; and a global log-space
    seasonal multiplier from year-over-year monthly GMV ratio if the target month contains a major
    sale window. Effort: medium, moderate risk — validate on snapshot months containing events.

13. **Multi-loss/target ensembling (cheap diversity)** — [Recruit/AMEX golds; M5 level-ensembles]
    Same features, same LGB, different objectives: huber on log1p, poisson on 30d order count
    (times expected order value), quantile-0.5 on log1p, tweedie with 2–3 variance powers +
    3-seed bagging each. Blend members via #6. Effort: light (config sweeps), queue-friendly.

14. **Per-history-segment models + per-segment calibration** — [AMEX 2nd (separate model for few-statement customers)]
    Dedicated models for short-history/new users (<60d observed) and long-dormant users; their
    feature distributions are degenerate in the global model (window features all-zero/NaN-like).
    Then per-segment binned/isotonic calibration instead of one global map. Effort: medium.

15. **Rank-average blending with isotonic value re-mapping** — [AMEX rank ensembles]
    Rank-normalize each member's predictions, average ranks, map blended rank back to log-values
    by isotonic fit on OOF. Robust when members' scales disagree (our MLP vs tweedie-LGB tails).
    Keep as an extra candidate member for #6, not a replacement. Effort: light.

## Anti-findings (checked, do not chase)
- Santander Value Prediction: score driven by a data leak (2D time-series reconstruction); the
  non-leak aggregation ideas are already covered by our RFM stack.
- H&M / Instacart / OTTO: two-stage retrieval+rank needs item IDs — we have none. The only
  transferable bit is the multi-snapshot resampling already captured in #1 (and Instacart's
  explicit None-probability = our hurdle, done).
- Ozon E-CUP 2024 (matching, moderation) and 2025 (recs, logistics, counterfeit): no public
  GMV-regression writeups on habr/github as of today — nothing to mine for task 3 specifics.
- AMEX 1st place never published details; its known ingredients (DART, aggregation+diff features,
  rank ensembling) are captured in #2, #8, #15.

#7 -> #5 -> #9 -> #8 -> #11 -> #13; heavy queue: #2 (DART), then #1 (snapshots,
if not already multi-cutoff), then #3; #4 and #6 after OOF infrastructure is aligned; #10 last.

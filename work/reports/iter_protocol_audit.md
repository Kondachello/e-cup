# Audit of the retrain iteration-scaling heuristic (`iter_mult`)

Question: our final protocol trains on clean cutoffs with early stopping, then retrains on
clean+gap+val with `iter_mult = 1 + 0.7*(row_ratio-1)` (`train_gbdt.py:270`, `train_xtw.py:150`).
Is that multiplier right? Measured on a shifted scheme where the truth is observable.

## Design

Everything is shifted back one validation step so that a *held-out future* anchor exists:

| role | anchors |
|---|---|
| clean train (6, gap-30 to V1) | 2025-10-01 .. 2025-11-05 |
| gap anchors | 2025-11-12, 11-19, 11-26, 12-03 |
| V1 (early stop target) | 2025-12-10 |
| retrain set | clean + gap + V1 = 11 anchors, 2.75M rows |
| V2 (independent judge) | 2026-01-14 |

`row_ratio = 11/6 = 1.8333` -> heuristic `iter_mult = 1.5833`.
Config: lgb tweedie vp1.45 on `log1p(target)`, nl 63, mdl 300, lr 0.08, ff .75, bag .8, l2 5,
USE_V2 only (170 features), 2 threads, nice 15, cap 800 trees.

**Phase A** early stop on V1: `best_it = 96` (not censored), V1 RMSLE 1.751278.
**Phase B/D** retrain to 175 trees (M up to 1.82) with V2 attached as a *passive* valid set,
recording `rmse` on `log1p` per iteration. Since the tweedie model predicts on the log target and
`expm1 -> clip -> log1p` round-trips to identity for positive predictions, that recorded rmse **is**
the pipeline RMSLE at every tree count — one fit yields the entire RMSLE(M) curve.
**Phase C** verified this: an independent refit at M=1.0 (96 trees) scored 1.6953716334 vs the curve's
1.6953716332, |diff| = 2.0e-10. So the curve is exactly equivalent to running a separate retrain per M,
and the 6-point grid cost 2 fits instead of 6.

## RMSLE on V2 = 2026-01-14 vs iteration multiplier M

| M | trees | seed 42 | seed 1337 | mean |
|---|---|---|---|---|
| 0.80 | 76 | 1.696458 | 1.696442 | 1.696450 |
| 1.00 | 96 | 1.695372 | 1.695802 | 1.695587 |
| 1.15 | 110 | 1.695420 | 1.695737 | 1.695578 |
| 1.30 | 124 | 1.694691 | 1.695339 | 1.695015 |
| **1.50** | 144 | **1.694432** | 1.695316 | 1.694874 |
| 1.58 (heuristic) | 152 | 1.694654 | 1.695374 | 1.695014 |
| 1.80 | 172 | 1.694822 | 1.695181 | 1.695002 |

Per-seed argmin over all tree counts: seed 42 -> n\*=144 (M\*=1.50, 1.694432);
seed 1337 -> n\*=170 (M\*=1.77, 1.695041). Mean-curve argmin: n\*=160, **M\* = 1.67**, 1.694828.

Flatness of the mean curve: within +1e-4 of its minimum for M in **1.33 .. 1.76**;
within +2e-4 for M in 1.29 .. 1.79.

## Findings

1. **The heuristic sits inside the flat optimum.** At M=1.583 the mean curve is +0.000186 above the
   mean optimum. Seed-to-seed spread at a *fixed* M reaches 0.00088, so the heuristic's shortfall is
   ~5x smaller than run-to-run noise — not measurable in practice.
2. **The curve is asymmetric: undershooting hurts, overshooting barely does.** Relative to the mean
   optimum, M=1.0 costs +0.00076 and M=0.8 costs +0.00162, while M=1.8 costs only +0.00017. Whatever
   error we make should be on the high side, which the current heuristic already is (1.58 > 1.0).
3. **If biased at all, the heuristic is mildly *conservative*, not aggressive.** Both the mean optimum
   (1.67) and seed 1337's optimum (1.77) sit above 1.583. Caveat: the curve was only traced to M=1.82,
   and seed 1337's argmin is near that edge, so its true optimum may lie beyond the measured range.
   A coefficient of 0.8 instead of 0.7 would land exactly on M\*=1.67 here — expected gain ~0.0002.
4. **The whole retrain step is worth ~0.0008 here.** The early-stopped model, applied to V2 with no
   retrain at all, scores 1.695581; the best possible retrain scores 1.694828. The choice of M inside
   [1.3, 1.8] moves a fraction of that.

## Limitations (why this does not license changing the coefficient)

- **One row_ratio point only (1.833).** The heuristic's shape is `1 + k*(row_ratio-1)`; a single
  row_ratio measures the *product* k*0.833, not the slope k. Production champions run at
  row_ratio 1.357 (14 anchors + 4 gap, `iter_mult` 1.25) and 1.83 is not that regime. Re-fitting k
  from one point is extrapolation.
- **One (fast) config.** lr 0.08 / nl 63 gives `best_it = 96`; the champion family runs lr 0.04 /
  nl 255 with far larger best_it, where the loss basin around the optimum is typically wider still,
  but that was not measured.
- Two seeds. Enough to show the M effect is comparable to seed noise, not enough to resolve M\* to
  better than ~+/-0.15.

## Recommendation

**Leave `iter_mult = 1 + 0.7*(row_ratio-1)` unchanged in `train_gbdt.py` and `train_xtw.py`.**
The measured optimum bracket (M 1.33..1.76 within 1e-4) contains the heuristic's 1.583; the residual
penalty (+0.0002) is several times below seed noise and far below the ~0.001-0.002 differences the
model search is chasing. The failure mode the heuristic protects against — undershooting toward
M=1.0/0.8, worth 0.0008-0.0016 — is genuinely avoided, so the heuristic is doing its job.

Do **not** retune k to 0.8 on this evidence: it would be fit from a single row_ratio, at a config far
from the champions, for a gain inside noise, with regression risk on every queued final retrain.
If someone wants to close the last 0.0002, the prerequisite is the same curve at a second row_ratio
(e.g. 14 clean anchors, ratio 1.357) to identify the slope; that is ~2 instrumented fits.

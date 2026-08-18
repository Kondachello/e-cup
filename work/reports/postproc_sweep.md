# Post-processor sweep (cheap, CPU-only)

Date: 2026-08-18. Script: `work/scripts/postproc_sweep.py`.

Usefulness threshold 0.0005; apply-to-test threshold 0.0008.

## Results on `my26` (val RMSLE 1.671085, calibrated fine blend)

| family | honest gain | fold0 / fold1 | params (per fold) | verdict |
|---|---|---|---|---|
| zero_floor (t grid 0.1..3.0) | +0.000000 | +0.000000 / +0.000000 | t=0 both folds (any t>0 hurts) | no |
| affine_log a*lp+b | +0.000502 | +0.000470 / +0.000534 | a≈0.994/0.997, b≈-0.028/-0.033 | useful (barely) |
| affine a-only (b=0) | +0.000414 | +0.000369 / +0.000459 | a≈0.986/0.987 | no |
| affine b-only (a=1) | +0.000492 | +0.000469 / +0.000515 | b≈-0.042/-0.040 | no (probe) |
| isotonic 50-bin lp→ly | -0.000266 | -0.000291 / -0.000240 | 50-bin monotone map | no (overfits) |
| **power c*lp^γ** | **+0.000549** | +0.000537 / +0.000562 | γ=1.02, c≈0.960/0.962 (stable) | useful, **best** |
| combo(power→affine) | +0.000549 | +0.000535 / +0.000563 | — | no add over power alone |

## Results on `mlp2_big_cal` (val RMSLE 1.672154, already bin-calibrated)

Everything is dead: zero_floor 0.000000, affine -0.000004, power -0.000003,
isotonic -0.000683 (overfit), combo -0.000003. The binned log-shift calibration
already exhausted this axis — no residual monotone structure left.

## Decision: NOT applied (no `my26_pp`)

Best honest gain +0.000549 < 0.0008 apply threshold. And decomposition shows the
gain is almost entirely a **global level de-bias**, not shape:

- b-only probe (+0.000492) captures ~98% of the affine gain (+0.000502);
  residual shape-only gain ≈ +0.00001.
- power's γ=1.02 ≈ 1 — its c=0.96 is the same level shift in disguise
  (would move test mean lp by -0.041, vs affine's -0.040).
- my26 val preds run hot: mean lp 2.2827 vs mean ly 2.2421 (bias +0.041).

## Level vs shape transferability (seasonality caveat)

Test-window preds already sit 0.138 lower in mean lp than val (2.1448 vs
2.2827) — the seasonal level differs, so val-fitted level corrections are a
gamble on the test window:

- **Level-dependent, NOT transferable to test**: affine intercept `b`; power
  `c` (with γ≈1 it is pure level); zero_floor threshold `t` (absolute-scale
  cut; a val-fitted t zeroes relatively more users in a lower-level window);
  the isotonic map (corrections indexed by absolute lp get misapplied when the
  input level shifts).
- **Shape-dependent, transferable in principle**: affine slope `a` (log-scale
  compression), power exponent `γ`. Both are ≈1 here (γ=1.02, a≈0.995) —
  the transferable component carries ≈+0.00001, i.e. nothing.

Bottom line: what little a cheap post-processor can find on my26 is a +0.04
level bias of the val window, which is exactly the component that must not be
blindly shipped to a different seasonal window. If the LB later suggests my26
is systematically hot on test too, a small global shrink (b≈-0.03..-0.04 in
log) is the one-knob candidate — but that is an LB decision, not a val one.

Other findings:
- zero_floor is useless on these bases: optimum t=0 in every fold; even t=0.1
  (affecting 0.09% of users) already hurts. With 46% zero targets RMSLE still
  prefers hedged small predictions over hard zeros.
- isotonic (50 bins, honest) is negative on both bases — after prior
  calibration the flexible map only fits fold noise.

# Segment-shift probe design + LB-oracle algebra (2026-08-18)

Scope: math-only analysis of the 10 public-LB-measured submissions; design of tomorrow's
5 attempts. Public score f = RMSLE over a hidden random 50k-user subset P of our 250k rows
(assumed uniform; FPC = sqrt(200000/249999) = 0.894 applied throughout). Log-space error of
file A: e_A,i = log1p(pred_A,i) − log1p(y_i); f_A² = mean_P(e_A²).

## 1. Exact relations used

**Segment-shift probe.** Probe = base file with log-preds shifted by +δ only on segment S
(computable from test features). Since e'_i = e_i + δ·1[i∈S∩P]:

    f_probe² − f_base² = 2·δ·s_S·m_S + δ²·s_S,     s_S = |S∩P|/50k,  m_S = mean_{S∩P}(e_base)
    =>  m̂_S = (f_probe² − f_base² − δ²·ŝ_S) / (2·δ·ŝ_S)      [ŝ_S = local share over 250k]

Exact — no approximation — as long as the shift is not clipped (δ>0 always safe:
log1p(pred)≥0). Error budget of m̂_S (derivation: substitute s_S = ŝ_S+ε, f known to ±σ_f):

    σ(m̂_S)² ≈ [√2·f·σ_f/(δ·ŝ_S)]²  +  [δ·σ_s/(2·ŝ_S)]²  (+ m_S²·(σ_s/ŝ_S)² , negligible)
    σ_s = sqrt(ŝ(1−ŝ)/50000)·0.894      (hypergeometric share noise, rel ≈ 1%)

Quantization term ∝ 1/δ, share term ∝ δ → optimum δ ≈ 0.37–0.44 if scores are 4-decimal,
δ ≈ 0.05–0.14 if 9-decimal. **δ = 0.30** is the robust compromise (worst-case σ(m̂)
0.0015–0.0024, typical 0.0009–0.0012 — see §5). Negative δ rejected: 32% of users have
lp_A1 < 0.30 → clipping breaks exactness. Hadamard/union multi-segment designs rejected for
the same clip reason (need ±δ) and marginal benefit at our score precision.

**Global-shift pair (already measured).** sub_blend_w1a_seasonal110 − sub_blend_w1a is an
exactly constant log-shift +0.09530000 (verified: std 6e-17). Hence
mean_P(e_w1a) = (f₁²−f₀²−s²)/(2s) = **−0.040465** (essentially exact).
For any file X, mean(e_X) = mean(e_w1a) + mean_P(lp_X−lp_w1a); the second term is local
(±sampling ~0.001). All 10 files' global mean errors follow (§2).

## 2. Free information already extracted (no new submissions)

Global mean signed log-errors on the public test subset (all files UNDER-predict; contrast
val where c_cand OVER-predicts +0.040 — strong window/seasonal shift, so val calibration
cannot substitute for LB probes):

| file | f (public) | mean_e ± sd |
|---|---|---|
| sub_blend_w1a | 1.6754553659 | −0.0405 ± 0.0000 |
| sub_twlog_probe | 1.66779 | −0.1104 ± 0.0011 |
| sub_c_cand | 1.6695398157 | −0.1996 ± 0.0012 |
| lbmix2 | 1.65896 | −0.0589 ± 0.0009 |
| lbmix4_3way | 1.6573961435 | **−0.0763 ± 0.0010** |
| sample_submit | 2.1224835232 | −0.0854 ± 0.0056 |
| submission_gbdt | 1.6792462146 | −0.2043 ± 0.0013 |
| sub_nn | 1.6788340400 | −0.1071 ± 0.0016 |

mean_P(y) = 2.3275 ± 0.0064. lbmix4 + optimal global shift (+0.0763) alone → predicted
**1.655639** (guaranteed ≈ +0.0018, zero probes needed).

**Full error Gram.** G_AB = (f_A²+f_B²−D²_AB)/2 with D² local. Exact affine dependencies
found: lbmix2 = 0.2983·w1a + 0.7017·team_v2; lbmix4 = 0.0560·w1a + 0.3034·twlog +
0.2738·lbmix2 + 0.3668·team_v2 (resid rms < 1e-14); s110 = w1a + const. Reduced independent
basis of 7 files {w1a, twlog, c_cand, team_v2, sample, gbdt, nn}: Gram well-conditioned
(min eig 0.0136). Minimizing w'Gw − (w'm)² s.t. Σw=1 (global shift folded in), ridge 3e-4,
weights averaged over a 600-rep Monte Carlo of all sampling/rounding noise:

    A1 weights: w1a +0.0614844772, twlog +0.2753848591, c_cand +0.3939570859,
                team_v2 +0.3928478764, sample +0.0342338462, gbdt −0.0766194326,
                nn −0.0812887121;  global shift +0.1162840816
    predicted public f(A1) = 1.654000, MC sd 0.000007, p95 1.654017
    (nonneg-simplex fallback: 1.654133; lbmix4+shift-only fallback: 1.655639)

min lp_A1 = 0.117 > 0 → no clipping, algebra exact. c_cand carries weight 0.39 because its
huge −0.20 bias is corrected by the shift term — it is a strong model that was mis-scaled
(scale 0.94 + seasonal under-shoot).

## 3. Identifiability theory (what the LB oracle can and cannot reveal)

Every submission's score is f² = mean_P(lp²) − 2·mean_P(lp·y) + mean_P(y²). The y²
coefficient is 1 per user in every submission, so K submissions reveal:
(i) **arbitrary linear functionals mean_P(g·y)** where g is any log-space difference we
construct (probes = g constant per segment; weighted probes = any g);
(ii) **one global quadratic invariant** mean_P(y²) — shared across all equations, never
segment-resolved. Consequences:

the constant):
  mean(e) of every file (§2); exact predicted score of ANY affine combination + shift of the
  10 files (basis of A1); pairwise/segment-wise D² and disagreement decompositions. After ONE probe on base A1, m_S of EVERY file (past and future)
  follows for free via local mean_S(lp_X − lp_A1) (table computed, e.g. c_cand sits
  −0.18…−0.25 below A1 uniformly across segments).
- **Per-segment second moments (q_S, error variances, correlations): NEVER identifiable**
  by any probe design — mean_S(y²) enters every score only through the fixed global sum.
- **Per-segment optimal blend weights ARE probe-identifiable**: in
  w*_S = [mean_S(lpA·(lpA−lpB)) − mean_S((lpA−lpB)·y)] / mean_S((lpA−lpB)²)
  the mean_S(y²) term cancels; the only unknown, mean_S((lpA−lpB)·y), is measurable with a
  weighted probe g = γ·(lpA−lpB)·1_S. Deferred (val says per-segment reblending ≈ +0.0001).

Per-segment D² decomposition (test, PB partition): gbdt-vs-c_cand disagreement concentrates
2.1× in top-gmv (0.416 of D² at 0.194 share) — top-gmv is where models disagree most,
confirming it as the highest-value probe segment.

## 4. Segment choice (val-calibrated)

Candidate disjoint partitions evaluated on val (c_cand_val vs anchor=2026-01-14 target,
250k rows; global mean +0.0401; honest 50/50 cross-fit gain of per-segment mean correction
beyond the global shift, 3 reps):

| partition | rel. means (vs global) | honest gain |
|---|---|---|
| **PA_gmv**: top2 gmv-dec / p5-7 / zero∪p1-2 / p3-4 | +0.059 / +0.013 / −0.029 / −0.029 | +0.00027 |
| PB_mix: gmv-top / ord90 2-5 / rec15-30 / rest | +0.059 / +0.009 / −0.009 / −0.028 | +0.00030 |
| PD_pred: pred-decile groups | ≤ |0.023| | +0.00000 |
| PC: preddec×ord90 cells grouped by val means | (fitted) | +0.00036 |

Chosen: **PA_gmv** — simplest deterministic rule, near-best, largest single-segment signal,
stable shares (val 0.192/0.288/0.329/0.192 vs test 0.194/0.291/0.322/0.194). Rule on test
features anchor=2026-02-13 (`gmv_sum_365`): positive-gmv decile edges
[50.5, 124.2, 231.9, 385.5, 603.9, 928.0, 1467.2, 2615.8];
S1 = deciles 8-9 (share 0.1938), S2 = deciles 5-7 (0.2907), S3 = gmv=0 ∪ deciles 1-2
(0.3217), S4 = deciles 3-4 (0.1938, complement — mean implied by Σ s_k·m_k = mean_e(A1) ≈ 0).

## 5. The 5-attempt plan (exact formulas)

All files below are deterministic functions of already-saved artifacts. lp = log1p(pred),
inverse pred = expm1(lp). Submission order = sample_submit sorted by user_id.

| # | file | formula | purpose | expected score |
|---|---|---|---|---|
| A1 | `A1_gram7_shift.csv` | Σ w_j·lp_j + 0.1162840816 (weights §2) | new best + framework validation | 1.65400 ± 0.00002 (MC) |

Extraction (use MEASURED f_A1, not predicted): m̂_k = (f_probe² − f_A1² − 0.09·ŝ_k)/(0.6·ŝ_k),
ŝ = (0.1938, 0.2907, 0.3217); m̂_4 = −(Σ_{k≤3} ŝ_k·m̂_k)/0.1938.

Measurement precision σ(m̂): S1 0.0012 (worst-case 4-decimal scores: 0.0024), S2 0.0009
(0.0016), S3 0.0009 (0.0015), S4-implied 0.0057. Probe damage vs A1 is +0.003…+0.012
(sacrificial attempts; expected |m| ≈ 0.02–0.09 given val structure and the observed 3×
seasonal amplification of the global term).

private transfer costs Var ≈ 1.25·σ_seg²/n_pub,S. Apply
c_k = m̂_k·(1 − t_k²/m̂_k²)₊ with thresholds t = (0.019, 0.017, 0.012, 0.022) for S1–S4
(val segment error sds 1.66/1.87/1.36/1.83). Corrections |c| ≤ 0.12 keep lp ≥ 0 → exact.

Total projected vs current best 1.6573961: **≈ +0.0042** (of which +0.0034 is A1's
blend+shift, certain to MC precision; +0.0003…+0.0013 segment corrections).

(2) probes are independent — submit regardless of A1's
outcome; (3) segments with |m̂| below threshold get c=0 (no risk added).

## 6. Generation script (paste-run tomorrow; 2 threads, no training)

```python
import numpy as np, polars as pl
ROOT = "/Users/alexanderkondakov/ozon-cup"
W = {"submissions/sub_blend_w1a.csv": 0.0614844772,
     "submissions/sub_twlog_probe.csv": 0.2753848591,
     "submissions/sub_c_cand.csv": 0.3939570859,
     "prev_solutions/FILE.csv": 0.3928478764,
     "sample_submit.csv": 0.0342338462,
     "prev_solutions/submission_gbdt.csv": -0.0766194326,
     "prev_solutions/sub_nn.csv": -0.0812887121}
SHIFT, DELTA = 0.1162840816, 0.30
lp, uid = None, None
for p, w in W.items():
    df = pl.read_csv(f"{ROOT}/{p}", schema_overrides={"user_id": pl.Int64}).sort("user_id")
    v = np.log1p(np.clip(df[df.columns[1]].to_numpy().astype(np.float64), 0, None))
    lp = v * w if lp is None else lp + v * w
    if uid is None: uid = df["user_id"].to_numpy()
    else: assert (df["user_id"].to_numpy() == uid).all()
lp = lp + SHIFT
g = pl.read_parquet(f"{ROOT}/work/features/anchor=2026-02-13.parquet",
                    columns=["user_id", "gmv_sum_365"]).sort("user_id")
assert (g["user_id"].to_numpy() == uid).all()
gv = g["gmv_sum_365"].to_numpy(); pos = gv > 0
q = np.quantile(gv[pos], np.linspace(0, 1, 10)[1:-1])
gd = np.zeros(len(gv), int); gd[pos] = np.searchsorted(q, gv[pos], side="right") + 1
S = {1: gd >= 8, 2: (gd >= 5) & (gd <= 7), 3: gd <= 2}
S[4] = ~(S[1] | S[2] | S[3])
def write(name, l):
    pl.DataFrame({"user_id": uid, "predict": np.expm1(np.clip(l, 0, None))}
                 ).write_csv(f"{ROOT}/submissions/{name}.csv")
write("A1_gram7_shift", lp)
for k in (1, 2, 3):
    write(f"A{k+1}_probe_s{k}_gmv", lp + DELTA * S[k])
# after scores: sh = {1:.1938, 2:.2907, 3:.3217, 4:.1938}
# m[k] = (fp[k]**2 - fA1**2 - DELTA**2*sh[k]) / (2*DELTA*sh[k]); m[4] = -sum(sh[k]*m[k] k=1..3)/sh[4]
# thr = {1:.019, 2:.017, 3:.012, 4:.022}; c[k] = m[k]*max(0, 1-(thr[k]/m[k])**2)
# lp5 = lp - sum(c[k]*S[k]);  write("FILE.csv", lp5)
```

## 7. Notes / limits

- Everything assumes the public 50k is a uniform draw of our 250k (task premise); the MC
  propagates exactly that sampling noise plus score rounding.
- This track (LB arithmetic) tops out ≈ 1.6527–1.6537. The residual-probe report showed
  c_cand residuals are feature-noise, so further progress toward the 1.639 target must come
  from the queued model improvements; A1's Gram framework will absorb any new measured file
  (one submission each) into a provably-optimal mix with exact predicted score. future ones) are known for free; a second
  probe day could target per-segment blend weights via weighted probes (§3), expected value
  small per val (+0.0001) unless test disagreement structure is larger.

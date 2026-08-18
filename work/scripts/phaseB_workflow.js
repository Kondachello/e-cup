export const meta = {
  name: 'phaseB-sub164',
  description: 'Phase B: seq-model max, deep tweedie HPO, horizon decomposition, hurdle-MLP — push mix under 1.64',
  phases: [
    { title: 'Heavy', detail: '4 parallel programs' },
    { title: 'Assemble', detail: 'mega-blend candidates' },
  ],
}

const CTX = `
CONTEXT — Ozon E-CUP LTV, phase B. Goal: components strong/diverse enough that the LB-math mix
projects < 1.639 public RMSLE. Team best 1.65740. All prior context in work/reports/scores.tsv.
Rules of the road:
- CLEAN PROTOCOL ONLY: every selection run uses --gap-days 30 (train_gbdt.py) or, for NN scripts,
  train anchors ending >= 30d before the eval anchor. Dual-window (VAL=2026-01-14 primary,
  --val-anchor 2025-12-31 secondary) for config selection where runtime permits.
- Feature tiers: USE_V2=1 always; USE_V3=1 second tier (rank/seasonal) — A/B it if untested in your family.
- Trainer: work/scripts/train_gbdt.py; helpers common.py/exp_lib.py; python .venv/bin/python from /Users/alexanderkondakov/ozon-cup; --threads 3.
- Clean references (Jan/Dec, 8 anchors): twlog vp1.3 1.6953/1.7175; direct 1.6977/1.7199; ts2 1.6968/1.7231.
- Finals ALWAYS: 2 seeds, test preds saved (retrain auto-includes gap+val anchors).
- Output raw JSON per the schema.
`

const SCHEMA = {
  type: 'object',
  properties: {
    best_name: { type: 'string' },
    best_val_rmsle: { type: 'number' },
    best_val2_rmsle: { type: 'number' },
    configs_tried: { type: 'array', items: { type: 'object', properties: {
      name: { type: 'string' }, val_rmsle: { type: 'number' }, val2_rmsle: { type: 'number' }, notes: { type: 'string' } },
      required: ['name', 'val_rmsle'] } },
    test_preds_saved: { type: 'boolean' },
    notes: { type: 'string' },
  },
  required: ['best_name', 'best_val_rmsle', 'configs_tried', 'test_preds_saved'],
}

phase('Heavy')
const heavy = await parallel([
  () => agent(CTX + `
PROGRAM: sequence model MAX (the diversity flagship; torch + mps).
Data: work/seq2/anchor=DATE.npy float16 [250k, 196 days, 8 ch] (channels in build_seq2.py docstring),
.target.npy = [y30, y7, y14] float32. 12 train anchors + VAL + TEST. User order = sorted user_id.
Write work/scripts/train_seq2.py:
- Arch A (GRU): GRU(8->128, 2 layers, do 0.1) -> concat(last, mean, max) -> 256 GELU -> heads.
- Arch B (transformer): conv1d patch embed (stride 7 -> 28 tokens, d=96) + learned pos emb ->
  3-layer causal encoder (4 heads) -> concat(last token, mean) -> heads. Pick by clean VAL.
- Multi-task heads: y30 log1p MSE (weight 1.0) + y7, y14 log1p MSE (0.3 each) + P(y30>0) BCE (0.3).
- Selection: train on the 8 OLDEST of the 12 anchors (they end >=30d before VAL... verify dates,
  use anchors <= 2025-12-10), early stop on VAL clean RMSLE (main head). Batch 2048-4096, AdamW lr 1e-3,
  cosine, <=10 epochs, mmap-load anchors per epoch.
- Final: retrain on ALL 12 + VAL for the stopped step count x1.2, 2 seeds averaged -> exp_lib.save_preds
  as seqmax_final (val + test), log_score with notes.
Report per schema.`,
    { label: 'B:seq-max', phase: 'Heavy', schema: SCHEMA }),

  () => agent(CTX + `
PROGRAM: deep HPO + bagging of tweedie-on-log LGB (our strongest family).
1. Explore (gap-30 dual-window): lr 0.025 n_estimators 15000 at vp {1.3, 1.45}; nl {255, 511} x mdl {100, 300};
   feature_fraction {0.6, 0.75}; USE_V3 on/off (if clearly untested in scores.tsv); best + --weight-tau {0, 150}.
2. One dart try: --params '{"boosting":"dart","drop_rate":0.1,"n_estimators":4000,...}' (skip early stop caveats; fixed iters ok).
3. FINAL BAG: best config x 5 runs varying --seed {42,1337,7,2024,555} AND --n-anchors {10,12,14 rotating} — each with test preds
   (names twbag_1..twbag_5). Then average the 5 in log1p space yourself and save as twdeep_final (val+test) via exp_lib; log_score.
Report per schema (best_name=twdeep_final).`,
    { label: 'B:tw-deep', phase: 'Heavy', schema: SCHEMA }),

  () => agent(CTX + `
PROGRAM: horizon decomposition — 4 tweedie-log LGB models for sub-windows of the 30d target: days 1-7, 8-14, 15-21, 22-30.
- Build sub-targets per anchor yourself from /Users/alexanderkondakov/ozon-cup/train.parquet with polars
  (sum gmv in [A+lo, A+hi] per user, absent=0) for the anchors you train on + VAL; cache to work/features_sub/.
- Reuse anchor FEATURES from work/features (USE_V2=1, load via common.load_anchor).
- Train each sub-model with the twlog recipe (vp 1.3, nl255 mdl300 lr .05, ES on its own sub-target at VAL, gap-30).
- Combined prediction = log1p(sum of expm1'd... NO: sum RAW sub-preds then log1p). Score combined on VAL (y30) — compare vs twlog reference 1.6953.
- If within 0.005 of reference or better: FINAL with test preds (retrain each sub-model incl gap+val anchors, iters x1.2), save combined as hordec_final via exp_lib.
Report per schema.`,
    { label: 'B:horizon', phase: 'Heavy', schema: SCHEMA }),

  () => agent(CTX + `
PROGRAM: hurdle-MLP (two-head NN: P(buy) + E[log1p|buy]) on tabular features (USE_V2=1 USE_V3=1; torch mps).
Write work/scripts/train_mlp2.py (start from train_mlp.py): shared trunk 512-256, two heads (sigmoid-logit BCE; magnitude MSE on log1p computed ONLY on positive rows), prediction = p * clip(mu,0,None), then expm1(clip(.,0,None)).
- Selection: gap-30 anchors (<= 2025-12-10, take 8-10), ES on clean VAL RMSLE. Batch 8192, AdamW 1e-3.
- Try 2 variants (deeper trunk 1024-512-256; loss weight bce {0.5, 1.0}). FINAL: 2 seeds averaged, retrain incl gap+val for stopped epochs, save mlp2_final (val+test) via exp_lib; log_score.
Report per schema.`,
    { label: 'B:hurdle-mlp', phase: 'Heavy', schema: SCHEMA }),
])
log(`Heavy done: ${heavy.filter(Boolean).length}/4`)

phase('Assemble')
const asm = await agent(CTX + `
YOUR TASK — assemble phase-B candidates on VAL:
1. Score every *_final val pred yourself (rmsle vs VAL target). Inventory: c_* (round 3b), twdeep_final, seqmax_final, hordec_final, mlp2_final, older finals.
2. Hill-climb blend (log1p space, 300 iters, repeats allowed) over ALL; also try "clean-only" subset. Report both.
3. Best blend -> binned calibration check (calibrate.py) -> apply if holdout-positive.
4. Save top candidate as B_cand_{val,test}.parquet, runner-up as B_cand2 (exp_lib.save_preds + log_score, notes = composition).
Report per schema (best_name=B_cand).`,
  { label: 'B:assemble', phase: 'Assemble', schema: SCHEMA })

return { heavy: heavy.filter(Boolean), assembly: asm }

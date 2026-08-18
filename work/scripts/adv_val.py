"""Adversarial validation: train anchors vs TEST anchor (2026-02-13).

Label 1 = TEST anchor rows (all 250k), label 0 = 50k users sampled from each of
the 8 clean train anchors (target window ends before VAL; 2025-10-22..2025-12-10).
LGB binary 5-fold CV -> AUC, drifting features (gain), iterative top-5 drops,
importance weights w = p/(1-p) (isotonic-calibrated, winsorized q05/q95, mean 1)
for the 14 recent train anchors.

Light-CPU job: 3 threads, 400 trees, <=1M rows. Coexists with the queue job.
"""
from __future__ import annotations

import gc
import json
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import lightgbm as lgb
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).parent))
from common import FEATURES_DIR, REPORTS_DIR, TEST_ANCHOR, VAL_ANCHOR, load_anchor  # noqa: E402

T0 = time.time()


def log(msg: str):
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


# ---------------------------------------------------------------- anchors
def weekly_anchors_before_val() -> list[date]:
    out = []
    for p in sorted(FEATURES_DIR.glob("anchor=*.parquet")):
        stem = p.stem.split("=")[1]
        if "." in stem:
            continue
        a = date.fromisoformat(stem)
        if a < VAL_ANCHOR:
            out.append(a)
    return sorted(out)


ALL_TRAIN = weekly_anchors_before_val()
CLEAN = [a for a in ALL_TRAIN if a + timedelta(days=30) <= VAL_ANCHOR]
AV_TRAIN = CLEAN[-8:]                       # 2025-10-22 .. 2025-12-10
WEIGHT_ANCHORS = ALL_TRAIN[-14:]            # 2025-10-08 .. 2026-01-07
SAMPLE_PER_ANCHOR = 50_000
SEED = 42

log(f"AV train anchors (8): {[a.isoformat() for a in AV_TRAIN]}")
log(f"weight anchors (14): {[a.isoformat() for a in WEIGHT_ANCHORS]}")

# ---------------------------------------------------------------- features
DROP_ALWAYS = {"user_id", "anchor_date", "target", "seasonal_index", "history_days"}

test_df = load_anchor(TEST_ANCHOR)
n_test = test_df.height
all_cols = test_df.columns
# anchor-level constants: constant inside TEST anchor AND inside each train anchor
const_test = {c for c in all_cols if test_df[c].n_unique() <= 1}
feat_cols = [c for c in all_cols if c not in DROP_ALWAYS]
extra_const = sorted((const_test - DROP_ALWAYS) & set(feat_cols))
if extra_const:  # verify constant in train anchors too before dropping
    tr_probe = load_anchor(AV_TRAIN[-1])
    extra_const = [c for c in extra_const if tr_probe[c].n_unique() <= 1]
    del tr_probe
feat_cols = [c for c in feat_cols if c not in extra_const]
log(f"n_features={len(feat_cols)} dropped_non_feat={sorted(DROP_ALWAYS)} "
    f"extra_anchor_const={extra_const}")

N_TR = SAMPLE_PER_ANCHOR * len(AV_TRAIN)
N = n_test + N_TR
log(f"rows: test={n_test} train={N_TR} total={N}")
assert N <= 1_000_000


def to_f32(df: pl.DataFrame, cols: list[str]) -> np.ndarray:
    return df.select([pl.col(c).cast(pl.Float32) for c in cols]).to_numpy()


X = np.empty((N, len(feat_cols)), dtype=np.float32)
y = np.zeros(N, dtype=np.int8)
row_anchor: list[str] = []          # anchor iso per row
row_uid = np.empty(N, dtype=np.int64)

X[:n_test] = to_f32(test_df, feat_cols)
y[:n_test] = 1
row_uid[:n_test] = test_df["user_id"].to_numpy()
row_anchor.extend([TEST_ANCHOR.isoformat()] * n_test)
del test_df
gc.collect()

rng = np.random.default_rng(SEED)
pos = n_test
for a in AV_TRAIN:
    df = load_anchor(a)
    idx = np.sort(rng.choice(df.height, SAMPLE_PER_ANCHOR, replace=False))
    sub = df[idx]
    del df
    X[pos:pos + SAMPLE_PER_ANCHOR] = to_f32(sub, feat_cols)
    row_uid[pos:pos + SAMPLE_PER_ANCHOR] = sub["user_id"].to_numpy()
    row_anchor.extend([a.isoformat()] * SAMPLE_PER_ANCHOR)
    pos += SAMPLE_PER_ANCHOR
    del sub
    gc.collect()
    log(f"loaded {a}")
row_anchor = np.array(row_anchor)

# ---------------------------------------------------------------- CV fit
PARAMS = dict(
    objective="binary", metric="auc", learning_rate=0.1, num_leaves=63,
    min_data_in_leaf=500, num_threads=3, verbosity=-1, seed=SEED,
    force_row_wise=True,
)
N_TREES = 400
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
FOLDS = list(skf.split(np.zeros(N), y))


def run_cv(cols_keep: list[str], keep_models: bool = False):
    full = cols_keep == feat_cols
    if full:
        Xk = X
    else:
        ci = [feat_cols.index(c) for c in cols_keep]
        Xk = np.ascontiguousarray(X[:, ci])
    oof = np.zeros(N, dtype=np.float64)
    gain = np.zeros(len(cols_keep), dtype=np.float64)
    models = []
    fold_aucs = []
    ds_full = lgb.Dataset(Xk, label=y, feature_name=cols_keep, free_raw_data=False)
    ds_full.construct()  # bin once, folds share binned data via subset()
    for k, (tr, va) in enumerate(FOLDS):
        bst = lgb.train(PARAMS, ds_full.subset(tr), num_boost_round=N_TREES)
        oof[va] = bst.predict(Xk[va], num_threads=3)
        fold_aucs.append(roc_auc_score(y[va], oof[va]))
        gain += bst.feature_importance("gain")
        if keep_models:
            models.append(bst)
        gc.collect()
        log(f"  fold {k}: auc={fold_aucs[-1]:.5f}")
    del ds_full
    if not full:
        del Xk
    gc.collect()
    auc = roc_auc_score(y, oof)
    return auc, fold_aucs, gain, oof, models


log("round 0: full feature set")
auc0, fold0, gain0, oof0, models0 = run_cv(feat_cols, keep_models=True)
log(f"round 0 pooled OOF AUC = {auc0:.5f} (folds {np.mean(fold0):.5f}+-{np.std(fold0):.5f})")

order0 = np.argsort(gain0)[::-1]
gain_share0 = gain0 / gain0.sum()
top20 = [(feat_cols[i], float(gain_share0[i])) for i in order0[:20]]

# ---------------------------------------------------------------- iterative drop
AUC_STOP, MAX_DROP = 0.62, 15
dropped: list[str] = []
kept = list(feat_cols)
rounds = [("round0_full", auc0, [])]
cur_auc, cur_gain, cur_cols = auc0, gain0, list(feat_cols)
if auc0 > 0.6:
    r = 0
    while cur_auc > AUC_STOP and len(dropped) < MAX_DROP:
        r += 1
        oi = np.argsort(cur_gain)[::-1]
        drop_now = [cur_cols[i] for i in oi[:5]]
        dropped.extend(drop_now)
        kept = [c for c in cur_cols if c not in drop_now]
        log(f"round {r}: dropping {drop_now}")
        cur_auc, fa, cur_gain, _, _ = run_cv(kept)
        cur_cols = kept
        rounds.append((f"round{r}_drop{len(dropped)}", cur_auc, drop_now))
        log(f"round {r} AUC = {cur_auc:.5f} (dropped total {len(dropped)})")
auc_final = cur_auc
stable = kept

# ---------------------------------------------------------------- weights
log("weights: isotonic calibration on round-0 OOF")
iso = IsotonicRegression(y_min=1e-6, y_max=1 - 1e-6, out_of_bounds="clip")
iso.fit(oof0, y.astype(np.float64))
p_cal = np.clip(iso.predict(oof0), 1e-6, 1 - 1e-6)
w_raw = p_cal / (1 - p_cal)

tr_mask = y == 0
q05, q95 = np.quantile(w_raw[tr_mask], [0.05, 0.95])
w_tr = np.clip(w_raw[tr_mask], q05, q95)
w_tr = w_tr / w_tr.mean()
n_eff = float(w_tr.sum() ** 2 / (w_tr ** 2).sum())
n_eff_ratio = n_eff / tr_mask.sum()
log(f"train-sample weights: q05={q05:.4g} q95={q95:.4g} n_eff={n_eff:.0f}/{tr_mask.sum()} "
    f"ratio={n_eff_ratio:.3f}")

# OOF override map for sampled train rows: (anchor, uid) -> oof pred
oof_key = {}
for i in np.where(tr_mask)[0]:
    oof_key[(row_anchor[i], row_uid[i])] = oof0[i]

del X
gc.collect()

log("scoring 14 weight anchors with fold-mean models (OOF where sampled)")
parts = []
for a in WEIGHT_ANCHORS:
    df = load_anchor(a)
    Xa = to_f32(df, feat_cols)
    uids = df["user_id"].to_numpy().astype(np.int64)
    del df
    p = np.zeros(Xa.shape[0], dtype=np.float64)
    for bst in models0:
        p += bst.predict(Xa, num_threads=3)
    p /= len(models0)
    del Xa
    gc.collect()
    a_iso = a.isoformat()
    if a_iso in set(row_anchor[tr_mask]):
        ov = np.array([oof_key.get((a_iso, u), np.nan) for u in uids])
        m = ~np.isnan(ov)
        p[m] = ov[m]
        log(f"  {a}: oof-override {int(m.sum())} rows")
    pc = np.clip(iso.predict(p), 1e-6, 1 - 1e-6)
    w = np.clip(pc / (1 - pc), q05, q95)
    parts.append(pl.DataFrame({
        "user_id": uids,
        "anchor_date": np.repeat(a_iso, len(uids)),
        "weight": w,
    }))
    log(f"  {a}: mean_raw_w={w.mean():.4f}")
wdf = pl.concat(parts)
wdf = wdf.with_columns((pl.col("weight") / wdf["weight"].mean()).alias("weight"))
wdf = wdf.with_columns(pl.col("anchor_date").str.to_date())
out_w = FEATURES_DIR / "av_weights.parquet"
wdf.write_parquet(out_w)
per_anchor = (
    wdf.group_by("anchor_date").agg(pl.col("weight").mean().alias("mean_w"),
                                    pl.col("weight").median().alias("med_w"))
    .sort("anchor_date")
)
log(f"saved {out_w} rows={wdf.height}")

# ---------------------------------------------------------------- outputs
stable_path = REPORTS_DIR / "av_stable_features.txt"
stable_path.write_text("\n".join(stable) + "\n")

rep = []
rep.append("# Adversarial validation: train anchors vs TEST (2026-02-13)\n")
rep.append(f"- Setup: label 1 = TEST anchor 2026-02-13 ({n_test} rows); label 0 = "
           f"{SAMPLE_PER_ANCHOR // 1000}k users x 8 clean anchors "
           f"{AV_TRAIN[0]}..{AV_TRAIN[-1]} ({N_TR} rows). Features USE_V2+V3+V4, "
           f"n={len(feat_cols)} after dropping {sorted(DROP_ALWAYS)} + {extra_const}.")
rep.append(f"- Model: LGB binary, 5-fold stratified CV, lr 0.1, 400 trees, nl 63, "
           f"min_data_in_leaf 500, 3 threads.")
rep.append(f"- Note: the prompt's '8 clean anchors 2025-10-15..2025-12-10' spans 9 weekly "
           f"anchors; used the project's canonical 8 most recent clean (gap30) anchors "
           f"{AV_TRAIN[0]}..{AV_TRAIN[-1]}.\n")
rep.append(f"## AUC\n\n- Round 0 (all features): pooled OOF **{auc0:.5f}** "
           f"(fold mean {np.mean(fold0):.5f} +- {np.std(fold0):.5f})")
for name, a, dr in rounds[1:]:
    rep.append(f"- {name}: **{a:.5f}** after dropping {dr}")
rep.append(f"\nStopping rule: AUC <= {AUC_STOP} or {MAX_DROP} features dropped. "
           f"Final AUC {auc_final:.5f}, dropped {len(dropped)}.\n")
rep.append("## Top-20 drifting features (round-0 gain share)\n")
rep.append("| rank | feature | gain share |")
rep.append("|---:|---|---:|")
for r_, (f_, g_) in enumerate(top20, 1):
    rep.append(f"| {r_} | {f_} | {g_:.3%} |")
rep.append(f"\n## Dropped drifters ({len(dropped)})\n")
for f_ in dropped:
    rep.append(f"- {f_}")
rep.append(f"\nStable subset: {len(stable)} features -> `work/reports/av_stable_features.txt`\n")
rep.append("## Importance weights (w = p/(1-p), isotonic-calibrated, "
           "winsorized [q05, q95], mean 1)\n")
rep.append(f"- Winsor bounds (raw odds, from the 8-anchor train sample): "
           f"[{q05:.4g}, {q95:.4g}]")
rep.append(f"- Train sample (400k rows): n_eff = {n_eff:,.0f} of {int(tr_mask.sum()):,} "
           f"(ratio {n_eff_ratio:.3f})")
rep.append(f"- Weights file: `work/features/av_weights.parquet` "
           f"({wdf.height:,} rows, 14 anchors {WEIGHT_ANCHORS[0]}..{WEIGHT_ANCHORS[-1]}, "
           f"normalized mean 1 over the file). Rows sampled into AV training are scored "
           f"with their OOF prediction, the rest with the 5-fold-mean model.")
rep.append("\nPer-anchor mean/median weight (recency profile the classifier implies):\n")
rep.append("| anchor | mean w | median w |")
rep.append("|---|---:|---:|")
for row in per_anchor.iter_rows():
    rep.append(f"| {row[0]} | {row[1]:.3f} | {row[2]:.3f} |")
rep.append("\n## Caveats\n")
rep.append("- gmv_ya_*/ordd_ya_* (year-ago) are all-constant in 2025 anchors (no 2024 "
           "data) but populated at TEST -> any importance they get is mechanical drift.")
rep.append("- High AUC here means 'anchors are distinguishable', not 'models fail': "
           "recency/tenure features shift by construction as the anchor moves forward.")
rep.append("- Weights are density-ratio estimates from the full-feature round-0 model; "
           "winsorization caps the tails, so treat them as soft recency/importance hints.")
(REPORTS_DIR / "adversarial_validation.md").write_text("\n".join(rep) + "\n")
log("saved work/reports/adversarial_validation.md")

result = {
    "auc": round(float(auc0), 5),
    "auc_after_drop": round(float(auc_final), 5),
    "n_dropped": len(dropped),
    "n_eff_ratio": round(float(n_eff_ratio), 4),
    "top_drifters": [f for f, _ in top20],
    "notes": (f"8 clean anchors {AV_TRAIN[0]}..{AV_TRAIN[-1]} vs TEST; "
              f"{len(feat_cols)} feats; stable={len(stable)}; "
              f"weights for 14 anchors {WEIGHT_ANCHORS[0]}..{WEIGHT_ANCHORS[-1]} "
              f"in av_weights.parquet"),
}
print("FINAL_JSON " + json.dumps(result), flush=True)

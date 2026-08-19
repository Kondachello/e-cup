"""Count x AOV decomposition trainer (TEAM_PLAN decomposition #1): gmv_30d = orders x check.

Head 1, COUNT (both modes): champion-base LGBM (nl255 mdl300 lr0.05 ff0.75,
n_estimators 6000), objective tweedie vp1.3 ON log1p(sum to_ord over (A, A+30]) —
the counter is zero-inflated too. Early stop on VAL counts.

Head 2, the check — two modes (--aov-mode):
  product (the literal spec): MSE on log(gmv_30/ord_30) trained ONLY on rows with
    orders in the window; final = expm1(pred_cnt) * exp(pred_aov) * damp.
    MEASURED STRUCTURAL FLAW (smoke 1.9737 vs champion 1.6927): the full check
    exp(pred_aov)~18 is applied UNDISCOUNTED to the zero mass -> non-buyer bias
    +2.40 in log1p (buyers are fine: rmse 1.234 vs channel2 1.614). A global damp
    cannot fix per-user gating (best grid damp 0.85 -> only 1.906).
  uplift (default): the check enters log1p space as the zero-inflated uplift
    u = log1p(gmv_30) - log1p(cnt_30)  (exact identity; 0 for non-buyers,
    ~log(check) for buyers), tweedie vp1.3 on ALL rows, early stop on VAL u;
    final = expm1(pred_cnt + pred_u) — additive in log space, the zero-mass
    discounting of the check is learned by the head itself.

Market cross-check (KNOWLEDGE.md, Ozon Q1'26): the mean check falls -26% YoY, so
the check head tends to overshoot forward. Optional damp: --aov-damp (default 1.0),
a multiplier on the check (product mode: on exp(pred_aov); uplift mode: +log(damp)
on positive uplift, floored at 0). Independently of the flag the best damp is
always fitted on the fixed fit-half of VAL users (grid 0.85..1.05 step 0.01, rng
protocol of calibrate.py) with the honest other-half score reported; if it differs
from --aov-damp, preds with the fitted damp are saved as NAME_dampfit_{val,test}.

Follows the exp_lib contract: gap-30 protocol, val preds -> NAME_val.parquet,
retrain (train + gap anchors + val, iters scaled by row ratio per head) ->
NAME_test.parquet, one line in scores.tsv. Logs val RMSLE of the total AND both
heads (count RMSLE on the count scale; check head RMSE in log space on VAL buyers).

Needs anchor=DATE.cnttgt.parquet (build_count_targets.py) for train/gap/VAL anchors.

Full champion run:
  USE_V2=1 USE_V3=1 USE_V4=1 OMP_NUM_THREADS=6 \
    train_countaov.py --name countaov --threads 6
Smoke:
  USE_V2=1 USE_V3=1 USE_V4=1 train_countaov.py --name countaov_smoke \
    --n-anchors 2 --threads 2 --params '{"n_estimators":300}' --no-test
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))
from common import FEATURES_DIR, TEST_ANCHOR, VAL_ANCHOR, feature_cols, load_anchor, rmsle
from exp_lib import log_score, save_preds
from model_io import booster_filename, save_lgb, save_meta
from train_gbdt import fit_lgb

DIRECT_CHAMPION = 1.6927  # twl_repair_ab (lgb tweedie1.45 on log1p, gap30, 14 anchors)

BASE_PARAMS = dict(
    num_leaves=255, min_data_in_leaf=300, learning_rate=0.05,
    feature_fraction=0.75, n_estimators=6000,
)
CNT_PARAMS = dict(BASE_PARAMS, objective="tweedie", tweedie_variance_power=1.3)
AOV_PARAMS = {  # per --aov-mode
    "product": dict(BASE_PARAMS),  # objective stays "regression" (MSE on log check)
    "uplift": dict(BASE_PARAMS, objective="tweedie", tweedie_variance_power=1.3),
}

# log(check) sanity clip in product mode: observed AOV range is ~[0.03, 13k]
AOV_LOG_CLIP = (-4.5, 10.0)
DAMP_GRID = np.round(np.arange(0.85, 1.0501, 0.01), 2)


def cnttgt_path(a: date) -> Path:
    return FEATURES_DIR / f"anchor={a.isoformat()}.cnttgt.parquet"


def cnttgt_anchors() -> list[date]:
    out = []
    for p in sorted(FEATURES_DIR.glob("anchor=*.cnttgt.parquet")):
        a = date.fromisoformat(p.stem.split("=")[1].split(".")[0])
        if a < VAL_ANCHOR:
            out.append(a)
    return out


def load_with_counts(anchors: list[date], cols: list[str]) -> pl.DataFrame:
    dfs = []
    for a in anchors:
        df = load_anchor(a, columns=["user_id", "anchor_date", "target"] + cols)
        ct = pl.read_parquet(cnttgt_path(a))
        j = df.join(ct, on="user_id", how="left")
        assert j["tgt_cnt"].null_count() == 0, f"cnttgt misses users at {a}"
        dfs.append(j)
    return pl.concat(dfs, how="vertical_relaxed")


def to_arrays(df: pl.DataFrame, cols: list[str]):
    X = df.select(cols).to_numpy().astype(np.float32)
    cnt = df["tgt_cnt"].to_numpy().astype(np.float64)
    aov = df["tgt_aov"].to_numpy().astype(np.float64)  # nan where cnt == 0
    y_tot = df["target"].to_numpy().astype(np.float64)
    return X, cnt, aov, y_tot


def uplift_target(y_tot: np.ndarray, cnt: np.ndarray) -> np.ndarray:
    """u = log1p(gmv) - log1p(cnt), clipped at 0 (negative only when check < 1)."""
    return np.clip(np.log1p(y_tot) - np.log1p(cnt), 0, None)


def combine_product(pc: np.ndarray, pa: np.ndarray, damp: float) -> np.ndarray:
    c = np.expm1(np.clip(pc, 0, None))
    a = np.exp(np.clip(pa, *AOV_LOG_CLIP))
    return np.clip(c * a * damp, 0, None)


def combine_uplift(pc: np.ndarray, pu: np.ndarray, damp: float) -> np.ndarray:
    u = np.clip(pu, 0, None)
    if damp != 1.0:
        u = np.where(u > 0, np.clip(u + np.log(damp), 0, None), 0.0)
    return np.expm1(np.clip(pc, 0, None) + u)


COMBINE = {"product": combine_product, "uplift": combine_uplift}


def cal_split(n: int) -> np.ndarray:
    """Fixed fit-half mask over VAL users (same rng protocol as calibrate.py)."""
    return np.random.default_rng(0).permutation(n) < n // 2


def fit_damp(mode: str, pc_val: np.ndarray, p2_val: np.ndarray,
             yv_tot: np.ndarray) -> dict:
    """Grid-search the check damp on the fit-half of VAL; honest score = other half."""
    comb = COMBINE[mode]
    half = cal_split(len(yv_tot))
    scores = {float(d): rmsle(yv_tot[half], comb(pc_val, p2_val, d)[half])
              for d in DAMP_GRID}
    best = min(scores, key=scores.get)
    return dict(
        best=best, half=half, fit_scores=scores,
        holdout=rmsle(yv_tot[~half], comb(pc_val, p2_val, best)[~half]),
        holdout_nodamp=rmsle(yv_tot[~half], comb(pc_val, p2_val, 1.0)[~half]),
        full=rmsle(yv_tot, comb(pc_val, p2_val, best)),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="countaov")
    ap.add_argument("--aov-mode", default="uplift", choices=["product", "uplift"])
    ap.add_argument("--params", type=str, default="{}",
                    help="JSON overrides on top of the champion base for BOTH heads")
    ap.add_argument("--params-cnt", type=str, default="{}",
                    help="JSON overrides for the count head only")
    ap.add_argument("--params-aov", type=str, default="{}",
                    help="JSON overrides for the check head only")
    ap.add_argument("--aov-damp", type=float, default=1.0,
                    help="check multiplier for the primary preds (default 1.0)")
    ap.add_argument("--n-anchors", type=int, default=14)
    ap.add_argument("--gap-days", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--no-test", action="store_true")
    ap.add_argument("--notes", type=str, default="")
    args = ap.parse_args()
    if args.threads:
        os.environ["OMP_NUM_THREADS"] = str(args.threads)
    mode = args.aov_mode
    over = json.loads(args.params)
    params_cnt = dict(CNT_PARAMS); params_cnt.update(over)
    params_cnt.update(json.loads(args.params_cnt))
    params_aov = dict(AOV_PARAMS[mode]); params_aov.update(over)
    params_aov.update(json.loads(args.params_aov))
    print(f"aov-mode={mode}", flush=True)
    print(f"params[cnt]: tweedie vp={params_cnt.get('tweedie_variance_power')} "
          f"nl={params_cnt['num_leaves']} mdl={params_cnt['min_data_in_leaf']} "
          f"lr={params_cnt['learning_rate']}", flush=True)
    print(f"params[aov]: obj={params_aov.get('objective', 'mse')} "
          f"nl={params_aov['num_leaves']} mdl={params_aov['min_data_in_leaf']} "
          f"lr={params_aov['learning_rate']}", flush=True)

    t0 = time.time()
    avail = cnttgt_anchors()
    assert avail, "no .cnttgt.parquet files; run build_count_targets.py first"
    assert cnttgt_path(VAL_ANCHOR).exists(), "VAL anchor has no count targets"
    cutoff = VAL_ANCHOR - timedelta(days=args.gap_days)
    tr_anchors = [a for a in avail if a <= cutoff][-args.n_anchors:]
    gap_anchors = [a for a in avail if cutoff < a < VAL_ANCHOR]
    print(f"train anchors ({len(tr_anchors)}): {[a.isoformat() for a in tr_anchors]}",
          flush=True)
    print(f"gap anchors for retrain ({len(gap_anchors)}): "
          f"{[a.isoformat() for a in gap_anchors]}", flush=True)

    cols = feature_cols(load_anchor(VAL_ANCHOR))  # BEFORE the cnttgt join
    cols = [c for c in cols if c not in ("tgt_cnt", "tgt_aov")]
    print(f"{len(cols)} features", flush=True)

    val = load_with_counts([VAL_ANCHOR], cols)
    Xv, cnt_v, aov_v, yv_tot = to_arrays(val, cols)
    uid_val = val["user_id"].to_numpy()
    mask_v = cnt_v > 0
    ident = float(np.abs(np.nan_to_num(cnt_v * aov_v) - yv_tot).max())
    print(f"val identity max|cnt*aov-target|={ident:.5f} "
          f"buyers={int(mask_v.sum())}/{len(cnt_v)}", flush=True)
    assert ident < 1.0

    tr = load_with_counts(tr_anchors, cols)
    X, cnt, aov, y_tot_tr = to_arrays(tr, cols)
    del tr, val
    mask = cnt > 0
    print(f"X {X.shape} (buyers {int(mask.sum())}), Xv {Xv.shape}, "
          f"load {time.time()-t0:.0f}s", flush=True)

    # --- COUNT head: tweedie vp1.3 on log1p(count), early stop on VAL counts ---
    print(f"--- count head: nz_rate={float(mask.mean()):.4f} "
          f"val_nz={float(mask_v.mean()):.4f}", flush=True)
    m_cnt, it_cnt = fit_lgb(X, np.log1p(cnt), None, Xv, np.log1p(cnt_v),
                            dict(params_cnt), "log_mse", args.seed)
    pc_val = m_cnt.predict(Xv)
    del m_cnt

    # --- check head ---
    if mode == "product":
        # MSE on log(check), buyers only, early stop on VAL buyers
        print(f"--- aov head (product): rows={int(mask.sum())} "
              f"val_rows={int(mask_v.sum())}", flush=True)
        m2, it_aov = fit_lgb(X[mask], np.log(aov[mask]), None,
                             Xv[mask_v], np.log(aov_v[mask_v]),
                             dict(params_aov), "log_mse", args.seed + 1)
        p2_val = m2.predict(Xv)  # full universe; extrapolated for non-buyers
        p2_head_val = np.clip(p2_val, *AOV_LOG_CLIP)
        u_ref_val = np.log(np.where(mask_v, aov_v, 1.0))
    else:
        # zero-inflated log-check uplift on ALL rows, early stop on VAL u
        u_tr = uplift_target(y_tot_tr, cnt)
        u_val = uplift_target(yv_tot, cnt_v)
        print(f"--- aov head (uplift): nz_rate={float((u_tr > 0).mean()):.4f} "
              f"clip_share={float((np.log1p(y_tot_tr) < np.log1p(cnt)).mean()):.5f}",
              flush=True)
        m2, it_aov = fit_lgb(X, u_tr, None, Xv, u_val,
                             dict(params_aov), "log_mse", args.seed + 1)
        p2_val = m2.predict(Xv)
        p2_head_val = np.clip(p2_val, 0, None)
        u_ref_val = u_val
    del m2

    # head diagnostics (check head measured in log space on VAL buyers)
    cnt_rmsle = rmsle(cnt_v, np.expm1(np.clip(pc_val, 0, None)))
    aov_rmse = float(np.sqrt(np.mean(
        (p2_head_val[mask_v] - u_ref_val[mask_v]) ** 2)))
    aov_bias = float(np.mean(p2_head_val[mask_v] - u_ref_val[mask_v]))

    # damp selection (honest half-split) + primary score with --aov-damp
    dampfit = fit_damp(mode, pc_val, p2_val, yv_tot)
    pv_tot = COMBINE[mode](pc_val, p2_val, args.aov_damp)
    score = rmsle(yv_tot, pv_tot)
    base_holdout = rmsle(yv_tot[~dampfit["half"]], pv_tot[~dampfit["half"]])
    print(f"[DAMP] best={dampfit['best']:.2f} holdout damp1.0 "
          f"{dampfit['holdout_nodamp']:.6f} -> best {dampfit['holdout']:.6f}; "
          f"full-val@best {dampfit['full']:.6f}", flush=True)

    notes = (args.notes or
             f"cnt*check {mode}; tw1.3-on-log1p(cnt) + "
             f"{'mse-log(aov) buyers' if mode == 'product' else 'tw1.3 uplift'} "
             f"nl255 mdl300 lr0.05 gap{args.gap_days} n{len(tr_anchors)} "
             f"damp={args.aov_damp}") + (
             f"; cnt_rmsle={cnt_rmsle:.4f} aov_rmse={aov_rmse:.4f} "
             f"aov_bias={aov_bias:+.4f} it={it_cnt}/{it_aov}; "
             f"best_damp={dampfit['best']:.2f} "
             f"(holdout {dampfit['holdout_nodamp']:.4f}->{dampfit['holdout']:.4f}); "
             f"direct_champ={DIRECT_CHAMPION} d={score - DIRECT_CHAMPION:+.4f}")
    save_preds(args.name, "val", uid_val, pv_tot)
    log_score(args.name, score, notes)

    use_dampfit = abs(dampfit["best"] - args.aov_damp) > 1e-9
    if use_dampfit:
        save_preds(f"{args.name}_dampfit", "val", uid_val,
                   COMBINE[mode](pc_val, p2_val, dampfit["best"]))
        log_score(f"{args.name}_dampfit", dampfit["full"],
                  f"check damp {dampfit['best']:.2f} fitted on half VAL "
                  f"(grid 0.85..1.05); honest holdout "
                  f"{base_holdout:.6f}->{dampfit['holdout']:.6f}")

    print("RESULT " + json.dumps({
        "name": args.name, "mode": mode, "total": round(score, 6),
        "cnt_rmsle": round(cnt_rmsle, 6), "aov_rmse": round(aov_rmse, 6),
        "aov_bias": round(aov_bias, 6),
        "best_damp": dampfit["best"],
        "damp_holdout": round(dampfit["holdout"], 6),
        "damp_holdout_nodamp": round(dampfit["holdout_nodamp"], 6),
        "full_at_best_damp": round(dampfit["full"], 6),
        "delta_vs_champion": round(score - DIRECT_CHAMPION, 6),
        "best_it": {"cnt": it_cnt, "aov": it_aov},
        "n_anchors": len(tr_anchors),
    }), flush=True)

    if args.no_test:
        return

    # --- retrain on train + gap + val, predict test (exp_lib contract) ---
    parts, cnt_parts, aov_parts, y_parts = [X], [cnt], [aov], [y_tot_tr]
    if gap_anchors:
        g = load_with_counts(gap_anchors, cols)
        Xg, cg, ag, yg = to_arrays(g, cols)
        del g
        parts.append(Xg); cnt_parts.append(cg); aov_parts.append(ag)
        y_parts.append(yg)
        print(f"retrain adds gap anchors: +{Xg.shape[0]} rows", flush=True)
    parts.append(Xv); cnt_parts.append(cnt_v); aov_parts.append(aov_v)
    y_parts.append(yv_tot)
    Xall = np.vstack(parts)
    cnt_all = np.concatenate(cnt_parts)
    aov_all = np.concatenate(aov_parts)
    y_all = np.concatenate(y_parts)
    mask_all = cnt_all > 0
    ratio_cnt = Xall.shape[0] / max(X.shape[0], 1)
    ratio_aov = (float(mask_all.sum()) / max(float(mask.sum()), 1.0)
                 if mode == "product" else ratio_cnt)
    mult_cnt = 1.0 + 0.7 * max(ratio_cnt - 1.0, 0.0)
    mult_aov = 1.0 + 0.7 * max(ratio_aov - 1.0, 0.0)
    print(f"retrain: row_ratio cnt={ratio_cnt:.3f} aov={ratio_aov:.3f} "
          f"iter_mult={mult_cnt:.3f}/{mult_aov:.3f}", flush=True)
    del X, Xv, parts, cnt_parts, aov_parts, y_parts

    test = load_anchor(TEST_ANCHOR)
    Xt = test.select(cols).to_numpy().astype(np.float32)
    uid_t = test["user_id"].to_numpy()
    del test

    p = dict(params_cnt); p["n_estimators"] = max(50, int(it_cnt * mult_cnt))
    print(f"--- retrain count: {p['n_estimators']} iters", flush=True)
    mf, _ = fit_lgb(Xall, np.log1p(cnt_all), None, None, None, p, "log_mse",
                    args.seed)
    pc_test = mf.predict(Xt)
    save_lgb(args.name, mf, tag="count")   # freeze: retrained count head
    del mf

    p = dict(params_aov); p["n_estimators"] = max(50, int(it_aov * mult_aov))
    if mode == "product":
        print(f"--- retrain aov (product): {p['n_estimators']} iters on "
              f"{int(mask_all.sum())} buyer rows", flush=True)
        mf, _ = fit_lgb(Xall[mask_all], np.log(aov_all[mask_all]), None, None,
                        None, p, "log_mse", args.seed + 1)
    else:
        print(f"--- retrain aov (uplift): {p['n_estimators']} iters", flush=True)
        mf, _ = fit_lgb(Xall, uplift_target(y_all, cnt_all), None, None, None,
                        p, "log_mse", args.seed + 1)
    p2_test = mf.predict(Xt)
    save_lgb(args.name, mf, tag="aov")     # freeze: retrained AOV head
    del mf, Xall

    # freeze: what inference needs to recombine the two heads
    save_meta(args.name, kind="countaov", model="lgb", mode=mode,
              feature_cols=cols, params_cnt=params_cnt, params_aov=params_aov,
              aov_damp=args.aov_damp, seed=args.seed, gap_days=args.gap_days,
              n_anchors=len(tr_anchors), val_rmsle=float(score),
              dampfit_best=None if not use_dampfit else float(dampfit["best"]),
              weights=[booster_filename("lgb", args.name, t)
                       for t in ("count", "aov")])
    save_preds(args.name, "test", uid_t, COMBINE[mode](pc_test, p2_test, args.aov_damp))
    if use_dampfit:
        save_preds(f"{args.name}_dampfit", "test", uid_t,
                   COMBINE[mode](pc_test, p2_test, dampfit["best"]))
    print(f"[DONE] {args.name} val_rmsle={score:.6f} total {time.time()-t0:.0f}s",
          flush=True)


if __name__ == "__main__":
    main()

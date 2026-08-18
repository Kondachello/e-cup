"""Heuristic baselines for Ozon E-CUP LTV, scored on the VAL anchor.

Protocol:
  - TUNE anchor 2025-12-31: features from data <= anchor, target = per-user gmv
    sum over 2026-01-01..2026-01-30. All free params fit here.
  - VAL anchor 2026-01-14: frozen params applied, RMSLE reported
    (target 2026-01-15..2026-02-13, absent users = 0).
  - TEST anchor 2026-02-13: best method's predictions saved via exp_lib.

User universe for all anchors: the 250k users of sample_submit.csv, in CSV order.
"""
from __future__ import annotations

import os

os.environ.setdefault("POLARS_MAX_THREADS", "3")
os.environ.setdefault("OMP_NUM_THREADS", "3")

import sys
from datetime import date, timedelta

import numpy as np
import polars as pl

sys.path.insert(0, "/Users/alexanderkondakov/ozon-cup/work/scripts")
from common import REPORTS_DIR, SAMPLE_SUBMIT, TEST_ANCHOR, TRAIN_PARQUET, VAL_ANCHOR, rmsle
from exp_lib import log_score, save_preds

TUNE_ANCHOR = date(2025, 12, 31)
HALFLIVES = [7, 14, 30, 60, 120]
FEATS = ["gmv_30", "gmv_90", "gmv_365", "gmv_ya"] + [f"dec_{h}" for h in HALFLIVES]


# ---------------------------------------------------------------- data
def features_for_anchor(anchor: date) -> pl.DataFrame:
    lf = pl.scan_parquet(TRAIN_PARQUET).select(["event_date", "user_id", "gmv"])
    sub = lf.filter(pl.col("event_date") <= anchor)
    days_ago = (pl.lit(anchor) - pl.col("event_date")).dt.total_days()
    aggs = [
        pl.col("gmv")
        .filter(pl.col("event_date") >= anchor - timedelta(days=29))
        .sum()
        .alias("gmv_30"),
        pl.col("gmv")
        .filter(pl.col("event_date") >= anchor - timedelta(days=89))
        .sum()
        .alias("gmv_90"),
        pl.col("gmv")
        .filter(pl.col("event_date") >= anchor - timedelta(days=364))
        .sum()
        .alias("gmv_365"),
        pl.col("gmv")
        .filter(
            (pl.col("event_date") >= anchor - timedelta(days=364))
            & (pl.col("event_date") <= anchor - timedelta(days=335))
        )
        .sum()
        .alias("gmv_ya"),
    ] + [
        (pl.col("gmv") * (pl.lit(0.5) ** (days_ago / float(h)))).sum().alias(f"dec_{h}")
        for h in HALFLIVES
    ]
    return sub.group_by("user_id").agg(aggs).collect(engine="streaming")


def target_for_anchor(anchor: date) -> pl.DataFrame:
    lf = pl.scan_parquet(TRAIN_PARQUET).select(["event_date", "user_id", "gmv"])
    t = lf.filter(
        (pl.col("event_date") > anchor)
        & (pl.col("event_date") <= anchor + timedelta(days=30))
    )
    return t.group_by("user_id").agg(pl.col("gmv").sum().alias("target")).collect(
        engine="streaming"
    )


def align(users: pl.DataFrame, df: pl.DataFrame, cols: list[str]) -> dict[str, np.ndarray]:
    out = users.join(df, on="user_id", how="left", maintain_order="left")
    assert out["user_id"].equals(users["user_id"]), "join broke user order"
    out = out.fill_null(0.0)
    return {c: out[c].to_numpy().astype(np.float64) for c in cols}


# ---------------------------------------------------------------- fitting helpers
def rmsle_log(z_pred: np.ndarray, ly: np.ndarray) -> float:
    """RMSLE when the prediction is expm1(z_pred) with z_pred >= 0."""
    return float(np.sqrt(np.mean((z_pred - ly) ** 2)))


def best_on_grid(grid, score_fn):
    scores = [score_fn(g) for g in grid]
    i = int(np.argmin(scores))
    return grid[i], scores[i]


def main() -> None:
    submit = pl.read_csv(SAMPLE_SUBMIT, schema_overrides={"user_id": pl.Int64})
    users = submit.select("user_id")
    uid = users["user_id"].to_numpy()
    assert len(uid) == 250_000 and len(np.unique(uid)) == 250_000
    print(f"submit users: {len(uid)}, sorted={bool(np.all(np.diff(uid) > 0))}", flush=True)

    print("building features/targets ...", flush=True)
    F_tu = align(users, features_for_anchor(TUNE_ANCHOR), FEATS)
    F_va = align(users, features_for_anchor(VAL_ANCHOR), FEATS)
    F_te = align(users, features_for_anchor(TEST_ANCHOR), FEATS)
    y_tu = align(users, target_for_anchor(TUNE_ANCHOR), ["target"])["target"]
    y_va = align(users, target_for_anchor(VAL_ANCHOR), ["target"])["target"]
    ly_tu, ly_va = np.log1p(y_tu), np.log1p(y_va)
    print(
        f"tune: active(target>0)={np.mean(y_tu > 0):.4f}  val: {np.mean(y_va > 0):.4f}",
        flush=True,
    )

    results = []  # (name, params_str, tune_rmsle, val_rmsle)

    # 1. zero
    zero = np.zeros_like(y_va)
    results.append(
        ("zero", "-", rmsle(y_tu, np.zeros_like(y_tu)), rmsle(y_va, zero))
    )

    # 2. best constant (closed form in log space on tune)
    lc = float(np.mean(ly_tu))
    const = float(np.expm1(lc))
    results.append(
        (
            "const",
            f"const={const:.2f}",
            rmsle_log(np.full_like(ly_tu, lc), ly_tu),
            rmsle_log(np.full_like(ly_va, lc), ly_va),
        )
    )

    # 3. naive AR: last-30d gmv sum
    results.append(
        (
            "ar30",
            "-",
            rmsle(y_tu, F_tu["gmv_30"]),
            rmsle(y_va, F_va["gmv_30"]),
        )
    )

    # 4. alpha * gmv_30
    alphas = np.round(np.arange(0.10, 1.5001, 0.05), 2)
    a4, tu4 = best_on_grid(list(alphas), lambda a: rmsle(y_tu, a * F_tu["gmv_30"]))
    results.append(
        ("alpha_ar30", f"alpha={a4:.2f}", tu4, rmsle(y_va, a4 * F_va["gmv_30"]))
    )

    # 5. shrunk log AR: expm1(c * log1p(gmv_30))
    cs = np.round(np.arange(0.10, 1.2001, 0.05), 2)
    x_tu, x_va = np.log1p(F_tu["gmv_30"]), np.log1p(F_va["gmv_30"])
    c5, tu5 = best_on_grid(list(cs), lambda c: rmsle_log(c * x_tu, ly_tu))
    results.append(("log_ar30", f"c={c5:.2f}", tu5, rmsle_log(c5 * x_va, ly_va)))

    # 6. decayed daily sum, shrunk in log space: 2D grid (c, halflife)
    best6 = None
    for h in HALFLIVES:
        xd = np.log1p(F_tu[f"dec_{h}"])
        c, s = best_on_grid(list(cs), lambda c: rmsle_log(c * xd, ly_tu))
        if best6 is None or s < best6[2]:
            best6 = (h, c, s)
    h6, c6, tu6 = best6
    results.append(
        (
            "decay",
            f"halflife={h6}, c={c6:.2f}",
            tu6,
            rmsle_log(c6 * np.log1p(F_va[f"dec_{h6}"]), ly_va),
        )
    )

    # 7. nonneg blend of log AR windows (NNLS, exact for this objective)
    from scipy.optimize import nnls

    def design(F):
        return np.stack(
            [
                np.log1p(F["gmv_30"]),
                np.log1p(F["gmv_90"] / 3.0),
                np.log1p(F["gmv_365"] / 12.17),
            ],
            axis=1,
        )

    X_tu, X_va = design(F_tu), design(F_va)
    w, _ = nnls(X_tu, ly_tu)
    results.append(
        (
            "blend_log",
            "w=[" + ", ".join(f"{v:.3f}" for v in w) + "]",
            rmsle_log(X_tu @ w, ly_tu),
            rmsle_log(X_va @ w, ly_va),
        )
    )

    # 8. year-ago window: expm1(c * log1p(gmv over [anchor-364, anchor-335]))
    xy_tu, xy_va = np.log1p(F_tu["gmv_ya"]), np.log1p(F_va["gmv_ya"])
    c8, tu8 = best_on_grid(list(cs), lambda c: rmsle_log(c * xy_tu, ly_tu))
    results.append(("yearago", f"c={c8:.2f}", tu8, rmsle_log(c8 * xy_va, ly_va)))

    # ---------------------------------------------------------------- rank + report
    ranked = sorted(results, key=lambda r: r[3])
    print("\n=== VAL ranking ===")
    for name, p, tu, va in ranked:
        print(f"{name:12s} val={va:.5f} tune={tu:.5f} {p}")

    # predictions of the best method for VAL and TEST (frozen params)
    method_pred = {
        "zero": lambda F: np.zeros(len(uid)),
        "const": lambda F: np.full(len(uid), const),
        "ar30": lambda F: F["gmv_30"],
        "alpha_ar30": lambda F: a4 * F["gmv_30"],
        "log_ar30": lambda F: np.expm1(c5 * np.log1p(F["gmv_30"])),
        "decay": lambda F: np.expm1(c6 * np.log1p(F[f"dec_{h6}"])),
        "blend_log": lambda F: np.expm1(design(F) @ w),
        "yearago": lambda F: np.expm1(c8 * np.log1p(F["gmv_ya"])),
    }
    best_name, best_params, best_tu, best_va = ranked[0]
    pred_va = np.clip(method_pred[best_name](F_va), 0, None)
    pred_te = np.clip(method_pred[best_name](F_te), 0, None)
    check = rmsle(y_va, pred_va)
    assert abs(check - best_va) < 1e-9, (check, best_va)

    save_preds("base_best", "val", uid, pred_va)
    save_preds("base_best", "test", uid, pred_te)
    log_score(
        "base_best",
        best_va,
        f"heuristic {best_name} ({best_params}), tuned on anchor {TUNE_ANCHOR}",
    )

    # sanity: saved order matches sample_submit order
    saved = pl.read_parquet("/Users/alexanderkondakov/ozon-cup/work/preds/base_best_test.parquet")
    assert np.array_equal(saved["user_id"].to_numpy(), uid), "saved order mismatch"
    print(
        f"test preds: mean={pred_te.mean():.2f} p50={np.median(pred_te):.2f} "
        f"share>0={np.mean(pred_te > 0):.4f}",
        flush=True,
    )

    # markdown report
    lines = [
        "# Heuristic baselines (VAL anchor 2026-01-14)",
        "",
        "Protocol: free params tuned on anchor **2025-12-31** (features from data <= anchor,",
        "target = per-user gmv over 2026-01-01..2026-01-30), then applied frozen to the",
        "VAL anchor **2026-01-14** (target 2026-01-15..2026-02-13). User universe: the",
        "250,000 users of `sample_submit.csv` (absent users have target 0). Metric: RMSLE.",
        "",
        f"- Share of users with target>0: tune {np.mean(y_tu > 0):.4f}, val {np.mean(y_va > 0):.4f}",
        "- Windows end at the anchor day inclusive; `gmv_ya` = gmv over [anchor-364, anchor-335]",
        "  (for TEST this is exactly one year before the prediction window).",
        f"- `dec_h` = sum(gmv * 0.5^(days_ago/h)); grids: alpha 0.10..1.50/0.05, c 0.10..1.20/0.05.",
        "",
        "## Ranked results (by VAL RMSLE)",
        "",
        "| rank | method | frozen params | tune RMSLE | VAL RMSLE |",
        "|---|---|---|---|---|",
    ]
    for i, (name, p, tu, va) in enumerate(ranked, 1):
        lines.append(f"| {i} | {name} | {p} | {tu:.5f} | {va:.5f} |")
    lines += [
        "",
        "Method notes:",
        "- `zero`: predict 0 for everyone.",
        "- `const`: best constant = expm1(mean(log1p(y_tune))) (closed form).",
        "- `ar30`: last-30d gmv sum as-is (naive AR).",
        "- `alpha_ar30`: alpha * gmv_30.",
        "- `log_ar30`: expm1(c * log1p(gmv_30)).",
        "- `decay`: expm1(c * log1p(sum gmv * 0.5^(days_ago/halflife))), 2D grid over (c, halflife in {7,14,30,60,120}).",
        "- `blend_log`: expm1(w1*log1p(gmv_30) + w2*log1p(gmv_90/3) + w3*log1p(gmv_365/12.17)), w>=0 via NNLS (exact for the squared-log objective).",
        "- `yearago`: expm1(c * log1p(gmv_ya)).",
        "",
        f"## Best method: `{best_name}` ({best_params})",
        "",
        f"- VAL RMSLE = **{best_va:.5f}** (tune {best_tu:.5f}).",
        f"- Predictions saved: `work/preds/base_best_val.parquet`, `work/preds/base_best_test.parquet`",
        f"  (TEST anchor 2026-02-13, same frozen params); row order verified to match `sample_submit.csv`.",
        f"- TEST preds: mean={pred_te.mean():.2f}, median={np.median(pred_te):.2f}, share>0={np.mean(pred_te > 0):.4f}.",
        "",
        "Caveats: the tuning anchor's target window (Jan 1-30) covers the post-New-Year",
        "period while features end in the December peak, so tuned shrinkage may be slightly",
        "biased low for later anchors; VAL numbers above are the honest frozen-param scores.",
    ]
    (REPORTS_DIR / "baselines.md").write_text("\n".join(lines) + "\n")
    print("report written to work/reports/baselines.md", flush=True)

    # machine-readable metrics for the caller
    import json

    metrics = {f"val_rmsle_{name}": round(va, 6) for name, _, _, va in results}
    metrics.update(
        best_val_rmsle=round(best_va, 6),
        alpha_ar30_alpha=a4,
        log_ar30_c=c5,
        decay_halflife=float(h6),
        decay_c=c6,
        yearago_c=c8,
        blend_w30=round(float(w[0]), 4),
        blend_w90=round(float(w[1]), 4),
        blend_w365=round(float(w[2]), 4),
    )
    print("METRICS_JSON=" + json.dumps(metrics))


if __name__ == "__main__":
    main()

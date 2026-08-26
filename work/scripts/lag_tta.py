"""Lag-TTA: the same trained model fed feature vintages frozen before the anchor.

Why this escapes the hull theorem: the theorem says the blend residual is not a
function of X_val (features AT the val anchor). A prediction computed from features
frozen 14/28 days EARLIER is a function of a different input, so its out-of-hull
share has never been measured. This is test-time augmentation over feature staleness.

One champion-recipe LGB (tweedie-on-log, gap30 protocol, early stop on fresh val),
then val predictions from three vintages:

  lag0   features at 2026-01-14 - the standard val prediction, in-hull baseline
  lag14  features at 2025-12-31 - model's window (01.01-30.01] read as a val forecast
  lag28  features at 2025-12-17 - model's window (18.12-16.01] read as a val forecast

plus log-space mixes with the fresh prediction. The stale windows are level- and
season-shifted; that is exactly what margin.py's internal honest calibration removes
before measuring, so acceptance = margin.py lag0 lag14 lag28 lagmix_a lagmix_b.
The DECISION contrast is margin(stale/mix) minus margin(lag0): does staleness add
out-of-hull signal to this model family, yes or no.

No leakage: stale anchors (12-17, 12-31) sit in the gap zone - their rows are never
trained on, and their features use data ending before the val window starts.

Run:
  USE_V2=1 USE_V3=1 .venv/bin/python work/scripts/lag_tta.py
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from common import VAL_ANCHOR, feature_cols, load_anchor, rmsle
from exp_lib import (available_train_anchors, load_matrix, log_score, note,
                     protocol_train_anchors, save_preds)
from model_io import save_lgb

# Vintage = feature anchor read as a val forecast. LAG_DAYS names them by staleness.
LAGS = (0, 14, 28, 42, 56, 70)

# Источник набора обучающих якорей; выставляется из --anchor-source в main().
ANCHOR_SOURCE = "protocol"
MIXES = {"lagmix_a": {"lag0": 0.5, "lag14": 0.5},
         "lagmix_b": {"lag0": 0.6, "lag14": 0.25, "lag28": 0.15}}


def champion_anchors(cut: date) -> list[date]:
    """Обучающие якоря чемпионского набора признаков не позже cut.

    Февральские якоря собраны без extra-тира (они нужны только год-назад окну
    febspec): чемпионские колонки на них не собираются, load_matrix падает на select.
    """
    # ПО ПРОТОКОЛУ, не по каталогу. Именно здесь проблему и нашли: 24.08 попытка
    # воспроизвести lagd28 подхватила 11 обучающих якорей вместо 9, потому что рядом
    # построили два лишних под другую проверку, — и совпало всё только после возврата
    # каталога к канону. Набор ОБЯЗАН зависеть от протокола, а не от того, что лежит
    # на диске. `--anchor-source disk` возвращает прежнее поведение для старых артефактов.
    anchors = [a for a in protocol_train_anchors(source=ANCHOR_SOURCE) if a <= cut]
    if os.environ.get("USE_V2"):
        from common import FEATURES_DIR
        anchors = [a for a in anchors
                   if (FEATURES_DIR / f"anchor={a.isoformat()}.extra.parquet").exists()]
    return anchors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lags", type=str, default="0,14,28",
                    help="feature vintages to score, in days of staleness before the val anchor")
    ap.add_argument("--anchor-source", choices=["protocol", "disk"], default="protocol",
                    help="откуда брать обучающие якоря: protocol — train_anchors(14), не "
                         "зависит от каталога (умолчание); disk — историческое поведение")
    ap.add_argument("--prefix", type=str, default="lag", help="prediction name prefix")
    ap.add_argument("--mixes", action="store_true", help="also emit the two log-space mixes")
    ap.add_argument("--rounds", type=int, default=2500)
    ap.add_argument("--test", action="store_true",
                    help="also emit test predictions for the WINNING vintage (exp_lib contract)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--objective", default="tweedie", choices=["tweedie", "two_stage"],
                    help="two_stage = classifier x regressor, a different loss on the same vintages")
    ap.add_argument("--test-vintage", default="2026-01-14",
                    help="feature date for the test-side prediction (30d stale by default)")
    ap.add_argument("--test-name", default="",
                    help="output name; default PREFIX28 for backward compatibility")
    ap.add_argument("--test-only", type=int, default=0, metavar="VAL_ITER",
                    help="skip the val pass and emit only test preds, reusing a known val it=")
    args = ap.parse_args()
    global ANCHOR_SOURCE
    ANCHOR_SOURCE = args.anchor_source
    assert os.environ.get("USE_V2") and os.environ.get("USE_V3"), \
        "run with USE_V2=1 USE_V3=1 (champion feature set available locally)"
    t0 = time.time()

    if args.test_only:
        val = load_anchor(VAL_ANCHOR).sort("user_id")
        # --test-only переиспользует известную точку остановки val-прогона, поэтому
        # число его обучающих якорей надо восстановить тем же правилом, что в main
        lags_o = [int(x) for x in args.lags.split(",")]
        stale_o = [VAL_ANCHOR - timedelta(days=d) for d in lags_o if d]
        cut_o = min([VAL_ANCHOR - timedelta(days=30)]
                    + [a - timedelta(days=1) for a in stale_o])
        emit_test(args, feature_cols(val), val["user_id"].to_numpy(), args.test_only,
                  len(champion_anchors(cut_o)))
        print(f"[DONE] {time.time()-t0:.0f}s", flush=True)
        return

    lags = [int(x) for x in args.lags.split(",")]
    stale = {f"{args.prefix}{d}": VAL_ANCHOR - timedelta(days=d) for d in lags if d}

    # A vintage may only be scored if the model never TRAINED on that anchor: otherwise the
    # booster has memorised those exact rows and the "prediction" is partly recall of that
    # anchor's own target. So training stops before the deepest vintage we intend to score.
    gap_cut = min([VAL_ANCHOR - timedelta(days=30)] + [a - timedelta(days=1) for a in stale.values()])
    tr_anchors = champion_anchors(gap_cut)
    print(f"deepest vintage {min(stale.values()) if stale else VAL_ANCHOR} "
          f"-> train cutoff {gap_cut}", flush=True)
    print(f"train anchors (gap30): {[a.isoformat() for a in tr_anchors]}", flush=True)

    val = load_anchor(VAL_ANCHOR).sort("user_id")
    cols = feature_cols(val)
    print(f"{len(cols)} features", flush=True)
    # Число обучающих якорей val-стороны задаёт множитель итераций тестового
    # переобучения, а сам набор берётся с диска — записываем оба факта.
    note(anchor_source=ANCHOR_SOURCE, val_train_anchors=[a.isoformat() for a in tr_anchors],
         n_val_anchors=len(tr_anchors), n_features=len(cols), lags=args.lags,
         objective=args.objective, seed=args.seed, train_cutoff=gap_cut.isoformat())

    tr = load_matrix(tr_anchors, columns=["target"] + cols)
    X = tr.select(cols).to_numpy().astype(np.float32)
    y = np.log1p(np.clip(tr["target"].to_numpy().astype(np.float64), 0, None))
    del tr
    Xv = val.select(cols).to_numpy().astype(np.float32)
    yv_raw = np.clip(val["target"].to_numpy().astype(np.float64), 0, None)
    yv = np.log1p(yv_raw)
    uid = val["user_id"].to_numpy()
    print(f"X {X.shape}, Xv {Xv.shape}, load {time.time()-t0:.0f}s", flush=True)

    import lightgbm as lgb
    base = dict(learning_rate=0.05, num_leaves=255, min_data_in_leaf=300,
                feature_fraction=0.75, bagging_fraction=0.8, bagging_freq=1,
                lambda_l2=5.0, max_bin=127, num_threads=7, seed=args.seed, verbosity=-1)

    def fit(yy, yyv, extra):
        p = dict(base, **extra)
        d = lgb.Dataset(X, yy, free_raw_data=False)
        dv = lgb.Dataset(Xv, yyv, reference=d, free_raw_data=False)
        mm = lgb.train(p, d, num_boost_round=args.rounds, valid_sets=[dv],
                       callbacks=[lgb.early_stopping(200, verbose=False),
                                  lgb.log_evaluation(400)])
        return mm, mm.best_iteration

    if args.objective == "two_stage":
        # P(y>0) and E[log1p | y>0] fitted separately; the vintage prediction is the product,
        # so staleness enters both the incidence and the magnitude channel.
        pos, posv = y > 0, yv > 0
        m1, it1 = fit(pos.astype(np.float64), posv.astype(np.float64),
                      dict(objective="binary", metric="auc"))
        Xp, Xvp = X[pos], Xv[posv]
        d2 = lgb.Dataset(Xp, y[pos], free_raw_data=False)
        dv2 = lgb.Dataset(Xvp, yv[posv], reference=d2, free_raw_data=False)
        m2 = lgb.train(dict(base, objective="regression", metric="rmse"), d2,
                       num_boost_round=args.rounds, valid_sets=[dv2],
                       callbacks=[lgb.early_stopping(200, verbose=False),
                                  lgb.log_evaluation(400)])
        it = m1.best_iteration
        print(f"two_stage: it1={it1} it2={m2.best_iteration}, train {time.time()-t0:.0f}s", flush=True)
        predict = lambda Z: np.clip(m1.predict(Z) * np.clip(m2.predict(Z), 0, None), 0, None)
        # Воспроизводимость: без сохранённых весов *_val.parquet не восстановить из чистого
        # клона, и модель остаётся «прогноз-артефактом» (inference.py --stage check). ОДНА
        # обученная модель порождает ВСЕ винтажи, поэтому её и достаточно.
        save_lgb(f"{args.prefix}_val", m1, tag="ts1")
        save_lgb(f"{args.prefix}_val", m2, tag="ts2")
    else:
        m, it = fit(y, yv, dict(objective="tweedie", tweedie_variance_power=1.45, metric="rmse"))
        print(f"best_iteration={it}, train {time.time()-t0:.0f}s", flush=True)
        predict = lambda Z: np.clip(m.predict(Z), 0, None)
        save_lgb(f"{args.prefix}_val", m, tag="tw")
    del X

    lp = {}   # name -> log1p-space prediction aligned to uid
    fresh = f"{args.prefix}0"
    lp[fresh] = predict(Xv)
    del Xv
    for name, anc in stale.items():
        df = load_anchor(anc, columns=["user_id"] + cols).sort("user_id")
        assert np.array_equal(df["user_id"].to_numpy(), uid), f"universe mismatch at {anc}"
        lp[name] = predict(df.select(cols).to_numpy().astype(np.float32))
        del df
    if args.mixes:
        for name, w in MIXES.items():
            if all(k in lp for k in w):
                lp[name] = np.clip(sum(v * lp[k] for k, v in w.items()), 0, None)

    for name, l in lp.items():
        pv = np.expm1(l)
        save_preds(name, "val", uid, pv)
        note = (f"lag-TTA {args.objective} cut{gap_cut} n{len(tr_anchors)} it={it}; "
                + (f"mix {MIXES[name]}" if name in MIXES else
                   f"features at {stale.get(name, VAL_ANCHOR)}"))
        log_score(name, rmsle(yv_raw, pv), note)

    e0 = lp[fresh] - yv
    for name in stale:
        r_pred = float(np.corrcoef(lp[fresh], lp[name])[0, 1])
        r_err = float(np.corrcoef(e0, lp[name] - yv)[0, 1])
        print(f"{name}: corr(pred)={r_pred:.4f} corr(err vs {fresh})={r_err:.4f}", flush=True)
    if args.test:
        emit_test(args, cols, uid, it, len(tr_anchors))
    print(f"[DONE] {time.time()-t0:.0f}s", flush=True)


def emit_test(args, cols, uid_val, val_iter, n_val_anchors: int):
    """Test-window counterpart of the winning vintage.

    The test anchor is 2026-02-13 and anchors are spaced 14 days, so the vintage
    nearest the measured optimum (28 days stale) is 2026-01-14 at 30 days - which is
    also the val anchor. That forces the training cutoff: a model that trained on the
    2026-01-14 rows has memorised that anchor's own target, and feeding it those very
    features would be recall, not forecast. So the retrain stops before it, exactly as
    the val-side run stopped before its deepest vintage.
    """
    import lightgbm as lgb
    from common import TEST_ANCHOR

    vintage = date.fromisoformat(args.test_vintage)
    cut = vintage - timedelta(days=1)
    anchors = champion_anchors(cut)
    print(f"\ntest: винтаж {vintage} ({(TEST_ANCHOR - vintage).days}д застоялости), "
          f"обучение до {cut}, якорей {len(anchors)}", flush=True)

    tr = load_matrix(anchors, columns=["target"] + cols)
    X = tr.select(cols).to_numpy().astype(np.float32)
    y = np.log1p(np.clip(tr["target"].to_numpy().astype(np.float64), 0, None))
    del tr
    base = dict(learning_rate=0.05, num_leaves=255, min_data_in_leaf=300,
                feature_fraction=0.75, bagging_fraction=0.8, bagging_freq=1,
                lambda_l2=5.0, max_bin=127, num_threads=7, seed=args.seed, verbosity=-1)
    # No held-out anchor is left to early-stop on, so reuse the val run's stopping point,
    # scaled by the data growth exactly as train_gbdt.py does on its retrain path.
    #
    # Делитель раньше был зашит константой 9.0 — числом обучающих якорей val-стороны на
    # момент написания. Но набор якорей берётся с диска и меняется: сейчас их 11, и
    # константа давала множитель 1.78 вместо 1.51, то есть +18% итераций. Все прочие
    # трейнеры считают row_ratio по факту; считаем и здесь.
    row_ratio = len(anchors) / max(n_val_anchors, 1)
    iter_mult = 1.0 + 0.7 * max(row_ratio - 1.0, 0.0)
    n_iter = max(50, int(val_iter * iter_mult))
    print(f"test: якорей {len(anchors)} против {n_val_anchors} на val -> "
          f"row_ratio={row_ratio:.3f} iter_mult={iter_mult:.3f} n_iter={n_iter}", flush=True)
    note(n_test_anchors=len(anchors), iter_mult=round(iter_mult, 4), test_n_iter=n_iter,
         test_vintage=args.test_vintage)
    print(f"test: {X.shape[0]} строк, {n_iter} итераций (val it={val_iter}), "
          f"objective={args.objective}", flush=True)
    if args.objective == "two_stage":
        pos = y > 0
        m1 = lgb.train(dict(base, objective="binary"), lgb.Dataset(X, pos.astype(np.float64)),
                       num_boost_round=n_iter)
        m2 = lgb.train(dict(base, objective="regression"), lgb.Dataset(X[pos], y[pos]),
                       num_boost_round=n_iter)
        predict = lambda Z: np.clip(m1.predict(Z) * np.clip(m2.predict(Z), 0, None), 0, None)
        # те же веса, что делают отгружаемый *_test.parquet
        save_lgb(args.test_name or f"{args.prefix}28", m1, tag="ts1")
        save_lgb(args.test_name or f"{args.prefix}28", m2, tag="ts2")
    else:
        m = lgb.train(dict(base, objective="tweedie", tweedie_variance_power=1.45),
                      lgb.Dataset(X, y), num_boost_round=n_iter)
        predict = lambda Z: np.clip(m.predict(Z), 0, None)
        save_lgb(args.test_name or f"{args.prefix}28", m, tag="tw")
    del X

    df = load_anchor(vintage, columns=["user_id"] + cols).sort("user_id")
    assert np.array_equal(df["user_id"].to_numpy(), uid_val), "universe mismatch at vintage"
    pv = np.expm1(predict(df.select(cols).to_numpy().astype(np.float32)))
    name = args.test_name or f"{args.prefix}28"
    save_preds(name, "test", uid_val, pv)
    print(f"test: сохранено {name}_test.parquet", flush=True)


if __name__ == "__main__":
    main()

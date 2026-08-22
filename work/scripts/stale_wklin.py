"""Застоялость входа для ЛИНЕЙНОГО семейства: ось общая или бустинговая?

Открытый пункт трека 4. На бустинге застоялость дала рекорд проекта по запасу
(lagd28, +0.00307): та же обученная модель, накормленная признаками, замороженными
за 28 дней до якоря, — прогноз с чужого винтажа есть функция ДРУГОГО входа, поэтому
теорема оболочки на него не распространяется. Вопрос, который это оставило открытым:
механизм общий для любой модели или свойство именно бустинга?

Саша замерил ответ со стороны секвенсов (tfm3_stale28 = шум, поглощён lagd28). Здесь
табличная линейная сторона: wklin — ridge на недельных матрицах, семейство настолько
далёкое от бустинга, насколько это возможно, не выходя из тех же признаков.

УСТРОЙСТВО. wklin считает сырые моменты Грама ПО ЯКОРЯМ, и любая модель — подблок-решение,
поэтому винтаж стоит один лишний проход, а не переобучение. Контраст берётся с ОДНОГО
обучения:

    свежий  X на 2026-01-14 -> прогноз окна 15.01-13.02   (контроль, внутри оболочки)
    винтаж  X на 2025-12-17 -> прогноз того же окна        (28 дней застоялости)

ЧИСТОТА. Обучающие якоря обрезаны так, чтобы модель не видела ни сам винтаж, ни его
целевое окно: иначе она узнаёт свои же строки, и это припоминание, а не прогноз.
Решает не абсолютный запас винтажа, а КОНТРАСТ запас(винтаж) − запас(свежий): именно он
говорит, добавляет ли застоялость сигнал вне оболочки ЭТОМУ семейству.

Запуск: python work/scripts/stale_wklin.py [--stale 2025-12-17]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))
from common import FEATURES_DIR, PREDS_DIR, REPORTS_DIR, ROOT, VAL_ANCHOR, feature_cols, load_anchor
from build_features_v5 import W_WEEKS, anchor_plan, joint_block, weekly_dense
from train_wklin import Acc, slog, solve, sse
from margin import calibrate_split, score

T0 = time.time()


def log(m):
    print(f"[{time.time()-T0:6.0f}s] {m}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stale", default="2025-12-17", help="якорь винтажа (28 дней до валидации)")
    ap.add_argument("--alpha", type=float, default=0.0, help="0 = выбрать по отложенному якорю")
    ap.add_argument("--json", default="stale_wklin.json")
    args = ap.parse_args()
    stale_a = date.fromisoformat(args.stale)
    lag_days = (VAL_ANCHOR - stale_a).days

    # ---- обучающие якоря: целевое окно должно кончиться ДО винтажа
    plan = anchor_plan()
    cut = stale_a - timedelta(days=30)
    fit_a = [a for a in plan["fit"] if a <= cut]
    assert len(fit_a) >= 4, f"слишком мало чистых якорей до {cut}: {fit_a}"
    log(f"винтаж {stale_a} ({lag_days} дней застоялости), обучение до {cut}")
    log(f"обучающих якорей {len(fit_a)}: {[a.isoformat() for a in fit_a]}")

    v0 = load_anchor(VAL_ANCHOR)
    base_cols = [c for c in feature_cols(v0) if not c.startswith("v5")]
    uid = np.sort(v0["user_id"].to_numpy())
    ly = np.log1p(np.clip(
        v0.sort("user_id")["target"].to_numpy().astype(np.float64), 0, None))
    n_wk, n_base = 5 * W_WEEKS, len(base_cols)
    p = n_wk + n_base
    del v0

    grid = fit_a + [stale_a, VAL_ANCHOR]
    max_off = max((VAL_ANCHOR - a).days // 7 for a in grid)
    Gv = weekly_dense(VAL_ANCHOR, max_off + W_WEEKS, uid)
    off = {a: (VAL_ANCHOR - a).days // 7 for a in grid}

    def design(a: date):
        Wb = joint_block(Gv, off[a])
        df = load_anchor(a).sort("user_id")
        assert np.array_equal(df["user_id"].to_numpy(), uid), f"порядок user_id разошёлся на {a}"
        X = np.empty((len(uid), p), dtype=np.float32)
        X[:, :n_wk] = Wb
        X[:, n_wk:] = slog(df.select(base_cols).to_numpy().astype(np.float64)).astype(np.float32)
        y = None
        if "target" in df.columns and df["target"].null_count() == 0:
            y = np.log1p(np.clip(df["target"].to_numpy().astype(np.float64), 0, None))
        return X, y

    accs = {}
    for a in fit_a:
        X, y = design(a)
        assert y is not None, f"нет таргета на обучающем якоре {a}"
        accs[a] = Acc(p).add(X, y)
        del X, y
        log(f"  моменты {a}")

    def pool(anchors):
        acc = Acc(p)
        for a in anchors:
            acc.A += accs[a].A; acc.g += accs[a].g; acc.yy += accs[a].yy; acc.n += accs[a].n
        return acc

    cols = np.arange(p)
    # alpha выбирается на ОТЛОЖЕННОМ обучающем якоре, валидацию не трогаем
    if args.alpha:
        alpha = args.alpha
    else:
        ho, tr = fit_a[-1], fit_a[:-1]
        best = (np.inf, None)
        for al in (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0):
            b = solve(pool(tr).A, pool(tr).g, pool(tr).n, cols, al)
            e = sse(accs[ho], b) / accs[ho].n
            if e < best[0]:
                best = (e, al)
        alpha = best[1]
        log(f"alpha={alpha} (отложенный якорь {ho}, mse {best[0]:.5f})")

    acc = pool(fit_a)
    beta = solve(acc.A, acc.g, acc.n, cols, alpha)

    def predict(a: date):
        X, _ = design(a)
        return X @ beta[:p] + beta[-1]

    out, preds = {}, {}
    pack = pl.read_parquet(ROOT / "work" / "preds_pack" / "val_preds.parquet").sort("user_id")
    assert np.array_equal(pack["user_id"].to_numpy(), uid), "юниверс пакета не совпал"
    lb = pack["blend"].to_numpy().astype(np.float64)
    sb = score(lb, ly)
    eb = lb - ly
    rng = np.random.default_rng(0)
    half = rng.permutation(len(ly)) < len(ly) // 2

    log(f"эталон: бленд пакета {sb:.6f}")
    print(f"\n{'винтаж':<16}{'скор(кал.)':>12}{'корр':>10}{'ЗАПАС':>11}")
    for tag, a in (("свежий", VAL_ANCHOR), (f"стал{lag_days}", stale_a)):
        lp = np.clip(predict(a), 0, None)
        lpc = calibrate_split(lp, ly, half, 24)
        sm = score(lpc, ly)
        e = lpc - ly
        rho = float(np.mean(e * eb) / max(sm * sb, 1e-12))
        margin = sb / sm - rho
        out[tag] = {"anchor": a.isoformat(), "rmsle_cal": sm, "err_corr": rho, "margin": margin}
        preds[tag] = lpc
        print(f"{tag:<16}{sm:12.6f}{rho:10.5f}{margin:+11.5f}")

    d = out[f"стал{lag_days}"]["margin"] - out["свежий"]["margin"]
    out["contrast"] = d
    out["alpha"] = alpha
    out["fit_anchors"] = [a.isoformat() for a in fit_a]
    print(f"\nКОНТРАСТ запас(винтаж) − запас(свежий) = {d:+.5f}")
    print("для сравнения на бустинге: lagd28 +0.00307 против lagd0 +0.00050, контраст +0.00257")
    m_stale = out[f"стал{lag_days}"]["margin"]
    # Контраст говорит про НАПРАВЛЕНИЕ, вес даёт только положительный АБСОЛЮТНЫЙ запас:
    # margin <= 0 тождественно означает beta >= 1, то есть нулевой вес у оптимизатора.
    # Положительный контраст при отрицательном запасе = ось шевелит семейство в нужную
    # сторону, но из оболочки не выводит. Это разные утверждения, и путать их нельзя.
    if m_stale > 0 and d > 0.0005:
        v = "застоялость выводит и линейное семейство из оболочки"
    elif d > 0.0005:
        v = (f"направление верное (контраст {d:+.5f}), но запас винтажа {m_stale:+.5f} <= 0 — "
             f"вес нулевой, в бленд не идёт")
    else:
        v = "ось на этом семействе не работает"
    print(f"ВЕРДИКТ: {v}")
    out["verdict"] = v

    for tag, lp in preds.items():
        nm = f"wklin_{'fresh' if tag == 'свежий' else f'stale{lag_days}'}"
        pl.DataFrame({"user_id": uid.astype(np.int64),
                      "pred": np.expm1(lp)}).write_parquet(PREDS_DIR / f"{nm}_val.parquet")
    (REPORTS_DIR / args.json).write_text(json.dumps(out, indent=1, ensure_ascii=False))
    log(f"JSON: {REPORTS_DIR / args.json}")


if __name__ == "__main__":
    main()

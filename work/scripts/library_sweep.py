"""Greedy set selection over the whole prediction library, with an honest floor.

Why this exists: acceptance in this project is per-model (margin vs the blend), and we
measured that the rule is structurally wrong - lagd28 and hz_v1_surv are each worthless
some may be alive as a SET. The library holds 65 val files; this sweeps them.

Two traps this design avoids, both already paid for by the team:

1. caruana.md: a set chosen and scored on the same users buys a gain that does not
   transfer (val 1.6240 -> public 1.6755). So users are split ONCE into DEV and EVAL;
   every selection decision and every weight is fitted on DEV, and the reported number
   comes from EVAL, which selection never touched.
2. Rule 4 of the team protocol: four arbitrary existing models added to the blend give
   +0.00011 by themselves. A positive gain is therefore meaningless without a floor, so
   the same pipeline is run on RANDOM sets of the same size. Greedy has to beat the
   95th percentile of random, not zero.

Usage:
  python work/scripts/library_sweep.py                 # full sweep
  python work/scripts/library_sweep.py --max-k 8 --random-sets 200
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))
from common import PREDS_DIR, REPORTS_DIR, ROOT
from margin import calibrate_honest, score

# A single model cannot beat a 30-model blend; anything that does is contaminated

SANITY_FLOOR = 1.66
# Узкая утечка проходит оба скоровых порога: соло-скор нормальный, а остаток знает
# валидацию (direct_val2chk: cal 1.6649 «лучше бленда», парный вклад +0.016 — в 35 раз
# выше рекорда tfm3 0.000443). Потолок правдоподобия парного вклада одиночки:
CONTAM_CEIL = 0.002

# Обучены до введения зазора 30 дней (val завышен на 0.05-0.10) либо мета-стек,
# подогнанный на валидации. Порог SANITY_FLOOR их не ловит: скоры выглядят законно.
BLACKLIST = {"gru_final", "lgblog_final", "xgblog_final", "mlp_final", "stack_meta"}
# Производные самого бленда: добавлять бленд к бленду — вырожденный кандидат.
BLEND_PREFIXES = ("blend", "caruana")


def nnls_weights(A: np.ndarray, y: np.ndarray) -> np.ndarray:
    from scipy.optimize import nnls
    G = A.T @ A
    L = np.linalg.cholesky(G + 1e-10 * np.eye(len(G)))
    return nnls(L.T, np.linalg.solve(L, A.T @ y))[0]


def fit_eval(A_dev, y_dev, A_ev, y_ev, sb_ev):
    """Weights from DEV, gain measured on EVAL. The only number that counts."""
    w = nnls_weights(A_dev, y_dev)
    return sb_ev - score(A_ev @ w, y_ev), w


def inner_gain(A_dev, y_dev, half):
    """Cross-fit gain INSIDE dev - used only to rank candidates during selection."""
    gs = []
    for m in (half, ~half):
        w = nnls_weights(A_dev[m], y_dev[m])
        gs.append(score(A_dev[~m][:, 0], y_dev[~m]) - score(A_dev[~m] @ w, y_dev[~m]))
    return float(np.mean(gs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", type=Path, default=ROOT / "work" / "preds_pack")
    ap.add_argument("--max-k", type=int, default=10)
    ap.add_argument("--random-sets", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", default="library_sweep.json",
                    help="имя json-артефакта в work/reports (прошлый прогон не оставил ничего)")
    ap.add_argument("--whitelist", default="pack", choices=["pack", "all"],
                    help="pack: только колонки пакета + --include (провенанс проверен). all: весь "
                         "зоопарк work/preds — ТОЛЬКО диагностика: там лежит pre-gap эпоха, чьи "
                         "остатки знают валидацию, и никакой скоровый порог это не ловит "
                         "(замерено 21.08: зоопарковый пол +0.0005 против честного +0.00003)")
    ap.add_argument("--include", default="",
                    help="доп. имена через запятую поверх пакета (свежие кандидаты: lagd28,...)")
    args = ap.parse_args()

    pack = pl.read_parquet(args.pack / "val_preds.parquet").sort("user_id")
    uid = pack["user_id"].to_numpy()
    ly = np.log1p(np.clip(pack["target"].to_numpy().astype(np.float64), 0, None))
    lb = pack["blend"].to_numpy().astype(np.float64)
    sb_full = score(lb, ly)
    print(f"эталон: бленд, скор {sb_full:.6f}, n={len(ly)}")

    # отдельная перестановка для проверки правдоподобия парного вклада кандидата
    chk = np.random.default_rng(args.seed + 999).permutation(len(ly)) < len(ly) // 2

    def pair_gain(lp: np.ndarray) -> float:
        A = np.column_stack([lb, lp])
        gs = []
        for m in (chk, ~chk):
            w = nnls_weights(A[m], ly[m])
            gs.append(score(lb[~m], ly[~m]) - score(A[~m] @ w, ly[~m]))
        return float(np.mean(gs))

    allowed: set[str] | None = None
    if args.whitelist == "pack":
        allowed = set(pack.columns) - {"user_id", "target", "blend"}
        allowed |= {n for n in args.include.split(",") if n}
        print(f"белый список: {len(allowed)} имён (колонки пакета + include)")

    names, cols = [], []
    excluded: dict[str, str] = {}
    for f in sorted(PREDS_DIR.glob("*_val.parquet")):
        n = f.name[: -len("_val.parquet")]
        if allowed is not None:
            if n not in allowed:
                continue                      # зоопарк молча мимо: провенанс не проверен
        else:
            if n in BLACKLIST or n.startswith(BLEND_PREFIXES):
                excluded[n] = "чёрный список (pre-gap / стек / производная бленда)"
                continue
            if n.endswith("_cal") and (PREDS_DIR / f"{n[:-4]}_val.parquet").exists():
                # свип калибрует сам; _cal при живом базовом файле — коллинеарный дубль
                excluded[n] = "производная _cal при живом базовом файле"
                continue
        d = pl.read_parquet(f).sort("user_id")
        if "pred" not in d.columns:
            excluded[n] = "нет колонки pred (служебный файл)"
            continue
        if d.height != len(uid) or not np.array_equal(d["user_id"].to_numpy(), uid):
            print(f"  пропуск {n}: чужой юниверс")
            excluded[n] = "чужой юниверс"
            continue
        lp = calibrate_honest(
            np.log1p(np.clip(d["pred"].to_numpy().astype(np.float64), 0, None)), ly, 24, args.seed)
        s = score(lp, ly)
        if s < SANITY_FLOOR:
            print(f"  ПРОПУСК {n}: скор {s:.4f} < {SANITY_FLOOR} — признак контаминации")
            excluded[n] = f"скор {s:.4f} ниже пола {SANITY_FLOOR}"
            continue
        if s < sb_full + 0.001:
            print(f"  ПРОПУСК {n}: калиброванный скор {s:.4f} на уровне бленда {sb_full:.4f} "
                  f"— честной одиночке недоступно")
            excluded[n] = f"калиброванный скор {s:.4f} на уровне бленда — контаминация"
            continue
        g1 = pair_gain(lp)
        if g1 > CONTAM_CEIL:
            print(f"  ПРОПУСК {n}: парный вклад {g1:+.6f} выше потолка {CONTAM_CEIL} "
                  f"— знает валидацию")
            excluded[n] = f"парный вклад {g1:+.6f} выше потолка правдоподобия"
            continue
        names.append(n); cols.append(lp)
    X = np.column_stack(cols)
    print(f"кандидатов в библиотеке: {len(names)} (исключено {len(excluded)})\n")

    rng = np.random.default_rng(args.seed)
    dev = rng.permutation(len(ly)) < len(ly) // 2      # отбор и веса живут здесь
    ev = ~dev                                          # это EVAL, отбор его не видел
    half = rng.permutation(dev.sum()) < dev.sum() // 2

    bd, be = lb[dev], lb[ev]
    yd, ye = ly[dev], ly[ev]
    sb_ev = score(be, ye)
    Xd, Xe = X[dev], X[ev]

    chosen, curve = [], []
    for k in range(1, args.max_k + 1):
        best = (-9, None)
        for j in range(len(names)):
            if j in chosen:
                continue
            A = np.column_stack([bd] + [Xd[:, i] for i in chosen + [j]])
            g = inner_gain(A, yd, half)
            if g > best[0]:
                best = (g, j)
        chosen.append(best[1])
        Ad = np.column_stack([bd] + [Xd[:, i] for i in chosen])
        Ae = np.column_stack([be] + [Xe[:, i] for i in chosen])
        g_ev, w = fit_eval(Ad, yd, Ae, ye, sb_ev)
        curve.append((k, names[best[1]], best[0], g_ev))
        print(f"k={k:2d}  +{names[best[1]]:<22} отбор(dev)={best[0]:+.6f}  ЧЕСТНО(eval)={g_ev:+.6f}")

    print("\n--- пол: случайные наборы той же ёмкости (правило 4) ---")
    floors = {}
    for k in range(1, args.max_k + 1):
        gs = []
        for _ in range(args.random_sets):
            idx = rng.choice(len(names), size=k, replace=False)
            Ad = np.column_stack([bd] + [Xd[:, i] for i in idx])
            Ae = np.column_stack([be] + [Xe[:, i] for i in idx])
            gs.append(fit_eval(Ad, yd, Ae, ye, sb_ev)[0])
        floors[k] = (float(np.mean(gs)), float(np.percentile(gs, 95)))
        print(f"k={k:2d}  случайный набор: среднее {floors[k][0]:+.6f}  95-й перцентиль {floors[k][1]:+.6f}")

    print("\n--- ВЕРДИКТ ---")
    print(f"{'k':>3}{'жадный (eval)':>16}{'пол (95%)':>14}  бьёт пол?")
    for k, nm, gd, ge in curve:
        print(f"{k:>3}{ge:>16.6f}{floors[k][1]:>14.6f}  {'ДА' if ge > floors[k][1] else 'нет'}")

    out = {
        "seed": args.seed,
        "reference_blend": round(score(lb, ly), 6),
        "n_candidates": len(names),
        "candidates": names,
        "excluded": excluded,
        "curve": [{"k": k, "added": nm, "dev": round(gd, 6), "eval": round(ge, 6),
                   "floor_mean": round(floors[k][0], 6), "floor_p95": round(floors[k][1], 6),
                   "beats_floor": bool(ge > floors[k][1])}
                  for k, nm, gd, ge in curve],
        "greedy_set": [names[i] for i in chosen],
    }
    (REPORTS_DIR / args.json).write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(f"\nJSON: work/reports/{args.json}")


if __name__ == "__main__":
    main()

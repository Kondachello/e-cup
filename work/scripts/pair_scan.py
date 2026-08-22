"""Исчерпывающий скан ПАР и ТРОЕК: то, что жадный отбор не видит структурно.

Зачем отдельно от library_sweep.py. Тот идёт жадно: сначала лучший одиночка, потом
лучшая добавка к нему. Пара, у которой КАЖДЫЙ член поодиночке даёт ноль, для жадного
невидима в принципе — первый шаг её просто не выберет. А это ровно тот механизм, что у
нас уже оплачен измерением: lagd28 + hz_v1_surv дали +0.000135 при нулях по отдельности,
kostya46 в наборе +0.000413 против +0.000155 соло. Пар всего C(n,2), по 18 мс каждая,
так что перебрать их дешевле, чем рассуждать о них.

ЧЕСТНОСТЬ ПРИ 5000 ПРОВЕРОК. Выбирая максимум из тысяч зашумлённых оценок, сравнивать
его с полом ОДНОГО случайного набора нельзя: при шуме 2.2e-05 максимум из 2000 бросков
даёт около +7e-05 сам по себе. Поэтому три защиты сразу:

  1. Отбор живёт на DEV, честное число берётся с EVAL, которого отбор не видел. Само
     eval-число для ОДНОГО выбранного набора несмещённое.
  2. Два РАЗНЫХ пола, потому что вопроса тоже два.
     (а) Плацебо (правило 4): 95-й перцентиль eval-выигрышей по случайным наборам той же
         ёмкости. С ним сравнивается dev-победитель. Множественность его НЕ раздувает:
         eval в отборе не участвовал, поэтому его число несмещённое для того набора,
         который выбран, сколько бы наборов ни просмотрели.
     (б) Цена нечестного скана: максимум по eval среди всех просмотренных наборов. Это
         то, что показал бы скан, если отбирать по eval, — печатается отдельно, чтобы
         был виден размер ловушки, но НЕ используется как порог для (а). Ранняя версия
         этого скрипта сравнивала (а) с бутстрапом максимума из тех же eval-выигрышей —
         так пол получался равен наблюдаемому максимуму, и вердикт «не бьёт» выходил
         тавтологически.
  3. Ранговый перенос: корреляция Спирмена между dev- и eval-выигрышами по ВСЕМ парам.
     Если структуры пар нет вовсе, она около нуля, и тогда любой победитель — шум,
     как бы красиво ни выглядело его eval-число. Это тест на существование сигнала,
     не зависящий от того, кто именно победил.

Кандидаты, эталон и стражи контаминации берутся из library_sweep.py — одни и те же
правила, чтобы числа были сравнимы между инструментами.

Запуск:
  python work/scripts/pair_scan.py                        # пары по колонкам пакета
  python work/scripts/pair_scan.py --triples 50           # + тройки поверх топ-50 пар
  python work/scripts/pair_scan.py --whitelist all        # зоопарк, ТОЛЬКО диагностика
"""
from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))
from common import PREDS_DIR, REPORTS_DIR, ROOT
from margin import calibrate_split, score
from library_sweep import (BLEND_PREFIXES, CONTAM_CEIL, CONTAMINATED, SANITY_FLOOR,
                           fit_eval, inner_gain, nnls_weights)


def load_library(pack_dir: Path, whitelist: str, include: str, seed: int, allow_dirty: bool):
    """Кандидаты по правилам library_sweep: те же стражи, тот же эталон, тот же разрез."""
    pack = pl.read_parquet(pack_dir / "val_preds.parquet").sort("user_id")
    uid = pack["user_id"].to_numpy()
    ly = np.log1p(np.clip(pack["target"].to_numpy().astype(np.float64), 0, None))
    lb = pack["blend"].to_numpy().astype(np.float64)
    rng = np.random.default_rng(seed)
    dev = rng.permutation(len(ly)) < len(ly) // 2
    chk = np.random.default_rng(seed + 999).permutation(len(ly)) < len(ly) // 2

    def pair_gain(lp):
        A = np.column_stack([lb, lp])
        gs = []
        for m in (chk, ~chk):
            w = nnls_weights(A[m], ly[m])
            gs.append(score(lb[~m], ly[~m]) - score(A[~m] @ w, ly[~m]))
        return float(np.mean(gs))

    allowed = None
    if whitelist == "pack":
        allowed = set(pack.columns) - {"user_id", "target", "blend"}
        allowed |= {n for n in include.split(",") if n}

    sb_full = score(lb, ly)
    names, cols, skipped = [], [], 0
    for f in sorted(PREDS_DIR.glob("*_val.parquet")):
        n = f.name[: -len("_val.parquet")]
        if n == "blend":
            continue
        if allowed is not None:
            if n not in allowed:
                continue
        elif n.startswith(BLEND_PREFIXES):
            skipped += 1
            continue
        if n in CONTAMINATED and not allow_dirty:
            skipped += 1
            continue
        d = pl.read_parquet(f).sort("user_id")
        if "pred" not in d.columns or d.height != len(uid) \
           or not np.array_equal(d["user_id"].to_numpy(), uid):
            skipped += 1
            continue
        lp = calibrate_split(
            np.log1p(np.clip(d["pred"].to_numpy().astype(np.float64), 0, None)), ly, dev, 24)
        s = score(lp, ly)
        if s < SANITY_FLOOR or s < sb_full + 0.001 or pair_gain(lp) > CONTAM_CEIL:
            skipped += 1
            continue
        names.append(n)
        cols.append(lp)
    return names, np.column_stack(cols), lb, ly, dev, sb_full, skipped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", type=Path, default=ROOT / "work" / "preds_pack")
    ap.add_argument("--whitelist", default="pack", choices=["pack", "all"])
    ap.add_argument("--include", default="")
    ap.add_argument("--triples", type=int, default=0,
                    help="строить тройки поверх топ-N пар (0 = не строить)")
    ap.add_argument("--boot", type=int, default=2000, help="бутстрап-повторов для пола максимума")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--allow-dirty", action="store_true")
    ap.add_argument("--save-all", action="store_true",
                    help="положить в json все просмотренные наборы (переанализ без пересчёта)")
    ap.add_argument("--json", default="pair_scan.json")
    args = ap.parse_args()

    names, X, lb, ly, dev, sb_full, skipped = load_library(
        args.pack, args.whitelist, args.include, args.seed, args.allow_dirty)
    ev = ~dev
    print(f"эталон: бленд, скор {sb_full:.6f}, n={len(ly)}")
    print(f"кандидатов: {len(names)} (отсеяно {skipped})\n")

    bd, be, yd, ye = lb[dev], lb[ev], ly[dev], ly[ev]
    Xd, Xe = X[dev], X[ev]
    sb_ev = score(be, ye)
    rng = np.random.default_rng(args.seed)
    half = rng.permutation(dev.sum()) < dev.sum() // 2

    def evaluate(idx):
        """(выигрыш на dev — для отбора, выигрыш на eval — честный)."""
        Ad = np.column_stack([bd] + [Xd[:, i] for i in idx])
        Ae = np.column_stack([be] + [Xe[:, i] for i in idx])
        return inner_gain(Ad, yd, half), fit_eval(Ad, yd, Ae, ye, sb_ev)[0]

    solo = {j: evaluate([j]) for j in range(len(names))}

    sets = list(combinations(range(len(names)), 2))
    print(f"--- перебор всех {len(sets)} пар ---")
    res = {s: evaluate(list(s)) for s in sets}

    if args.triples:
        top = sorted(res, key=lambda s: -res[s][0])[: args.triples]
        tri = {tuple(sorted(set(s) | {j})) for s in top for j in range(len(names)) if j not in s}
        print(f"--- перебор {len(tri)} троек поверх топ-{args.triples} пар ---")
        res.update({t: evaluate(list(t)) for t in tri})

    devs = np.array([res[s][0] for s in res])
    evs = np.array([res[s][1] for s in res])
    keys = list(res)

    # (3) существует ли переносимая структура наборов вообще
    from scipy.stats import spearmanr
    rho, pv = spearmanr(devs, evs)

    # (1) победитель отобран ТОЛЬКО по dev
    win = keys[int(np.argmax(devs))]
    g_dev, g_ev = res[win]

    # (2а) плацебо той же ёмкости: случайные наборы размера |win|, честное eval-число
    k_win = len(win)
    same_k = np.array([res[s][1] for s in keys if len(s) == k_win])
    floor_placebo = float(np.percentile(same_k, 95))
    # (2б) цена нечестного скана: максимум по eval среди всех просмотренных
    naive_max = float(evs.max())
    naive_set = keys[int(np.argmax(evs))]

    print(f"\n--- ПЕРЕНОС СТРУКТУРЫ (существует ли сигнал наборов) ---")
    print(f"Спирмен dev vs eval по {len(keys)} наборам: rho={rho:+.4f}  p={pv:.2e}")

    print(f"\n--- ПОБЕДИТЕЛЬ (отобран по dev, честное число с eval) ---")
    print(f"{'+'.join(names[i] for i in win)}")
    print(f"  отбор(dev)={g_dev:+.6f}   ЧЕСТНО(eval)={g_ev:+.6f}")
    for i in win:
        print(f"    поодиночке {names[i]:<24} dev={solo[i][0]:+.6f}  eval={solo[i][1]:+.6f}")
    print(f"  плацебо: 95-й перцентиль по {len(same_k)} наборам ёмкости {k_win} = {floor_placebo:+.6f}")
    print(f"  ВЕРДИКТ: {'БЬЁТ плацебо' if g_ev > floor_placebo else 'не бьёт плацебо'}")
    print(f"\n  [цена нечестного скана] лучший ПО EVAL: "
          f"{'+'.join(names[i] for i in naive_set)} = {naive_max:+.6f}")
    print(f"  Столько наврал бы отбор по eval; честное число выше — {g_ev:+.6f}.")

    # то, ради чего всё затевалось: набор жив, а его члены поодиночке мертвы
    print(f"\n--- НАБОРЫ, ЖИВЫЕ ПРИ МЁРТВЫХ ЧЛЕНАХ (топ-10 по eval) ---")
    emerge = [(s, res[s][1]) for s in keys if all(solo[i][1] <= 0.00002 for i in s)]
    emerge.sort(key=lambda x: -x[1])
    if not emerge:
        print("  таких наборов нет")
    for s, g in emerge[:10]:
        solos = ", ".join(f"{names[i]} {solo[i][1]:+.6f}" for i in s)
        mark = "  <-- выше плацебо" if g > floor_placebo else ""
        print(f"  {g:+.6f}  [{solos}]{mark}")

    out = {"blend_rmsle": sb_full, "n_candidates": len(names), "n_sets": len(keys),
           "spearman_dev_eval": float(rho), "spearman_p": float(pv),
           "winner": {"members": [names[i] for i in win], "gain_dev": g_dev, "gain_eval": g_ev,
                      "solo_eval": {names[i]: solo[i][1] for i in win}},
           "floor_placebo_p95": floor_placebo, "k_winner": k_win,
           "naive_eval_max": naive_max,
           "naive_eval_set": [names[i] for i in naive_set],
           "beats_floor": bool(g_ev > floor_placebo),
           "emergent_top": [{"members": [names[i] for i in s], "gain_eval": g}
                            for s, g in emerge[:20]]}
    if args.save_all:
        out["all_sets"] = [{"members": [names[i] for i in s],
                            "gain_dev": res[s][0], "gain_eval": res[s][1]} for s in keys]
    p = REPORTS_DIR / args.json
    p.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(f"\nJSON: {p}")


if __name__ == "__main__":
    main()

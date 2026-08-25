"""train_weak.py — НАМЕРЕННО ОБЕДНЁННЫЕ модели для разнообразия ансамбля.

Зачем. Ансамбль насыщен: улучшение члена пула на 0.0005 двигает бленд на 0.0001
(KNOWLEDGE, четыре независимых подтверждения). Дважды измерено, что УЛУЧШЕНИЕ
модели убивает её вклад: febspec_cal (слабая, 3 среза, 53 коротких признака,
скор 1.83) держит тест-вес 0.054-0.063 при частоте отбора 0.99 и корреляции с
ближайшим соседом 0.9691; его улучшенные версии (скор 1.771) получают тест-вес
0.000-0.013 при корреляции с соседом 0.9943-0.9989. Ценно УНИКАЛЬНОЕ НАПРАВЛЕНИЕ,
а не качество.

Отсюда гипотеза, которую проверяет этот скрипт: модели, обеднённые НАМЕРЕННО и
РАЗНЫМИ способами, дают в сумме то, чего не даёт ни одна сильная модель.

Четыре механизма обеднения (--mech):

  subspace  случайное подпространство признаков: модель видит случайные
            15-25% из 203 признаков (--frac, --sel-seed).
  anchors   случайное подмножество обучающих срезов: 2-4 случайных среза
            вместо 14 (--k-anchors, --sel-seed).
  ftype     ограничение по ТИПУ признака (--ftype):
              short14  только окна до 14 дней (+ снимок последнего дня)
              long90   только окна от 90 дней (включая прошлогодние)
              recency  только давность/интервалы/стаж
              counts   только счётчики и дни, ни одного рубля
  tiny      без обеднения признаков — обеднение ЁМКОСТЬЮ: мелкие деревья,
            мало итераций, огромный min_data_in_leaf (задаётся --params).

Конфиг, протокол, ретрейн и сохранение делегируются train_gbdt.py, поэтому
скрипт не дублирует ни строки обучающего кода (тот же приём, что в
train_behavonly.py). По умолчанию — чемпионский tweedie-on-log и --gap-days 30.

Примеры:
  train_weak.py --name weak_rs_a --mech subspace --frac 0.20 --sel-seed 101
  train_weak.py --name weak_an_b --mech anchors --k-anchors 3 --sel-seed 22
  train_weak.py --name weak_ft_short14 --mech ftype --ftype short14
  train_weak.py --name weak_tiny_a --mech tiny \
      --params '{"objective":"tweedie","tweedie_variance_power":1.45,
                 "n_estimators":300,"num_leaves":15,"min_data_in_leaf":5000}'
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from common import VAL_ANCHOR, feature_cols, load_anchor  # noqa: E402

CHAMPION = ['--model', 'lgb', '--objective', 'log_mse', '--params',
            '{"objective":"tweedie","tweedie_variance_power":1.45,"n_estimators":6000}']

FTYPES = ("short14", "long90", "recency", "counts")


def ftype_keep(cols: list[str], kind: str) -> list[str]:
    """Признаки одного типа. Всё остальное дропается — это и есть обеднение."""
    s = set(cols)
    order = {c: i for i, c in enumerate(cols)}
    if kind == "short14":
        keep = [c for c in cols if re.search(r"_(1|3|7|14)$", c)]
        keep += [c for c in ("dec_gmv_h7", "last_day_gmv", "last_day_ord",
                             "last_day_cart", "last_day_searches") if c in s]
    elif kind == "long90":
        keep = [c for c in cols if re.search(
            r"(_90|_180|_365|_b90_179|_b180_364|_ya_tgt|_ya_wide|_full)$", c)]
        keep += [c for c in ("dec_gmv_h120", "gmv_ya_t1", "ordd_ya_t1", "gmv_ya_t2",
                             "ordd_ya_t2", "gmv_ya_t3", "ordd_ya_t3", "ya_cov_ya_tgt",
                             "ya_cov_ya_wide", "tenure", "history_days") if c in s]
    elif kind == "recency":
        keep = [c for c in cols if c.startswith("rec_") or "gap" in c]
        keep += [c for c in ("tenure", "history_days", "act_density", "burstiness",
                             "btyd_recency", "btyd_T", "btyd_p_alive",
                             "rk_rec_order") if c in s]
    elif kind == "counts":
        keep = [c for c in cols
                if ("cnt" in c or "days" in c or "searches" in c)
                and "gmv" not in c and not re.match(r"^hv\d+_days_", c)]
    else:
        raise SystemExit(f"unknown --ftype {kind!r}, expected one of {FTYPES}")
    return sorted(set(keep), key=lambda c: order[c])


def pick_anchors(k: int, seed: int, pool_last: int, gap: int = 30,
                 source: str = "protocol") -> list:
    """k случайных обучающих срезов из ЧИСТОГО пула (зазор gap дней до val).

    Пул чистится по зазору ЗДЕСЬ, чтобы k был честным числом обучающих срезов:
    иначе train_gbdt отфильтровал бы часть выбранных срезов как gap-срезы и вернул
    бы их только на ретрейне, и модель обучалась бы на k−2 срезах вместо k.
    """
    import os
    from datetime import timedelta

    from common import FEATURES_DIR
    from exp_lib import protocol_train_anchors

    def tiers_ok(a) -> bool:
        """Срезы фев-2025 не имеют тира v2 (.extra) — данные начинаются 2025-01-01.
        Без этой проверки load_matrix падает на ColumnNotFoundError уже ПОСЛЕ
        обучения предыдущих моделей задания."""
        need = [t for flag, t in (("USE_V2", "extra"), ("USE_V3", "v3"), ("USE_V4", "v4"))
                if os.environ.get(flag)]
        return all((FEATURES_DIR / f"anchor={a.isoformat()}.{t}.parquet").exists()
                   for t in need)

    cut = VAL_ANCHOR - timedelta(days=gap)
    # ПУЛ ПО ПРОТОКОЛУ, а не по каталогу. В таблице ретрейна 25.08 weak_an_d разошёлся
    # сильнее всех (сырой скор 1.683269 против 1.695393, Δ бленда +0.000093) именно из-за
    # этой строки: sel_seed выбирает k якорей ИЗ ПУЛА, а пул был размером с каталог, и на
    # другой машине выбиралась другая четвёрка — то есть обучалась другая модель.
    allow = [a for a in protocol_train_anchors(source=source) if a <= cut and tiers_ok(a)]
    if pool_last:
        allow = allow[-pool_last:]
    rng = np.random.default_rng(seed)
    idx = sorted(rng.choice(len(allow), size=min(k, len(allow)), replace=False).tolist())
    return [allow[i] for i in idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--mech", required=True,
                    choices=["subspace", "anchors", "ftype", "tiny"])
    ap.add_argument("--frac", type=float, default=0.20,
                    help="subspace: доля из 203 признаков (0.15..0.25)")
    ap.add_argument("--anchor-source", choices=["protocol", "disk"], default="protocol",
                    help="откуда берётся ПУЛ обучающих срезов: protocol — train_anchors(14), "
                         "не зависит от каталога (умолчание); disk — историческое поведение")
    ap.add_argument("--sel-seed", type=int, default=0,
                    help="сид ОТБОРА (подпространства / срезов); не сид обучения")
    ap.add_argument("--k-anchors", type=int, default=3, help="anchors: сколько срезов")
    ap.add_argument("--anchor-pool", type=int, default=0,
                    help="anchors: брать из последних N доступных срезов (0 = все)")
    ap.add_argument("--ftype", type=str, default="", choices=list(FTYPES) + [""])
    ap.add_argument("--dry-run", action="store_true")
    args, rest = ap.parse_known_args()

    cols = feature_cols(load_anchor(VAL_ANCHOR))
    keep, drop = cols, []
    sub_anchors = None

    if args.mech == "subspace":
        rng = np.random.default_rng(args.sel_seed)
        n = max(5, int(round(args.frac * len(cols))))
        idx = sorted(rng.choice(len(cols), size=n, replace=False).tolist())
        keep = [cols[i] for i in idx]
    elif args.mech == "ftype":
        if not args.ftype:
            raise SystemExit("--mech ftype requires --ftype")
        keep = ftype_keep(cols, args.ftype)
    elif args.mech == "anchors":
        sub_anchors = pick_anchors(args.k_anchors, args.sel_seed, args.anchor_pool,
                                   source=args.anchor_source)

    drop = [c for c in cols if c not in set(keep)]
    print(f"[weak] {args.name}: mech={args.mech} keep={len(keep)}/{len(cols)} feats", flush=True)
    print("[weak] kept:", ",".join(keep), flush=True)
    if sub_anchors is not None:
        print(f"[weak] anchors ({len(sub_anchors)}): "
              f"{[a.isoformat() for a in sub_anchors]}", flush=True)
    if args.dry_run:
        return

    argv = list(rest)
    if "--name" not in argv:
        argv = ["--name", args.name] + argv
    if "--gap-days" not in argv:
        argv += ["--gap-days", "30"]
    if not any(a in argv for a in ("--model", "--objective", "--params")):
        argv += CHAMPION
    if drop:
        argv += ["--drop-cols", ",".join(drop)]
    if "--notes" not in argv:
        note = (f"weak/{args.mech}"
                + (f" frac={args.frac} sel_seed={args.sel_seed}" if args.mech == "subspace" else "")
                + (f" ftype={args.ftype}" if args.mech == "ftype" else "")
                + (f" k={args.k_anchors} sel_seed={args.sel_seed}" if args.mech == "anchors" else "")
                + f"; {len(keep)} feats")
        argv += ["--notes", note]

    argv += ["--anchor-source", args.anchor_source]

    import train_gbdt
    if sub_anchors is not None:
        # обеднение по срезам: подменяем источник срезов И для обучения, И для
        # gap-фазы ретрейна, чтобы модель никогда не увидела остальные срезы
        train_gbdt.anchor_pool = lambda _a=sub_anchors: list(_a)
    sys.argv = ["train_gbdt.py"] + argv
    train_gbdt.main()


if __name__ == "__main__":
    main()

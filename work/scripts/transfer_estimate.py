"""Дозирование новой оси без зонда: сколько вал-оптимального шага нести на тест.

ЧТО ВЫЯСНИЛОСЬ (transfer_law.py, 8 замеренных осей). Предиктора каппы из вал-наблюдаемых
свойств НЕТ. Проверено три кандидата, все мимо:

  концентрация выигрыша  ОПРОВЕРГНУТА на контролируемой паре: ridge_v1 и cnt_rank — один
                         класс, один протокол; менее концентрированный cnt_rank при
                         БОЛЬШЕМ вал-выигрыше перенёсся ХУЖЕ (kappa -0.08 против +0.31)
  вал-выигрыш            rho=+0.40, 95% [-1.00, +1.00] — точек 4
  класс механизма        LOO-MAE 0.229, ХУЖЕ общего среднего 0.214; внутри класса «стек
                         по признакам» разброс каппы 0.388 при двух точках

Ни один вариант не дотягивает до порога 0.15. Поэтому этот скрипт НЕ притворяется, что
предсказывает каппу. Он делает то, что на этих данных обосновано: считает вал-профиль
кандидата и выдаёт ОПТИМАЛЬНУЮ СЛЕПУЮ ДОЗУ вместе с ценой отказа от зонда.

АЛГЕБРА ДОЗЫ. Шаг d вдоль оси с истинной каппой даёт выигрыш (2*d*kappa - d^2)*Q, где
Q = mean(шаг^2) — то же Q, что в probes_5.md. Если каппа неизвестна и берётся из
эмпирического распределения замеренных осей, ожидаемый выигрыш равен
(2*d*E[kappa] - d^2)*Q и максимален при

    d* = E[kappa]                          <- слепая доза
    ожидаемый выигрыш = E[kappa]^2 * Q     <- что достанется без зонда
    выигрыш при знании каппы = E[kappa^2] * Q

Отношение E[kappa^2]/E[kappa]^2 и есть ЦЕНА ЗОНДА в чистом виде: во столько раз зонд
увеличивает ожидаемый сбор с оси. Оно велико именно потому, что каппа непредсказуема.

Запуск:
  python work/scripts/transfer_estimate.py --cand work/preds/foo_val.parquet
  python work/scripts/transfer_estimate.py --dose-only                      # только правило
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))
from common import REPORTS_DIR, ROOT
from margin import score

LAW = REPORTS_DIR / "transfer_law.json"
N_USERS = 250000


def kappa_population(tight_only: bool = True):
    """Эмпирическое распределение каппы по замеренным осям (единственная опора прогноза)."""
    d = json.loads(LAW.read_text())
    ks = [p["kappa"] for p in d["points"] if not tight_only or p["sigma"] <= 0.10]
    k = np.array(ks, float)
    return k


def read_lp(path: Path, uid_ref=None):
    """log1p прогноза из parquet (колонка pred) или csv сабмита."""
    if path.suffix == ".csv":
        df = pl.read_csv(path)
        col = [c for c in df.columns if c != "user_id"][0]
    else:
        df = pl.read_parquet(path)
        col = "pred" if "pred" in df.columns else [c for c in df.columns if c != "user_id"][0]
    df = df.sort("user_id")
    uid = df["user_id"].to_numpy()
    if uid_ref is not None and not np.array_equal(uid, uid_ref):
        raise SystemExit(f"порядок user_id в {path.name} не совпал с эталоном")
    return uid, np.log1p(np.clip(df[col].to_numpy().astype(np.float64), 0, None))


def conc(step: np.ndarray, eb: np.ndarray):
    """Концентрация вклада шага в MSE-выигрыш: доля топ-1% и топ-0.1% юзеров.

    Методика night_corrector_variants.py. Доля может быть > 1: часть юзеров вносит
    отрицательный вклад, и топ перекрывает суммарный нетто-выигрыш.
    """
    d = eb ** 2 - (eb + step) ** 2
    tot = float(d.sum())
    if tot <= 0:
        return float("nan"), float("nan")
    ds = np.sort(d)[::-1]
    n = len(d)
    return float(ds[:int(0.01 * n)].sum() / tot), float(ds[:int(0.001 * n)].sum() / tot)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cand", type=Path, help="кандидат: parquet с колонкой pred или csv сабмита")
    ap.add_argument("--base", type=Path, help="база; без неё шагом считается (кандидат − бленд)")
    ap.add_argument("--dose-only", action="store_true", help="только правило дозирования")
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    k = kappa_population()
    Ek, Ek2 = float(k.mean()), float((k ** 2).mean())
    lo, hi = np.percentile(k, [10, 90])
    out = {"kappa_population": k.tolist(), "E_kappa": Ek, "E_kappa2": Ek2,
           "dose_blind": Ek, "probe_value_ratio": Ek2 / max(Ek ** 2, 1e-12),
           "kappa_range_10_90": [float(lo), float(hi)]}

    print("--- ПРАВИЛО ДОЗИРОВАНИЯ (закона переноса нет, см. докстринг) ---")
    print(f"замеренных осей: {len(k)};  каппа от {k.min():+.3f} до {k.max():+.3f}")
    print(f"E[kappa] = {Ek:+.3f}   E[kappa^2] = {Ek2:.4f}   разброс 10-90% "
          f"[{lo:+.3f}, {hi:+.3f}]")
    print(f"\nСЛЕПАЯ ДОЗА для новой оси: d* = {Ek:.2f} от вал-оптимального шага.")
    print(f"Ожидаемый выигрыш без зонда: {Ek**2:.4f}*Q")
    print(f"Ожидаемый выигрыш с зондом:  {Ek2:.4f}*Q")
    print(f"ЦЕНА ЗОНДА: он увеличивает ожидаемый сбор с оси в {Ek2/Ek**2:.2f} раза.")
    print("Читается так: доза 0.2 вслепую собирает меньше половины того, что собрал бы")
    print("зонд, и это не лечится вал-наблюдаемыми — каппа из них не выводится.")

    if args.dose_only or not args.cand:
        if args.json:
            (REPORTS_DIR / args.json).write_text(json.dumps(out, indent=1, ensure_ascii=False))
        return

    pack = pl.read_parquet(ROOT / "work" / "preds_pack" / "val_preds.parquet").sort("user_id")
    uid = pack["user_id"].to_numpy()
    ly = np.log1p(np.clip(pack["target"].to_numpy().astype(np.float64), 0, None))
    lb = pack["blend"].to_numpy().astype(np.float64)
    eb = lb - ly
    sb = score(lb, ly)

    _, lc = read_lp(args.cand, uid)
    base = lb if args.base is None else read_lp(args.base, uid)[1]
    step = lc - base

    Q = float(np.mean(step ** 2))
    val_gain = sb - score(base + step, ly)
    t1, t01 = conc(step, eb)
    d = eb ** 2 - (eb + step) ** 2
    pos = d[d > 0]
    n_eff = float(pos.sum() ** 2 / np.sum(pos ** 2)) if len(pos) else float("nan")
    rho_b = float(np.corrcoef(step, lb)[0, 1])

    print(f"\n--- ВАЛ-ПРОФИЛЬ КАНДИДАТА {args.cand.name} ---")
    print(f"  вал-выигрыш шага      {val_gain:+.6f}")
    print(f"  Q = mean(шаг^2)       {Q:.8f}")
    if np.isnan(t1):
        print(f"  концентрация          не определена: нетто-выигрыш шага <= 0, делить не на что.")
        print(f"                        Инструмент рассчитан на ПОПРАВКУ поверх базы, а не на")
        print(f"                        подмену базы моделью — для второго считайте joint_gain.py.")
    else:
        print(f"  концентрация топ-1%   {t1:.2f}   топ-0.1% {t01:.3f}")
    print(f"  эфф. носителей n_eff  {n_eff:.0f}  (N/n_eff = {N_USERS/max(n_eff,1):.1f})")
    print(f"  корреляция с блендом  {rho_b:+.4f}")
    print(f"\n  ПРОГНОЗ КАППЫ: {Ek:+.3f}, интервал 10-90% [{lo:+.3f}, {hi:+.3f}].")
    print(f"  Интервал НЕ сужается профилем выше — ни одно из этих свойств каппу не")
    print(f"  предсказывает (проверено на 8 осях). Профиль печатается для протокола.")
    print(f"\n  Рекомендуемый шаг вслепую: {Ek:.2f} * вал-оптимум, ожидание "
          f"{Ek**2*Q/(2*sb):.7f} RMSLE.")
    print(f"  С зондом ожидание {Ek2*Q/(2*sb):.7f} — если ось стоит попытки, зонд окупается.")

    out.update({"candidate": str(args.cand), "val_gain": val_gain, "Q": Q,
                "conc_top1": t1, "conc_top01": t01, "n_eff": n_eff, "corr_blend": rho_b})
    if args.json:
        (REPORTS_DIR / args.json).write_text(json.dumps(out, indent=1, ensure_ascii=False))
        print(f"\nJSON: {REPORTS_DIR / args.json}")


if __name__ == "__main__":
    main()

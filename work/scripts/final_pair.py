"""Выбор двух финалистов: σ_d меряется, а не предполагается.

ЗАЧЕМ. В зачёт идёт ЛУЧШИЙ из двух приватных скоров, поэтому ценность второго слота


где D — насколько второй файл хуже по ожиданию, σ_d — разброс РАЗНИЦЫ приватных скоров
двух файлов. Часть E Жени вывела этот функционал и правильный вывод «непохожесть важнее
отставания», но σ_d там НЕ ИЗМЕРЕНА: взята сетка предположений 0.0002 / 0.0005 / 0.0010 /
0.0020, и рекомендация построена на строке 0.0010.

Между тем σ_d ровно один раз замерена по-настоящему — night_blend_stability.json, пара
«старый бленд 1.6663022 против нового 1.6656470», полная пересборка состава:

    split_sim  priv200k  σ_d = 5.05e-05
    bootstrap  n200k     σ_d = 1.11e-04

Это в 2-20 раз меньше сетки Жени, а функционал в этой области крайне чувствителен к σ_d.
Поэтому здесь σ_d КАЛИБРУЕТСЯ по реальным векторам: как она зависит от непохожести пары.

ДВЕ СХЕМЫ ПЕРЕСЭМПЛИРОВАНИЯ, И ОНИ МЕРЯЮТ РАЗНОЕ:
  split  — приват как случайные 200k из наших 250k. Это НАША ситуация: пользователи
           фиксированы, случаен только раскол на публику/приват.
  boot   — 200k с возвращением, то есть «другая выборка пользователей из популяции».
Для выбора финалистов верна SPLIT: юзеры уже даны, разыгрывается только раскол.

Режимы:
  --calibrate            (локально) проверка на известной паре + кривая σ_d(непохожесть)
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import polars as pl
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).parent))
from common import REPORTS_DIR, ROOT

N_PRIV = 200_000          # приватная часть
B_DEFAULT = 4000
OLD_PACK_COMMIT = "pack-old"   # пак со старым блендом 1.6663022
NEW_PACK_COMMIT = "pack-new"   # пак с новым 1.6656470
NOISE = 0.000022


def sigma_d(lp_a: np.ndarray, lp_b: np.ndarray, ly: np.ndarray, *,
            scheme: str = "split", B: int = B_DEFAULT, seed: int = 20260823):
    """Разброс разницы приватных скоров двух файлов.

    Считается на повользовательских квадратах ошибки: скор подвыборки — корень из их
    среднего, поэтому пересэмплирование стоит одно усреднение и B=4000 идёт секунды.
    """
    ea2 = (lp_a - ly) ** 2
    eb2 = (lp_b - ly) ** 2
    n = len(ly)
    rng = np.random.default_rng(seed)
    out = np.empty(B)
    for i in range(B):
        idx = (rng.permutation(n)[:N_PRIV] if scheme == "split"
               else rng.integers(0, n, N_PRIV))
        out[i] = np.sqrt(ea2[idx].mean()) - np.sqrt(eb2[idx].mean())
    return float(out.std()), float(out.mean())


def slot_value(D: float, sd: float) -> float:
    """Ценность второго слота. D>0 — второй файл хуже по ожиданию."""
    if sd <= 0:
        return 0.0
    z = D / sd
    return float(sd * norm.pdf(z) - D * norm.cdf(-z))


def p_inversion(D: float, sd: float) -> float:
    """Вероятность, что второй файл окажется на привате ЛУЧШЕ первого."""
    return float(norm.cdf(-D / sd)) if sd > 0 else 0.0


def _pack_from_commit(commit: str, path: Path) -> pl.DataFrame:
    blob = subprocess.run(["git", "show", f"{commit}:work/preds_pack/val_preds.parquet"],
                          capture_output=True, cwd=ROOT).stdout
    path.write_bytes(blob)
    return pl.read_parquet(path).sort("user_id")


def calibrate(scratch: Path, B: int):
    new = _pack_from_commit(NEW_PACK_COMMIT, scratch / "pack_new.parquet")
    old = _pack_from_commit(OLD_PACK_COMMIT, scratch / "pack_old.parquet")
    ly = np.log1p(np.clip(new["target"].to_numpy().astype(np.float64), 0, None))
    b_new = new["blend"].to_numpy().astype(np.float64)
    b_old = old["blend"].to_numpy().astype(np.float64)

    print("--- ПРОВЕРКА НА ИЗВЕСТНОЙ ПАРЕ (пересборка состава бленда) ---")
    rms = float(np.sqrt(np.mean((b_new - b_old) ** 2)))
    print(f"скоры: старый {np.sqrt(((b_old-ly)**2).mean()):.7f}  "
          f"новый {np.sqrt(((b_new-ly)**2).mean()):.7f}   непохожесть rms {rms:.5f}")
    ref = {}
    for scheme, known in (("split", 5.0511e-05), ("boot", 1.1100e-04)):
        sd, mu = sigma_d(b_new, b_old, ly, scheme=scheme, B=B)
        ref[scheme] = sd
        print(f"  {scheme:6} σ_d = {sd:.3e}   в отчёте {known:.3e}   "
              f"отношение {sd/known:.2f}")

    print("\n--- КАЛИБРОВКА σ_d ПО НЕПОХОЖЕСТИ ---")
    print("пары строятся как бленд + a*(модель − бленд): a задаёт непохожесть.")
    donors = ["kostya46_cal", "fusion_v3c_avg_cal", "febspec_cal"]
    rows = []
    for dn in donors:
        if dn not in new.columns:
            continue
        step = new[dn].to_numpy().astype(np.float64) - b_new
        for a in (0.02, 0.05, 0.1, 0.25, 0.5):
            cand = b_new + a * step
            r = float(np.sqrt(np.mean((cand - b_new) ** 2)))
            sd, _ = sigma_d(cand, b_new, ly, scheme="split", B=max(600, B // 6))
            rows.append((dn, a, r, sd))
    rows.sort(key=lambda z: z[2])
    print(f"{'донор':<20}{'a':>6}{'rms':>10}{'σ_d(split)':>13}{'σ_d/rms':>10}")
    for dn, a, r, sd in rows:
        print(f"{dn:<20}{a:>6.2f}{r:>10.5f}{sd:>13.3e}{sd/r:>10.5f}")

    ratios = np.array([sd / r for _, _, r, sd in rows])
    k = float(np.median(ratios))
    print(f"\nσ_d ≈ {k:.4f} · rms(разница файлов)   (медиана отношения, разброс "
          f"{ratios.min():.4f}-{ratios.max():.4f})")
    print(f"проверка на известной паре: rms {rms:.5f} -> предсказано "
          f"{k*rms:.3e}, замерено {ref['split']:.3e}")

    out = {"known_pair": {"rms": rms, **{f"sigma_d_{s}": v for s, v in ref.items()}},
           "calibration": [{"donor": d, "a": a, "rms": r, "sigma_d": sd} for d, a, r, sd in rows],
           "k_sigma_per_rms": k}
    (REPORTS_DIR / "final_pair_calibration.json").write_text(
        json.dumps(out, indent=1, ensure_ascii=False))
    return k, ref["split"], rms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--pair", nargs=2, metavar=("A", "B"))
    ap.add_argument("--scores", nargs=2, type=float, metavar=("SA", "SB"),
                    help="публичные скоры пары, если известны (для D)")
    ap.add_argument("-B", type=int, default=B_DEFAULT)
    args = ap.parse_args()

    scratch = Path("/tmp/claude-1000/-home-olya-ozon-cup/"
                   "3410b186-4f1d-4097-b46c-9fd918faacc8/scratchpad")
    scratch.mkdir(parents=True, exist_ok=True)

    if args.calibrate or not args.pair:
        k, sd_known, rms_known = calibrate(scratch, args.B)
        print("\n--- ЧТО ЭТО ЗНАЧИТ ДЛЯ ВЫБОРА ПАРЫ ---")
        print(f"{'σ_d':>10}{'D=0':>11}{'0.0002':>11}{'0.0005':>11}{'0.0010':>11}")
        for sd in (sd_known, 2 * sd_known, 5 * sd_known, 0.0010):
            tag = f"{sd:.2e}"
            print(f"{tag:>10}" + "".join(f"{slot_value(D, sd):>11.6f}"
                                         for D in (0, 0.0002, 0.0005, 0.0010)))
        print(f"\nпорог осмысленности — шум замера {NOISE:.6f}")

        print("\n--- РЕШАЮЩАЯ ТАБЛИЦА: какой должна быть непохожесть, чтобы слот окупился ---")
        print("(нужно, чтобы ценность слота превысила шум замера 0.000022)")
        print(f"{'отставание D':>14}{'нужен σ_d':>13}{'нужен rms':>12}   что это значит")
        from scipy.optimize import brentq
        for D in (0.0, 0.00005, 0.0001, 0.0002, 0.0005, 0.00074):
            f = lambda sd: slot_value(D, sd) - NOISE
            hi = 0.05
            if f(hi) <= 0:
                print(f"{D:>14.5f}{'—':>13}{'—':>12}   недостижимо")
                continue
            sd_req = brentq(f, 1e-8, hi)
            rms_req = sd_req / k
            note = ("та же величина, что полная пересборка бленда" if 0.03 < rms_req < 0.06
                    else "больше любой мыслимой пары финалистов" if rms_req > 0.2
                    else "")
            print(f"{D:>14.5f}{sd_req:>13.2e}{rms_req:>12.4f}   {note}")
        print(f"\nДЛЯ СРАВНЕНИЯ: полная пересборка состава бленда дала rms {rms_known:.4f}.")
        print("Два финалиста, отличающиеся цепочкой поправок, лежат СИЛЬНО ниже этого.")
        return

    print("режим пары требует самих файлов сабмитов (submissions/ здесь нет).")


if __name__ == "__main__":
    main()

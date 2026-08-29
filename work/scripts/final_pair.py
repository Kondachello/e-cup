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
  --calibrate            (локально) проверка на известной паре + кривая σ_d(непохожесть).
                         ВНИМАНИЕ: перезаписывает work/reports/final_pair_calibration.json
                         (этот же режим включается, если запустить БЕЗ аргументов).
                         σ_d = k·rms по сохранённой калибровке, ожидания — E[priv] из
                         реестра finalist_guard.FINALISTS (или --scores SA SB).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
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


def _scratch_dir() -> Path:
    """Каталог для временных файлов: переменная окружения, иначе системный tmp.

    Раньше здесь был зашит скретчпад чужой машины (/tmp/claude-1000/-home-olya-...),
    из-за чего скрипт мусорил в несуществующий путь на любой другой машине.
    """
    env = os.environ.get("OZON_SCRATCH") or os.environ.get("CLAUDE_SCRATCHPAD")
    p = Path(env) if env else Path(tempfile.gettempdir()) / "ozon_final_pair"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _resolve_sub(fn: str) -> Path:
    """Путь как есть, иначе submissions/<имя> (можно передавать одно имя файла)."""
    p = Path(fn)
    if p.exists():
        return p
    for cand in (ROOT / "submissions" / p.name, ROOT / "submissions" / f"{p.name}.csv"):
        if cand.exists():
            return cand
    return p


def _read_lp(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """(user_id, log1p(predict)) отсортировано — та же шкала, в которой калибрована σ_d."""
    d = pl.read_csv(path, schema_overrides={"user_id": pl.Int64}).sort("user_id")
    col = "predict" if "predict" in d.columns else d.columns[1]
    return (d["user_id"].to_numpy(),
            np.log1p(np.clip(d[col].to_numpy().astype(np.float64), 0, None)))


def _k_from_calibration() -> tuple[float, str]:
    """σ_d ≈ k·rms — коэффициент из сохранённой калибровки (--calibrate)."""
    f = REPORTS_DIR / "final_pair_calibration.json"
    if f.exists():
        j = json.loads(f.read_text())
        return float(j["k_sigma_per_rms"]), str(f)
    return float("nan"), "нет — прогони --calibrate"


def _expectations(names: list[str], scores: list[float] | None) -> tuple[list[float], str]:
    """Ожидаемые ПРИВАТНЫЕ скоры пары. Валюта штаба — finalist_guard.private_ev."""
    if scores:
        return list(scores), "заданы вручную (--scores)"
    try:
        import finalist_guard as G
        evs, src = [], []
        for n in names:
            rec = G.FINALISTS.get(n)
            if rec is None:
                raise KeyError(n)
            evs.append(G.private_ev(rec["pub"], rec["k"]))
            src.append(f"{n}: pub {rec['pub']:.7f}, k {rec['k']}")
        return evs, "E[priv] по реестру finalist_guard (" + "; ".join(src) + ")"
    except Exception:
        pass
    try:
        import predict_lb as P
        known = {n: s for n, _, s in P.MEASURED}
        return [known[n] for n in names], "публичные скоры из predict_lb.MEASURED (не E[priv]!)"
    except Exception as e:
        raise SystemExit(f"нет ожиданий для пары ({e}) — задай --scores SA SB")


def decide_pair(pa: Path, pb: Path, scores: list[float] | None):
    """Решение по конкретной паре: σ_d из непохожести, ценность второго слота."""
    if not pa.exists() or not pb.exists():
        print("режим пары требует самих файлов сабмитов "
              f"(нет: {', '.join(str(p) for p in (pa, pb) if not p.exists())}).")
        return 1
    ua, la = _read_lp(pa)
    ub, lb = _read_lp(pb)
    if not np.array_equal(ua, ub):
        print("ОТКАЗ: user_id файлов не совпадают — пара несравнима.")
        return 1
    names = [pa.stem, pb.stem]
    ev, ev_src = _expectations(names, scores)
    k, k_src = _k_from_calibration()
    rms = float(np.sqrt(np.mean((la - lb) ** 2)))
    sd = k * rms

    print("--- РЕШЕНИЕ ПО ПАРЕ ---")
    print(f"файлы: {names[0]} (слот 1) и {names[1]} (слот 2), n = {len(ua)}")
    print(f"ожидания: {ev_src}")
    print(f"  E[{names[0]}] = {ev[0]:.7f}   E[{names[1]}] = {ev[1]:.7f}")
    lead = 0 if ev[0] <= ev[1] else 1
    D = abs(ev[1] - ev[0])
    print(f"ведёт {names[lead]}, отставание второго D = {D:.7f}")
    print(f"непохожесть rms(log1p) = {rms:.5f}")
    print(f"σ_d ≈ {k:.6f}·rms = {sd:.3e}   (калибровка: {k_src})")

    sv = slot_value(D, sd)
    pi = p_inversion(D, sd)
    print(f"\nценность второго слота E[max]−E[] = {sv:.3e}")
    print(f"вероятность инверсии на привате p = {pi:.2e}   (z = D/σ_d = {D/sd:.1f})")
    print(f"порог осмысленности — шум замера {NOISE:.6f}")
    print(f"E[зачёт] ≈ {ev[lead] - sv:.7f} против {ev[lead]:.7f} у одного ведущего файла")
    print("\nВЕРДИКТ: " + (
        f"второй слот окупается ({sv:.3e} > шума {NOISE:.6f}) — пара оправдана"
        if sv > NOISE else
        f"второй слот ниже шума замера ({sv:.3e} ≤ {NOISE:.6f}): страховка почти "
        f"бесплатна, но и почти бесполезна — слот всё равно занимаем лучшим законным файлом"))
    print("ОГОВОРКА: σ_d — только разброс раскола публика/приват. Ошибка самой модели "
          "ожиданий (k в валюте штаба) в него НЕ входит; страховка второго слота живёт "
          "именно за счёт неё, а не за счёт σ_d.")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--pair", nargs=2, metavar=("A", "B"))
    ap.add_argument("--scores", nargs=2, type=float, metavar=("SA", "SB"),
                    help="ожидаемые скоры пары, если известны (для D); "
                         "по умолчанию берутся E[priv] из реестра finalist_guard")
    ap.add_argument("-B", type=int, default=B_DEFAULT)
    args = ap.parse_args()

    scratch = _scratch_dir()

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

    sys.exit(decide_pair(_resolve_sub(args.pair[0]), _resolve_sub(args.pair[1]), args.scores))


if __name__ == "__main__":
    main()

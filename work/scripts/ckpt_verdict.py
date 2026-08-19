"""Вердикт по усреднению КОНТРОЛЬНЫХ ТОЧЕК внутри одного прогона fusion_v3.

Что проверяется.  Ранняя остановка по калиброванному скору выбирает ОДНУ точку
кривой, которая не монотонна и колеблется с амплитудой ~0.002 (сид 555: 1.67334,
1.67286, 1.67033, 1.67128, 1.67095, 1.66987, 1.67197, 1.66960, 1.66923, 1.66870,
1.66868).  Выбор одной точки шумной кривой берёт максимум шума, а не его среднее.
Гипотеза: усреднение нескольких поздних точек снимает ту же долю шума, что и
усреднение сидов, но внутри одного прогона и без лишнего обучения.

  метод (а) --ckpt-avg  усреднение ПРОГНОЗОВ в log1p (пространство бленда);
  метод (б) --swa       усреднение ВЕСОВ.

Режимы: last (последние k точек) и top (лучшие k по критерию).  Разница между
ними не косметическая: фаза 2 (--final) переобучается БЕЗ валидации, выбрать там
«лучшие по критерию» нечем, поэтому на тест переносится только last.  top меряется
как верхняя граница, а решение принимается по last.

База сравнения — строка (top, k=1): это ровно та точка, которую выбрала бы ранняя
остановка, то есть действующее поведение.  Всё считается по КАЛИБРОВАННОМУ скору
с честным разрезом (половина юзеров подбирает сдвиги, половина меряется) — тем
самым, что воспроизводит опубликованные 1.668676 и 1.667846 до знака.

Дополнительно доказывается, что путь по умолчанию не тронут: прогоны с
--ckpt-sweep не включают ни --ckpt-avg, ни --swa, поэтому их валидационные
прогнозы обязаны совпасть с fusion_v3_escal{seed} ПОБИТОВО.  Это проверка на
полном масштабе, а не на смоуке.

Запуск: POLARS_MAX_THREADS=2 .venv/bin/python work/scripts/ckpt_verdict.py
        [--archive-identical]
"""
from __future__ import annotations

import os

os.environ.setdefault("POLARS_MAX_THREADS", "2")

import argparse  # noqa: E402
import json  # noqa: E402
import shutil  # noqa: E402
import sys  # noqa: E402

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import PREDS_DIR, REPORTS_DIR  # noqa: E402

SEEDS = [555, 7]
SWEEP_FMT = "fusion_v3_swp{}"          # прогон с --ckpt-sweep
BASE_FMT = "fusion_v3_escal{}"         # его же двойник без флага, уже посчитанный
REF_SINGLE = 1.668676                  # сид 555, --es-metric cal, одна точка
REF_3SEED = 1.667846                   # усреднение трёх сидов в log1p
NOISE = 0.0003                         # порог осмысленности проекта


def load_sweep(seed: int) -> list[dict] | None:
    p = REPORTS_DIR / f"ckpt_sweep_{SWEEP_FMT.format(seed)}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())["rows"]


def best_row(rows, mode, key):
    have = [r for r in rows if r["mode"] == mode and key in r]
    return min(have, key=lambda r: r[key]) if have else None


def _verdict(d_best: float, d_last: float | None, ks: list[int]) -> str:
    """Вердикт формулируется ПО ДВУМ БАЗАМ отдельно: относительно лучшей точки
    (фаза 1, валидация) и относительно последней (фаза 2, тест)."""
    p1 = ("даёт" if d_best < -NOISE else "не даёт")
    p2 = ("нет данных" if d_last is None else
          "даёт" if d_last < -NOISE else "не даёт")
    stab = ("оптимум k одинаков на всех сидах" if ks and len(set(ks)) == 1
            else f"оптимум k ПЛЯШЕТ между сидами {ks}, фиксировать нельзя")
    return (f"фаза 1 (против ЛУЧШЕЙ точки) {p1} {d_best:+.6f}; "
            f"фаза 2 (против ПОСЛЕДНЕЙ точки) {p2} "
            f"{'н/д' if d_last is None else f'{d_last:+.6f}'}; {stab}")


def identical(a: str, b: str) -> bool | None:
    pa, pb = PREDS_DIR / f"{a}_val.parquet", PREDS_DIR / f"{b}_val.parquet"
    if not (pa.exists() and pb.exists()):
        return None
    da = pl.read_parquet(pa).sort("user_id")
    db = pl.read_parquet(pb).sort("user_id")
    return bool(np.array_equal(da["user_id"].to_numpy(), db["user_id"].to_numpy())
                and np.array_equal(da["pred"].to_numpy(), db["pred"].to_numpy()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive-identical", action="store_true",
                    help="убрать прогнозы sweep-прогонов из work/preds после того, "
                         "как доказано их побитовое совпадение с fusion_v3_escal: "
                         "это точные дубликаты, а дубликат в пуле — коллинеарная "
                         "помеха для blend_reopt")
    args = ap.parse_args()

    per_seed, missing, ident = [], [], {}
    for sd in SEEDS:
        rows = load_sweep(sd)
        if rows is None:
            missing.append(SWEEP_FMT.format(sd))
            continue
        # ДВЕ РАЗНЫЕ БАЗЫ, и их нельзя смешивать:
        #   b_best — ЛУЧШАЯ точка по калиброванному критерию. Это то, что делает
        #     фаза 1, откуда берутся ВАЛИДАЦИОННЫЕ прогнозы.
        #   b_last — ПОСЛЕДНЯЯ точка. Это то, что делает фаза 2 (--final), откуда
        #     берутся ТЕСТОВЫЕ прогнозы: там валидации нет вообще, ранняя
        #     остановка не работает, выбрать лучшую точку физически нечем.
        # Выигрыш относительно этих двух баз отличается на порядок, поэтому
        # единственная цифра «выигрыш усреднения» бессмысленна.
        b = next(r for r in rows if r["mode"] == "top" and r["k"] == 1)["pred_avg_cal"]
        b_last = next(r for r in rows
                      if r["mode"] == "last" and r["k"] == 1)["pred_avg_cal"]
        rec = {"seed": sd, "baseline_cal": b, "baseline_last_cal": b_last,
               "n_ckpts": max(r["k"] for r in rows)}

        # k=1 в любом режиме — это одна точка, поэтому усреднение весов при k=1
        # обязано дать ровно её же прогноз; расхождение означает ошибку в коде
        k1 = next((r for r in rows if r["mode"] == "top" and r["k"] == 1), None)
        rec["swa_k1_matches_pred_k1"] = (
            None if k1 is None or "swa_cal" not in k1
            else abs(k1["swa_cal"] - k1["pred_avg_cal"]) < 1e-6)

        for mode in ("last", "top"):
            for meth, key in (("a", "pred_avg_cal"), ("b", "swa_cal")):
                r = best_row(rows, mode, key)
                if r is None:
                    continue
                rec[f"{meth}_{mode}_best_k"] = r["k"]
                rec[f"{meth}_{mode}_cal"] = r[key]
                rec[f"{meth}_{mode}_delta"] = round(r[key] - b, 6)
                if mode == "last":   # база фазы 2 — последняя точка
                    rec[f"{meth}_last_delta_vs_last"] = round(r[key] - b_last, 6)
        # ФИКСИРОВАННОЕ k=3: оптимум, найденный на сиде 555. Берём его значение
        # на КАЖДОМ сиде, иначе «лучшее k» подбирается по тем же данным, на
        # которых меряется, и выигрыш завышен отбором.
        for meth, key in (("a", "pred_avg_cal"), ("b", "swa_cal")):
            r3 = next((r for r in rows
                       if r["mode"] == "last" and r["k"] == 3 and key in r), None)
            if r3 is not None:
                rec[f"{meth}_k3_cal"] = r3[key]
                rec[f"{meth}_k3_delta_vs_best"] = round(r3[key] - b, 6)
                rec[f"{meth}_k3_delta_vs_last"] = round(r3[key] - b_last, 6)
        per_seed.append(rec)
        ident[str(sd)] = {"sweep_val_bit_identical_to_escal":
                          identical(SWEEP_FMT.format(sd), BASE_FMT.format(sd))}
        print(f"seed {sd}: база ФАЗЫ 1 (лучшая точка) {b:.6f}, "
              f"база ФАЗЫ 2 (последняя точка) {b_last:.6f}", flush=True)
        print(f"        (а) last лучшее k={rec.get('a_last_best_k')} "
              f"{rec.get('a_last_cal')}: против лучшей {rec.get('a_last_delta'):+.6f}, "
              f"против последней {rec.get('a_last_delta_vs_last'):+.6f}", flush=True)
        print(f"        (а) при k=3 фиксировано {rec.get('a_k3_cal')}: "
              f"против лучшей {rec.get('a_k3_delta_vs_best'):+.6f}, "
              f"против последней {rec.get('a_k3_delta_vs_last'):+.6f}", flush=True)
        print(f"        (б) last лучшее k={rec.get('b_last_best_k')} "
              f"{rec.get('b_last_cal')} против лучшей "
              f"{rec.get('b_last_delta'):+.6f}", flush=True)
        print(f"        верхняя граница top: (а) {rec.get('a_top_delta'):+.6f} "
              f"k={rec.get('a_top_best_k')} | (б) {rec.get('b_top_delta'):+.6f} "
              f"k={rec.get('b_top_best_k')} | побитовое совпадение "
              f"{ident[str(sd)]}", flush=True)

    if missing:
        print(f"НЕТ развёрток, сначала прогоны очереди: {missing}", flush=True)
        return

    def avg(key):
        v = [r[key] for r in per_seed if key in r]
        return round(float(np.mean(v)), 6) if v else None

    def mode_k(key):
        v = [r[key] for r in per_seed if key in r]
        return int(round(float(np.mean(v)))) if v else 0

    a_delta, b_delta = avg("a_last_delta"), avg("b_last_delta")
    best = min([d for d in (a_delta, b_delta) if d is not None], default=0.0)
    best_cal = min([c for c in (avg("a_last_cal"), avg("b_last_cal"))
                    if c is not None], default=float("nan"))
    ident_ok = all(v is not False for d in ident.values() for v in d.values())
    k1_ok = all(r.get("swa_k1_matches_pred_k1") is not False for r in per_seed)

    ks = [r["a_last_best_k"] for r in per_seed if "a_last_best_k" in r]
    out = {
        "per_seed": per_seed,
        # база ФАЗЫ 1 (лучшая точка) — определяет валидационные прогнозы
        "method_a_delta": a_delta,          # усреднение прогнозов, режим last
        "method_b_delta": b_delta,          # усреднение весов, режим last
        "method_a_delta_top": avg("a_top_delta"),
        "method_b_delta_top": avg("b_top_delta"),
        # база ФАЗЫ 2 (последняя точка) — определяет ТЕСТОВЫЕ прогнозы
        "method_a_delta_vs_last": avg("a_last_delta_vs_last"),
        "method_b_delta_vs_last": avg("b_last_delta_vs_last"),
        # то же при ЖЁСТКО зафиксированном k=3, без подбора k по тем же данным
        "k3_delta_vs_best": avg("a_k3_delta_vs_best"),
        "k3_delta_vs_last": avg("a_k3_delta_vs_last"),
        "best_k_per_seed": ks,
        "best_k_stable": bool(ks and len(set(ks)) == 1),
        "best_k": mode_k("a_last_best_k" if (a_delta or 0) <= (b_delta or 0)
                         else "b_last_best_k"),
        "baseline_cal": avg("baseline_cal"),
        "baseline_matches_published": (
            None if not per_seed else
            abs(per_seed[0]["baseline_cal"] - REF_SINGLE) < 1e-6),
        "best_cal": best_cal,
        "vs_3seed_avg": round(best_cal - REF_3SEED, 6),
        "default_identical": bool(ident_ok),
        "swa_k1_sanity": bool(k1_ok),
        "identity": ident,
        "verdict": ("СЛОМАНО: путь по умолчанию изменился, сравнение недействительно"
                    if not ident_ok else
                    "СЛОМАНО: SWA при k=1 не равен одной точке" if not k1_ok else
                    _verdict(best, avg("a_k3_delta_vs_last"), ks)),
    }

    if args.archive_identical and ident_ok:
        dst = PREDS_DIR / "ckpt_sweep_dupes"
        moved = []
        for sd in SEEDS:
            if ident.get(str(sd), {}).get("sweep_val_bit_identical_to_escal"):
                dst.mkdir(exist_ok=True)
                for nm in (SWEEP_FMT.format(sd), SWEEP_FMT.format(sd) + "_cal"):
                    for split in ("val", "test"):
                        p = PREDS_DIR / f"{nm}_{split}.parquet"
                        if p.exists():
                            shutil.move(str(p), dst / p.name)
                            moved.append(p.name)
        out["archived"] = moved
        print(f"убрано {len(moved)} дубликатов -> {dst}", flush=True)

    (REPORTS_DIR / "ckpt_verdict.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1))
    print("\n=== RAW JSON ===")
    print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()

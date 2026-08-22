"""Закон переноса val->LB: что предсказывает каппу переноса по вал-наблюдаемым.

ЗАЧЕМ. У нас накоплены пары «шаг, оптимальный на валидации -> сколько его дожило до
теста». Коэффициент дожития kappa = b*/b дозирует новые оси финала. Пользуемся мы им
как константой класса, но не знаем, что его предсказывает. Если предиктор есть, новая
ось не обязана платить зондом лидерборда.

ДВЕ ВЕЛИЧИНЫ С ОДНИМ ИМЕНЕМ — ГЛАВНАЯ ЛОВУШКА ЭТОЙ ЗАДАЧИ. В проекте «каппой» зовут
две разные вещи:

  ранняя (probe_plan_v2.md:69)  kappa = LB gain / val gain      PA_gmv 2.5, febdir 7.7

Первая — отношение ВЫИГРЫШЕЙ, вторая — отношение ШАГОВ. Это не одна величина в разных
единицах. Если сложить их в одну регрессию, febdir с 7.7 в одиночку задаст наклон.
Здесь берётся ТОЛЬКО поздняя: она восстановлена по параболе из двух LB-замеров на ось,
у неё есть sigma, и именно ею дозируют шаг.

ВЕСА. sigma(kappa) различаются в 40 раз: 0.016 у ridge против 0.695 у шейдинга. Точка с
sigma 0.695 — это «от -0.4 до +2.4», то есть неизмеренная. Класть её наравне с точными
нельзя, поэтому основной прогон идёт по точкам с sigma <= 0.10, а полный набор считается
отдельно и печатается для сравнения.

Запуск: python work/scripts/transfer_law.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from common import REPORTS_DIR

# ---------------------------------------------------------------------------
# Точки. Каждая — ось, у которой ЗАМЕРЕН перенос на лидерборде. Провенанс в поле src:
# число берётся оттуда и ниоткуда больше (правило «константы из чужих отчётов не брать»).
# conc_top1/top01 — доля вклада топ-1%/0.1% юзеров в MSE-выигрыш на валидации
# (методика night_corrector_variants.py: d = eb^2 - (eb+step)^2, доля может быть > 1,
# потому что часть юзеров вносит отрицательный вклад).
# n_eff — эффективное число носителей шага; conc_ipr = N / n_eff, общая шкала концентрации.
POINTS = [
    dict(name="blend_delta", cls="состав бленда", kappa=0.618, sigma=0.020,
         val_gain=0.000655, conc_top1=None, conc_top01=None, n_eff=None, novelty=None,
         src="r6_joint_opt.json b_opt[0]; вал night_blend_stability.json s_old->s_new"),
    dict(name="ridge_v1", cls="стек по признакам", kappa=0.308, sigma=0.016,
         val_gain=0.000509, conc_top1=4.2386, conc_top01=0.8764, n_eff=None, novelty=None,
         src="r6_joint_opt.json b_opt[1]; night_corrector_variants.json variants.base"),
    dict(name="shade", cls="форма", kappa=1.002, sigma=0.695,
         val_gain=None, conc_top1=None, conc_top01=None, n_eff=None, novelty=None,
         src="r6_joint_opt.json b_opt[2]; KNOWLEDGE «Залив 21.08» 1.32+-0.51 по двум замерам"),
    dict(name="W1_e_new", cls="форма", kappa=0.089, sigma=0.055,
         val_gain=None, conc_top1=None, conc_top01=None, n_eff=8709.0, novelty=0.902,
         src="KNOWLEDGE «»; w1_silence_e.json n_eff/новизна"),
    dict(name="S1_segwall", cls="форма", kappa=0.046, sigma=0.055,
         val_gain=None, conc_top1=None, conc_top01=None, n_eff=121332.1, novelty=0.6799,
         src="KNOWLEDGE «Залив 21.08»; segwall_probe.json n_eff/novelty18"),
]
N_USERS = 250000
TIGHT = 0.10          # порог «точка измерена»: sigma(kappa) не больше этого
NAIVE = 0.5           # наивный прогноз из постановки задачи


def loo_mae(pts, predict):
    """LOO: для каждой точки прогноз строится по ОСТАЛЬНЫМ, ошибка считается на ней."""
    errs = []
    for i in range(len(pts)):
        rest = pts[:i] + pts[i + 1:]
        errs.append(abs(predict(pts[i], rest) - pts[i]["kappa"]))
    return float(np.mean(errs)), errs


def pred_class(p, rest):
    """Среднее по своему классу; если класс в остатке пуст — общее среднее."""
    same = [r["kappa"] for r in rest if r["cls"] == p["cls"]]
    return float(np.mean(same)) if same else float(np.mean([r["kappa"] for r in rest]))


def pred_global(p, rest):
    return float(np.mean([r["kappa"] for r in rest]))


def pred_naive(p, rest):
    return NAIVE


def spearman_boot(x, y, B=5000, seed=0):
    """Ранговая корреляция с бутстрапом ПО ТОЧКАМ (их мало — интервал будет широким)."""
    from scipy.stats import spearmanr
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = ~(np.isnan(x) | np.isnan(y))
    x, y = x[m], y[m]
    if len(x) < 3:
        return None
    rho = float(spearmanr(x, y).statistic)
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(B):
        idx = rng.integers(0, len(x), len(x))
        if len(np.unique(x[idx])) < 2 or len(np.unique(y[idx])) < 2:
            continue
        bs.append(spearmanr(x[idx], y[idx]).statistic)
    lo, hi = np.percentile(bs, [2.5, 97.5]) if bs else (np.nan, np.nan)
    return {"n": int(len(x)), "rho": rho, "ci95": [float(lo), float(hi)]}


def main():
    for p in POINTS:
        p["conc_ipr"] = N_USERS / p["n_eff"] if p["n_eff"] else None

    tight = [p for p in POINTS if p["sigma"] <= TIGHT]
    print(f"точек всего {len(POINTS)}, измеренных (sigma<= {TIGHT}) {len(tight)}\n")
    print(f"{'ось':<14}{'класс':<20}{'kappa':>8}{'sigma':>8}{'вал':>10}"
          f"{'top1%':>8}{'N/n_eff':>9}{'новизна':>9}")
    for p in sorted(POINTS, key=lambda z: -z["kappa"]):
        f = lambda v, w, d: (f"{v:{w}.{d}f}" if v is not None else " " * (w - 1) + "-")
        mark = "" if p["sigma"] <= TIGHT else "  (не измерена)"
        print(f"{p['name']:<14}{p['cls']:<20}{p['kappa']:>+8.3f}{p['sigma']:>8.3f}"
              f"{f(p['val_gain'],10,6)}{f(p['conc_top1'],8,2)}{f(p['conc_ipr'],9,1)}"
              f"{f(p['novelty'],9,3)}{mark}")

    # ---- гипотеза 1: kappa ~ концентрация. Контролируемая пара внутри одного класса.
    print("\n--- ГИПОТЕЗА 1: «диффузное переносится, ставка на китов нет» ---")
    pair = [p for p in POINTS if p["conc_top1"] is not None]
    for p in sorted(pair, key=lambda z: -z["conc_top1"]):
        print(f"  {p['name']:<12} класс «{p['cls']}»  топ-1% {p['conc_top1']:.2f}  "
              f"топ-0.1% {p['conc_top01']:.3f}  вал {p['val_gain']:.6f}  kappa {p['kappa']:+.3f}")
    a, b = sorted(pair, key=lambda z: -z["conc_top1"])
    ok = (a["conc_top1"] > b["conc_top1"]) and (a["kappa"] < b["kappa"])
    print(f"  Гипотеза требует: менее концентрированный переносится ЛУЧШЕ.")
    print(f"  Факт: {b['name']} менее концентрирован и при БОЛЬШЕМ вал-выигрыше "
          f"перенёсся ХУЖЕ ({b['kappa']:+.3f} против {a['kappa']:+.3f}).")
    print(f"  ВЕРДИКТ: {'подтверждена' if ok else 'ОПРОВЕРГНУТА на контролируемой паре'}")

    # ---- ранговые проверки остальных предикторов
    print("\n--- РАНГОВЫЕ ПРОВЕРКИ (бутстрап по точкам) ---")
    tests = {}
    for label, key in (("концентрация N/n_eff", "conc_ipr"), ("вал-выигрыш", "val_gain"),
                       ("новизна", "novelty")):
        r = spearman_boot([p[key] for p in POINTS if p[key] is not None],
                          [p["kappa"] for p in POINTS if p[key] is not None])
        tests[key] = r
        if r is None:
            print(f"  {label:<24} точек < 3 — проверить нечем")
        else:
            print(f"  {label:<24} n={r['n']}  rho={r['rho']:+.3f}  "
                  f"95% [{r['ci95'][0]:+.2f}, {r['ci95'][1]:+.2f}]  — интервал накрывает 0: "
                  f"{'да' if r['ci95'][0] < 0 < r['ci95'][1] else 'НЕТ'}")

    # ---- гипотеза 2: класс механизма
    print("\n--- ГИПОТЕЗА 2: kappa определяется КЛАССОМ механизма ---")
    for c in dict.fromkeys(p["cls"] for p in POINTS):
        ks = [p["kappa"] for p in tight if p["cls"] == c]
        if ks:
            print(f"  {c:<20} n={len(ks)}  kappa {np.mean(ks):+.3f}"
                  + (f"  разброс {max(ks)-min(ks):.3f}" if len(ks) > 1 else ""))

    print("\n--- LOO (критерий приёмки: MAE <= 0.15) ---")
    out = {"points": POINTS, "hypothesis1_refuted": not ok, "rank_tests": tests, "loo": {}}
    for setname, pts in (("измеренные (sigma<=0.10)", tight), ("все точки", POINTS)):
        print(f"  {setname}, n={len(pts)}:")
        for pname, fn in (("по классу", pred_class), ("общее среднее", pred_global),
                          ("наивный 0.5", pred_naive)):
            mae, _ = loo_mae(pts, fn)
            verdict = "ГОДИТСЯ" if mae <= 0.15 else "не проходит"
            print(f"     {pname:<16} LOO-MAE {mae:.3f}   {verdict}")
            out["loo"].setdefault(setname, {})[pname] = round(mae, 4)

    (REPORTS_DIR / "transfer_law.json").write_text(
        json.dumps(out, indent=1, ensure_ascii=False, default=str))
    print(f"\nJSON: {REPORTS_DIR / 'transfer_law.json'}")


if __name__ == "__main__":
    main()

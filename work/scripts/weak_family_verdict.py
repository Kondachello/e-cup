"""weak_family_verdict.py — сводит замеры семейства обеднённых моделей в один вердикт.

Читает то, что произвели предыдущие шаги очереди:
  work/reports/weak_family_eval.json      — корреляция с ближайшим соседом (а)
  work/reports/blend_reopt_before.json    — честный OOF бленда БЕЗ семейства (б)
  work/reports/blend_reopt_after.json     — честный OOF бленда С семейством  (б)
  work/reports/blend_testopt_wstab.json   — тест-веса и частота отбора       (в)

Порог: одно подобранное направление на лидерборде стоит 0.000022 (noise_floor.py);
прирост всего семейства ниже 0.0003 = направление закрыто.

Запуск: .venv/bin/python work/scripts/weak_family_verdict.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import REPORTS_DIR  # noqa: E402

THRESH = 0.0003
PREFIX = "weak_"


def jload(name: str) -> dict:
    p = REPORTS_DIR / name
    return json.loads(p.read_text()) if p.exists() else {}


def main() -> int:
    ev = jload("weak_family_eval.json")
    before = jload("blend_reopt_before.json")
    after = jload("blend_reopt_after.json")
    wst = jload("blend_testopt_wstab.json")

    def oof(d):
        try:
            return float(d["winner"]["oof"])
        except Exception:
            return None

    def oof_m(d, method, lib="B_plus_cal"):
        """OOF конкретного метода. ols_free = ЗНАКО-СВОБОДНЫЕ веса.

        Это обязательно мерить отдельно: обеднённые модели получают ОТРИЦАТЕЛЬНЫЙ
        оптимальный вес (замерено на подпространствах: w* от -0.006 до -0.121),
        поэтому NNLS их просто обнуляет и headline-вклад выходит 0 по построению.
        Отрицательные веса в этом проекте легитимны — R12 в KNOWLEDGE (gbdt -0.077,
        nn -0.081) подтверждён фактическим скором A1.
        """
        try:
            return float(d["results"][lib][method]["oof"])
        except Exception:
            return None

    o_b, o_a = oof(before), oof(after)
    gain = round(o_b - o_a, 6) if (o_b is not None and o_a is not None) else None
    sf_b, sf_a = oof_m(before, "ols_free"), oof_m(after, "ols_free")
    gain_signfree = round(sf_b - sf_a, 6) if (sf_b is not None and sf_a is not None) else None

    pm = (wst or {}).get("per_model", {})
    weak = {k: v for k, v in pm.items() if k.startswith(PREFIX)}
    tw = round(sum(max(v.get("test", 0.0), 0.0) for v in weak.values()), 4)
    picked = {k: dict(test=round(v.get("test", 0.0), 4), freq=round(v.get("freq", 0.0), 2),
                      corr=round(v.get("corr", 0.0), 4), nearest=v.get("nearest"))
              for k, v in sorted(weak.items(), key=lambda kv: -kv[1].get("test", 0.0))}

    summ = (ev or {}).get("summary", {})
    weights_after = ((after or {}).get("winner", {}) or {}).get("weights", {})
    in_blend = {k: round(v, 5) for k, v in weights_after.items() if k.startswith(PREFIX)}

    best = max([g for g in (gain, gain_signfree) if g is not None], default=None)
    if best is None:
        verdict = "неполный замер: нет одного из blend_reopt_before/after"
    elif best >= THRESH:
        verdict = (f"направление живое: семейство даёт {best:+.6f} "
                   f"(неотриц. веса {gain}, знако-свободные {gain_signfree}; "
                   f"порог {THRESH}, шум одного замера 0.000022)")
    else:
        verdict = (f"направление закрыто: прирост семейства {best:+.6f} "
                   f"(неотриц. веса {gain}, знако-свободные {gain_signfree}) "
                   f"ниже порога {THRESH}")

    out = {
        "n_models": summ.get("n_models"),
        "mechanisms": ["random feature subspace", "random anchor subset",
                       "feature-type restriction", "tiny over-regularised"],
        "min_corr_with_nearest": summ.get("min_corr_with_nearest"),
        "median_corr_with_nearest": summ.get("median_corr_with_nearest"),
        "n_below_097": summ.get("n_below_097"),
        "ref_febspec_corr": summ.get("ref_corr"),
        "blend_oof_before": o_b,
        "blend_oof_after": o_a,
        "family_gain": gain,
        "blend_oof_before_signfree": sf_b,
        "blend_oof_after_signfree": sf_a,
        "family_gain_signfree": gain_signfree,
        "weak_weights_in_val_blend": in_blend,
        "test_weights_total": tw,
        "test_weights_per_model": picked,
        "verdict": verdict,
    }
    (REPORTS_DIR / "weak_family_verdict.json").write_text(
        json.dumps(out, indent=1, ensure_ascii=False))
    print(json.dumps(out, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

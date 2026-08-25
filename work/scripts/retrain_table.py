"""Таблица «отгружено / ретрейн / Δ» — честный ответ жюри вместо побитовости.

ЗАЧЕМ. Побитовое воспроизведение отгруженного файла недостижимо и не нужно: часть членов
бленда — нейросети, у одной обучение обрывалось на 4500 шаге из 11736. Правильное
утверждение для проверки — «код в репозитории порождает модели решения, а переобучение
отличается от отгруженного на столько-то, и это ИЗМЕРЕНО». Скрипт считает это «столько-то»
на трёх уровнях сразу:

  соло       насколько разошёлся собственный скор члена
  вектор     rms и максимум расхождения предсказаний в лог-пространстве
  бленд      насколько сдвинется ИТОГОВЫЙ скор, если подставить переобученного члена
             на его же вес (веса не переподбираются — иначе Δ маскируется подгонкой)

Последняя колонка и есть ответ на вопрос «изменится ли решение, если прогнать код заново».

Запуск:
  python work/scripts/retrain_table.py --pair weak_an_d_cal:weak_an_d_rt_cal ...
  python work/scripts/retrain_table.py --auto        # все *_rt_* в work/preds
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


def load_lp(name: str, uid: np.ndarray) -> np.ndarray | None:
    p = PREDS_DIR / f"{name}_val.parquet"
    if not p.exists():
        return None
    d = pl.read_parquet(p).sort("user_id")
    if not np.array_equal(d["user_id"].to_numpy(), uid):
        return None
    return np.log1p(np.clip(d["pred"].to_numpy().astype(np.float64), 0, None))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", action="append", default=[],
                    help="ОТГРУЖЕННОЕ:ПЕРЕОБУЧЕННОЕ, имена как в work/preds без _val")
    ap.add_argument("--json", default="retrain_table.json")
    a = ap.parse_args()

    pack = pl.read_parquet(ROOT / "work" / "preds_pack" / "val_preds.parquet").sort("user_id")
    uid = pack["user_id"].to_numpy()
    ly = np.log1p(np.clip(pack["target"].to_numpy().astype(np.float64), 0, None))
    blend = pack["blend"].to_numpy().astype(np.float64)
    sb = float(np.sqrt(np.mean((blend - ly) ** 2)))

    W = json.loads((REPORTS_DIR / "blend_reopt.json").read_text())["winner"]
    w = {k: v for k, v in (W.get("weights") or W.get("w")).items() if abs(v) > 1e-4}

    print(f"эталон бленда {sb:.6f}; членов {len(w)}, суммарный вес {sum(w.values()):.3f}\n")
    print(f"{'член':<22}{'вес':>7}{'отгружено':>11}{'ретрейн':>10}{'Δсоло':>10}"
          f"{'rms':>9}{'Δбленда':>10}")
    print("-" * 79)
    rows, cover = [], 0.0
    for spec in a.pair:
        ship, rt = spec.split(":")
        base = ship if ship in w else next((k for k in w if k.startswith(ship.split("_cal")[0])), None)
        wt = w.get(ship, w.get(base, 0.0))
        lp_s, lp_r = load_lp(ship, uid), load_lp(rt, uid)
        if lp_s is None or lp_r is None:
            print(f"{ship:<22}{wt:>7.4f}   пропуск: нет файла "
                  f"{'отгруженного' if lp_s is None else 'переобученного'}")
            continue
        ss = float(np.sqrt(np.mean((lp_s - ly) ** 2)))
        sr = float(np.sqrt(np.mean((lp_r - ly) ** 2)))
        rms = float(np.sqrt(np.mean((lp_s - lp_r) ** 2)))
        # подстановка на тот же вес: бленд минус старый член плюс новый
        nb = blend + wt * (lp_r - lp_s)
        sn = float(np.sqrt(np.mean((nb - ly) ** 2)))
        cover += wt
        rows.append(dict(member=ship, retrain=rt, weight=wt, shipped=ss, retrained=sr,
                         d_solo=sr - ss, rms_log=rms, d_blend=sn - sb))
        print(f"{ship:<22}{wt:>7.4f}{ss:>11.6f}{sr:>10.6f}{sr-ss:>+10.6f}"
              f"{rms:>9.4f}{sn-sb:>+10.6f}")

    if rows:
        tot = blend.copy()
        for r in rows:
            tot += r["weight"] * (load_lp(r["retrain"], uid) - load_lp(r["member"], uid))
        st = float(np.sqrt(np.mean((tot - ly) ** 2)))
        print("-" * 79)
        print(f"{'ВСЕ ПОДСТАВЛЕНЫ':<22}{cover:>7.4f}{sb:>11.6f}{st:>10.6f}{st-sb:>+10.6f}")
        print(f"\nпокрыто переобучением {cover:.1%} веса бленда; итоговый сдвиг решения "
              f"{st - sb:+.6f} RMSLE")
        print(f"для масштаба: шум одного замера лидерборда 0.000022")
        out = dict(blend_ref=sb, coverage=cover, blend_after_all=st, d_blend_total=st - sb,
                   rows=rows)
        (REPORTS_DIR / a.json).write_text(json.dumps(out, indent=1, ensure_ascii=False))
        print(f"JSON: work/reports/{a.json}")


if __name__ == "__main__":
    main()

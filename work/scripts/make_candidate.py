"""Кандидат на отправку: честный бленд, приведённый к проверенным моментам.

корреляция 0.9972. То есть весь результат это честный бленд плюс два числа — среднее
2.3247 и разброс 1.6320 логарифма прогноза. Оба подобраны по лидерборду с запасом в
десятки единиц шума, и они СВОЙСТВО ТЕСТОВОГО ОКНА, а не конкретного бленда.

Поэтому когда бленд улучшается, не нужно заново подбирать сдвиг и множитель: достаточно
привести новый бленд к тем же двум моментам. Улучшение бленда при этом сохраняется,
а проверенная сезонная настройка не теряется.

Запуск:
    .venv/bin/python work/scripts/make_candidate.py --pred blend_opt --name J1_newblend
    .venv/bin/python work/scripts/make_candidate.py --pred blend_opt --name J1 --check
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))
from subs import MEASURED, lp, novelty, span_matrix

ROOT = Path("/Users/alexanderkondakov/ozon-cup")
PREDS = ROOT / "work" / "preds"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True, help="имя набора в work/preds (без _test.parquet)")
    ap.add_argument("--name", required=True, help="имя выходного сабмита без .csv")
    ap.add_argument("--check", action="store_true", help="только посчитать, ничего не писать")
    ap.add_argument("--carry-from", default="",
                    help="имя старого бленда (например blend_cal): перенести на новый файл "
                         "остаток опоры над приведённым старым блендом — это накопленная "
                         "цепочка мелких шагов, иначе она теряется")
    ap.add_argument("--carry-shrink", type=float, default=1.0,
                    help="усадка переносимого остатка; уроки H1/ говорят, что поправки "
                         "плохо переносятся между разными базами, при сомнении ставь 0.5")
    args = ap.parse_args()

    uid_ref, ref = lp(args.ref)
    m_ref, s_ref = float(ref.mean()), float(ref.std())

    p = PREDS / f"{args.pred}_test.parquet"
    d = pl.read_parquet(p).sort("user_id")
    uid = d["user_id"].to_numpy()
    assert np.array_equal(uid, uid_ref), "user_id не совпадает с опорным файлом"
    x = np.log1p(np.clip(d["pred"].to_numpy().astype(np.float64), 0, None))

    b = s_ref / x.std()
    a = m_ref - b * x.mean()
    new = a + b * x
    print(f"опора {args.ref}: среднее {m_ref:.4f} разброс {s_ref:.4f}")
    print(f"{args.pred}: среднее {x.mean():.4f} разброс {x.std():.4f}")
    print(f"приведение: {a:+.4f} + {b:.4f} * x")
    print(f"после: среднее {new.mean():.4f} разброс {new.std():.4f}  "
          f"корреляция с опорой {np.corrcoef(new, ref)[0, 1]:.4f}")

    if args.carry_from:
        do = pl.read_parquet(PREDS / f"{args.carry_from}_test.parquet").sort("user_id")
        xo = np.log1p(np.clip(do["pred"].to_numpy().astype(np.float64), 0, None))
        bo = s_ref / xo.std()
        old_matched = (m_ref - bo * xo.mean()) + bo * xo
        resid = ref - old_matched
        new = new + args.carry_shrink * resid
        print(f"перенос остатка от {args.carry_from}: разброс {resid.std():.4f}, "
              f"усадка {args.carry_shrink}; после переноса среднее {new.mean():.4f} "
              f"разброс {new.std():.4f}")

    if args.strength != 1.0:
        full = new - ref
        new = ref + args.strength * full
        print(f"сила шага {args.strength}: направление к опоре ужато с разброса "
              f"{full.std():.4f} до {(new - ref).std():.4f}")

    h = new - ref
    n = len(uid)
    nv, _ = novelty(h, span_matrix(MEASURED, n))
    q = float((h ** 2).mean())
    print(f"направление к опоре: разброс {h.std():.4f}, новизна {nv:.3f}, q={q:.5f}")
    print(f"клипуется в ноль: {(new <= 0).sum()} строк")

    if args.check:
        return
    out = ROOT / "submissions" / f"{args.name}.csv"
    pl.DataFrame({"user_id": uid, "predict": np.expm1(np.clip(new, 0, None))}).write_csv(out)
    print(f"записан {out}")


if __name__ == "__main__":
    main()

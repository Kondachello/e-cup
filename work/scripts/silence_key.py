"""ЗАДАЧА Б: ключ таблицы вероятности молчания — не складывать блоки.

Сейчас ключ это пара «активные дни за 90 дней x давность», причём active_days_90
ТОЖДЕСТВЕННО равен сумме активных дней трёх 30-дневных блоков (проверено). Сумма
теряет форму: 10+0+0 и 3+3+4 попадают в одну ячейку, а вероятность молчания у них
разная. Здесь перебираются ключи, где блоки не складываются.

ТРИ ОБЯЗАТЕЛЬНЫХ УСЛОВИЯ ОФЛАЙНОВОЙ ОЦЕНКИ (выведены и проверены раньше):
  2. уровень молчания ставится 3.4-3.5% (он привязан к лидерборду), а не 2-3%,
     иначе сравниваются формы при чужом сдвиге логита;
  3. хвост «много активных дней» НЕ зануляется: среднесохраняющее направление
     обязано его слегка поднимать.

Метрика — та же, что решает на лидерборде: c^2/q, где u = p*m,
c = cov(u, y*m), q = var(u).  Проверка несмещённая: таблица обучается на пяти
якорях, чьи окна цели заканчиваются до начала окна оценки.

    .venv/bin/python work/scripts/silence_key.py
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from common import ROOT, load_anchor                                   # noqa: E402
from silence_model import DAY_B, REC_B, sel_mask, sig_level            # noqa: E402
from silence_split import build_real_cumsum, labels                    # noqa: E402
from silence_target import DAY0, build_cumsum, window_active_days      # noqa: E402
from subs import lp                                                    # noqa: E402

OUT = ROOT / "work" / "reports"
EVAL_ANCHORS = [date(2025, 9, 17), date(2025, 10, 1), date(2025, 10, 15)]
FIT_OFFSETS = (105, 91, 77, 63, 49)      # окна цели кончаются до начала окна оценки
LEVEL = 0.0345                            # уровень молчания, привязанный к лидерборду
SMOOTH = 20.0

B30 = np.array([0, 1, 2, 3, 5, 8, 13, 21, 31])          # активные дни в 30-дневном блоке
B60 = np.array([0, 1, 3, 6, 10, 16, 25, 40, 61])        # активные дни в 60-дневном окне
B4 = np.array([0, 1, 6, 16, 31])
REC6 = np.array([0, 1, 3, 7, 14, 30, 10 ** 6])


def _bin(x, b):
    return np.clip(np.searchsorted(b, x, "right") - 1, 0, len(b) - 2)


def blocks_of(C, a):
    out = []
    return out          # [последние 30, предыдущие 30, ещё предыдущие 30]


def recency(C, a):
    """Дней с последней активности (0 = активен в день якоря, 10**6 = не был никогда)."""
    rev = A[:, ::-1] > 0
    del A
    has = rev.any(1)
    k = rev.argmax(1).astype(np.int64)
    del rev
    return np.where(has, k, 10 ** 6)


def key_funcs():
    """name -> f(C, a) -> (плоский индекс ячейки, число ячеек)"""
    def old(C, a):
        """Ровно текущий ключ. active_days_90 и rec_active берутся из C, а не из
        паркета: проверено, что они совпадают ПОБИТОВО (rec_active=NaN <-> 10**6),
        зато не нужен файл признаков и якори можно брать любые."""






    def sum_rec8(C, a):
        """контроль ёмкости: столько же ячеек, но по СУММЕ блоков и давности."""



def fit_table(f, C, R, anchors, ncell):
    num = np.zeros(ncell); den = np.zeros(ncell)
    glob = num.sum() / den.sum()
    return (num + SMOOTH * glob) / (den + SMOOTH)


def gain_of(p, y, m, level=LEVEL):
    u = sig_level(p, level) * m
    c = float(np.cov(u, y.astype(np.float64) * m, ddof=0)[0, 1])
    q = float(u.var())
    return c * c / max(q, 1e-30), c, q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=200)
    KF = key_funcs()

    res = {"level": LEVEL, "anchors": {}}

    names = [n for n in KF]
    print("\n== сводка отношений к старому ключу по трём якорям ==")
    for n in names:
        rs = [res["anchors"][str(a)][n]["ratio_to_old"] for a in EVAL_ANCHORS]
        print(f"  {n:<10} " + "  ".join(f"{r:+7.1%}" for r in rs) +
              f"   среднее {np.mean(rs):+.1%}")
        res.setdefault("summary", {})[n] = {"ratios": rs, "mean": float(np.mean(rs))}
    (OUT / "silence_key.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(f"\nзаписан {OUT / 'silence_key.json'}")


if __name__ == "__main__":
    main()

"""Тестовая сторона: во что превращается исправление на уровне отправляемого файла.

Берёт сохранённые тестовые прогнозы двух плеч (work/reports/mb_fix_preds.npz),
складывает суррогатный бленд, приводит оба к моментам опорного файла ровно так,
как это делает make_candidate.py, и меряет получившееся направление: разброс,
q = среднее квадрата, новизну относительно уже замеренного базиса.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from common import REPORTS_DIR
from subs import MEASURED, lp, novelty, span_matrix

# веса победившего бленда (work/reports/blend_reopt.json) для тех членов, что удалось повторить
W = {"weak_an_d": 0.043814, "weak_ft_recency": 0.023857, "weak_ft_counts": 0.017269,
     "weak_ft_long90": 0.007295, "countaov_s7": 0.023260, "behavonly_s7": 0.057488 / 2,
     "behavonly_s1337": 0.057488 / 2}

z = np.load(REPORTS_DIR / "mb_fix_preds.npz")
uid_ref, ref = lp(REF)
uid = z["user_id"]
assert np.array_equal(uid, uid_ref), "user_id не совпадает с опорным файлом"
m_ref, s_ref = float(ref.mean()), float(ref.std())
n = len(uid)


def mm(x: np.ndarray) -> np.ndarray:
    """Приведение к моментам опоры — то же, что делает make_candidate.py."""
    b = s_ref / x.std()
    return (m_ref - b * x.mean()) + b * x


def report(tag: str, old: np.ndarray, new: np.ndarray) -> None:
    a, b = mm(old), mm(new)
    h = b - a
    q = float((h ** 2).mean())
    nv, _ = novelty(h, span_matrix(MEASURED, n))
    print(f"\n{tag}")
    print(f"  до приведения: sd(разницы) {(new-old).std():.5f}, среднее {(new-old).mean():+.5f}, "
          f"корр {np.corrcoef(old,new)[0,1]:.5f}")
    print(f"  ПОСЛЕ приведения к моментам: sd {h.std():.5f}, q={q:.6f}, "
          f"новизна к замеренному базису {nv:.3f}")
    print(f"  корреляция приведённых плеч {np.corrcoef(a,b)[0,1]:.5f}; "
          f"юзеров со сдвигом >0.01: {float((np.abs(h)>0.01).mean()):.4f}")


names = sorted({k.split("__")[0] for k in z.files if "__" in k})
old_eq = np.mean([z[f"{m}__old"] for m in names], axis=0)
new_eq = np.mean([z[f"{m}__new"] for m in names], axis=0)
report(f"РАВНЫЕ ВЕСА по {len(names)} повторённым моделям", old_eq, new_eq)

ws = np.array([W[m] for m in names if m in W])
sel = [m for m in names if m in W]
ws = ws / ws.sum()
old_w = np.sum([w * z[f"{m}__old"] for w, m in zip(ws, sel)], axis=0)
new_w = np.sum([w * z[f"{m}__new"] for w, m in zip(ws, sel)], axis=0)
report(f"ВЕСА БЛЕНДА (нормированные) по {len(sel)} членам", old_w, new_w)

print(f"\nдля справки: разброс самого опорного файла {s_ref:.4f}, среднее {m_ref:.4f}")

"""!!! ЧИСЛА ЭТОГО СКРИПТА НЕДЕЙСТВИТЕЛЬНЫ — см. предупреждение в mb_fix_mirror_eval.py.

Он считается по плечам, полученным ДООБУЧЕННЫМИ бустерами, которые видели
валидационный якорь. Внутривыборочный замер. Честная цена — mb_fix_clean_probe3.py.

Исходное описание: отменяет ли БЛЕНД цену обрезки (считается по плечам валидации).

Порядок в точности как в боевом пайплайне: калибровка каждого члена, затем веса
подбираются на ПРАВИЛЬНЫХ признаках валидации, затем эти же веса применяются к
тестовому плечу — где признаки обрезаны. Подбор на одной половине юзеров, замер
на другой.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from calibrate import apply_shifts, fit_shifts
from common import REPORTS_DIR, rmsle

BINS = 24
z = np.load(REPORTS_DIR / "mb_fix_mirror_val.npz")
y = z["target"]
ly = np.log1p(y)
half = z["half"]
ev = ~half
names = sorted({k.split("__")[0] for k in z.files if "__" in k})
print(f"членов: {len(names)} -> {names}\n")

F, C = [], []
for n in names:
    lf_, lc_ = z[f"{n}__full"], z[f"{n}__cut"]
    c, s = fit_shifts(lf_[half], ly[half], BINS)      # калибратор с ПРАВИЛЬНЫХ признаков
    F.append(apply_shifts(lf_, c, s))
    C.append(apply_shifts(lc_, c, s))                 # заморожен, как на тесте
F, C = np.array(F), np.array(C)

# веса подбираются на ПРАВИЛЬНОМ плече (валидация не заражена дефектом)
Ah = np.vstack([np.ones(half.sum()), F[:, half]]).T
w = np.linalg.lstsq(Ah, ly[half], rcond=None)[0]
bf_ = w[0] + w[1:] @ F[:, ev]
bc_ = w[0] + w[1:] @ C[:, ev]

r_f = rmsle(y[ev], np.expm1(np.clip(bf_, 0, None)))
r_c = rmsle(y[ev], np.expm1(np.clip(bc_, 0, None)))
print(f"бленд {len(names)} членов, веса с правильного плеча:")
print(f"  FULL {r_f:.6f}   CUT {r_c:.6f}   ЦЕНА {r_c - r_f:+.6f}")

# простое среднее — контроль, что дело не в подборе весов
mf, mc = F[:, ev].mean(0), C[:, ev].mean(0)
print(f"простое среднее: FULL {rmsle(y[ev], np.expm1(mf)):.6f} "
      f"CUT {rmsle(y[ev], np.expm1(mc)):.6f} "
      f"ЦЕНА {rmsle(y[ev], np.expm1(mc)) - rmsle(y[ev], np.expm1(mf)):+.6f}")

# что переживает приведение к моментам: аффин по цели убирает уровень и масштаб
def after_affine(p):
    m, s = ly[ev].mean(), ly[ev].std()
    q = (p - p.mean()) / p.std() * s + m
    return rmsle(y[ev], np.expm1(np.clip(q, 0, None)))

print(f"\nПОСЛЕ приведения к моментам цели (то, что делает make_candidate):")
print(f"  FULL {after_affine(bf_):.6f}   CUT {after_affine(bc_):.6f}   "
      f"ЦЕНА {after_affine(bc_) - after_affine(bf_):+.6f}")

d = bc_ - bf_
rho = float(np.corrcoef(bf_, bc_)[0, 1])
print(f"\nнаправление в логарифме: sd {d.std():.5f}, среднее {d.mean():+.5f}, "
      f"корреляция плеч {rho:.5f}")
print(f"доля юзеров с |сдвигом| > 0.01: {float((np.abs(d) > 0.01).mean()):.4f}")

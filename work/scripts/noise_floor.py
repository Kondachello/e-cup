"""Уровень шума одного направления, подобранного по публичному лидерборду.

Зачем. Каждый наш «применённый» файл получен так: берём направление h, отправляем
base + step*h, по изменению скора считаем c = mean(e*h) и применяем оптимальный шаг
c/q. Выигрыш в MSE равен c^2/q. Но c оценивается по 50 000 публичных строк, значит
у него есть своя ошибка. Направление, которое не несёт вообще ничего, всё равно даст
оценку c порядка её стандартной ошибки и «выигрыш» c^2/q > 0 — целиком фиктивный,
существующий только на публичной части.

Скрипт считает этот уровень шума честно: разброс произведения остатка на направление
берётся с валидации (там известен настоящий таргет), делится на 50 000, и переводится
в единицы RMSLE. Дальше каждый реально измеренный шаг сравнивается с этим уровнем.

Запуск: .venv/bin/python work/scripts/noise_floor.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))
from subs import lp

# Корень репозитория: OZON_ROOT, иначе поднимаемся от этого файла.
# Захардкоженный путь одной машины делал скрипт неработающим у всех
# остальных членов команды и на чистом клоне.
ROOT = Path(os.environ.get("OZON_ROOT", str(Path(__file__).resolve().parents[2])))
N_PUB = 50_000

# цепочка применённых файлов: (имя, публичный скор). Между соседями — подобранные шаги.
CHAIN = [
    ("A1_gram7_shift", 1.6535955005),
    ("F4_applied", 1.64916806),
]


def main():
    vp = pl.read_parquet(ROOT / "work/preds_pack/val_preds.parquet").sort("user_id")
    y = np.log1p(vp["target"].to_numpy().astype(np.float64))
    cand = [c for c in vp.columns if c.endswith("_cal")]
    p = np.log1p(np.clip(vp[cand[0]].to_numpy().astype(np.float64), 0, None))
    e = y - p
    print(f"остаток валидации по {cand[0]}: sd={e.std():.4f}, E[e^2]={float((e**2).mean()):.4f}")

    # типичное направление берём из реально применявшихся: разности соседних файлов цепочки
    print(f"\n{'направление':32s} {'sd(h)':>7s} {'SE(c)':>9s} {'шум в RMSLE':>12s}")
    floors = []
    for i in range(len(CHAIN) - 1):
        a, b = CHAIN[i][0], CHAIN[i + 1][0]
        try:
            _, la = lp(a + ".csv")
            _, lb_ = lp(b + ".csv")
        except FileNotFoundError:
            continue
        h = lb_ - la
        if h.std() < 1e-12:
            continue
        h = h / h.std()                    # нормируем, q = mean(h^2) = 1
        q = float((h ** 2).mean())
        var_c = float(((e * h) ** 2).mean() - ((e * h).mean()) ** 2) / N_PUB
        se_c = float(np.sqrt(var_c))
        # чисто шумовое направление даёт ожидаемый «выигрыш» E[c^2]/q = SE^2/q в MSE
        noise_mse = se_c ** 2 / q
        noise_rmsle = noise_mse / (2 * 1.649)
        floors.append(noise_rmsle)
        print(f"{a[:10]}->{b[:14]:20s} {(lb_-la).std():7.4f} {se_c:9.5f} {noise_rmsle:12.6f}")

    fl = float(np.median(floors)) if floors else float("nan")
    print(f"\nмедианный уровень шума одного подобранного направления: {fl:.6f} RMSLE")

    print(f"\n{'шаг цепочки':34s} {'выигрыш':>10s} {'в шумах':>9s}  вердикт")
    for i in range(len(CHAIN) - 1):
        (na, sa), (nb, sb) = CHAIN[i], CHAIN[i + 1]
        g = sa - sb
        r = g / fl if fl == fl else float("nan")
        v = "реальный" if r > 4 else ("на грани" if r > 2 else "НЕОТЛИЧИМ ОТ ШУМА")
        print(f"{na[:15]:16s}->{nb[:15]:16s} {g:10.6f} {r:9.1f}  {v}")

    tot = CHAIN[0][1] - CHAIN[-1][1]
    print(f"\nвсего от {CHAIN[0][0]} до {CHAIN[-1][0]}: {tot:.6f}")
    print(f"из них шагов ниже 2 шумов: "
          f"{sum(1 for i in range(len(CHAIN)-1) if (CHAIN[i][1]-CHAIN[i+1][1]) < 2*fl)} из {len(CHAIN)-1}")


if __name__ == "__main__":
    main()

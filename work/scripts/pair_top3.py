# -*- coding: utf-8 -*-
"""pair_top3.py — выбор ВТОРОГО слота: перебор по всем готовым состояниям GLS.

Чем отличается от того, что было. `p_top3.py` считает карту по абстрактным
(mu, sigma); перебор слота-2 30.08 шёл по девяти собранным CSV. Здесь перебираются
ВСЕ состояния солвера, у которых есть готовый вектор lp, — их 26, то есть кандидатов
втрое больше, и любой из них собирается в CSV за секунду (`assemble_from_lp.py`).

ГЛАВНАЯ ЛОВУШКА, ради которой написан модуль. Ранжировать кандидатов по прогнозу
паблика МОЖНО ТОЛЬКО ВНУТРИ ГРУППЫ С ОДИНАКОВЫМИ ПРИОРАМИ. Файл с более широким
приором tau почти всегда выглядит лучше по пабликy — и ровно настолько же хуже
переносится на приват, потому что у него больше k (эффективное число подогнанных
направлений). `eb_decomp.md` §6 фиксирует это прямо: на 46 осях расширение
tau_decomp даёт +5.9 шума паблика против +0.6 шума привата.

Признак сравнимости точный: **k один и тот же тогда и только тогда, когда совпадают
Lam и cQ**. Lam = diag(q(1-w)tau^2) и cQ = 1.25m - 0.25cP несут в себе все приоры;
gamma в них не входит — она добавляет ridge к диагонали системы. Поэтому внутри
группы варьируется только gamma, и сравнение честное.

k по группам берётся из дифференциальных оценок кампании (finalists.md §0):
    k(F8) = 8.3 (аудит K1);
    расширение tau_model 0.196 -> 0.2855  ...  Δk = +2.97;
    F12 = то же расширение + расщепление сегментного приора (сегментные tau ВСЕ
          сужаются) ... Δk <= +0.99, в реестр взят пессимистический край k = 9.3.
Отсюда линия A3 (расширенный tau_model, сегментный НЕ расщеплён) имеет k = 11.27.
Именно здесь живёт gamma=0 — он НЕ является gamma=0-версией F12.

    .venv/bin/python work/scripts/pair_top3.py            # перебор слота-2
    .venv/bin/python work/scripts/pair_top3.py --map      # карта решения по замеру
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import os
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))
from p_top3 import NOISE, SIGMA_US, sample_field, sample_us, threshold_k

ROOT = Path(__file__).resolve().parents[2]
LINEA = ROOT / "work" / "reports" / "lineA"
NS = 400_000
NOISE_PER_DIR, PRIV_TRANSFER = 2.63e-5, 1.25
MU_SLOT1 = 1.646073          # E[приват F12] в валюте части M2 (в ней все отчёты штаба)
PUB_F12 = 1.6456761695614883
K_F12 = 9.3
SLIP = 4.543e-5              # среднее промаха алгебры по двум замерам (F8, F12)
K_BY_GROUP = {"ed71684c": 8.30,          # приор F8: tau_model 0.196, сегментный не расщеплён
              "a2086aa2": 8.30 + 2.97,   # линия A3: tau_model 0.2855 (тут gamma=0)
              "5e89093c": 9.30}          # F12: A3 + расщеплённый сегментный приор


def group_key(z) -> str:
    return hashlib.md5(np.round(z["Lam"], 14).tobytes()
                       + np.round(z["cQ"], 14).tobytes()).hexdigest()[:8]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", action="store_true", help="карта решения по замеру γ=0")
    a = ap.parse_args()

    field = sample_field(NS)
    f3, f5 = threshold_k(field, 3), threshold_k(field, 5)
    z, w = sample_us(NS)

    def p_pair(delta: float, sd: float) -> tuple[float, float]:
        g1 = MU_SLOT1 + SIGMA_US * z
        x = np.minimum(g1, g1 + delta + sd * w)
        return float((x < f3).mean()), float((x < f5).mean())

    b3, b5 = p_pair(0.0, 0.0)
    ref = np.load(LINEA / "gls_state_f12.npz", allow_pickle=True)
    rn = [str(x) for x in ref["names"]]
    d12, V, FS = ref["d_fin"], ref["mdl_vivian"], float(ref["F_SCALE"])
    Q, cP, F0 = ref["Q"], ref["cP"], float(ref["F0"])
    mu1_gard = PUB_F12 + PRIV_TRANSFER * K_F12 * NOISE_PER_DIR

    def alg(d):
        return float(np.sqrt(max(F0 ** 2 - 2 * d @ cP + d @ Q @ d, 0)))

    print(f"поле: E[C_(3)] = {f3.mean():.7f}   слот-1 F12 соло: "
          f"P(топ-3) {b3*100:.2f}%  P(топ-5) {b5*100:.2f}%")
    print(f"\n{'тег':10s} {'γ':>5s} {'k':>6s} {'алгебра':>11s} {'прогноз pub':>12s} "
          f"{'delta':>10s} {'sd_d,ш':>7s} {'P(топ-3)':>9s} {'ΔP3':>6s} {'ΔP5':>6s}")
    rows, skipped = [], []
    for p in sorted(glob.glob(str(LINEA / "gls_state_*.npz"))):
        tag = os.path.basename(p)[len("gls_state_"):-4]
        zz = np.load(p, allow_pickle=True)
        if [str(x) for x in zz["names"]] != rn:
            continue
        key = group_key(zz)
        if key not in K_BY_GROUP:
            skipped.append(tag)
            continue
        d = zz["d_fin"]
        dd = d - d12
        if np.abs(dd).max() < 1e-12:
            continue
        k = K_BY_GROUP[key]
        pub = alg(d) + SLIP
        delta = (pub + PRIV_TRANSFER * k * NOISE_PER_DIR) - mu1_gard
        sd = float(np.sqrt(1.25 ** 2 * dd @ V @ dd) / FS)
        p3, p5 = p_pair(delta, sd)
        rows.append((p3, p5, tag, float(zz["gamma"]), k, alg(d), pub, delta, sd))
    for p3, p5, tag, g, k, al, pub, delta, sd in sorted(rows, reverse=True):
        print(f"{tag:10s} {g:5.2f} {k:6.2f} {al:11.7f} {pub:12.7f} {delta:+10.6f} "
              f"{sd/NOISE:7.2f} {p3*100:8.2f}% {(p3-b3)*100:+6.2f} {(p5-b5)*100:+6.2f}")
    if skipped:
        print(f"\nпропущены (k группы не выведен, сравнивать по пабликy НЕЛЬЗЯ): "
              f"{', '.join(sorted(skipped))}")

    if a.map:
        zz = np.load(LINEA / "gls_state_a3g0.npz", allow_pickle=True)
        dd = zz["d_fin"] - d12
        sd = float(np.sqrt(1.25 ** 2 * dd @ V @ dd) / FS)
        print(f"\nКАРТА РЕШЕНИЯ по замеру γ=0 (sd_d = {sd/NOISE:.2f} шума). "
              f"База — слот-2 = F8: {p_pair((1.6458057389+PRIV_TRANSFER*8.3*NOISE_PER_DIR)-mu1_gard, 2.41*NOISE)[0]*100:.2f} %")
        print(f"{'замер γ=0':>12s} |" + "".join(f"{f'k={k}':>10s}" for k in (9.3, 11.27, 13.0)))
        print("-" * 44)
        for pub in (1.645600, 1.645630, 1.645655, 1.645678, 1.645700, 1.645730, 1.645760):
            cells = []
            for k in (9.3, 11.27, 13.0):
                delta = (pub + PRIV_TRANSFER * k * NOISE_PER_DIR) - mu1_gard
                cells.append(f"{p_pair(delta, sd)[0]*100:9.2f}%")
            print(f"{pub:12.6f} |" + "".join(cells))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

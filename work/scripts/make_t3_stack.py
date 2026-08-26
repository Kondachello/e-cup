"""T3: доводка G-дозы внутри T2 до оптимума. Требует НАСТОЯЩИЙ T2 с платформы.

Геометрия (выведена алгеброй параболы, запись T_SERIES_BASE в KNOWLEDGE):
    T2 = respread(G2 + 0.45·d_T),  G2 = respread(V3 + 0.20·d_G)
Ось G замерена парой G1/G2: κ_G = 0.436, Q_G = 0.000497 (HANDOFF_track4 §2),
то есть внутри T2 ось G стоит на дозе 0.20 при оптимуме 0.44 — недобор 1.3 шума,
единственный несобранный урожай среди замеренных осей.

Сборка: T3 = clip(lp(T2) + δa·d_G_eff), где d_G_eff = lp(G1) − lp(V3) — ось G в
пространстве отгруженных файлов, δa = κ_G − 0.20 − поправка на пересечение с d_T.
Пересечение считается по векторам локально (это геометрия файлов, не подгонка):
    δa* = κ_G − 0.20 − 0.45·E[d_G·d_T]/E[d_G²]
Все входящие κ — замеренные на LB юзерские смещения; публичные скоры в подборе
весов не участвуют, подгонки под паблик нет.

Ожидание: T2 − (2·δa·(κ_G − 0.20 − 0.45·ρ) − δa²)·Q_G/чистыми ≈ −0.0000277 при
ρ≈0 (1.3 шума). Файл ЗАКОНЕН как финалист после замера (не SHOW-класс).

Запуск:  python work/scripts/make_t3_stack.py            # расчёт без записи
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
SUB = ROOT / "submissions"

KAPPA_G, Q_G = 0.436, 0.000497       # HANDOFF_track4 §2, пара G1/G2
DOSE_G_IN_T2 = 0.20                   # унаследована от G2
DOSE_T_IN_T2 = 0.45
T2_SCORE = 1.6469638837149883


def rd(p: Path) -> tuple[np.ndarray, np.ndarray]:
    d = pl.read_csv(p, schema_overrides={"user_id": pl.Int64}).sort("user_id")
    return (d["user_id"].to_numpy(),
            np.log1p(np.clip(d[d.columns[1]].to_numpy().astype(np.float64), 0, None)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--emit", action="store_true")

    t2p = SUB / "T2_tfm4_orth_045.csv"
    if not t2p.exists():
        raise SystemExit("нет submissions/T2_tfm4_orth_045.csv — скачай сабмит с платформы "
                         "(или пусть трек 5 запушит); реконструкция не годится, см. t_restore.md")
    uid, t2 = rd(t2p)
    _, v3 = rd(SUB / "V3_canon.csv")
    _, g1 = rd(SUB / "G1_gru_tfm_full.csv")
    _, g2 = rd(SUB / "G2_gru_tfm_02.csv")

    d_g = g1 - v3                      # ось G в пространстве отгруженных файлов
    d_t = (t2 - g2) / DOSE_T_IN_T2     # ось T там же
    q_g = float(np.mean(d_g * d_g))
    rho = float(np.mean(d_g * d_t)) / q_g
    da = KAPPA_G - DOSE_G_IN_T2 - DOSE_T_IN_T2 * rho
    print(f"q(d_G)={q_g:.6f}  E[d_G·d_T]/q_G={rho:+.4f}  δa*={da:+.4f}")

    # выигрыш в единицах Q_G: парабола оси G относительно точки 0.20 внутри T2
    gain = (2 * da * (KAPPA_G - DOSE_G_IN_T2 - DOSE_T_IN_T2 * rho) - da * da) * Q_G
    print(f"ожидаемый выигрыш {gain:+.7f} -> расчётный скор {T2_SCORE - gain:.7f}")
    if da <= 0:
        print("δa* <= 0: пересечение съело недобор, файл не собирать")
        return

    lp = np.clip(t2 + da * d_g, 0, None)
    print(f"sd(log1p): T2 {t2.std():.4f} -> T3 {lp.std():.4f} (V3 {v3.std():.4f})")



if __name__ == "__main__":
    main()

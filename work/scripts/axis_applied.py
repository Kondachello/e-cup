"""УЖЕ ЛИ ПРИМЕНЕНА ось? Проекция поправки цепочки на направление кандидата.

ЗАЧЕМ. `joint_gain.py` меряет прирост к колонке `blend` пакета — то есть к
НЕИСПРАВЛЕННОЙ базе. Отправляемый файл это не бленд, а бленд + 46 осей поправок,
измеренных зондами лидерборда. Модель может давать большой joint_gain и при этом
не стоить ни одной посылки, потому что цепочка уже стоит на её направлении.

Ровно так и вышло 30.08 с бестабличной осью: joint_gain дал маржинал +0.000230,
по формуле ценности зонда это 4.1 шума = +8.1 п.п. P(топ-3), гейт пройден, две
посылки из пяти были бы потрачены. Проекция показала **1.07 шага** — ось применена.

ЧТО СЧИТАЕТ.

    d     = направление кандидата (из make_tabless_dir.py или directions/*.parquet)
    corr  = lp(отправленный файл) − blend_test        (вся применённая поправка)
    proj  = <corr, d> / <d, d>                        (доза вдоль оси, в шагах)
    cos   = <corr, d> / (|corr|·|d|)

`proj ≈ 0` — ось не тронута, кандидат живой.
`proj ≈ 1` — цепочка уже прошла полный шаг, остался в лучшем случае передоз.

Контроль от самообмана: та же проекция для 200 СЛУЧАЙНЫХ направлений той же нормы.
Если наблюдаемое не отстоит от них на много сигм — это артефакт масштаба, а не
настоящее выравнивание.

Запуск:
  .venv/bin/python work/scripts/axis_applied.py --dir work/data/dir_N2_tabless.parquet
  .venv/bin/python work/scripts/axis_applied.py --model kevf_tl_gru    # направление из модели на лету
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
L1 = lambda x: np.log1p(np.clip(np.asarray(x, np.float64), 0, None))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", help="parquet направления (колонка d или step)")
    p.add_argument("--model", help="имя в work/preds: направление = lp(model_test) − blend_test")
    p.add_argument("--file", default="F12_ebint", help="отправленный файл в submissions/ (без .csv)")
    p.add_argument("--pack", default=str(ROOT / "work" / "preds_pack"))
    a = p.parse_args()
    if not (a.dir or a.model):
        raise SystemExit("нужен --dir или --model")

    t = pl.read_parquet(f"{a.pack}/test_preds.parquet").sort("user_id")
    uid = t["user_id"].to_numpy()
    blend = t["blend"].to_numpy().astype(np.float64)      # колонки пакета уже в log1p

    if a.dir:
        z = pl.read_parquet(a.dir).sort("user_id")
        col = "d" if "d" in z.columns else "step"
        d = z[col].to_numpy().astype(np.float64)
        tag = Path(a.dir).stem
    else:
        m = pl.read_parquet(ROOT / "work" / "preds" / f"{a.model}_test.parquet").sort("user_id")
        col = "pred" if "pred" in m.columns else "predict"
        assert np.array_equal(m["user_id"].to_numpy(), uid), "модель: другой user_id"
        d = L1(m[col].to_numpy()) - blend
        tag = a.model
    d = d - d.mean()
    q = float(np.mean(d * d))

    c = pl.read_csv(ROOT / "submissions" / f"{a.file}.csv").sort("user_id")
    assert np.array_equal(c["user_id"].to_numpy(), uid), "файл: другой user_id"
    corr = L1(c["predict"].to_numpy()) - blend
    corr = corr - corr.mean()

    proj = float(np.mean(corr * d) / q)
    cos = float(np.mean(corr * d) / np.sqrt(q * float(np.mean(corr * corr))))

    rng = np.random.default_rng(0)
    ps = []
    for _ in range(200):
        z_ = rng.standard_normal(len(d))
        z_ -= z_.mean()
        z_ *= np.sqrt(q / np.mean(z_ * z_))
        ps.append(float(np.mean(corr * z_) / q))
    ps = np.array(ps)
    sig = abs(proj - ps.mean()) / ps.std()

    print(f"ось {tag}  против {a.file}")
    print(f"  q = mean(d²) = {q:.4e}   |d| rms {np.sqrt(q):.5f}   |поправка| rms {np.sqrt(np.mean(corr*corr)):.5f}")
    print(f"  ПРОЕКЦИЯ = {proj:+.4f} шага     cos = {cos:+.4f}")
    print(f"  контроль (200 случайных направлений той же нормы): {ps.mean():+.4f} ± {ps.std():.4f}")
    print(f"  наблюдаемое отстоит на {sig:.0f} сигм -> "
          f"{'ВЫРАВНИВАНИЕ НАСТОЯЩЕЕ' if sig > 5 else 'может быть артефактом масштаба'}")
    if abs(proj) < 0.15 and sig > 5:
        print("\n  ВЕРДИКТ: ось НЕ применена — кандидат живой, зонд имеет смысл.")
    elif abs(proj) < 0.15:
        print("\n  ВЕРДИКТ: проекция мала, но и не отличима от случайной. Смотреть q и новизну.")
    else:
        print(f"\n  ВЕРДИКТ: ось УЖЕ ПРИМЕНЕНА на {proj:.0%} шага. Зонд не покупает новую ось;")
        print("  в лучшем случае это передоз, и его цена — q·(b_opt−b_now)²/(2F0), обычно доли шума.")


if __name__ == "__main__":
    main()

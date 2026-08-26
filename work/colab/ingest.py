"""Приёмка предсказаний, обученных на GPU Colab, в наш пул моделей.

ЗАЧЕМ ОТДЕЛЬНЫЙ СКРИПТ. Сырой скор модели ничего не значит: калибровка переписывает
уровень, и на сырых сравнениях мы обманывались девять раз. Поэтому файл из Colab
не попадает в пул раньше, чем про него сказаны три вещи:

  1. калиброванный скор — единственный сопоставимый с остальным зоопарком;
  2. ЗАПАС над блендом: по выведенному тождеству corr(ошибка бленда, ошибка модели)
     тождественно равна sb/sm, поэтому корреляция ошибок не несёт никакой своей
     информации. Значение имеет только разность δ = sb/sm − ρ (корреляция
     НЕЦЕНТРИРОВАННАЯ — тождество выполняется для E[e·e_b]/(sm·sb), не для corrcoef).
     Модель с δ ≤ 0 не даёт ничего, каким бы ни был её скор;
  2а. ВКЛАД — точной алгеброй пары, а НЕ по формуле 7.1·δ². Формула похоронена
     20–21.08: она откалибрована на слабых моделях (sm/sb≈1.06), объясняет 9%
     разброса и занижает сильные модели в 3–20 раз — по ней браковалась kostya46,
     крупнейший член бленда. Здесь используется та же функция, что в margin.py,
     чтобы у проекта была ОДНА реализация приёмки;
  3. сравнение рук: small против big на одинаковых сидах и срезах. Это и есть
     ответ на вопрос, покупает ли ёмкость что-нибудь.

Запуск:
    POLARS_MAX_THREADS=3 .venv/bin/python work/colab/ingest.py --name gseq_big_s42
    POLARS_MAX_THREADS=3 .venv/bin/python work/colab/ingest.py --compare
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import polars as pl

# Корень репозитория: OZON_ROOT, иначе два уровня вверх от этого файла
# (work/colab/ -> repo). Захардкоженный путь одной машины делал скрипт
# неработающим у всех остальных, включая проверяющего на чистом клоне.
ROOT = Path(os.environ.get("OZON_ROOT", str(Path(__file__).resolve().parents[2])))
sys.path.insert(0, str(ROOT / "work" / "scripts"))
from common import PREDS_DIR, VAL_ANCHOR, load_anchor, rmsle          # noqa: E402
from calibrate import fit_shifts, apply_shifts                        # noqa: E402
from margin import NOISE, THRESHOLD, pair_contribution                # noqa: E402

OUT = ROOT / "work" / "colab" / "out"


def cal_2fold(lp: np.ndarray, ly: np.ndarray, y: np.ndarray,
              half: np.ndarray, bins: int = 24) -> np.ndarray:
    """Калибровка без самокалибровки: половина A настраивает сдвиги для B и наоборот."""
    out = np.empty_like(lp)
    c_a, s_a = fit_shifts(lp[half], ly[half], bins)
    out[~half] = apply_shifts(lp[~half], c_a, s_a)
    c_b, s_b = fit_shifts(lp[~half], ly[~half], bins)
    out[half] = apply_shifts(lp[half], c_b, s_b)
    return out


def blend_val(uid: np.ndarray) -> np.ndarray | None:
    """Действующий бленд на валидации — база для расчёта запаса."""
    p = ROOT / "work" / "preds_pack" / "val_preds.parquet"
    if not p.exists():
        return None
    d = pl.read_parquet(p).sort("user_id")
    if not np.array_equal(d["user_id"].to_numpy(), uid):
        return None
    return d["blend"].to_numpy().astype(np.float64)


def assess(name: str, verbose: bool = True) -> dict | None:
    val = load_anchor(VAL_ANCHOR, columns=["user_id", "target"]).sort("user_id")
    uid = val["user_id"].to_numpy()
    y = val["target"].to_numpy().astype(np.float64)
    ly = np.log1p(y)

    pv = OUT / f"{name}_val.parquet"
    if not pv.exists():
        print(f"нет файла {pv}")
        return None
    d = pl.read_parquet(pv).sort("user_id")
    if not np.array_equal(d["user_id"].to_numpy(), uid):
        print(f"{name}: user_id не совпадает с валидацией")
        return None
    lp = np.log1p(np.clip(d["pred"].to_numpy().astype(np.float64), 0, None))

    rng = np.random.default_rng(0)
    half = rng.random(len(uid)) < 0.5
    lc = cal_2fold(lp, ly, y, half)
    raw = rmsle(y, np.expm1(lp))
    cal = rmsle(y, np.expm1(lc))

    res = {"name": name, "raw": raw, "cal": cal}
    bl = blend_val(uid)
    if bl is not None:
        eb, em = bl - ly, lc - ly
        sb, sm = float(np.sqrt(np.mean(eb ** 2))), float(np.sqrt(np.mean(em ** 2)))
        # НЕцентрированная корреляция: тождество ЗАПАС = sb/sm − ρ выполняется для
        # E[e·e_b]/(sm·sb). np.corrcoef центрирует обе ошибки и искажает результат у
        # любой модели с ненулевым средним ошибки (Женя намерил: 6 из 30).
        rho = float(np.mean(em * eb) / max(sm * sb, 1e-12))
        delta = sb / max(sm, 1e-12) - rho
        res |= {"sb": sb, "sm": sm, "rho": rho, "margin": delta,
                "contribution": pair_contribution(sb, sm, delta)}

    j = OUT / f"{name}.json"
    if j.exists():
        m = json.loads(j.read_text())
        res |= {"arm": m.get("arm"), "seed": m.get("seed"),
                "cal_on_colab": m.get("cal_rmsle_ckptavg"), "minutes": m.get("minutes")}

    if verbose:
        print(f"{name}: сырой {raw:.6f}  калиброванный {cal:.6f}")
        if "cal_on_colab" in res and res["cal_on_colab"]:
            d_ = abs(res["cal_on_colab"] - cal)
            print(f"  скор, посчитанный на Colab: {res['cal_on_colab']:.6f} "
                  f"(расхождение {d_:.6f}{'  ВНИМАНИЕ' if d_ > 1e-4 else ''})")
        if "margin" in res:
            print(f"  бленд {res['sb']:.6f}, модель {res['sm']:.6f}, ρ {res['rho']:.5f}")
            c = res["contribution"]
            verdict = ("ВЫШЕ ПОРОГА" if c >= THRESHOLD else
                       "слабо, но не шум" if c >= 2 * NOISE else "шум")
            print(f"  ЗАПАС δ = {res['margin']:+.5f}  ->  вклад в бленд "
                  f"{c:.6f}   {verdict} (порог {THRESHOLD}, шум {NOISE})")
            print("  наборы кандидатов мерить joint_gain.py: запасы НЕ складываются")
    return res


def cmd_ingest(args) -> None:
    res = assess(args.name)
    if res is None:
        sys.exit(1)
    if args.copy:
        for suf in ("val", "test"):
            src = OUT / f"{args.name}_{suf}.parquet"
            if not src.exists():
                print(f"нет {src}, копирование прервано")
                sys.exit(1)
            shutil.copy2(src, PREDS_DIR / f"{args.name}_{suf}.parquet")
        print(f"скопировано в {PREDS_DIR} — теперь модель видна пересборке бленда")
    else:
        print("файлы НЕ скопированы в пул; добавь --copy, когда вердикт устроит")


def cmd_compare(args) -> None:
    """Сравнение рук. Единственный честный вопрос к этому опыту."""
    names = sorted({p.name.rsplit("_val.parquet", 1)[0] for p in OUT.glob("*_val.parquet")})
    if not names:
        print("в work/colab/out пока ничего нет")
        return
    rows = [r for r in (assess(n, verbose=False) for n in names) if r]
    rows.sort(key=lambda r: r["cal"])
    w = max(len(r["name"]) for r in rows)
    print(f"{'модель':<{w}}  {'рука':<6} {'калибр.':>9} {'запас δ':>9} {'вклад':>9}")
    for r in rows:
        print(f"{r['name']:<{w}}  {str(r.get('arm','?')):<6} {r['cal']:>9.6f} "
              f"{r.get('margin', float('nan')):>+9.5f} "
              f"{r.get('contribution', float('nan')):>9.6f}")
    big = [r for r in rows if r.get("arm") == "big"]
    small = [r for r in rows if r.get("arm") == "small"]
    if big and small:
        b = min(r["cal"] for r in big)
        s = min(r["cal"] for r in small)
        print(f"\nлучшая big {b:.6f} против лучшей small {s:.6f}: "
              f"ёмкость даёт {s - b:+.6f} калиброванного скора")
        print("если разница около нуля — направление закрыто, GPU не помогает;")
        print("если big заметно лучше, следующий шаг это ещё ёмкость и табличная часть")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd")
    ap.add_argument("--name", help="имя набора в work/colab/out")
    ap.add_argument("--copy", action="store_true",
                    help="скопировать в work/preds после осмотра вердикта")
    ap.add_argument("--compare", action="store_true", help="сравнить все руки")
    args = ap.parse_args()
    if args.compare:
        cmd_compare(args)
    elif args.name:
        cmd_ingest(args)
    else:
        ap.error("укажи --name ИМЯ или --compare")


if __name__ == "__main__":
    main()

"""ЗАПАС трансформера против ДЕЙСТВУЮЩЕГО бленда, без внешних зависимостей.

Зачем: штатная цепочка calibrate.py -> err_corr.py на этой машине не пойдёт по
двум причинам сразу —

  1. calibrate.py импортирует exp_lib.py, а тот делает `import fcntl`: на Windows
     это ImportError ещё до расчёта;
  2. err_corr.py требует девять файлов *_cal_val.parquet в work/preds/, и трёх из
     них (fusion_f_cal 0.32, behavonly_cal 0.08, countaov_cal 0.07) нет ни у нас,
     ни в work/preds_pack/. blend_lp() вызывается безусловно, поэтому --file не
     спасает.

Этот скрипт делает то же самое, опираясь только на work/preds_pack/val_preds.parquet
и test_preds.parquet, которые лежат в репозитории. Калибровка воспроизведена из
calibrate.py дословно (24 квантильных бина, честный разрез по половинам юзеров).

ВАЖНО про эталон. err_corr.py сравнивает с ЗАХАРДКОЖЕННЫМ девятичленным блендом
(в докстринге 1.666718). Действующий чемпион — blend_opt со скором 1.666395, и
именно он лежит колонкой `blend` в паке (см. work/reports/scores.tsv). Это разные
объекты, разница 0.000323, то есть 15 шумовых единиц лидерборда. Против более
слабого эталона ЗАПАС механически выше примерно на 0.00019 — не потому, что
модель лучше. Скрипт печатает оба числа, чтобы это было видно.

  python work/scripts/seq/margin_vs_pack.py --seeds tfm_s1 tfm_s2 tfm_s3
  python work/scripts/seq/margin_vs_pack.py --pred tfm3        # уже усреднённый
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import polars as pl

BINS = 24
ERRCORR_BLEND_RMSLE = 1.666718   # из докстринга err_corr.py, для справки
PROJECT_RECORD = 0.00307   # lagd28, ось застоялости входа. Прежние 0.00193 (febspec2)
                           # мерились против более слабого бленда прошлой эпохи.


def exact_contrib(sb: float, sm: float, margin: float) -> float:
    """Точная алгебра пары вместо похороненного правила 7.1*запас**2.

    Старое правило объясняло 9% разброса и занижало сильные модели в 15-20 раз:
    по нему браковалась kostya46 (запас 0.00137, реальный вклад 0.000226) -
    крупнейший член действующего бленда. Отрицательный запас означает beta >= 1,
    то есть нулевой вес у оптимизатора, поэтому пол не косметический.
    """
    z = max(margin, 0.0)
    den = (sm * sm - sb * sb + 2.0 * sb * sm * z) * 2.0 * sb
    return (sb * sb * sm * sm * z * z) / den if den > 1e-12 else 0.0


def find_root(explicit: str | None) -> Path:
    if explicit:
        return Path(explicit).resolve()
    if os.environ.get("OZON_ROOT"):
        return Path(os.environ["OZON_ROOT"]).resolve()
    guess = Path(__file__).resolve().parents[3]
    if (guess / "work" / "preds_pack").exists():
        return guess
    return Path.cwd().resolve()


def L1(x) -> np.ndarray:
    return np.log1p(np.clip(np.asarray(x, dtype=np.float64), 0, None))


def rmsle_log(lp: np.ndarray, ly: np.ndarray) -> float:
    return float(np.sqrt(np.mean((lp - ly) ** 2)))


# --- калибровка, дословно из work/scripts/calibrate.py ---------------------
def fit_shifts(lp: np.ndarray, ly: np.ndarray, bins: int = BINS):
    qs = np.quantile(lp, np.linspace(0, 1, bins + 1))
    qs[0] -= 1e-9
    qs[-1] += 1e-9
    centers, shifts = [], []
    for i in range(bins):
        m = (lp > qs[i]) & (lp <= qs[i + 1])
        if m.sum() < 500:
            continue
        centers.append(lp[m].mean())
        shifts.append(ly[m].mean() - lp[m].mean())
    return np.array(centers), np.array(shifts)


def apply_shifts(lp: np.ndarray, centers: np.ndarray, shifts: np.ndarray) -> np.ndarray:
    return np.clip(lp + np.interp(lp, centers, shifts), 0, None)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="")
    ap.add_argument("--seeds", nargs="*", default=[],
                    help="имена сидов в work/preds/ -> усреднить в лог-пространстве")
    ap.add_argument("--pred", default="",
                    help="имя уже усреднённой модели в work/preds/ (вместо --seeds)")
    ap.add_argument("--save", default="",
                    help="сохранить усреднённые сырые предсказания под этим именем в work/preds/")
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    if not a.seeds and not a.pred:
        ap.error("нужен --seeds или --pred")

    root = find_root(a.root or None)
    preds = root / "work" / "preds"
    pack = root / "work" / "preds_pack"

    dv = pl.read_parquet(pack / "val_preds.parquet").sort("user_id")
    uid = dv["user_id"].to_numpy()
    ly = L1(dv["target"].to_numpy())
    lb = dv["blend"].to_numpy().astype(np.float64)      # уже log1p
    sb = rmsle_log(lb, ly)
    eb = lb - ly
    print(f"эталон: колонка `blend` из пакета (действующий blend_opt)  RMSLE={sb:.6f}  n={len(uid)}")

    # --- собрать прогноз модели на валидации --------------------------------
    def load(name: str, split: str) -> tuple[np.ndarray, np.ndarray]:
        d = pl.read_parquet(preds / f"{name}_{split}.parquet").sort("user_id")
        col = "pred" if "pred" in d.columns else "predict"
        return d["user_id"].to_numpy(), d[col].to_numpy().astype(np.float64)

    if a.seeds:
        acc_v = np.zeros(len(uid))
        acc_t = None
        uid_t = None
        for s in a.seeds:
            u, p = load(s, "val")
            assert np.array_equal(u, uid), f"{s}_val: другой набор user_id"
            acc_v += L1(p) / len(a.seeds)
            ut, pt = load(s, "test")
            if acc_t is None:
                uid_t, acc_t = ut, np.zeros(len(ut))
            assert np.array_equal(ut, uid_t), f"{s}_test: другой набор user_id"
            acc_t += L1(pt) / len(a.seeds)
            print(f"  + {s}")
        lp, lt = acc_v, acc_t
    else:
        u, p = load(a.pred, "val")
        assert np.array_equal(u, uid), f"{a.pred}_val: другой набор user_id"
        lp = L1(p)
        uid_t, pt = load(a.pred, "test")
        lt = L1(pt)

    if a.save:
        preds.mkdir(parents=True, exist_ok=True)
        pl.DataFrame({"user_id": uid.astype(np.int64), "pred": np.expm1(lp)}).write_parquet(
            preds / f"{a.save}_val.parquet")
        pl.DataFrame({"user_id": uid_t.astype(np.int64), "pred": np.expm1(lt)}).write_parquet(
            preds / f"{a.save}_test.parquet")
        print(f"сохранено (СЫРЫЕ, без поправки +0.17): {a.save}_val.parquet, {a.save}_test.parquet")

    # --- калибровка с честным разрезом --------------------------------------
    rng = np.random.default_rng(0)
    half = rng.permutation(len(uid)) < len(uid) // 2
    c1, s1 = fit_shifts(lp[half], ly[half])
    hold_base = rmsle_log(lp[~half], ly[~half])
    hold_cal = rmsle_log(apply_shifts(lp[~half], c1, s1), ly[~half])
    print(f"\nчестный holdout: {hold_base:.6f} -> {hold_cal:.6f} "
          f"({'ОК' if hold_cal < hold_base else 'КАЛИБРОВКА НЕ ПОМОГАЕТ'})")

    centers, shifts = fit_shifts(lp, ly)
    lpc = apply_shifts(lp, centers, shifts)

    out = {"blend_rmsle_pack": sb, "n": int(len(uid))}
    print(f"\n{'':22} {'скор':>10} {'r ошибок':>10} {'тождество':>11} {'ЗАПАС':>10} {'вклад~':>10}")
    for label, v in (("сырой", lp), ("калиброванный", lpc)):
        sm = rmsle_log(v, ly)
        e = v - ly
        # НЕцентрированная корреляция: тождество ЗАПАС = sb/sm - rho выполняется
        # для E[e*eb]/(sm*sb), а не для corrcoef. Центрирование искажало 6 моделей
        # из 30 - ровно все некалиброванные (ненулевое среднее ошибки). Строка
        # "сырой" ниже как раз некалиброванная, то есть страдала именно она.
        rho = float(np.mean(e * eb) / max(sm * sb, 1e-12))
        ident = sb / max(sm, 1e-12)
        margin = ident - rho
        contrib = exact_contrib(sb, sm, margin)
        print(f"  {label:20} {sm:10.6f} {rho:10.5f} {ident:11.6f} {margin:+10.5f} {contrib:10.6f}")
        out[label] = {"rmsle": sm, "err_corr": rho, "corr_expected": ident,
                      "margin": margin, "contrib_est": contrib}

    m = out["калиброванный"]["margin"]
    print(f"\nрекорд проекта по запасу {PROJECT_RECORD}: "
          f"{'ПОБИТ' if m > PROJECT_RECORD else 'не побит'} ({m:+.5f})")

    # --- поправка на разницу эталонов ---------------------------------------
    sm_cal = out["калиброванный"]["rmsle"]
    shift = (ERRCORR_BLEND_RMSLE - sb) / sm_cal
    print(f"\nсправочно: против ЗАХАРДКОЖЕННОГО девятичленного бленда ({ERRCORR_BLEND_RMSLE})")
    print(f"ЗАПАС был бы выше примерно на {shift:+.5f} -> около {m + shift:+.5f}.")
    print("Это разница ЭТАЛОНОВ, а не качества модели. Нужно это теперь только чтобы")
    print("сверяться с числами из СТАРЫХ отчётов: с 21.08 err_corr.py сам читает колонку")
    print("blend из пакета, так что оба инструмента считают против одного эталона.")
    out["errcorr_reference_shift_est"] = shift

    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1, ensure_ascii=False))
        print(f"\nзаписано {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

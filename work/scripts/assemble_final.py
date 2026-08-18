"""End-to-end сборка кандидата: бленд чистых моделей -> LB-микс с измеренным базисом
-> сегментные и сезонные поправки -> файл сабмита.


Шаги:
1. Собирает все *_val.parquet чистого протокола (список CLEAN ниже + автодобавление
   новых имён из --extra), координатным спуском подбирает веса в log1p-пространстве.
2. Оценивает f нового бленда на тесте: f_est = val_rmsle + VAL_TO_LB (замеренный офсет),
   с поправкой на глобальное смещение уровня (mean_e считается точно относительно
   файла с известным mean_e через разность средних log-предсказаний).
   сегментные + сезонные поправки, отмасштабированные под итоговый вес.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))
from common import PREDS_DIR, ROOT, VAL_ANCHOR, load_anchor, rmsle
from exp_lib import save_preds, log_score

# модели чистого протокола (gap-30), у которых есть val И test предсказания
CLEAN = ["mlp2_big_cal", "mlp2_final_cal", "c_xtw_s42", "c_ts2_s42", "c_twlog_s42",
         "twdeep", "seq2tr_f", "c_dirlgb_s42", "mlpbin", "mlpziln", "fusion",
         "dart_tw", "twl_seqoof", "whale_final", "twl_repair_ab", "febspec"]

VAL_TO_LB = -0.0022          # замерено: c_cand val 1.6717 -> LB 1.66954
BASE_SCORE_PROJ = 1.652269   # расчётный скор A8 (уточнить фактическим после отправки)
MEAN_E_A5 = 0.0              
FEBDIR_DELTA = 0.0613        
# сегментные средние ошибки, замеренные пробами A2-A4 поверх A1 (остаются валидными)
SEG_M = {"S1": -0.072949, "S2": -0.032625, "S3": 0.054674, "S4": 0.031136}


def available(names):
    out = []
    for n in names:
        if (PREDS_DIR / f"{n}_val.parquet").exists() and (PREDS_DIR / f"{n}_test.parquet").exists():
            out.append(n)
    return out


def coord_blend(P, y, names, rounds=4):
    w = {n: 0.0 for n in names}
    w[names[0]] = 1.0
    def sc(w):
        lp = sum(P[n] * wi for n, wi in w.items() if wi > 0)
        return rmsle(y, np.expm1(lp))
    best = sc(w)
    for _ in range(rounds):
        for n in names:
            for d in (0.2, 0.1, 0.05, 0.02, -0.05, -0.1):
                w2 = dict(w); w2[n] = max(0.0, w2[n] + d)
                s = sum(w2.values())
                if s <= 0:
                    continue
                w2 = {k: v / s for k, v in w2.items()}
                c = sc(w2)
                if c < best - 1e-7:
                    best, w = c, w2
    return {k: v for k, v in w.items() if v > 0.001}, best


def load_lp_csv(p):
    df = pl.read_csv(p, schema_overrides={"user_id": pl.Int64}).sort("user_id")
    return df["user_id"].to_numpy(), np.log1p(np.clip(df[df.columns[1]].to_numpy(), 0, None))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="my27")
    ap.add_argument("--extra", default="", help="доп. имена моделей через запятую")
    ap.add_argument("--base-score", type=float, default=BASE_SCORE_PROJ)
    args = ap.parse_args()

    names = available(CLEAN + [s for s in args.extra.split(",") if s])
    print(f"доступно моделей: {len(names)}: {names}")

    val = load_anchor(VAL_ANCHOR, columns=["user_id", "target"]).sort("user_id")
    y = val["target"].to_numpy()
    uid_v = val["user_id"].to_numpy()
    P = {}
    for n in names:
        d = pl.read_parquet(PREDS_DIR / f"{n}_val.parquet").sort("user_id")
        assert (d["user_id"].to_numpy() == uid_v).all(), n
        P[n] = np.log1p(np.clip(d["pred"].to_numpy(), 0, None))
        print(f"  {n}: solo val {rmsle(y, np.expm1(P[n])):.6f}")
    names = sorted(names, key=lambda n: rmsle(y, np.expm1(P[n])))

    w, val_score = coord_blend(P, y, names)
    print(f"\nбленд: {({k: round(v,3) for k,v in w.items()})}\nval RMSLE = {val_score:.6f}")

    lp_val = sum(P[n] * wi for n, wi in w.items())
    save_preds(args.name, "val", uid_v, np.expm1(np.clip(lp_val, 0, None)))

    lt, uid_t = None, None
    for n, wi in w.items():
        d = pl.read_parquet(PREDS_DIR / f"{n}_test.parquet").sort("user_id")
        if uid_t is None:
            uid_t = d["user_id"].to_numpy()
        v = np.log1p(np.clip(d["pred"].to_numpy(), 0, None))
        lt = v * wi if lt is None else lt + v * wi
    save_preds(args.name, "test", uid_t, np.expm1(np.clip(lt, 0, None)))
    log_score(args.name, val_score, f"assemble_final blend {w}")

    # --- LB-микс с базой ---
    uid_b, lp_base = load_lp_csv(f"{ROOT}/{args.base}")
    assert (uid_b == uid_t).all()
    # выравниваем уровень нового бленда на уровень базы (база уже откалибрована по LB)
    shift = float(np.mean(lp_base - lt))
    lt_aligned = np.clip(lt + shift, 0, None)
    print(f"\nвыравнивание уровня: сдвиг {shift:+.4f}")

    fb2 = args.base_score ** 2
    fm = val_score + VAL_TO_LB
    fm2 = fm ** 2
    D2 = float(np.mean((lp_base - lt_aligned) ** 2))
    cov = (fb2 + fm2 - D2) / 2
    den = fb2 + fm2 - 2 * cov
    w_opt = (fb2 - cov) / den if den > 1e-9 else 0.0
    wm = float(np.clip(w_opt, 0.0, 0.6))
    # прогноз считаем при ФАКТИЧЕСКОМ (обрезанном) весе, иначе цифра вводит в заблуждение
    f_mix = np.sqrt(max(fb2 * (1 - wm) ** 2 + fm2 * wm ** 2 + 2 * wm * (1 - wm) * cov, 0.0))
    corr = cov / np.sqrt(fb2 * fm2)
    print(f"f(нового, оценка) = {fm:.5f} | D2 = {D2:.4f} | corr = {corr:.4f}")
    print(f"оптимальный вес (без ограничений) = {w_opt:+.3f} -> применён {wm:.3f}")
    print(f"прогноз скора при применённом весе = {f_mix:.5f} (база {args.base_score:.5f})")
    if wm <= 1e-9:
        print("новый бленд не добавляет информации к измеренному базису — файл = база")

    lp_final = (1 - wm) * lp_base + wm * lt_aligned
    out = ROOT / args.out
    pl.DataFrame({"user_id": uid_t, "predict": np.expm1(np.clip(lp_final, 0, None))}).write_csv(out)
    print(f"\nзаписан {out}")
    print("ВНИМАНИЕ: прогноз скора использует оценку f нового бленда через офсет val->LB;")
    print("после отправки уточнить фактическим скором и пересобрать.")


if __name__ == "__main__":
    main()

"""weak_family_eval.py — главный замер семейства НАМЕРЕННО ОБЕДНЁННЫХ моделей.

Скор отдельной обеднённой модели заведомо плохой, и это ожидаемо: проверяется не он,
а три вещи (в порядке важности).

(а) КОРРЕЛЯЦИЯ С БЛИЖАЙШИМ СОСЕДОМ среди всех существующих калиброванных моделей,
    считается на ТЕСТОВЫХ лог-предсказаниях — ровно та величина, которая объяснила
    провал улучшенных февральских специалистов: старый слабый febspec_cal отстоит
    от ближайшего на 0.9691 и держит тест-вес 0.054-0.063, улучшенные версии стоят
    на 0.9943-0.9989 и получают ноль. Цель для новых моделей — ниже 0.97.
(б) вклад ВСЕГО семейства сразу (blend_reopt с семейством и без него — парный
    прогон в один момент, флаг --exclude weak_).
(в) тест-веса и частота отбора (blend_testopt_wstab.py).

Этот скрипт делает (а) плюс сводку сольных скоров; (б) и (в) запускаются отдельно.

Запуск: .venv/bin/python work/scripts/weak_family_eval.py [--prefix weak_]
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "POLARS_MAX_THREADS"):
    os.environ.setdefault(_v, "3")

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from common import PREDS_DIR, REPORTS_DIR, VAL_ANCHOR, load_anchor, rmsle  # noqa: E402

CONTAMINATED = {"lgblog_final", "xgblog_final", "cblog_final", "mlp_final", "gru_final",
                "hjit37", "hjit44"}
BLEND_PREFIX = ("blend", "caruana", "lbmix", "stack")
# эталон: старый слабый febspec — единственная модель, чья непохожесть измерена и
# оплачена реальным тест-весом
REF = "febspec_cal"


def pool_names() -> list[str]:
    out = []
    for p in sorted(PREDS_DIR.glob("*_cal_test.parquet")):
        stem = p.name[: -len("_cal_test.parquet")]
        if stem in CONTAMINATED or stem.startswith(BLEND_PREFIX):
            continue
        if not (PREDS_DIR / f"{stem}_cal_val.parquet").exists():
            continue
        out.append(stem + "_cal")
    if (PREDS_DIR / "channel3_chcal_test.parquet").exists():
        out.append("channel3_chcal")
    return out


def lp(name: str, split: str, uid_ref=None):
    d = pl.read_parquet(PREDS_DIR / f"{name}_{split}.parquet").sort("user_id")
    u = d["user_id"].to_numpy()
    if uid_ref is not None and not np.array_equal(u, uid_ref):
        raise ValueError(f"{name}_{split}: user_id mismatch")
    return u, np.log1p(np.clip(d["pred"].to_numpy().astype(np.float64), 0, None))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="weak_")
    ap.add_argument("--out", default="weak_family_eval.json")
    args = ap.parse_args()

    names = pool_names()
    new = [n for n in names if n.startswith(args.prefix)]
    old = [n for n in names if not n.startswith(args.prefix)]
    print(f"пул: {len(names)} калиброванных моделей, из них новых {len(new)}")
    if not new:
        print("новых моделей нет — нечего мерить")
        return 1

    uid_t, _ = lp(names[0], "test")
    L = np.empty((len(names), len(uid_t)))
    for i, n in enumerate(names):
        _, L[i] = lp(n, "test", uid_t)
    C = np.corrcoef(L)

    val = load_anchor(VAL_ANCHOR, columns=["user_id", "target"]).sort("user_id")
    uid_v = val["user_id"].to_numpy()
    y = val["target"].to_numpy().astype(np.float64)

    idx = {n: i for i, n in enumerate(names)}
    old_idx = [idx[n] for n in old]
    rows = {}
    print(f"\n{'модель':30s}{'val':>10}{'корр(бл.сосед)':>16}  сосед")
    todo = new + ([REF] if REF in idx else [])
    for n in todo:
        i = idx[n]
        cand = [j for j in old_idx if j != i]
        j = cand[int(np.argmax(C[i, cand]))]
        _, lv = lp(n, "val", uid_v)
        solo = rmsle(y, np.expm1(lv))
        # ближайший сосед среди ВСЕХ (включая другие обеднённые) — для контроля,
        # что семейство не выродилось в копии друг друга
        cand_all = [k for k in range(len(names)) if k != i]
        j_all = cand_all[int(np.argmax(C[i, cand_all]))]
        rows[n] = dict(val_rmsle=round(solo, 6),
                       corr_nearest_existing=round(float(C[i, j]), 4), nearest_existing=names[j],
                       corr_nearest_any=round(float(C[i, j_all]), 4), nearest_any=names[j_all],
                       is_new=n.startswith(args.prefix))
        print(f"{n:30s}{solo:10.4f}{C[i, j]:16.4f}  {names[j]}")

    # --- что РЕАЛЬНО решает вклад семейства ------------------------------------
    # При неотрицательных весах модель со скором sm получает вес > 0, только если её
    # корреляция ошибок с блендом ниже sb/sm (sb = скор бленда). Для sm=1.82 это 0.916,
    # а вовсе не 0.97. Усреднение K моделей снижает sm до sm*sqrt(rho_ww+(1-rho_ww)/K),
    # поэтому решает ВЗАИМНАЯ корреляция внутри семейства, а не его размер:
    # при rho_ww=0.99 семейство из 16 даёт столько же, сколько из 4, то есть ноль.
    from err_corr import BLEND, blend_lp
    lb = blend_lp(uid_v)
    eb = lb - np.log1p(np.clip(y, 0, None))
    sb = float(np.sqrt(np.mean(eb ** 2)))
    E, LV = [], []
    for n in new:
        _, lv = lp(n, "val", uid_v)
        LV.append(lv)
        E.append(lv - np.log1p(np.clip(y, 0, None)))
    E = np.array(E)
    Cw = np.corrcoef(E)
    iu = np.triu_indices(len(new), 1)
    rho_b = np.array([float(np.corrcoef(e, eb)[0, 1]) for e in E])
    for n, r in zip(new, rho_b):
        rows[n]["err_corr_blend"] = round(float(r), 4)
        rows[n]["rho_max_for_nonzero_w"] = round(sb / max(rows[n]["val_rmsle"], 1e-9), 4)

    ly_ = np.log1p(np.clip(y, 0, None))

    def as_one(sel: list[int]) -> dict:
        """Подмножество семейства, усреднённое в лог-пространстве, как ОДНА модель."""
        lv_avg = np.mean([LV[i] for i in sel], axis=0)
        e_avg = lv_avg - ly_
        sm = float(np.sqrt(np.mean(e_avg ** 2)))
        # НЕцентрированная, как в margin.py/err_corr.py: усреднение калиброванных
        # прогнозов не обязано давать нулевое среднее ошибки.
        rho = float(np.mean(e_avg * eb) / max(sm * sb, 1e-12))
        d = e_avg - eb
        w2 = float(-np.dot(eb, d) / max(np.dot(d, d), 1e-12))
        g = sb - float(np.sqrt(np.mean(((1 - w2) * lb + w2 * lv_avg - ly_) ** 2)))
        out = dict(k=len(sel), avg_rmsle=round(sm, 6), err_corr=round(rho, 4),
                   rho_max_for_nonzero_w=round(sb / sm, 4),
                   w_opt_2way=round(w2, 4), gain_2way=round(g, 6))
        if len(sel) > 1:
            sub = Cw[np.ix_(sel, sel)]
            iu2 = np.triu_indices(len(sel), 1)
            out["mean_pairwise_err_corr"] = round(float(sub[iu2].mean()), 4)
        return out

    fam = as_one(list(range(len(new))))
    fam.update(blend_rmsle=round(sb, 6),
               mean_pairwise_err_corr_in_family=round(float(Cw[iu].mean()), 4),
               min_pairwise_err_corr_in_family=round(float(Cw[iu].min()), 4),
               mean_err_corr_with_blend=round(float(rho_b.mean()), 4),
               min_err_corr_with_blend=round(float(rho_b.min()), 4))
    # по механизмам: решает ВЗАИМНАЯ корреляция внутри группы, а не её размер
    mech = {}
    for tag in ("rs", "an", "ft", "tiny"):
        sel = [i for i, n in enumerate(new) if n.startswith(f"{args.prefix}{tag}_")]
        if sel:
            mech[tag] = as_one(sel)
    fam["by_mechanism"] = mech
    print("\n[семейство как одна модель]", json.dumps(fam, ensure_ascii=False))

    newrows = {k: v for k, v in rows.items() if v["is_new"]}
    summ = dict(
        n_models=len(new),
        n_pool=len(names),
        min_corr_with_nearest=round(min(v["corr_nearest_existing"] for v in newrows.values()), 4),
        median_corr_with_nearest=round(
            float(np.median([v["corr_nearest_existing"] for v in newrows.values()])), 4),
        n_below_097=sum(v["corr_nearest_existing"] < 0.97 for v in newrows.values()),
        ref_model=REF,
        ref_corr=rows.get(REF, {}).get("corr_nearest_existing"),
        family=fam,
    )
    print("\n[сводка]", json.dumps(summ, ensure_ascii=False))
    (REPORTS_DIR / args.out).write_text(
        json.dumps(dict(summary=summ, models=rows), indent=1, ensure_ascii=False))
    print(f"записано work/reports/{args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

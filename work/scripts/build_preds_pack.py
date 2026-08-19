"""Сборка компактного набора предсказаний для сокомандников.

Что кладём и зачем:
  * target на валидации — без него ничего измерить нельзя;
  * blend — действующий бленд, потому что главный критерий (запас) считается
    относительно него, а собирать его самому у человека нет ни данных, ни смысла;
  * члены бленда и опорные модели — чтобы можно было ставить опыты со смешиванием.

Исключаем: артефакты (smoke/probe/cand/applied/path), модели эпохи до зазора 30 дней
(их валидационный скор завышен на 0.05-0.10), побитовые дубликаты.

Запуск: POLARS_MAX_THREADS=3 .venv/bin/python work/scripts/build_preds_pack.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))
from common import PREDS_DIR, REPORTS_DIR, VAL_ANCHOR, load_anchor, rmsle

OUT = Path("/Users/alexanderkondakov/ozon-cup/work/preds_pack")
# обучены до введения зазора 30 дней: валидационный скор завышен, в пул нельзя
OLD_ERA = {"lgblog_final", "xgblog_final", "mlp_final", "gru_final"}
EXCL = ("smoke", "probe", "cand", "applied", "hjit", "path")
# опорные модели: не в бленде, но полезны как ориентиры непохожести
EXTRA = ["febspec_cal", "short14_cal", "channel2_cal", "mlpbin_cal", "mlpziln_cal",
         "fusion_v3c_avg_cal", "fusion_v3_avg_cal", "c_ts2_avg_cal", "wklin_base_cal"]


def blend_members() -> dict[str, float]:
    d = json.loads((REPORTS_DIR / "blend_reopt.json").read_text())
    w = d["winner"].get("weights") or d["winner"].get("w") or {}
    return {k: v for k, v in w.items() if abs(v) > 1e-4}


def main():
    val = load_anchor(VAL_ANCHOR, columns=["user_id", "target"]).sort("user_id")
    uid = val["user_id"].to_numpy()
    y = val["target"].to_numpy().astype(np.float64)

    w = blend_members()
    names = list(w) + [n for n in EXTRA if n not in w]
    names = [n for n in names if n not in OLD_ERA and not any(s in n for s in EXCL)]

    vcols: dict[str, np.ndarray] = {}
    tcols: dict[str, np.ndarray] = {}
    blend_v = np.zeros(len(uid))
    blend_t = np.zeros(len(uid))
    seen: list[tuple[str, np.ndarray]] = []
    skipped: list[str] = []

    for n in names:
        pv, pt = PREDS_DIR / f"{n}_val.parquet", PREDS_DIR / f"{n}_test.parquet"
        if not (pv.exists() and pt.exists()):
            skipped.append(f"{n} (нет файлов)")
            continue
        dv = pl.read_parquet(pv).sort("user_id")
        dt = pl.read_parquet(pt).sort("user_id")
        if not np.array_equal(dv["user_id"].to_numpy(), uid):
            skipped.append(f"{n} (user_id не совпадает)")
            continue
        lv = np.log1p(np.clip(dv["pred"].to_numpy().astype(np.float64), 0, None))
        lt = np.log1p(np.clip(dt["pred"].to_numpy().astype(np.float64), 0, None))
        dup = next((m for m, arr in seen if np.array_equal(arr, lv)), None)
        if dup:
            skipped.append(f"{n} (побитовый дубликат {dup})")
            continue
        seen.append((n, lv))
        vcols[n], tcols[n] = lv, lt
        if n in w:
            blend_v += w[n] * lv
            blend_t += w[n] * lt

    vd = pl.DataFrame({"user_id": uid, "target": y, "blend": blend_v,
                       **{k: v for k, v in vcols.items()}})
    td = pl.DataFrame({"user_id": uid, "blend": blend_t,
                       **{k: v for k, v in tcols.items()}})
    cast = [pl.col(c).cast(pl.Float32) for c in vd.columns if c not in ("user_id", "target")]
    OUT.mkdir(exist_ok=True)
    vd.with_columns(cast).write_parquet(OUT / "val_preds.parquet")
    td.with_columns([pl.col(c).cast(pl.Float32) for c in td.columns if c != "user_id"]
                    ).write_parquet(OUT / "test_preds.parquet")

    sb = rmsle(y, np.expm1(blend_v))
    print(f"моделей в паке: {len(vcols)}, из них членов бленда: {sum(1 for n in vcols if n in w)}")
    print(f"скор бленда на валидации: {sb:.6f}")
    print(f"val {(OUT / 'val_preds.parquet').stat().st_size / 1e6:.1f} МБ, "
          f"test {(OUT / 'test_preds.parquet').stat().st_size / 1e6:.1f} МБ")
    if skipped:
        print("пропущено: " + "; ".join(skipped))
    return sb, len(vcols)


if __name__ == "__main__":
    main()

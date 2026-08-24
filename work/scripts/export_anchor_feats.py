"""Экспорт табличных фич по якорям в npz float16 для Kaggle (joint fusion, сессия 3).

Для каждого якоря пишет <out>/tabf16_<ISO>.npz:
    feats    float16 [250000, N]  — знаковый логарифм sign(x)*log1p(|x|), NaN сохранён
    cols     список имён N колонок (одинаков на всех якорях, проверяется)
    user_id  int64 [250000]       — строго sorted, совпадает с user_ids.npy kaggle_seq
    anchor   ISO-дата якоря; transform="signed_log1p"; tiers="base+v2+v3"

Почему знаковый логарифм, а не сырые значения: float16 переполняется на 65504, а
gmv_sum_365 бывает ~1e6+; в лог-шкале максимум ~28 и относительная ошибка f16
~5e-4 — для нейросетевого входа с LayerNorm достаточно. Прецедент проекта: wklin
подаёт «203 базовых признака в знаковом логарифме» (KNOWLEDGE.md).

Набор фич = чемпионский: USE_V2=1 USE_V3=1, common.feature_cols (сейчас 196 колонок
после дрейфа tab203->196; имена лежат в каждом npz, потребитель обязан читать их,
а не хардкодить число).

СЕТКИ ЯКОРЕЙ (--grid):
  wed  (дефолт) — сетка сессии 3: 24 среды, последняя VAL-35 = 2025-12-10, шаг 7,
        плюс VAL 2026-01-14 (день 378) и TEST 2026-02-13 (день 408). Все 26x3
        parquet-тиров УЖЕ существуют — это канонические файлы чемпионского пула.
        kaggle_seq в сессии 3 переводится на эту сетку одним флагом --gap 35
        (см. work/reports/eve2_joint_fusion_design.md).
  mon  — сетка kaggle_seq v1/v2 как в коде сейчас: последний якорь VAL-30 =
        2025-12-15 (понедельники). Табличных parquet для неё НЕТ; скрипт
        перечислит недостающие и команды их сборки (~40 мин, только через очередь).

Запуск (лёгкий: чтение parquet + запись npz, по одному якорю за раз):
  POLARS_MAX_THREADS=2 .venv/bin/python work/scripts/export_anchor_feats.py \
      --out /path/kaggle_feats --zip /path/kaggle_tabfeats_wed_v1.zip
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
import time
import zipfile
from datetime import date, timedelta
from pathlib import Path

import numpy as np

# чемпионский набор фиксируем ДО импорта common: load_anchor читает env при вызове
os.environ.setdefault("USE_V2", "1")
os.environ.setdefault("USE_V3", "1")
sys.path.insert(0, str(Path(__file__).parent))
from common import FEATURES_DIR, VAL_ANCHOR, TEST_ANCHOR, feature_cols, load_anchor  # noqa: E402

N_USERS = 250_000
LOG_CLIP = 60.0          # |signed log1p| выше не бывает (log1p(f32max)=88.7), страховка от inf


def grid(kind: str) -> list[date]:
    last = {"wed": VAL_ANCHOR - timedelta(days=35),   # 2025-12-10, среда, как весь пул
            "mon": VAL_ANCHOR - timedelta(days=30)}[kind]  # 2025-12-15, kaggle_seq v1
    train = sorted(last - timedelta(days=7 * k) for k in range(24))
    assert all(a + timedelta(days=30) <= VAL_ANCHOR for a in train)  # зазор 30 соблюдён
    return train + [VAL_ANCHOR, TEST_ANCHOR]


def check_parquets(anchors: list[date]) -> list[str]:
    missing = []
    for a in anchors:
        for suf in ("", ".extra", ".v3"):
            p = FEATURES_DIR / f"anchor={a.isoformat()}{suf}.parquet"
            if not p.exists():
                missing.append(p.name)
    return missing


def signed_log1p_f16(x: np.ndarray) -> np.ndarray:
    """float32 колонка -> float16 знакового логарифма; NaN остаётся NaN."""
    with np.errstate(invalid="ignore", over="ignore"):
        lg = np.sign(x) * np.log1p(np.abs(x, dtype=np.float32))
        lg = np.clip(lg, -LOG_CLIP, LOG_CLIP)      # clip сохраняет NaN
    return lg.astype(np.float16)


def export_one(a: date, out_dir: Path, ref_cols: list[str] | None,
               ref_uid: np.ndarray | None):
    t0 = time.time()
    df = load_anchor(a).sort("user_id")
    cols = feature_cols(df)
    if ref_cols is not None and cols != ref_cols:
        sys.exit(f"{a}: набор колонок отличается от первого якоря "
                 f"(+{set(cols) - set(ref_cols)} -{set(ref_cols) - set(cols)})")
    uid = df["user_id"].to_numpy().astype(np.int64)
    assert len(uid) == N_USERS and np.all(np.diff(uid) > 0), f"{a}: не 250k sorted юзеров"
    if ref_uid is not None:
        assert np.array_equal(uid, ref_uid), f"{a}: юниверс юзеров отличается"

    feats = np.empty((N_USERS, len(cols)), dtype=np.float16)
    stats = {}
    for j, c in enumerate(cols):
        v = df[c].to_numpy().astype(np.float32, copy=False)
        feats[:, j] = signed_log1p_f16(v)
        fin = np.isfinite(v)
        stats[c] = {"finite_frac": round(float(fin.mean()), 4),
                    "std_log": round(float(np.nanstd(feats[:, j].astype(np.float32))), 4)}
    del df
    gc.collect()

    p = out_dir / f"tabf16_{a.isoformat()}.npz"
    np.savez_compressed(p, feats=feats, cols=np.array(cols), user_id=uid,
                        anchor=np.array(a.isoformat()),
                        transform=np.array("signed_log1p"),
                        tiers=np.array("base+v2+v3"))
    mb = p.stat().st_size / 1e6
    print(f"  {a}: {feats.shape} -> {p.name} {mb:.0f} МБ за {time.time() - t0:.1f}с",
          flush=True)
    del feats
    gc.collect()
    return cols, uid, stats, mb


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--grid", choices=["wed", "mon"], default="wed")
    ap.add_argument("--anchors", default=None,
                    help="явный список ISO-дат через запятую (перекрывает --grid)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--zip", default=None, help="собрать zip (store: npz уже сжаты)")
    args = ap.parse_args()

    anchors = ([date.fromisoformat(s) for s in args.anchors.split(",")]
               if args.anchors else grid(args.grid))
    missing = check_parquets(anchors)
    if missing:
        iso = ",".join(a.isoformat() for a in anchors)
        cmds = "\n".join(
            f"  USE_V2=1 USE_V3=1 .venv/bin/python work/scripts/{s}.py --anchors {iso}"
            for s in ("build_features", "build_features_v2"))
        # у build_features_v3.py нет --anchors: вызывать build() по якорю
        v3 = ("  USE_V2=1 .venv/bin/python -c \"import sys; sys.path.insert(0,"
              "'work/scripts'); import polars as pl; from datetime import date; "
              "from common import TRAIN_PARQUET, user_universe; "
              "import build_features_v3 as b3; lf=pl.scan_parquet(TRAIN_PARQUET); "
              "uni=user_universe(); [b3.build(date.fromisoformat(s), uni, lf) "
              f"for s in '{iso}'.split(',')]\"")
        sys.exit(f"нет {len(missing)} parquet ({missing[:4]}...). Сборка ~40 мин — "
                 f"ТОЛЬКО через work/queue (base и extra до v3!). Команды:\n{cmds}\n{v3}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"якорей {len(anchors)} ({anchors[0]}..{anchors[-1]}), сетка {args.grid}, "
          f"тиры USE_V2={os.environ['USE_V2']} USE_V3={os.environ['USE_V3']}", flush=True)

    ref_cols, ref_uid, total_mb, col_stats = None, None, 0.0, {}
    for a in anchors:
        cols, uid, stats, mb = export_one(a, out_dir, ref_cols, ref_uid)
        ref_cols, ref_uid = cols, uid
        col_stats[a.isoformat()] = stats
        total_mb += mb

    # какие колонки вырождены на обучающих срезах (ya_* и т.п.) — потребителю на дроп
    tr = [a.isoformat() for a in anchors if a not in (VAL_ANCHOR, TEST_ANCHOR)]
    degen = [c for c in ref_cols
             if all(col_stats[a][c]["finite_frac"] == 0.0
                    or col_stats[a][c]["std_log"] == 0.0 for a in tr)]
    meta = {
        "grid": args.grid, "anchors": [a.isoformat() for a in anchors],
        "n_train": len(anchors) - 2, "val": VAL_ANCHOR.isoformat(),
        "test": TEST_ANCHOR.isoformat(), "n_features": len(ref_cols),
        "cols_sha1": hashlib.sha1("|".join(ref_cols).encode()).hexdigest()[:12],
        "cols": ref_cols, "transform": "signed_log1p", "tiers": "base+v2+v3",
        "degenerate_on_train_slices": degen,
        "total_npz_mb": round(total_mb, 1),
    }
    (out_dir / "tabf16_meta.json").write_text(json.dumps(meta, indent=1))
    print(f"итого {total_mb / 1000:.2f} ГБ npz; вырожденных на срезах колонок "
          f"{len(degen)}: {degen}", flush=True)

    if args.zip:
        t0 = time.time()
        zp = Path(args.zip)
        with zipfile.ZipFile(zp, "w", zipfile.ZIP_STORED) as z:
            for f in sorted(out_dir.glob("tabf16_*.npz")) + [out_dir / "tabf16_meta.json"]:
                z.write(f, f.name)
        print(f"zip: {zp} {zp.stat().st_size / 1e9:.2f} ГБ за {time.time() - t0:.0f}с",
              flush=True)


if __name__ == "__main__":
    main()

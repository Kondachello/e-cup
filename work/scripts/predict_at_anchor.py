#!/usr/bin/env python3
"""Прогноз сохранённой модели на ПРОИЗВОЛЬНОМ историческом якоре.

Зачем: часть C зеркального оценщика κ (Женя) — предикты нескольких НЕПОХОЖИХ
семейств на исторических якорях с ПОЛНЫМ таргетом, чтобы оценивать перенос
качества между окнами без траты LB-попыток.

Ничего не дублируем: функции предсказания per-базы (predict_gbdt,
predict_mlp_family) и поиск весов (find/load_meta) импортируются из
final_submission/inference.py — теми же путями собирается действующий сабмит.
Здесь своя только сборка матрицы признаков на произвольном якоре
(inference.load_test_matrix жёстко зашита на TEST_ANCHOR).

Выход (каталог work/preds_hist/):
  {model}_a{ISO}.parquet   user_id:int64, pred:float64 — юниверс 250к, sorted
  target_a{ISO}.parquet    user_id:int64, target:float64 — факт 30-дневного окна
                           якоря (пишется один раз на якорь)
  _results.json            накопительный журнал прогонов: mean_log1p, RMSLE по
                           таргету якоря, санити-числа — сырьё для манифеста

Санити (--sanity): прогноз из весов на VAL (2026-01-14) и TEST (2026-02-13)
сверяется с work/preds/{model}_{val,test}.parquet по max|Δlog1p|.
ВАЖНО ПРО ВОСПРОИЗВОДИМОСТЬ: все трейнеры (train_gbdt.py, train_mlpziln.py)
сохраняют веса ФИНАЛЬНОГО РЕТРЕЙНА (train + gap + val), а *_val.parquet писали
val-фазной моделью ДО ретрейна. Поэтому битовое совпадение возможно только с
*_test.parquet; расхождение с *_val.parquet ожидаемо большое и означает не
поломку, а то, что val-якорь для сохранённых весов — обучающий (in-sample).
По той же причине in-sample и все исторические якоря <= val — RMSLE на них
занижен, использовать для СРАВНЕНИЯ семейств, а не как абсолютную оценку.

Запуск (все лёгкие, минуты; очередь не нужна):
  .venv/bin/python work/scripts/predict_at_anchor.py --sanity
  .venv/bin/python work/scripts/predict_at_anchor.py --models weak_ft_recency \
      --anchors 2025-11-19
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import date
from pathlib import Path

# torch и lightgbm в одном процессе: без этого второй импорт libomp падает
# (OMP Error #179) — тот же обход, что в inference.py
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

HERE = Path(__file__).resolve().parent          # work/scripts
ROOT = HERE.parents[1]                          # корень репозитория
# Веса грузим из work/models (туда пишут трейнеры). inference.find() смотрит
# сначала в MODELS_DIR, затем в work/models — делаем их одним и тем же местом,
# чтобы копии в final_submission/models не могли разойтись с рабочими.
os.environ.setdefault("MODELS_DIR", str(ROOT / "work" / "models"))
os.environ.setdefault("OZON_ROOT", str(ROOT))

sys.path.insert(0, str(ROOT / "final_submission"))
sys.path.insert(0, str(HERE))

import numpy as np                              # noqa: E402
import inference as inf                         # noqa: E402  (predict_*, find, load_meta)

OUT_DEFAULT = ROOT / "work" / "preds_hist"
PREDS_DIR = ROOT / "work" / "preds"

# c_ts2_s42 и behavonly (сид 42) обучены до появления model_io — весов нет
# (см. work/reports/eve2_packaging.md), поэтому в ростере их сиды-близнецы.
DEFAULT_MODELS = ["twl_v7_s1337", "mlpziln_c42", "behavonly_s1337", "weak_ft_recency"]
DEFAULT_ANCHORS = ["2025-11-19", "2025-12-03", "2025-12-17"]

# Всё, что common.load_anchor читает из окружения: перед каждой моделью чистим
# и ставим ровно её флаги из меты (порядок колонок модели зависит от них).
FLAG_KEYS = ("USE_V2", "USE_V3", "USE_V4", "USE_V5", "USE_V5S", "USE_V5CAP",
             "USE_V6", "USE_V7", "USE_V8", "USE_V10", "USE_SEQOOF")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_anchor_matrix(base: str, meta: dict, anchor: date):
    """(X, user_id, target|None) на произвольном якоре, колонки в порядке обучения.

    Копия inference.load_test_matrix по духу, но якорь — параметр, строки
    отсортированы по user_id, и отдельно возвращается таргет (в X он не входит:
    feature_cols в мете записан из common.feature_cols, который его исключает —
    здесь это ещё раз проверяется явным assert).
    """
    import polars as pl
    for k in FLAG_KEYS:
        os.environ.pop(k, None)
    os.environ.update(meta.get("feature_flags") or {})
    from common import load_anchor
    df = load_anchor(anchor).sort("user_id")
    cols = meta["feature_cols"]
    forbidden = {"target", "user_id", "anchor_date"} & set(cols)
    assert not forbidden, f"{base}: в feature_cols попали служебные колонки {forbidden}"
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise inf.MissingArtifact(
            f"{base} @ {anchor}: в срезе нет {len(missing)} признаков, "
            f"например {missing[:5]} (флаги {meta.get('feature_flags')}) — "
            f"не собран нужный тир признаков этого якоря")
    X = df.select([pl.col(c).cast(pl.Float32) for c in cols]).to_numpy()
    uid = df["user_id"].to_numpy()
    target = df["target"].to_numpy().astype(np.float64) if "target" in df.columns else None
    del df
    return np.ascontiguousarray(X), uid, target


def predict_model(base: str, meta: dict, X: np.ndarray) -> np.ndarray:
    kind = meta.get("kind")
    if kind in ("mlpziln", "mlp2"):
        pred = inf.predict_mlp_family(base, meta, X)
    elif kind == "gbdt":
        if meta.get("detrend") and float(meta.get("m_hat_test") or 0.0):
            log(f"ВНИМАНИЕ: {base} детрендирован (m_hat_test="
                f"{meta['m_hat_test']}) — сдвиг подобран под ТЕСТОВЫЙ якорь, "
                f"на историческом уровень будет смещён")
        pred = inf.predict_gbdt(base, meta, X)
    else:
        raise inf.MissingArtifact(
            f"{base}: kind={kind!r} не поддержан (этот скрипт умеет gbdt и mlpziln/mlp2)")
    return np.clip(np.asarray(pred, dtype=np.float64), 0, None)


def rmsle(y: np.ndarray, p: np.ndarray) -> float:
    lt = np.log1p(np.clip(y, 0, None))
    lp = np.log1p(np.clip(p, 0, None))
    return float(np.sqrt(np.mean((lt - lp) ** 2)))


def compare_saved(base: str, split: str, uid: np.ndarray, pred: np.ndarray) -> dict | None:
    """max/mean |Δlog1p| против work/preds/{base}_{split}.parquet (если он есть)."""
    import polars as pl
    p = PREDS_DIR / f"{base}_{split}.parquet"
    if not p.exists():
        log(f"  санити {split}: нет {p} — сверять не с чем")
        return None
    d = pl.read_parquet(p).sort("user_id")
    if not np.array_equal(d["user_id"].to_numpy(), uid):
        log(f"  санити {split}: наборы user_id не совпали — сверка невозможна")
        return {"split": split, "error": "user_id mismatch"}
    ref = np.log1p(np.clip(d["pred"].to_numpy().astype(np.float64), 0, None))
    got = np.log1p(pred)
    dl = np.abs(got - ref)
    return {"split": split, "max_dlog": float(dl.max()), "mean_dlog": float(dl.mean()),
            "ref_mean_log1p": float(ref.mean()), "got_mean_log1p": float(got.mean())}


def merge_results(out_dir: Path, key: str, payload: dict) -> None:
    f = out_dir / "_results.json"
    data = json.loads(f.read_text()) if f.exists() else {}
    data[key] = payload
    f.write_text(json.dumps(data, indent=1, ensure_ascii=False, sort_keys=True))


def write_target(out_dir: Path, iso: str, force: bool) -> None:
    import polars as pl
    dst = out_dir / f"target_a{iso}.parquet"
    if dst.exists() and not force:
        return
    src = ROOT / "work" / "features" / f"anchor={iso}.parquet"
    d = pl.read_parquet(src, columns=["user_id", "target"]).sort("user_id")
    assert d["target"].null_count() == 0, f"{iso}: у якоря неполный таргет"
    d.select(pl.col("user_id").cast(pl.Int64),
             pl.col("target").cast(pl.Float64)).write_parquet(dst)
    log(f"таргет якоря -> {dst.name} ({d.height} строк, "
        f"mean_log1p {float(np.log1p(d['target'].to_numpy().clip(0)).mean()):.4f})")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", default=",".join(DEFAULT_MODELS),
                    help="имена моделей через запятую (веса+мета в work/models)")
    ap.add_argument("--anchors", default=",".join(DEFAULT_ANCHORS),
                    help="ISO-даты якорей через запятую")
    ap.add_argument("--sanity", action="store_true",
                    help="сверка прогноза из весов на VAL и TEST с work/preds "
                         "(совпадение ожидается только на TEST — см. докстринг)")
    ap.add_argument("--force", action="store_true", help="пересчитать существующие файлы")
    ap.add_argument("--out", default=str(OUT_DEFAULT))
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    models = [m for m in args.models.split(",") if m]
    anchors = [date.fromisoformat(a) for a in args.anchors.split(",") if a]

    from common import user_universe
    uni = user_universe()["user_id"].to_numpy()

    import polars as pl
    for iso in [a.isoformat() for a in anchors]:
        write_target(out_dir, iso, args.force)

    n_pairs = 0
    for base in models:
        meta = inf.load_meta(base)          # ищет в MODELS_DIR -> work/models
        need = meta.get("weights") or []
        for w in need:
            inf.find(w, f"веса {base}", base)   # упадём внятно, если файла нет
        log(f"=== {base}: kind={meta.get('kind')}, "
            f"{len(meta['feature_cols'])} признаков, флаги {meta.get('feature_flags')}")

        todo = list(anchors)
        if args.sanity:
            todo += [date.fromisoformat(inf.VAL_ANCHOR_ISO),
                     date.fromisoformat(inf.TEST_ANCHOR_ISO)]
        for a in todo:
            iso = a.isoformat()
            is_check = iso in (inf.VAL_ANCHOR_ISO, inf.TEST_ANCHOR_ISO)
            dst = out_dir / f"{base}_a{iso}.parquet"
            if dst.exists() and not args.force and not is_check:
                log(f"{base} @ {iso}: уже есть {dst.name}, пропускаю (--force для пересчёта)")
                n_pairs += 1
                continue
            t0 = time.time()
            X, uid, target = load_anchor_matrix(base, meta, a)
            assert np.array_equal(uid, uni), f"{base} @ {iso}: user_id != юниверс 250к"
            pred = predict_model(base, meta, X)
            del X
            secs = time.time() - t0
            res: dict = {"model": base, "anchor": iso, "secs": round(secs, 1),
                         "mean_log1p": float(np.log1p(pred).mean())}
            line = (f"{base} @ {iso}: готово за {secs:.0f}s, "
                    f"mean_log1p={res['mean_log1p']:.4f}")
            if target is not None and not is_check:
                res["rmsle_vs_target"] = rmsle(target, pred)
                line += f", RMSLE(таргет якоря)={res['rmsle_vs_target']:.4f} [in-sample]"
            if is_check:
                split = "val" if iso == inf.VAL_ANCHOR_ISO else "test"
                cmp_ = compare_saved(base, split, uid, pred)
                if target is not None and np.isfinite(target).all():
                    res["rmsle_vs_target"] = rmsle(target, pred)
                if cmp_:
                    res["sanity"] = cmp_
                    verdict = "OK" if cmp_.get("max_dlog", 9) < 1e-4 else "РАСХОЖДЕНИЕ"
                    line += (f"; санити {split}: max|Δlog1p|={cmp_['max_dlog']:.2e} "
                             f"mean={cmp_['mean_dlog']:.2e} [{verdict}]")
                    if split == "val" and verdict != "OK":
                        line += (" — ожидаемо: веса финального ретрейна "
                                 "(train+gap+val), а *_val.parquet писала "
                                 "val-фазная модель")
            else:
                (pl.DataFrame({"user_id": uid.astype(np.int64), "pred": pred})
                   .write_parquet(dst))
                res["file"] = dst.name
                n_pairs += 1
            log(line)
            merge_results(out_dir, f"{base}_a{iso}", res)
            del pred

    on_disk = sum(1 for f in out_dir.glob("*_a*.parquet")
                  if not f.name.startswith("target_a"))
    log(f"итого пар (модель x якорь) на диске: {on_disk} "
        f"(за прогон обработано {n_pairs})")
    return 0


if __name__ == "__main__":
    sys.exit(main())

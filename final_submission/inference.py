#!/usr/bin/env python3
"""Инференс финального решения E-CUP 2026, задача 3 (прогноз GMV за 30 дней).

Читает train.parquet + sample_submit.csv, строит признаки тестового среза
2026-02-13, загружает СОХРАНЁННЫЕ веса девяти моделей бленда, считает их
прогнозы, применяет поквантильную калибровку каждой модели, смешивает с
зафиксированными весами и применяет финальную аффинную перенастройку.
Сетевых вызовов нет.

Стадии (`--stage`):
  check       только проверить наличие артефактов и выйти (ничего не считает)
  features    признаки тестового среза (base/v2/v3/v4/v7) + тензор seq2
  predict     прогнозы девяти моделей из сохранённых весов -> кэш .npy
  ensemble    калибровка + бленд + аффин -> кэш final_lp.npy
  all         всё подряд (по умолчанию)

Переменные окружения:
  OZON_ROOT   корень с train.parquet / sample_submit.csv (по умолчанию — родитель
              этого каталога)
  MODELS_DIR  каталог с весами (по умолчанию final_submission/models; если файла
              там нет, ищем в work/models — туда пишут трейнеры)
  CACHE_DIR   каталог промежуточных прогнозов (по умолчанию MODELS_DIR/preds_test)

Архитектуры и функции предсказания НЕ дублируются: они импортируются из тех же
work/scripts/train_*.py, которыми модели обучены. Это единственный способ
гарантировать, что инференс и обучение не разъедутся.

Если какого-то файла весов нет — скрипт падает с явным сообщением, какой именно
файл отсутствует и какой командой он создаётся (см. reproduce_training.md).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("OZON_ROOT", str(HERE.parent))).resolve()
os.environ["OZON_ROOT"] = str(ROOT)          # common.py читает это при импорте

SCRIPTS = ROOT / "work" / "scripts"
WORK_MODELS = ROOT / "work" / "models"
MODELS_DIR = Path(os.environ.get("MODELS_DIR", str(HERE / "models"))).resolve()
CACHE_DIR = Path(os.environ.get("CACHE_DIR", str(MODELS_DIR / "preds_test")))

sys.path.insert(0, str(SCRIPTS))

import numpy as np  # noqa: E402

TEST_ANCHOR_ISO = "2026-02-13"

# --- зафиксированная конструкция ансамбля --------------------------------------
# Веса: NNLS на валидации, честный OOF 1.666791 (work/reports/scores.tsv).
# Канонический источник — work/scripts/blend_testopt.py, константа W_VAL.
BLEND_WEIGHTS = {
    "fusion_f":   0.316,
    "c_ts2_s42":  0.246,
    "mlpziln":    0.122,
    "behavonly":  0.080,
    "countaov":   0.074,
    "seq2tr_f":   0.070,
    "twl_v7":     0.055,
    "hmmsim":     0.028,
    "channel2":   0.012,
}
# Финальная аффинная перенастройка в log1p-пространстве:
#   lp_final = SLOPE * lp_blend + SHIFT
# SLOPE — недодисперсность бленда (sd 1.510 -> нужные 1.628, KNOWLEDGE);
# SHIFT — уровень: среднее log1p 2.155 -> mean_P(t) = 2.3275, замерено на LB.
# Источник: work/reports/blend_testopt_honest.json, ключ "_affine".
AFFINE_SLOPE = 1.0775792958468002
AFFINE_SHIFT = 0.006176042172469855
# Контрольные значения на нашем blend_cal_test (250k строк) — см. самопроверку.
EXPECT_MEAN_AFTER = 2.3287
EXPECT_SD_AFTER = 1.6278

# Наборы признаков, с которыми обучалась каждая модель (проверяется по meta).
FEATURE_ENV = {
    "fusion_f":  {"USE_V2": "1", "USE_V3": "1", "USE_V4": "1"},
    "c_ts2_s42": {"USE_V2": "1", "USE_V3": "1"},
    "mlpziln":   {"USE_V2": "1", "USE_V3": "1", "USE_V4": "1"},
    "behavonly": {"USE_V2": "1", "USE_V3": "1", "USE_V4": "1"},
    "countaov":  {"USE_V2": "1", "USE_V3": "1", "USE_V4": "1"},
    "twl_v7":    {"USE_V2": "1", "USE_V3": "1", "USE_V4": "1", "USE_V7": "1"},
    "channel2":  {"USE_V2": "1", "USE_V3": "1", "USE_V4": "1"},
    "seq2tr_f":  {},
    "hmmsim":    {},
}

# Команда, которой модель создаётся заново (для текста ошибки).
HOWTO = {
    "fusion_f":  "reproduce_training.md §2.1  (train_fusion.py --name fusion_f --final ...)",
    "c_ts2_s42": "reproduce_training.md §2.2  (train_gbdt.py --name c_ts2_s42 --objective two_stage ...)",
    "mlpziln":   "reproduce_training.md §2.3  (train_mlpziln.py --name mlpziln ...)",
    "behavonly": "reproduce_training.md §2.4  (train_behavonly.py --name behavonly ...)",
    "countaov":  "reproduce_training.md §2.5  (train_countaov.py --name countaov ...)",
    "seq2tr_f":  "reproduce_training.md §2.6  (train_seq2.py --name seq2tr_f --arch tr --final ...)",
    "twl_v7":    "reproduce_training.md §2.7  (train_gbdt.py --name twl_v7 ...)",
    "hmmsim":    "reproduce_training.md §2.8  (train_hmm_sim.py --name hmmsim ...)",
    "channel2":  "reproduce_training.md §2.9  (train_channel.py --name channel2 ...)",
}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


class MissingArtifact(RuntimeError):
    """Нет файла, без которого прогноз был бы мусором. Падаем громко."""


def find(fname: str, what: str, model: str | None = None) -> Path:
    """Ищет артефакт в MODELS_DIR, затем в work/models. Иначе — внятная ошибка."""
    for d in (MODELS_DIR, WORK_MODELS):
        p = d / fname
        if p.exists():
            return p
    hint = f"\n  создаётся: {HOWTO[model]}" if model in HOWTO else ""
    raise MissingArtifact(
        f"НЕ ХВАТАЕТ ФАЙЛА: {fname}  ({what})\n"
        f"  искали в: {MODELS_DIR}\n"
        f"            {WORK_MODELS}{hint}\n"
        f"  без него прогноз посчитать нельзя — прекращаю, чтобы не выдать мусор."
    )


def have(fname: str) -> bool:
    return any((d / fname).exists() for d in (MODELS_DIR, WORK_MODELS))


def load_meta(model: str) -> dict:
    return json.loads(find(f"{model}_meta.json",
                           f"конфиг модели {model} (порядок признаков, архитектура)",
                           model).read_text())


def run_script(script: str, *args: str, env: dict | None = None) -> None:
    cmd = [sys.executable, str(SCRIPTS / script), *args]
    e = dict(os.environ)
    e.update(env or {})
    log(f"$ {' '.join(cmd[1:])}" + (f"   env={env}" if env else ""))
    subprocess.run(cmd, check=True, env=e)


# ------------------------------------------------------------------ признаки --

def stage_features() -> None:
    for f in ("train.parquet", "sample_submit.csv"):
        if not (ROOT / f).exists():
            raise MissingArtifact(f"НЕ ХВАТАЕТ ВХОДНОГО ФАЙЛА: {ROOT / f}")
    a = TEST_ANCHOR_ISO
    run_script("build_features.py", "--anchors", a)
    run_script("build_features_v2.py", "--anchors", a)
    run_script("build_features_v3.py")          # сам пропускает уже собранные срезы
    run_script("build_features_v4.py", "--anchors", a)
    if "twl_v7" in BLEND_WEIGHTS:
        run_script("build_features_v7.py", "--anchors", a, "--states", "4",
                   "--sims", "300", "--win", "120", "--em-cap", "15000", "--seed", "42")
    if {"seq2tr_f", "fusion_f"} & set(BLEND_WEIGHTS):
        run_script("build_seq2.py")             # сам пропускает уже собранные тензоры
    log("признаки тестового среза готовы")


def load_test_matrix(model: str, meta: dict):
    """(X, user_id) для модели: колонки строго в том порядке, в каком обучали."""
    import polars as pl
    for k in ("USE_V2", "USE_V3", "USE_V4", "USE_V6", "USE_V7", "USE_V8",
              "USE_V10", "USE_SEQOOF"):
        os.environ.pop(k, None)
    os.environ.update(meta.get("feature_flags") or FEATURE_ENV.get(model, {}))
    import common
    from common import TEST_ANCHOR, load_anchor
    df = load_anchor(TEST_ANCHOR)
    cols = meta["feature_cols"]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise MissingArtifact(
            f"модель {model}: в тестовом срезе нет {len(missing)} признаков, "
            f"например {missing[:5]}.\n"
            f"  вероятно не собран нужный набор признаков "
            f"(флаги {meta.get('feature_flags')}) — см. reproduce_training.md §1")
    X = df.select([pl.col(c).cast(pl.Float32) for c in cols]).to_numpy()
    uid = df["user_id"].to_numpy()
    del df, common
    return np.ascontiguousarray(X), uid


# -------------------------------------------------------------- предсказатели --
# Каждая функция возвращает прогноз в СЫРОЙ шкале GMV (>=0), ровно как это делает
# retrain-фаза соответствующего трейнера (включая пространство усреднения сидов).

def _torch_device() -> str:
    import torch
    return "mps" if torch.backends.mps.is_available() else "cpu"


def _apply_stats(mod, X: np.ndarray, stats_file: Path) -> None:
    z = np.load(stats_file)
    mod.apply_stats(X, {k: z[k] for k in ("med", "lo", "hi", "mean", "std")})


def predict_mlp_family(model: str, meta: dict, X: np.ndarray) -> np.ndarray:
    """mlpziln / mlp2: усреднение сидов в СЫРОЙ шкале (как в трейнере)."""
    import torch
    mod = __import__("train_mlpziln" if meta["kind"] == "mlpziln" else "train_mlp2")
    _apply_stats(mod, X, find(meta["stats_npz"], f"статистики препроцессинга {model}", model))
    dev, cfg = _torch_device(), meta["cfg"]
    preds = []
    for seed in meta["seeds"]:
        w = find(f"{model}_seed{seed}.pt", f"веса {model}, сид {seed}", model)
        net = mod.build_model(X.shape[1], cfg["hidden"], cfg["dropout"]).to(dev)
        net.load_state_dict(torch.load(w, map_location=dev))
        preds.append(np.expm1(np.clip(mod.predict_log(net, X, dev), 0, None)))
        del net
    return np.mean(preds, axis=0)


def predict_mlpbin(model: str, meta: dict, X: np.ndarray) -> np.ndarray:
    """Усреднение сидов в E[log1p]-пространстве (как в трейнере), потом expm1."""
    import torch
    import train_mlpbin as mod
    sf = find(meta["stats_npz"], f"статистики/центры бинов {model}", model)
    _apply_stats(mod, X, sf)
    centers = np.load(sf)["centers"]
    dev, cfg = _torch_device(), meta["cfg"]
    centers_t = torch.tensor(centers, dtype=torch.float32, device=dev)
    elogs = []
    for seed in meta["seeds"]:
        w = find(f"{model}_seed{seed}.pt", f"веса {model}, сид {seed}", model)
        net = mod.build_model(X.shape[1], cfg["hidden"], cfg["dropout"],
                              len(centers), cfg.get("norm", "layer")).to(dev)
        net.load_state_dict(torch.load(w, map_location=dev))
        elogs.append(mod.predict_elog(net, X, centers_t, dev).astype(np.float64))
        del net
    return np.expm1(np.clip(np.mean(elogs, axis=0), 0, None))


def predict_fusion(model: str, meta: dict) -> tuple[np.ndarray, np.ndarray]:
    import polars as pl
    import torch
    import train_fusion as mod
    from common import TEST_ANCHOR, user_universe
    os.environ.update(meta.get("feature_flags") or FEATURE_ENV[model])
    uids = user_universe()["user_id"].to_numpy()
    cols = meta["feature_cols"]
    f32 = [pl.col(c).cast(pl.Float32) for c in cols]
    z = np.load(find(meta["stats_npz"], f"статистики препроцессинга {model}", model))
    stats = {k: z[k] for k in ("med", "lo", "hi", "mean", "std")}
    tab = mod.apply_stats_f16(
        mod.load_tab_raw(TEST_ANCHOR, cols, f32, uids, check_target=False), stats)
    x_mm = mod.open_x(TEST_ANCHOR)
    dev, idx = _torch_device(), np.arange(len(uids))
    trunk = tuple(int(t) for t in str(meta.get("trunk", "384,256")).split(","))
    preds = []
    for seed in meta["seeds"]:
        w = find(f"{model}_seed{seed}.pt", f"веса {model}, сид {seed}", model)
        net = mod.build_model(meta["d_tab"], meta.get("dropout", 0.15), dev,
                              meta.get("tab_dim", 256), trunk)
        net.load_state_dict(torch.load(w, map_location=dev))
        lp = mod.predict_log(net, x_mm, tab, idx, dev, 4096)
        preds.append(np.expm1(np.clip(lp, 0, None)).astype(np.float64))
        del net
    return np.mean(preds, axis=0), uids


def predict_seq2(model: str, meta: dict) -> tuple[np.ndarray, np.ndarray]:
    import torch
    import train_seq2 as mod
    from common import TEST_ANCHOR, user_universe
    uids = user_universe()["user_id"].to_numpy()
    x_mm = mod.open_x(TEST_ANCHOR)
    dev, preds = _torch_device(), []
    for seed in meta["seeds"]:
        w = find(f"{model}_seed{seed}.pt", f"веса {model}, сид {seed}", model)
        net = mod.build_model(meta["arch"], dev)
        net.load_state_dict(torch.load(w, map_location=dev))
        _, lp = mod.predict_main(net, x_mm, dev, 4096, 1)
        preds.append(np.expm1(np.clip(lp, 0, None)).astype(np.float64))
        del net
    return np.mean(preds, axis=0), uids


def _lgb(fname: str, what: str, model: str):
    import lightgbm as lgb
    return lgb.Booster(model_file=str(find(fname, what, model)))


def predict_gbdt(model: str, meta: dict, X: np.ndarray) -> np.ndarray:
    """train_gbdt/train_xtw: two_stage = p*mu, иначе expm1(raw + m_hat)."""
    kind = meta.get("model", "lgb")
    if meta["objective"] == "two_stage":
        m1 = _lgb(f"{model}__stage1.txt", f"{model}: бустер P(y>0)", model)
        m2 = _lgb(f"{model}__stage2.txt", f"{model}: бустер E[log1p|y>0]", model)
        p = m1.predict(X)
        mu = m2.predict(X)
        return np.expm1(np.clip(p * np.clip(mu, 0, None), 0, None))
    if kind == "xgb":
        import xgboost as xgb
        b = xgb.Booster()
        b.load_model(str(find(f"{model}.xgb.json", f"бустер XGBoost {model}", model)))
        raw = b.predict(xgb.DMatrix(X))
    else:
        raw = _lgb(f"{model}.txt", f"бустер LightGBM {model}", model).predict(X)
    if meta["objective"] == "log_mse":
        return np.expm1(np.clip(raw + float(meta.get("m_hat_test", 0.0)), 0, None))
    return np.clip(raw, 0, None)


def predict_channel(model: str, meta: dict, X: np.ndarray) -> np.ndarray:
    from train_channel import combine
    return combine({c: _lgb(f"{model}__{c}.txt", f"{model}: канал «{c}»", model).predict(X)
                    for c in meta["channels"]})


def predict_countaov(model: str, meta: dict, X: np.ndarray) -> np.ndarray:
    from train_countaov import COMBINE
    pc = _lgb(f"{model}__count.txt", f"{model}: голова числа заказов", model).predict(X)
    pa = _lgb(f"{model}__aov.txt", f"{model}: голова среднего чека", model).predict(X)
    return COMBINE[meta["mode"]](pc, pa, meta["aov_damp"])


def predict_hmmsim(model: str, meta: dict) -> tuple[np.ndarray, np.ndarray]:
    """У модели нет весов: пересчитываем симулятор с теми же гиперпараметрами."""
    import polars as pl
    from common import PREDS_DIR
    out = PREDS_DIR / f"{model}_test.parquet"
    if not out.exists():
        log(f"{model}: сохранённых весов нет по построению (генеративный "
            f"симулятор) — пересчитываю, ~6 мин")
        run_script("train_hmm_sim.py", "--name", model,
                   "--states", str(meta["states"]), "--sims", str(meta["sims"]),
                   "--win", str(meta["win"]), "--em-cap", str(meta["em_cap"]),
                   "--seed", str(meta["seed"]), "--splits", "test",
                   env={"THREADS": os.environ.get("OMP_NUM_THREADS", "6")})
    d = pl.read_parquet(out).sort("user_id")
    return d["pred"].to_numpy(), d["user_id"].to_numpy()


def predict_model(model: str) -> tuple[np.ndarray, np.ndarray]:
    meta = load_meta(model)
    kind = meta["kind"]
    if kind == "hmm_sim":
        return predict_hmmsim(model, meta)
    if kind == "fusion":
        return predict_fusion(model, meta)
    if kind == "seq2":
        return predict_seq2(model, meta)
    X, uid = load_test_matrix(model, meta)
    if kind in ("mlpziln", "mlp2"):
        return predict_mlp_family(model, meta, X), uid
    if kind == "mlpbin":
        return predict_mlpbin(model, meta, X), uid
    if kind == "channel":
        return predict_channel(model, meta, X), uid
    if kind == "countaov":
        return predict_countaov(model, meta, X), uid
    if kind == "gbdt":
        return predict_gbdt(model, meta, X), uid
    raise MissingArtifact(f"неизвестный тип модели {kind!r} в {model}_meta.json")


# ----------------------------------------------------------------- проверка ---

def stage_check() -> int:
    """Что уже есть, а чего не хватает. Ничего не считает."""
    print(f"MODELS_DIR: {MODELS_DIR}")
    print(f"work/models: {WORK_MODELS}")
    print(f"{'модель':<12} {'вес':>6}  {'meta':<5} {'веса':<26} калибровка")
    print("-" * 78)
    nmiss = 0
    for m, w in sorted(BLEND_WEIGHTS.items(), key=lambda kv: -kv[1]):
        has_meta = have(f"{m}_meta.json")
        weights_ok = False
        if has_meta:
            need_w = load_meta(m).get("weights") or []
            got = [f for f in need_w if have(f)]
            weights_ok = len(got) == len(need_w)
            wtxt = "нет весов (stateless)" if not need_w else f"{len(got)}/{len(need_w)}"
        else:
            wtxt = "?"
        cal = have(f"{m}_cal.npz")
        nmiss += 0 if (has_meta and weights_ok and cal) else 1
        print(f"{m:<12} {w:>6.3f}  {'да' if has_meta else 'НЕТ':<5} "
              f"{wtxt:<26} {'да' if cal else 'НЕТ'}")
    print("-" * 78)
    print(f"готовы к инференсу: {len(BLEND_WEIGHTS) - nmiss}/{len(BLEND_WEIGHTS)}")
    if nmiss:
        print("\nНедостающие артефакты создаются переобучением соответствующих\n"
              "моделей — команды в final_submission/reproduce_training.md §2.\n"
              "Трейнеры сохраняют веса сами (work/scripts/model_io.py).")
    return nmiss


# ------------------------------------------------------------------ стадии ----

def stage_predict() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    uid_ref = None
    for model in BLEND_WEIGHTS:
        dst = CACHE_DIR / f"{model}.npy"
        if dst.exists():
            log(f"{model}: прогноз уже в кэше, пропускаю")
            continue
        t0 = time.time()
        pred, uid = predict_model(model)
        order = np.argsort(uid)
        uid, pred = uid[order], np.asarray(pred, dtype=np.float64)[order]
        if uid_ref is None:
            uid_ref = uid
            np.save(CACHE_DIR / "user_ids.npy", uid)
        elif not np.array_equal(uid, uid_ref):
            raise MissingArtifact(f"{model}: набор user_id не совпал с остальными")
        np.save(dst, np.clip(pred, 0, None))
        log(f"{model}: готово за {time.time()-t0:.0f}s, "
            f"mean_log1p={np.log1p(np.clip(pred,0,None)).mean():.4f}")


def stage_ensemble() -> np.ndarray:
    from calibrate import apply_shifts
    uid_path = CACHE_DIR / "user_ids.npy"
    if not uid_path.exists():
        raise MissingArtifact(f"нет {uid_path} — сначала выполните стадию predict")
    uid = np.load(uid_path)
    lp_blend = np.zeros(len(uid), dtype=np.float64)
    total_w = 0.0
    for model, w in BLEND_WEIGHTS.items():
        f = CACHE_DIR / f"{model}.npy"
        if not f.exists():
            raise MissingArtifact(f"нет прогноза {model} — сначала стадия predict")
        lp = np.log1p(np.clip(np.load(f), 0, None))
        z = np.load(find(f"{model}_cal.npz", f"таблица калибровки {model}", model))
        lp_cal = apply_shifts(lp, z["centers"], z["shifts"])
        lp_blend += w * lp_cal
        total_w += w
        log(f"{model:<12} w={w:.3f} mean_log1p {lp.mean():.4f} -> {lp_cal.mean():.4f}")
    log(f"бленд: сумма весов {total_w:.4f}, mean_log1p {lp_blend.mean():.4f}, "
        f"sd {lp_blend.std():.4f}")

    lp_final = AFFINE_SLOPE * lp_blend + AFFINE_SHIFT
    log(f"аффин ({AFFINE_SLOPE:.7f} * lp + {AFFINE_SHIFT:.7f}): "
        f"mean {lp_blend.mean():.4f} -> {lp_final.mean():.4f}, "
        f"sd {lp_blend.std():.4f} -> {lp_final.std():.4f}")
    for what, got, exp in (("среднее", lp_final.mean(), EXPECT_MEAN_AFTER),
                           ("разброс", lp_final.std(), EXPECT_SD_AFTER)):
        if abs(got - exp) > 0.05:
            print(f"ВНИМАНИЕ: {what} итогового прогноза {got:.4f}, "
                  f"ожидалось ~{exp:.4f} — проверьте состав бленда", file=sys.stderr)
    np.save(CACHE_DIR / "final_lp.npy", lp_final)
    return lp_final


def stage_submission(lp: np.ndarray | None = None) -> None:
    import polars as pl
    from common import SAMPLE_SUBMIT
    if lp is None:
        lp = np.load(CACHE_DIR / "final_lp.npy")
    uid = np.load(CACHE_DIR / "user_ids.npy")
    vals = np.expm1(np.clip(lp, 0, None))
    sample = pl.read_csv(SAMPLE_SUBMIT, schema_overrides={"user_id": pl.Int64})
    out = (sample.select("user_id")
           .join(pl.DataFrame({"user_id": uid.astype(np.int64), "predict": vals}),
                 on="user_id", how="left"))
    assert out.height == sample.height, "число строк не совпало с sample_submit"
    assert out["predict"].null_count() == 0, "есть user_id без прогноза"
    assert float(out["predict"].min()) >= 0.0, "есть отрицательные прогнозы"


STAGES = ["check", "features", "predict", "ensemble", "submission"]


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=STAGES + ["all"], default="all")
    args = ap.parse_args()

    if args.stage == "check":
        return 1 if stage_check() else 0
    try:
        if args.stage in ("features", "all"):
            stage_features()
        if args.stage in ("predict", "all"):
            stage_predict()
        lp = stage_ensemble() if args.stage in ("ensemble", "all") else None
        if args.stage in ("submission", "all"):
            stage_submission(lp)
    except MissingArtifact as e:
        print(f"\nОШИБКА: {e}\n", file=sys.stderr)
        print("Состояние артефактов: python inference.py --stage check", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())

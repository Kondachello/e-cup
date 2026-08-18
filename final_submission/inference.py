#!/usr/bin/env python3
"""Полный инференс финального решения E-CUP 2026 Задача 3 (LTV) с нуля.

Стадии (каждую можно запустить отдельно через --stage):
  features   : признаки base/v2/v3/v4 для тестового среза        [~5-10 мин CPU]
  sequences  : тензор дневных последовательностей (для GRU)      [~1-2 мин]
  predict    : прогнозы пяти моделей ансамбля из артефактов models/  [~3-6 мин]
  ensemble   : бленд + калибровка + поправки -> итоговый прогноз [<1 мин]

Пути параметризованы (ни одного захардкоженного абсолютного пути):
  OZON_ROOT   корень с train.parquet и sample_submit.csv; кэш признаков
              создаётся в $OZON_ROOT/work/{features,seq}/ (по умолчанию —
              родительский каталог этого файла, т.е. корень репозитория)
  MODELS_DIR  каталог обученных артефактов (по умолчанию final_submission/models)
  SCRIPTS_DIR каталог скриптов пайплайна (по умолчанию <root>/work/scripts)

Сетевых вызовов нет ни в одной стадии. Используются только данные соревнования.
Артефакты моделей и конфиги ансамбля описаны в models/README.md.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Пути (всё переопределяется окружением; абсолютных путей в коде нет)
# ---------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent                      # .../final_submission
ROOT = Path(os.environ.get("OZON_ROOT", str(HERE.parent)))  # корень с train.parquet
MODELS_DIR = Path(os.environ.get("MODELS_DIR", str(HERE / "models")))
SCRIPTS_DIR = Path(os.environ.get("SCRIPTS_DIR", str(ROOT / "work" / "scripts")))

TEST_ANCHOR = "2026-02-13"          # прогнозируемое окно: 2026-02-14 .. 2026-03-15
N_USERS = 250_000

# Окружение дочерних процессов: наследует OZON_ROOT, включает наборы признаков.
CHILD_ENV = {**os.environ, "OZON_ROOT": str(ROOT),
             "USE_V2": "1", "USE_V3": "1", "USE_V4": "1"}


def log(msg: str) -> None:
    print(f"[inference +{time.time() - T0:7.1f}s] {msg}", flush=True)


def run_script(script: str, *args: str) -> None:
    """Запуск скрипта пайплайна тем же интерпретатором, без сети, с OZON_ROOT."""
    cmd = [sys.executable, str(SCRIPTS_DIR / script), *args]
    log("run: " + " ".join(cmd))
    subprocess.run(cmd, check=True, env=CHILD_ENV, cwd=str(ROOT))


def need(path: Path, what: str) -> Path:
    if not path.exists():
        sys.exit(f"ОШИБКА: не найден {what}: {path}\n"
                 f"Что и откуда кладётся в models/ — см. models/README.md; "
                 f"данные (train.parquet, sample_submit.csv) — в OZON_ROOT={ROOT}")
    return path


# ---------------------------------------------------------------------------
# Стадия 1-2: признаки и последовательности тестового среза
# ---------------------------------------------------------------------------
def stage_features() -> None:
    need(ROOT / "train.parquet", "train.parquet (данные соревнования)")
    need(ROOT / "sample_submit.csv", "sample_submit.csv (вселенная user_id)")
    # base: только тестовый срез (~1-2 мин)
    run_script("build_features.py", "--anchors", TEST_ANCHOR)
    # v2/v3 строят все стандартные срезы и пропускают уже посчитанные;
    # для чистого инференса нужен только тестовый (полная сборка ~5-8 мин)
    run_script("build_features_v2.py")
    run_script("build_features_v3.py")
    # v4 (BTYD): только тестовый срез (~2-4 мин)
    run_script("build_features_v4.py", "--anchors", TEST_ANCHOR)
    log("features: готово")


def stage_sequences() -> None:
    # Тензор [250k x 112 дней x 6 каналов]; уже посчитанные срезы пропускаются.
    run_script("build_seq.py")
    log("sequences: готово")


# ---------------------------------------------------------------------------
# Загрузка тестовой матрицы признаков (тот же код, что при обучении)
# ---------------------------------------------------------------------------
def load_test_matrix(feature_columns: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Возвращает (user_ids [250k], X float32 [250k, F]) в порядке sample_submit."""
    sys.path.insert(0, str(SCRIPTS_DIR))
    os.environ.update({k: CHILD_ENV[k] for k in ("OZON_ROOT", "USE_V2", "USE_V3", "USE_V4")})
    from datetime import date
    import common  # noqa: E402  (пайплайновый common.py, пути из OZON_ROOT)

    df = common.load_anchor(date.fromisoformat(TEST_ANCHOR)).sort("user_id")
    assert df.height == N_USERS, f"тестовый срез: {df.height} строк вместо {N_USERS}"
    missing = [c for c in feature_columns if c not in df.columns]
    assert not missing, f"в тестовом срезе нет колонок (несовпадение версий признаков): {missing[:5]}"
    X = df.select(feature_columns).to_numpy().astype(np.float32)
    return df["user_id"].to_numpy(), X


# ---------------------------------------------------------------------------
# Прогнозы отдельных моделей (артефакты — models/README.md)
# ---------------------------------------------------------------------------
def _tab_preprocess(X: np.ndarray, stats: np.lib.npyio.NpzFile) -> np.ndarray:
    """Препроцессинг табличных MLP: median-impute -> clip[p1,p99] -> standardize."""
    med, lo, hi = stats["med"], stats["lo"], stats["hi"]
    mean, std = stats["mean"], stats["std"]
    X = np.where(np.isnan(X), med, X)
    X = np.clip(X, lo, hi)
    return (X - mean) / np.maximum(std, 1e-6)


def _mlp_trunk(torch, n_feats: int, hidden=(512, 256)):
    """Общий ствол табличных MLP (как в обучении): Linear-GELU-LayerNorm-Dropout."""
    import torch.nn as nn
    layers, d = [], n_feats
    for h in hidden:
        layers += [nn.Linear(d, h), nn.GELU(), nn.LayerNorm(h), nn.Dropout(0.15)]
        d = h
    return nn.Sequential(*layers), d


def _forward_batched(torch, model, X: np.ndarray, bs: int = 8192) -> np.ndarray:
    outs = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            outs.append(model(torch.from_numpy(X[i:i + bs])).numpy())
    return np.concatenate(outs)


def predict_mlp_ziln(X: np.ndarray) -> np.ndarray:
    """ZILN MLP: 3 выхода (logit p, mu, sigma); E[log1p] квадратурой Гаусса-Эрмита.
    Среднение по сидам — на уровне E[log1p]. Возвращает log1p-прогноз."""
    import torch
    import torch.nn as nn
    stats = np.load(need(MODELS_DIR / "mlp_ziln_stats.npz", "статистики препроцессинга ZILN-MLP"))
    Xn = _tab_preprocess(X, stats).astype(np.float32)
    gh_x, gh_w = np.polynomial.hermite.hermgauss(20)
    acc = []
    for pt in sorted(MODELS_DIR.glob("mlp_ziln_seed*.pt")) or [need(MODELS_DIR / "mlp_ziln_seed42.pt", "веса ZILN-MLP (mlp_ziln_seed*.pt)")]:
        trunk, d = _mlp_trunk(torch, Xn.shape[1])
        model = nn.Sequential(trunk, nn.Linear(d, 3))
        model.load_state_dict(torch.load(pt, map_location="cpu"))
        model.eval()
        out = _forward_batched(torch, model, Xn)
        p = 1.0 / (1.0 + np.exp(-out[:, 0]))
        mu, sigma = out[:, 1], np.log1p(np.exp(out[:, 2])) + 1e-3  # softplus
        z = mu[:, None] + np.sqrt(2.0) * sigma[:, None] * gh_x[None, :]
        equad = (np.logaddexp(0.0, z) * gh_w[None, :]).sum(1) / np.sqrt(np.pi)
        acc.append(np.clip(p * equad, 0, None))
    return np.mean(acc, axis=0)


def predict_mlp_bin(X: np.ndarray) -> np.ndarray:
    """Биновая MLP-классификация: softmax по K+1 бинам, E[log1p] = sum p_k * c_k."""
    import torch
    import torch.nn as nn
    stats = np.load(need(MODELS_DIR / "mlp_bin_stats.npz", "статистики/центры бинов binned-MLP"))
    Xn = _tab_preprocess(X, stats).astype(np.float32)
    centers = stats["centers"].astype(np.float64)
    if centers[0] != 0.0:                      # c_0 (нулевой бин) = 0
        centers = np.concatenate([[0.0], centers])
    acc = []
    for pt in sorted(MODELS_DIR.glob("mlp_bin_seed*.pt")) or [need(MODELS_DIR / "mlp_bin_seed42.pt", "веса binned-MLP (mlp_bin_seed*.pt)")]:
        trunk, d = _mlp_trunk(torch, Xn.shape[1])
        model = nn.Sequential(trunk, nn.Linear(d, len(centers)))
        model.load_state_dict(torch.load(pt, map_location="cpu"))
        model.eval()
        logits = _forward_batched(torch, model, Xn)
        e = np.exp(logits - logits.max(1, keepdims=True))
        proba = e / e.sum(1, keepdims=True)
        acc.append(np.clip(proba @ centers, 0, None))
    return np.mean(acc, axis=0)


def predict_xgb(X: np.ndarray, feature_columns: list[str]) -> np.ndarray:
    """XGBoost tweedie-on-log1p: прогноз уже в log1p-пространстве."""
    import xgboost as xgb
    bst = xgb.Booster()
    bst.load_model(str(need(MODELS_DIR / "xgb_tweedie_log.json", "бустер XGBoost (xgb_tweedie_log.json)")))
    return np.clip(bst.predict(xgb.DMatrix(X, feature_names=feature_columns)), 0, None)


def predict_channels(X: np.ndarray) -> np.ndarray:
    """Канальная декомпозиция: log1p(GMV_search) и log1p(GMV_cat) двумя LightGBM;
    сумма — в линейном пространстве, возврат в log1p."""
    import lightgbm as lgb
    ps = lgb.Booster(model_file=str(need(MODELS_DIR / "channel_search.txt", "LightGBM канала «поиск»"))).predict(X)
    pc = lgb.Booster(model_file=str(need(MODELS_DIR / "channel_cat.txt", "LightGBM канала «каталог»"))).predict(X)
    total = np.expm1(np.clip(ps, 0, None)) + np.expm1(np.clip(pc, 0, None))
    return np.log1p(total)


def predict_gru(user_ids: np.ndarray) -> np.ndarray:
    """GRU по дневным последовательностям (112 дней x 6 каналов)."""
    import torch
    import torch.nn as nn

    seq_path = need(ROOT / "work" / "seq" / f"anchor={TEST_ANCHOR}.npy",
                    "тензор последовательностей тестового среза (стадия sequences)")
    arr = np.load(seq_path).astype(np.float32)          # [250k, 112, 6]
    assert arr.shape[0] == len(user_ids) == N_USERS

    class GruNet(nn.Module):
        def __init__(self, hidden=96, layers=2):
            super().__init__()
            self.gru = nn.GRU(6, hidden, num_layers=layers, batch_first=True, dropout=0.1)
            self.head = nn.Sequential(nn.Linear(hidden * 3, hidden), nn.GELU(), nn.Linear(hidden, 1))

        def forward(self, x):
            h, _ = self.gru(x)
            z = torch.cat([h[:, -1], h.mean(1), h.max(1).values], dim=1)
            return self.head(z).squeeze(1)

    model = GruNet()
    model.load_state_dict(torch.load(
        need(MODELS_DIR / "gru_seed42.pt", "веса GRU (gru_seed42.pt)"), map_location="cpu"))
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(arr), 8192):
            preds.append(model(torch.from_numpy(arr[i:i + 8192])).numpy())
    return np.clip(np.concatenate(preds), 0, None)


def stage_predict() -> None:
    """Прогнозы всех моделей ансамбля -> models/preds_test/*.npy (log1p-шкала)."""
    cfg = json.loads(need(MODELS_DIR / "blend_config.json", "конфиг ансамбля (blend_config.json)").read_text())
    out_dir = MODELS_DIR / "preds_test"
    out_dir.mkdir(exist_ok=True)

    feats_tab = cfg["feature_columns"]["tabular"]        # 203 колонки (base+v2+v3+v4)
    feats_xgb = cfg["feature_columns"].get("xgb", feats_tab)  # у XGB срез без v4 (194)
    uid, X = load_test_matrix(feats_tab)
    np.save(out_dir / "user_ids.npy", uid)
    Xxgb = X if feats_xgb == feats_tab else load_test_matrix(feats_xgb)[1]

    runners = {
        "mlp_ziln": lambda: predict_mlp_ziln(X),
        "mlp_bin": lambda: predict_mlp_bin(X),
        "xgb_tweedie_log": lambda: predict_xgb(Xxgb, feats_xgb),
        "channels": lambda: predict_channels(X),
        "gru": lambda: predict_gru(uid),
    }
    for name in cfg["weights"]:
        assert name in runners, f"в blend_config.json неизвестная модель: {name}"
    for name, fn in runners.items():
        if name not in cfg["weights"]:
            log(f"predict {name}: пропуск (нет в blend_config.json)")
            continue
        t = time.time()
        lp = fn()
        assert lp.shape == (N_USERS,) and np.isfinite(lp).all()
        np.save(out_dir / f"{name}.npy", lp.astype(np.float64))
        log(f"predict {name}: {time.time() - t:.0f}s, mean log1p={lp.mean():.4f}")


# ---------------------------------------------------------------------------
# Стадия 4: ансамбль = бленд -> калибровка -> поправки (всё в log1p)
# ---------------------------------------------------------------------------
def apply_calibration(lp: np.ndarray, table: Path) -> np.ndarray:
    """Поквантильные сдвиги: np.interp по центрам бинов (calibrate.py)."""
    t = np.load(table)
    return np.clip(lp + np.interp(lp, t["centers"], t["shifts"]), 0, None)


def stage_ensemble() -> np.ndarray:
    cfg = json.loads(need(MODELS_DIR / "blend_config.json", "blend_config.json").read_text())
    corr = json.loads(need(MODELS_DIR / "lb_corrections.json", "lb_corrections.json").read_text())
    pred_dir = MODELS_DIR / "preds_test"
    uid = np.load(need(pred_dir / "user_ids.npy", "user_ids.npy (стадия predict)"))

    # 1. взвешенное среднее в log1p (веса подобраны на валидации, см. README §5)
    lp = np.zeros(N_USERS)
    wsum = 0.0
    for name, w in cfg["weights"].items():
        comp = np.load(need(pred_dir / f"{name}.npy", f"прогноз {name} (стадия predict)"))
        cal = MODELS_DIR / f"calibration_{name}.npz"
        if cal.exists():                       # покомпонентная калибровка (ziln/bin)
            comp = apply_calibration(comp, cal)
        lp += w * comp
        wsum += w
    assert abs(wsum - 1.0) < 1e-6, f"веса бленда должны суммироваться к 1, сейчас {wsum}"

    # 2. калибровка итогового бленда (если зафиксирована)
    final_cal = MODELS_DIR / "calibration_blend.npz"
    if final_cal.exists():
        lp = apply_calibration(lp, final_cal)

    # 3. поправки, уточнённые по публичному лидерборду (README §5.3-5.4):
    #    глобальный сезонный сдвиг + сегментные сдвиги; все уже включают
    #    консервативную усадку (shrinkage) для переноса на private.
    lp = lp + float(corr["global_log_shift"])
    for seg in corr.get("segments", []):
        mask = eval_segment_mask(seg, uid)
        lp[mask] += float(seg["log_shift"])
        log(f"ensemble: сегмент {seg['name']}: {mask.sum()} юзеров, сдвиг {seg['log_shift']:+.4f}")
    return np.clip(lp, 0, None)


def eval_segment_mask(seg: dict, uid: np.ndarray) -> np.ndarray:
    """Сегмент задаётся порогом по одному признаку тестового среза
    ({"column": ..., "op": ">=|<", "threshold": ...}) — прозрачно и воспроизводимо."""
    col, op, thr = seg["column"], seg["op"], float(seg["threshold"])
    uid2, X = load_test_matrix([col])
    assert (uid2 == uid).all()
    v = X[:, 0].astype(np.float64)
    if op == ">=":
        return v >= thr
    if op == "<":
        return v < thr
    raise ValueError(f"сегмент {seg['name']}: неизвестный оператор {op}")


# ---------------------------------------------------------------------------
# Стадия 5: submission + самопроверки
# ---------------------------------------------------------------------------
def stage_submission(lp: np.ndarray) -> None:
    import polars as pl
    uid = np.load(MODELS_DIR / "preds_test" / "user_ids.npy")
    pred = np.expm1(lp)
    assert pred.shape == (N_USERS,)
    assert np.isfinite(pred).all() and (pred >= 0).all()
    sample = pl.read_csv(ROOT / "sample_submit.csv", schema_overrides={"user_id": pl.Int64})
    assert sorted(sample["user_id"].to_list()) == sorted(uid.tolist()), "вселенная user_id не совпала с sample_submit"


STAGES = ["features", "sequences", "predict", "ensemble", "submission"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=STAGES + ["all"], default="all",
                    help="какую стадию выполнить (по умолчанию весь пайплайн)")
    args = ap.parse_args()

    todo = STAGES if args.stage == "all" else [args.stage]
    lp = None
    if "features" in todo:
        stage_features()
    if "sequences" in todo:
        stage_sequences()
    if "predict" in todo:
        stage_predict()
    if "ensemble" in todo:
        lp = stage_ensemble()
    if "submission" in todo:
        if lp is None:
            lp = stage_ensemble()
        stage_submission(lp)
    log("ГОТОВО")


T0 = time.time()
if __name__ == "__main__":
    main()

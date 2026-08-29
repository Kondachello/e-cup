"""Инференс сохранённых моделей на СТАРЫХ и ИСПРАВЛЕННЫХ тестовых признаках.

Ничего не переобучает и ничего не перезаписывает. Для каждой модели с
сохранёнными весами строит две матрицы (MAX_BACK=379 и 409) в порядке колонок
из её meta и печатает, насколько разъезжается прогноз в log1p.

Результат кладёт в work/reports/mb_fix_preds.npz (uid + по два вектора на модель).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))
from common import FEATURES_DIR, REPORTS_DIR, TEST_ANCHOR, WORK

MODELS = WORK / "models"
# Артефакты живут в ДВУХ местах: рабочем каталоге и упакованном пакете сдачи. Искать
# только в первом — тот же дефект, что был в pack_models.py: 28.08 этот скрипт пропустил
# ВСЕ модели с «нет meta», хотя шесть мет лежали в final_submission/models.
PKG_MODELS = WORK.parent / "final_submission" / "models"


def find_artifact(fname: str):
    """Файл артефакта: сначала пакет (он канонический), потом рабочий каталог."""
    for d in (PKG_MODELS, MODELS):
        p = d / fname
        if p.exists():
            return p
    return None
A = TEST_ANCHOR.isoformat()
MB = 409

# модели с настоящими весами, которые едят табличную матрицу
LGB_MODELS = ["weak_an_d", "weak_ft_recency", "weak_ft_counts", "weak_ft_long90",
              "twl_v7_s7", "twl_v7_s1337", "behavonly_s7", "behavonly_s1337"]
MULTIHEAD = {"countaov_s7": ("countaov", ["count", "aov"]),
             "countaov_s1337": ("countaov", ["count", "aov"]),
             "channel_s1337": ("channel", ["search", "cat"])}


def frame(fixed: bool) -> pl.DataFrame:
    """Тестовый срез с тирами v2/v3/v4 (флаги во всех meta одинаковы)."""
    sfx = f".mb{MB}" if fixed else ""
    df = pl.read_parquet(FEATURES_DIR / f"anchor={A}{sfx}.parquet")
    df = df.join(pl.read_parquet(FEATURES_DIR / f"anchor={A}{sfx}.extra.parquet"),
                 on="user_id", how="left")
    df = df.join(pl.read_parquet(FEATURES_DIR / f"anchor={A}{sfx}.v3.parquet"),
                 on="user_id", how="left")
    # v4 (BTYD) фильтрует только сверху (event_date <= anchor), обрезки нет — общий файл
    df = df.join(pl.read_parquet(FEATURES_DIR / f"anchor={A}.v4.parquet"),
                 on="user_id", how="left")
    return df.sort("user_id")


def mat(df: pl.DataFrame, cols: list[str]) -> np.ndarray:
    miss = [c for c in cols if c not in df.columns]
    assert not miss, f"нет колонок: {miss[:6]}"
    return np.ascontiguousarray(df.select([pl.col(c).cast(pl.Float32) for c in cols]).to_numpy())


def lgb_booster(stem: str):
    import lightgbm as lgb
    p = find_artifact(f"{stem}.txt")
    if p is None:
        raise FileNotFoundError(f"нет весов {stem}.txt ни в пакете, ни в work/models")
    return lgb.Booster(model_file=str(p))


def gbdt_predict(meta: dict, name: str, X: np.ndarray) -> np.ndarray:
    """Повторяет final_submission/inference.py::predict_gbdt."""
    obj = meta["objective"]
    if obj == "two_stage":
        p = lgb_booster(f"{name}__stage1").predict(X)
        mu = lgb_booster(f"{name}__stage2").predict(X)
        return np.expm1(np.clip(p * np.clip(mu, 0, None), 0, None))
    raw = lgb_booster(name).predict(X)
    if obj == "log_mse":
        return np.expm1(np.clip(raw + float(meta.get("m_hat_test", 0.0)), 0, None))
    return np.clip(raw, 0, None)


def main() -> None:
    old, new = frame(False), frame(True)
    uid = old["user_id"].to_numpy()
    assert np.array_equal(uid, new["user_id"].to_numpy())
    out = {"user_id": uid}
    print(f"{'модель':22s} {'sd(dlog)':>9s} {'mean':>9s} {'corr':>8s} "
          f"{'|d|>0.01':>9s} {'sd после аффина':>16s}")

    todo = [(n, None) for n in LGB_MODELS] + [(n, v) for n, v in MULTIHEAD.items()]
    for name, multi in todo:
        mp = find_artifact(f"{name}_meta.json")
        if mp is None:
            print(f"{name:22s} нет meta — пропуск")
            continue
        meta = json.loads(mp.read_text())
        cols = meta["feature_cols"]
        try:
            Xo, Xn = mat(old, cols), mat(new, cols)
        except AssertionError as e:
            print(f"{name:22s} {e}")
            continue
        if multi is None:
            po, pn = gbdt_predict(meta, name, Xo), gbdt_predict(meta, name, Xn)
        elif multi[0] == "countaov":
            from train_countaov import COMBINE
            f = COMBINE[meta["mode"]]
            po = f(lgb_booster(f"{name}__count").predict(Xo),
                   lgb_booster(f"{name}__aov").predict(Xo), meta["aov_damp"])
            pn = f(lgb_booster(f"{name}__count").predict(Xn),
                   lgb_booster(f"{name}__aov").predict(Xn), meta["aov_damp"])
        else:
            from train_channel import combine
            po = combine({c: lgb_booster(f"{name}__{c}").predict(Xo) for c in meta["channels"]})
            pn = combine({c: lgb_booster(f"{name}__{c}").predict(Xn) for c in meta["channels"]})
        lo = np.log1p(np.clip(np.asarray(po, dtype=np.float64), 0, None))
        ln = np.log1p(np.clip(np.asarray(pn, dtype=np.float64), 0, None))
        d = ln - lo
        rho = float(np.corrcoef(lo, ln)[0, 1])
        # то, что переживает приведение к моментам: остаток после аффина
        b = np.cov(ln, lo)[0, 1] / np.var(lo)
        resid = ln - (ln.mean() + b * (lo - lo.mean()))
        print(f"{name:22s} {d.std():9.5f} {d.mean():+9.5f} {rho:8.5f} "
              f"{float((np.abs(d) > 0.01).mean()):9.4f} {resid.std():16.5f}")
        out[f"{name}__old"], out[f"{name}__new"] = lo, ln

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(REPORTS_DIR / "mb_fix_preds.npz", **out)
    print(f"\nсохранено {REPORTS_DIR / 'mb_fix_preds.npz'}")


if __name__ == "__main__":
    main()

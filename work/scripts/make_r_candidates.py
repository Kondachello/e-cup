"""Сборка кандидатов на сабмит mdl_flint/mdl_gypsum/mdl_gneis2 поверх Q1 (пересборка бленда 21.08).

Подход — как у add_direction.py: не смесь, а ДОБАВКА в лог-пространстве поверх
готовой базы Q1_probes5 (в ней уже сидят уровень, поправка на молчащих и пять
оптимумов проб). Дельта = log1p(новый бленд) − log1p(старый бленд) на тесте,
ЦЕНТРИРОВАННАЯ (уровень мерился зондами под старый бленд, его не трогаем).
После каждого шага сборки разброс приводится к канону 1.631108 (оптимум mdl_amber;
правило ночи 21.08: sd(log1p) проверять ПОСЛЕ ВСЕХ шагов).

R2_newblend  Q1 + дельта нового бленда (kostya46 0.246, gseq_small 0.109,
             lagd28 0.035, gseq_big 0.024; honest OOF −0.000655 на валидации)
R3_ridge     mdl_flint + центрированный ridge-стек на 117 колонках, побитово не
             задетых обрезкой MAX_BACK=379 (честный OOF замер против НОВОГО
             бленда печатается при сборке)
R5_shade     как mdl_flint, но kostya46 заменён на kostya46shade (шейдинг явки;
             валидация различие не видит ПО ПОСТРОЕНИЮ — это LB-проба)

Запуск: USE_V2=1 USE_V3=1 .venv/bin/python work/scripts/make_r_candidates.py \
            --old-pack <каталог со старым паком>
Артефакты: submissions/R{2,3,5}_*.csv, work/reports/r_candidates.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))
from common import REPORTS_DIR, ROOT, TEST_ANCHOR, VAL_ANCHOR, feature_cols, load_anchor
from margin import score
from exp_resid_ridge import ridge_oof, FOLDS
from subs import lp

SUB = ROOT / "submissions"
SD_CANON = 1.631108          # оптимум разброса по пробе mdl_amber
W_KOSTYA = 0.246021          # вес kostya46_cal в новом winner (blend_reopt 21.08)


def L1(x):
    return np.log1p(np.clip(np.asarray(x, np.float64), 0, None))


def respread(lp_, sd_target=SD_CANON):
    """Привести разброс к канону, не трогая среднее (движение вдоль замеренной оси mdl_amber)."""
    m = lp_.mean()
    out = np.clip(m + (lp_ - m) * (sd_target / lp_.std()), 0, None)
    return out


def write_sub(name, uid, lp_, rep):
    pred = np.expm1(lp_)
    assert len(pred) == 250000 and np.isfinite(pred).all() and (pred >= 0).all()
    pl.DataFrame({"user_id": uid, "predict": pred}).write_csv(SUB / name)
    rep[name] = {"mean": round(float(lp_.mean()), 6), "sd": round(float(lp_.std()), 6),
                 "clipped": int((lp_ <= 0).sum())}
    print(f"{name}: mean {lp_.mean():.6f} sd {lp_.std():.6f} clip {(lp_ <= 0).sum()}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old-pack", required=True, help="каталог со старым preds_pack (эпоха 1.666302)")
    ap.add_argument("--new-pack", default=str(ROOT / "work" / "preds_pack"))
    args = ap.parse_args()

    uid, q1 = lp("Q1_probes5.csv")
    rep: dict = {"base": "Q1_probes5.csv", "sd_canon": SD_CANON}

    def pack_blend(d, side):
        f = pl.read_parquet(Path(d) / f"{side}_preds.parquet").sort("user_id")
        assert np.array_equal(f["user_id"].to_numpy(), uid)
        return f["blend"].to_numpy().astype(np.float64), f

    old_t, _ = pack_blend(args.old_pack, "test")
    new_t, _ = pack_blend(args.new_pack, "test")
    old_v, _ = pack_blend(args.old_pack, "val")
    new_v, vpack = pack_blend(args.new_pack, "val")
    ly = L1(vpack["target"].to_numpy())
    print(f"валидация: старый бленд {score(old_v, ly):.6f} -> новый {score(new_v, ly):.6f} "
          f"(дельта {score(old_v, ly) - score(new_v, ly):+.6f})")

    # ---- mdl_flint: центрированная дельта нового бленда
    d_t = new_t - old_t
    d_t_c = d_t - d_t.mean()
    rep["delta"] = {"mean": round(float(d_t.mean()), 6), "sd": round(float(d_t.std()), 6)}
    r2 = respread(np.clip(q1 + d_t_c, 0, None))
    write_sub("R2_newblend.csv", uid, r2, rep)

    # ---- mdl_gypsum: mdl_flint + ridge-стек на безопасных колонках, обучен на остатке НОВОГО бленда
    assert os.environ.get("USE_V2") and os.environ.get("USE_V3"), "нужны USE_V2=1 USE_V3=1"
    bad = {x.split(" ")[0] for x in json.loads(
        (REPORTS_DIR / "maxback_affected_cols.json").read_text())}
    # пакет отсортирован по user_id и совпал с uid (assert выше) => uid отсортирован,
    # и отсортированные матрицы признаков выравниваются с ним напрямую
    val = load_anchor(VAL_ANCHOR).sort("user_id")
    assert np.array_equal(val["user_id"].to_numpy(), uid)
    cols = [c for c in feature_cols(val) if c not in bad]

    def matrix(df):
        X = df.select(cols).to_numpy().astype(np.float64)
        return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    Xv = matrix(val)
    mu, sd = Xv.mean(0), Xv.std(0)
    sd[sd == 0] = 1.0
    Xv = np.column_stack([(Xv - mu) / sd, np.ones(len(Xv))])

    resid = ly - new_v
    rng = np.random.default_rng(0)
    folds = rng.permutation(len(ly)) % FOLDS
    oof, alphas = ridge_oof(Xv, resid, folds, rng)
    gain_oof = score(new_v, ly) - score(new_v + oof, ly)
    print(f"ridge против нового бленда: честный OOF {gain_oof:+.6f}, альфы {alphas}")
    rep["ridge"] = {"oof_gain_vs_new_blend": round(float(gain_oof), 6),
                    "n_cols": len(cols), "alphas": alphas}

    al = max(set(alphas), key=alphas.count)
    w = np.linalg.solve(Xv.T @ Xv + al * np.eye(Xv.shape[1]), Xv.T @ resid)
    test = load_anchor(TEST_ANCHOR).sort("user_id")
    assert np.array_equal(test["user_id"].to_numpy(), uid)
    Xt = matrix(test)
    Xt = np.column_stack([(Xt - mu) / sd, np.ones(len(Xt))])
    corr = Xt @ w
    corr_c = corr - corr.mean()
    r3 = respread(np.clip(r2 + corr_c, 0, None))
    write_sub("R3_ridge.csv", uid, r3, rep)

    # ---- mdl_gneis2: mdl_flint с kostya46 -> kostya46shade (проба явки, val различий не видит)
    z = np.load(ROOT / "work" / "models" / "kostya46_cal.npz")
    c_, s_ = z["centers"], z["shifts"]

    def cal(lp_):
        return np.clip(lp_ + np.interp(lp_, c_, s_), 0, None)

    k_t = L1(pl.read_parquet(ROOT / "work" / "preds" / "kostya46_test.parquet")
             .sort("user_id")["pred"].to_numpy())
    sh_t = L1(pl.read_parquet(ROOT / "work_kostya" / "preds" / "kostya46shade_test.parquet")
              .sort("user_id")["pred"].to_numpy())
    shade_delta = W_KOSTYA * (cal(sh_t) - cal(k_t))
    rep["shade"] = {"w_kostya": W_KOSTYA, "mean_delta": round(float(shade_delta.mean()), 6),
                    "sd_delta": round(float(shade_delta.std()), 6)}
    r5 = respread(np.clip(q1 + d_t_c + (shade_delta - shade_delta.mean()), 0, None))
    write_sub("R5_shade.csv", uid, r5, rep)

    (REPORTS_DIR / "r_candidates.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1))
    print("JSON: work/reports/r_candidates.json")


if __name__ == "__main__":
    main()

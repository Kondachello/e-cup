"""Персональный случайный эффект (random effect) в остатках: ОПРОВЕРГНУТ.

Гипотеза: одни и те же 250k юзеров есть во всех срезах, включая тест, поэтому у юзера
может быть устойчивое личное смещение, которое признаки не ловят. В LTV-задачах это
стандартный per-user random effect, и у нас его нет ни в каком виде.

Источник остатков БЕЗ обучения: work/features/anchor=*.seqoof.parquet — честные
time-split OOF-прогнозы seq-трансформера (обучен только на срезах <= 2025-11-12,
предсказывает 2025-11-19..2026-01-07). Это E[log1p(y30)] прямо в лог-пространстве,
8 срезов с известным таргетом -> остаток r = log1p(y) - lp без единой минуты обучения.

ЛОВУШКА (главная): шаг срезов 7 дней при окне таргета 30 дней -> целевые окна соседних
срезов пересекаются на 23 дня и остатки коррелируют механически. Здесь группы разнесены
на >=35 дней по якорю (окна не пересекаются вообще) + приведён контроль на пересекающихся.

ВТОРАЯ ЛОВУШКА: сегментное смещение калибровочной кривой тоже даёт корреляцию, но это не
персональный эффект. Поэтому внутри каждого среза снимается E[r | бин прогноза] (40 бинов).

Запуск (секунды, 1 поток):
  POLARS_MAX_THREADS=2 OMP_NUM_THREADS=1 .venv/bin/python work/scripts/resid_re.py
"""
from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))
from common import FEATURES_DIR, PREDS_DIR, REPORTS_DIR, VAL_ANCHOR, load_anchor

ANCHORS = [date(2025, 11, 19), date(2025, 11, 26), date(2025, 12, 3), date(2025, 12, 10),
           date(2025, 12, 17), date(2025, 12, 24), date(2025, 12, 31), date(2026, 1, 7)]
# срезы-предикторы для применения на валидации: их целевые окна кончаются <= 09.01,
# валидационное окно начинается 15.01 -> пересечения нет
PRED_IDX = [0, 1, 2, 3]
NOISE = 2.2e-5  # цена одного подобранного направления в RMSLE (noise_floor.py)

# лучший честный бленд (blend_reopt.json -> winner, OOF 1.666746)
BLEND_W = {"fusion_avg_cal": 0.182158, "fusion_f_cal": 0.145958, "c_ts2_s42_cal": 0.11936,
           "c_ts2_avg_cal": 0.117003, "mlpziln_avg_cal": 0.084174, "behavonly_cal": 0.078989,
           "seq2tr_f_cal": 0.06909, "countaov_cal": 0.066059, "twl_v7_cal": 0.055507,
           "mlpziln_cal": 0.041982, "hmmsim_cal": 0.022222, "channel2_cal": 0.015367,
           "hmmsim": 0.004673}


def load_resid():
    R, LP, uid = [], [], None
    for a in ANCHORS:
        s = pl.read_parquet(FEATURES_DIR / f"anchor={a}.seqoof.parquet").sort("user_id")
        t = load_anchor(a, columns=["user_id", "target"]).sort("user_id")
        uid = t["user_id"].to_numpy()
        assert (s["user_id"].to_numpy() == uid).all()
        lp = s["seqoof_pred"].to_numpy().astype(np.float64)
        LP.append(lp)
        R.append(np.log1p(t["target"].to_numpy().astype(np.float64)) - lp)
    return uid, np.stack(R), np.stack(LP)


def decal(R, LP, bins=40):
    """Снять внутри каждого среза среднее остатка по бину прогноза (кривую калибровки)."""
    out = R.copy()
    for j in range(R.shape[0]):
        lp, r = LP[j], out[j]
        qs = np.quantile(lp, np.linspace(0, 1, bins + 1))
        qs[0] -= 1e-9
        qs[-1] += 1e-9
        b = np.clip(np.searchsorted(qs, lp, side="left") - 1, 0, bins - 1)
        cnt = np.bincount(b, minlength=bins)
        sm = np.bincount(b, weights=r, minlength=bins)
        out[j] = r - np.where(cnt > 0, sm / np.maximum(cnt, 1), 0.0)[b]
    return out


def cen(x):
    return x - x.mean()


def cr(a, b):
    return float(np.corrcoef(a, b)[0, 1])


def fit_shifts(x, y, bins=24):
    qs = np.quantile(x, np.linspace(0, 1, bins + 1))
    qs[0] -= 1e-9
    qs[-1] += 1e-9
    c, s = [], []
    for i in range(bins):
        m = (x > qs[i]) & (x <= qs[i + 1])
        if m.sum() < 500:
            continue
        c.append(x[m].mean())
        s.append(y[m].mean() - x[m].mean())
    return np.array(c), np.array(s)


def main():
    uid, R, LP = load_resid()
    Rd = decal(R, LP)
    n = len(uid)
    print(f"юзеров {n}, срезов {len(ANCHORS)}, sd остатка {R.std():.4f} "
          f"(после декалибровки {Rd.std():.4f})")

    out = {"n_users": n, "se_corr": float(1 / np.sqrt(n)), "pairs": {}}
    pairs = [("11-19+11-26", "12-31+01-07", [0, 1], [6, 7], 35),
             ("11-19", "01-07", [0], [7], 49),
             ("11-19+11-26+12-03", "01-07", [0, 1, 2], [7], 35),
             ("КОНТРОЛЬ пересекающихся", "12-03+12-10", [0, 1], [2, 3], 7)]
    print(f"\n{'группы':44s} {'дней':>5s} {'corr':>8s} {'corr_знак':>10s}")
    for na, nb, ia, ib, gap in pairs:
        a, b = cen(Rd[ia].mean(0)), cen(Rd[ib].mean(0))
        sa, sb = cen(np.sign(Rd[ia]).mean(0)), cen(np.sign(Rd[ib]).mean(0))
        out["pairs"][f"{na}|{nb}"] = dict(gap_days=gap, corr=cr(a, b), corr_sign=cr(sa, sb))
        print(f"{na + ' | ' + nb:44s} {gap:5d} {cr(a, b):8.4f} {cr(sa, sb):10.4f}")

    # var(u) из ковариации непересекающихся групп; шум окна из дисперсии одного среза
    m1, m2 = cen(Rd[[0, 1]].mean(0)), cen(Rd[[6, 7]].mean(0))
    var_u, var_e = float((m1 * m2).mean()), float(Rd[7].var())
    print(f"\nvar(u)={var_u:.5f} sd(u)={np.sqrt(max(var_u, 0)):.4f}; "
          f"дисперсия остатка одного окна {var_e:.4f}")
    proj = {}
    for k in (1, 2, 4, 8, 16, 10 ** 6):
        g = var_u ** 2 / (var_u + var_e / k)
        proj[k] = g / (2 * 1.667)
        print(f"  потолок при k={k:>7d} независимых окон: {proj[k]:.6f} RMSLE "
              f"({proj[k] / NOISE:.1f} шумов)")
    out.update(var_u=var_u, var_window=var_e,
               ceiling_by_k={str(k): v for k, v in proj.items()})

    # ---- честная проверка на валидации: поправка оценена ТОЛЬКО по обучающим срезам ----
    m = cen(Rd[PRED_IDX].mean(0))
    lam = max(var_u, 0.0) / float((m * m).mean())
    u_eb = lam * m
    u_sign = cen(np.sign(Rd[PRED_IDX]).mean(0))
    val = load_anchor(VAL_ANCHOR, columns=["user_id", "target"]).sort("user_id")
    ly = np.log1p(val["target"].to_numpy().astype(np.float64))
    assert (val["user_id"].to_numpy() == uid).all()
    lp, tot = np.zeros(n), 0.0
    for name, w in BLEND_W.items():
        d = pl.read_parquet(PREDS_DIR / f"{name}_val.parquet").sort("user_id")
        lp += w * np.log1p(np.clip(d["pred"].to_numpy().astype(np.float64), 0, None))
        tot += w
    lp /= tot
    rng = np.random.default_rng(0)
    half = rng.permutation(n) < n // 2

    def cal(v):
        t = 0.0
        for f in (half, ~half):
            c, s = fit_shifts(v[~f], ly[~f])
            adj = np.clip(v[f] + np.interp(v[f], c, s), 0, None)
            t += float(((ly[f] - adj) ** 2).mean()) * f.sum()
        return float(np.sqrt(t / n))

    base = cal(lp)
    print(f"\nбаза (бленд после калибровки, 2-fold honest): {base:.6f}")
    out["val_rmsle_base"] = base
    out["val"] = {}
    grid = [-0.04, -0.01, 0.005, 0.01, 0.02, 0.04, 0.1, 0.5, 1.0]
    for nm, u in (("eb_mean", u_eb), ("sign", u_sign)):
        curve = {float(c): cal(lp + c * u) for c in grid}
        bc = min(curve, key=curve.get)
        d = base - curve[bc]
        out["val"][nm] = dict(best_c=bc, best=curve[bc], delta=d, in_noise=d / NOISE,
                              curve={str(k): v for k, v in curve.items()})
        print(f"  [{nm}] лучший c={bc:+.3f}: {curve[bc]:.6f}, delta {d:+.6f} "
              f"({d / NOISE:.1f} шумов)")

    # почему устойчивость знака (0.23) ничего не даёт: её знает сам бленд
    def resid_on(x, b):
        b0 = cen(b)
        return x - (np.dot(cen(x), b0) / np.dot(b0, b0)) * b0

    a, b = np.sign(Rd[[0, 1]]).mean(0), np.sign(Rd[[6, 7]]).mean(0)
    out["sign_corr_raw"] = cr(a, b)
    out["sign_corr_after_blend"] = cr(resid_on(a, lp), resid_on(b, lp))
    out["corr_sign_with_blend_pred"] = cr(u_sign, lp)
    print(f"\nустойчивость знака остатка: сырая {out['sign_corr_raw']:.4f} -> "
          f"после снятия прогноза бленда {out['sign_corr_after_blend']:.4f} "
          f"(corr со прогнозом {out['corr_sign_with_blend_pred']:.4f})")
    (REPORTS_DIR / "resid_re.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print("сохранено work/reports/resid_re.json")


if __name__ == "__main__":
    main()

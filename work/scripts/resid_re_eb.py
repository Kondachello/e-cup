"""Эмпирический Байес: персональная поправка из остатков обучающих срезов + честная проверка.

Поправка u_i оценивается ТОЛЬКО по обучающим срезам (2025-07..12), валидационное окно
2026-01-14 в оценке не участвует. Усадка: lambda = var(u)/var(m), где var(u) берётся из
ковариации средних остатков на НЕПЕРЕСЕКАЮЩИХСЯ группах срезов (там шум независим),
а var(m) — дисперсия среднего остатка по всем срезам. Усадка считается ещё и поблочно
(по разбросу остатков юзера), чтобы шумных юзеров усаживать сильнее.

Проверка на валидации — ПОСЛЕ калибровки, честно: и калибровка, и шаг c подбираются
на половине юзеров, оцениваются на другой.

Запуск: POLARS_MAX_THREADS=3 .venv/bin/python work/scripts/resid_re_eb.py
"""
from __future__ import annotations

import json
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))
from common import PREDS_DIR, REPORTS_DIR, VAL_ANCHOR, load_anchor

RES = PREDS_DIR / "resid_re"
ANCHORS = [date(2025, 7, 2) + timedelta(days=7 * i) for i in range(24)]
S1 = [i for i, a in enumerate(ANCHORS) if date(2025, 7, 2) <= a <= date(2025, 8, 27)]
S2 = [i for i, a in enumerate(ANCHORS) if date(2025, 10, 1) <= a <= date(2025, 12, 10)]

# лучший честный бленд (blend_reopt.json -> winner, OOF 1.666746)
BLEND_W = {
    "fusion_avg_cal": 0.182158, "fusion_f_cal": 0.145958, "c_ts2_s42_cal": 0.11936,
    "c_ts2_avg_cal": 0.117003, "mlpziln_avg_cal": 0.084174, "behavonly_cal": 0.078989,
    "seq2tr_f_cal": 0.06909, "countaov_cal": 0.066059, "twl_v7_cal": 0.055507,
    "mlpziln_cal": 0.041982, "hmmsim_cal": 0.022222, "channel2_cal": 0.015367,
    "hmmsim": 0.004673,
}
N_CALB = 24


def fit_shifts(lp, ly, bins=N_CALB):
    qs = np.quantile(lp, np.linspace(0, 1, bins + 1))
    qs[0] -= 1e-9
    qs[-1] += 1e-9
    c, s = [], []
    for i in range(bins):
        m = (lp > qs[i]) & (lp <= qs[i + 1])
        if m.sum() < 500:
            continue
        c.append(lp[m].mean())
        s.append(ly[m].mean() - lp[m].mean())
    return np.array(c), np.array(s)


def apply_shifts(lp, c, s):
    return np.clip(lp + np.interp(lp, c, s), 0, None)


def rmsle_log(ly, lp):
    return float(np.sqrt(np.mean((ly - lp) ** 2)))


def load_blend_val(uid_ref):
    lp = np.zeros(len(uid_ref))
    tot = 0.0
    for name, w in BLEND_W.items():
        p = PREDS_DIR / f"{name}_val.parquet"
        if not p.exists():
            print(f"  ПРОПУСК {name} (нет файла)")
            continue
        d = pl.read_parquet(p).sort("user_id")
        assert (d["user_id"].to_numpy() == uid_ref).all(), name
        lp += w * np.log1p(np.clip(d["pred"].to_numpy().astype(np.float64), 0, None))
        tot += w
    return lp / tot


def main():
    Rd = np.load(RES / "R_decal.npy").astype(np.float64)
    uid = np.load(RES / "uid.npy")
    ok = np.isfinite(Rd)
    m_all = np.nansum(np.where(ok, Rd, 0), axis=1) / np.maximum(ok.sum(axis=1), 1)
    m1 = np.nanmean(Rd[:, S1], axis=1)
    m2 = np.nanmean(Rd[:, S2], axis=1)
    m_all -= m_all.mean()
    m1 -= m1.mean()
    m2 -= m2.mean()

    var_u = float(np.mean(m1 * m2))            # ковариация непересекающихся групп = var(u)
    var_m = float(np.mean(m_all ** 2))
    lam = max(var_u, 0.0) / var_m
    print(f"var(u)={var_u:.5f}  var(m_all)={var_m:.5f}  lambda_global={lam:.4f}  "
          f"sd(u)={np.sqrt(max(var_u,0)):.4f}")

    # поблочная усадка: чем шумнее юзер, тем сильнее усадка
    disp = np.nanstd(Rd, axis=1)
    qb = np.quantile(disp, np.linspace(0, 1, 11))
    qb[0] -= 1e-9
    qb[-1] += 1e-9
    b = np.clip(np.searchsorted(qb, disp, side="left") - 1, 0, 9)
    lam_b = np.zeros(10)
    print(f"\n{'блок':>4s} {'n':>7s} {'sd(r)':>7s} {'var_u':>9s} {'var_m':>9s} {'lambda':>7s}")
    for k in range(10):
        s = b == k
        vu = float(np.mean(m1[s] * m2[s]))
        vm = float(np.mean(m_all[s] ** 2))
        lam_b[k] = float(np.clip(vu / vm, 0.0, 1.0)) if vm > 0 else 0.0
        print(f"{k:4d} {s.sum():7d} {disp[s].mean():7.3f} {vu:9.5f} {vm:9.5f} {lam_b[k]:7.4f}")
    u_hat = lam_b[b] * m_all
    u_flat = lam * m_all
    print(f"\nsd(u_hat) блочный {u_hat.std():.4f}, плоский {u_flat.std():.4f}")

    # ---- честная проверка на валидации ----
    val = load_anchor(VAL_ANCHOR, columns=["user_id", "target"]).sort("user_id")
    uid_v = val["user_id"].to_numpy()
    ly = np.log1p(val["target"].to_numpy().astype(np.float64))
    order = np.searchsorted(uid, uid_v)
    assert (uid[order] == uid_v).all()
    U = {"eb_block": u_hat[order], "eb_flat": u_flat[order], "raw_mean": m_all[order]}

    lp_base = load_blend_val(uid_v)
    print(f"база (бленд, сырой лог): rmsle {rmsle_log(ly, lp_base):.6f}")

    rng = np.random.default_rng(0)
    half = rng.permutation(len(uid_v)) < len(uid_v) // 2
    grid = np.round(np.arange(-0.2, 1.001, 0.05), 3)
    res = {}
    for uname, u in U.items():
        curve = {}
        for c in grid:
            lp = lp_base + c * u
            tot = 0.0
            for fold in (half, ~half):
                cc, ss = fit_shifts(lp[~fold], ly[~fold])
                tot += rmsle_log(ly[fold], apply_shifts(lp[fold], cc, ss)) ** 2 * fold.sum()
            curve[float(c)] = float(np.sqrt(tot / len(ly)))
        base = curve[0.0]
        best_c = min(curve, key=curve.get)
        # честный выбор c: подбираем на одной половине, оцениваем на другой
        honest = 0.0
        for fit_mask, ev_mask in ((~half, half), (half, ~half)):
            best, bc = None, 0.0
            for c in grid:
                lp = lp_base + c * u
                cc, ss = fit_shifts(lp[fit_mask], ly[fit_mask])
                v = rmsle_log(ly[fit_mask], apply_shifts(lp[fit_mask], cc, ss))
                if best is None or v < best:
                    best, bc = v, float(c)
            lp = lp_base + bc * u
            cc, ss = fit_shifts(lp[fit_mask], ly[fit_mask])
            honest += rmsle_log(ly[ev_mask], apply_shifts(lp[ev_mask], cc, ss)) ** 2 * ev_mask.sum()
            print(f"  [{uname}] честный c={bc:+.2f} на половине")
        honest = float(np.sqrt(honest / len(ly)))
        res[uname] = dict(base_cal=base, best_c=float(best_c), best_cal=curve[best_c],
                          honest_cal=honest, delta_best=base - curve[best_c],
                          delta_honest=base - honest, curve=curve)
        print(f"[{uname}] после калибровки: база {base:.6f} -> лучший c={best_c:+.2f} "
              f"{curve[best_c]:.6f} (delta {base-curve[best_c]:+.6f}); честно {honest:.6f} "
              f"(delta {base-honest:+.6f})")

    out = dict(var_u=var_u, var_m=var_m, lambda_global=lam, lambda_blocks=lam_b.tolist(),
               sd_u_hat=float(u_hat.std()), results=res)
    (REPORTS_DIR / "resid_re_eb.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    np.save(RES / "u_hat.npy", u_hat)
    np.save(RES / "u_flat.npy", u_flat)
    print("\nсохранено: work/reports/resid_re_eb.json, work/preds/resid_re/u_hat.npy")


if __name__ == "__main__":
    main()

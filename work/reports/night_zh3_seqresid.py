"""Ночная проба (задача Жени, п.3): разложение дисперсии на НАСТОЯЩИХ OOF-остатках
секвенсной модели (work/features/anchor=*.seqoof.parquet) и сравнение с остатками
бленда и колонок пакета.

Методика Жени (i2/i3): кросс-секционная корреляция r(k) лог-величин между
НЕпересекающимися 30-дневными окнами, подгонка r(k)=p+q*lam^k, разложение
на постоянный уровень p / медленную компоненту q (lam) / белый шум 1-p-q.

Здесь два зонда:
  A. автокорреляция ОСТАТКОВ seq-OOF между якорями 2025-11-19..2026-01-14
     (36 пар, непересекающиеся = разнос >=35 дней);
  B. корреляция остатка на ВАЛИДАЦИОННОМ якоре с лог-таргетом прошлых
     непересекающихся окон (лаги 35..224 дней) — для бленда, seqoof и колонок
     пакета; та же кривая для самого таргета как эталон затухания.

Обучений нет: только чтение готовых parquet и numpy. Запуск из корня:
  POLARS_MAX_THREADS=2 OMP_NUM_THREADS=2 .venv/bin/python work/reports/night_zh3_seqresid.py
Пишет ТОЛЬКО work/reports/night_zh3_seqresid.json (+ печать).
"""
from __future__ import annotations

import json
import os
from datetime import date, timedelta
from pathlib import Path

for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "POLARS_MAX_THREADS"):
    os.environ.setdefault(v, "2")

import numpy as np
import polars as pl
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[2]
FEAT = ROOT / "work" / "features"
PACK = ROOT / "work" / "preds_pack" / "val_preds.parquet"
OUT = ROOT / "work" / "reports" / "night_zh3_seqresid.json"

VAL_ANCHOR = date(2026, 1, 14)
SEQ_ANCHORS = [date(2025, 11, 19), date(2025, 11, 26), date(2025, 12, 3), date(2025, 12, 10),
               date(2025, 12, 17), date(2025, 12, 24), date(2025, 12, 31), date(2026, 1, 7),
               VAL_ANCHOR]
DEEP_GAPS = [35, 42, 49, 56, 63, 70, 84, 98, 112, 126, 140, 168, 196, 224]
PACK_COLS = ["blend", "fusion_v3c_avg_cal", "fusion_v3_avg_cal", "fusion_f_cal",
             "gseq_big_s42_cal", "gseq_small_s42_cal",
             "kostya46_cal", "c_ts2_avg_cal", "febspec2_cal", "wklin_wk_cal",
             "mlpziln_cal", "behavonly_avg_cal"]
CLASS = {"blend": "blend", "fusion_v3c_avg_cal": "seq", "fusion_v3_avg_cal": "seq",
         "fusion_f_cal": "seq", "gseq_big_s42_cal": "seq", "gseq_small_s42_cal": "seq",
         "seqoof_cal": "seq", "kostya46_cal": "tab", "c_ts2_avg_cal": "tab",
         "febspec2_cal": "tab", "wklin_wk_cal": "tab", "mlpziln_cal": "tab",
         "behavonly_avg_cal": "tab"}


def cr(a, b):
    return float(np.corrcoef(a, b)[0, 1])


def decal(lp, r, bins=40):
    """Снять E[остаток | бин прогноза] внутри окна (кривая калибровки), как в resid_re.py."""
    qs = np.quantile(lp, np.linspace(0, 1, bins + 1))
    qs[0] -= 1e-9
    qs[-1] += 1e-9
    b = np.clip(np.searchsorted(qs, lp, side="left") - 1, 0, bins - 1)
    cnt = np.bincount(b, minlength=bins)
    sm = np.bincount(b, weights=r, minlength=bins)
    return r - np.where(cnt > 0, sm / np.maximum(cnt, 1), 0.0)[b]


def fit_pql(k, r, x0=(0.4, 0.16, 0.8)):
    f = lambda t: t[0] + t[1] * t[2] ** np.asarray(k, float) - np.asarray(r, float)
    s = least_squares(f, list(x0), bounds=([0, 0, 0], [1, 1, 0.999]))
    p, q, lam = map(float, s.x)
    return dict(p=p, q=q, lam=lam, noise=1 - p - q, maxres=float(np.abs(s.fun).max()))


def fit_pq_fixedlam(k, r, lam):
    """Линейная МНК p + q*lam^k при фиксированном lam (честно про узкий диапазон k)."""
    x = lam ** np.asarray(k, float)
    A = np.stack([np.ones_like(x), x], 1)
    (p, q), *_ = np.linalg.lstsq(A, np.asarray(r, float), rcond=None)
    return float(p), float(q)


def main():
    rep = {}
    pack = pl.read_parquet(PACK).sort("user_id")
    uid = pack["user_id"].to_numpy()
    n = len(uid)
    y_raw = pack["target"].to_numpy().astype(np.float64)
    ly_val = np.log1p(np.clip(y_raw, 0, None))
    lb = pack["blend"].to_numpy().astype(np.float64)
    sb = float(np.sqrt(np.mean((lb - ly_val) ** 2)))
    rep["blend_score_val"] = sb
    rep["n_users"] = n
    rep["se_corr_iid"] = float(1 / np.sqrt(n))
    print(f"эталон: blend из пакета {sb:.6f} (ожидалось 1.665647), n={n}")

    # ---------- окна из train.parquet ----------
    print("читаю train.parquet (3 колонки, с 2025-04-15)...")
    tr = (pl.scan_parquet(ROOT / "train.parquet")
          .select(["user_id", "event_date", "gmv"])
          .filter(pl.col("event_date") >= date(2025, 4, 15))
          .collect())
    uframe = pl.DataFrame({"user_id": uid})

    def win_lp(s: date, e: date) -> np.ndarray:
        w = (tr.filter(pl.col("event_date").is_between(s, e))
             .group_by("user_id").agg(pl.col("gmv").sum().alias("g")))
        g = uframe.join(w, on="user_id", how="left")["g"].to_numpy().astype(np.float64)
        return np.log1p(np.nan_to_num(g))

    # санити: сумма train за валид. окно == target пакета
    lv = win_lp(date(2026, 1, 15), date(2026, 2, 13))
    rep["sanity_val_target_max_abs_diff"] = float(np.abs(lv - ly_val).max())
    print(f"санити target(pack) vs train-сумма: max|diff| = {rep['sanity_val_target_max_abs_diff']:.2e}")

    # ---------- воспроизведение таргет-разложения Жени (его 5 пар окон) ----------
    zh_pairs = [(date(2025, 11, 16), date(2025, 12, 15), 1),
                (date(2025, 10, 17), date(2025, 11, 15), 2),
                (date(2025, 9, 17), date(2025, 10, 16), 3),
                (date(2025, 7, 19), date(2025, 8, 17), 5),
                (date(2025, 4, 20), date(2025, 5, 19), 8)]
    base = win_lp(date(2025, 12, 16), date(2026, 1, 14))
    kk, rr = [], []
    for s, e, k in zh_pairs:
        a = win_lp(s, e)
        kk.append(float(k))
        rr.append(cr(a, base))
    rep["target_zhenya"] = dict(k=kk, r=rr, fit=fit_pql(kk, rr))
    ft = rep["target_zhenya"]["fit"]
    print("таргет, окна Жени: r(k) =", " ".join(f"{v:.4f}" for v in rr))
    print(f"  фит: p={ft['p']:.4f} q={ft['q']:.4f} lam={ft['lam']:.4f} шум={ft['noise']:.4f} "
          f"(у Жени 0.416/0.180/0.788/0.404)")

    # ---------- A. остатки seq-OOF по якорям ----------
    R_raw, R_cal, scores = {}, {}, {}
    for a in SEQ_ANCHORS:
        sq = pl.read_parquet(FEAT / f"anchor={a.isoformat()}.seqoof.parquet").sort("user_id")
        t = pl.read_parquet(FEAT / f"anchor={a.isoformat()}.parquet",
                            columns=["user_id", "target"]).sort("user_id")
        assert np.array_equal(sq["user_id"].to_numpy(), uid)
        assert np.array_equal(t["user_id"].to_numpy(), uid)
        lp = sq["seqoof_pred"].to_numpy().astype(np.float64)
        lyy = np.log1p(np.clip(t["target"].to_numpy().astype(np.float64), 0, None))
        r = lyy - lp
        R_raw[a] = r
        R_cal[a] = decal(lp, r)
        scores[a.isoformat()] = dict(raw=float(np.sqrt(np.mean(r ** 2))),
                                     cal=float(np.sqrt(np.mean(R_cal[a] ** 2))))
    rep["seqoof_scores"] = scores
    print("\nseq-OOF скоры по якорям (raw -> после декалибровки):")
    for k2, v in scores.items():
        print(f"  {k2}: {v['raw']:.4f} -> {v['cal']:.4f}")

    pairs = []
    for i in range(len(SEQ_ANCHORS)):
        for j in range(i + 1, len(SEQ_ANCHORS)):
            a, b = SEQ_ANCHORS[i], SEQ_ANCHORS[j]
            gap = (b - a).days
            pairs.append(dict(gap=gap, a=a.isoformat(), b=b.isoformat(),
                              r_raw=cr(R_raw[a], R_raw[b]), r_cal=cr(R_cal[a], R_cal[b]),
                              overlap_frac=max(0, 30 - gap) / 30))
    rep["A_pairs"] = pairs
    print("\nA. корреляция остатков между якорями (cal), по разносу:")
    for g in sorted({p["gap"] for p in pairs}):
        vals = [p["r_cal"] for p in pairs if p["gap"] == g]
        ov = max(0, 30 - g) / 30
        print(f"  gap={g:3d}d k={g/30:.2f}  r_cal={np.mean(vals):+.4f} "
              f"(n={len(vals)}, min..max {min(vals):+.4f}..{max(vals):+.4f}; "
              f"доля общих дней {ov:.3f})")

    non = [p for p in pairs if p["gap"] >= 35]
    kA = [p["gap"] / 30 for p in non]
    rA_cal = [p["r_cal"] for p in non]
    rA_raw = [p["r_raw"] for p in non]
    lamz = ft["lam"]
    pA, qA = fit_pq_fixedlam(kA, rA_cal, lamz)
    pAr, qAr = fit_pq_fixedlam(kA, rA_raw, lamz)
    rep["A_nonoverlap"] = dict(
        n_pairs=len(non), k_range=[min(kA), max(kA)],
        mean_r_cal=float(np.mean(rA_cal)), max_abs_r_cal=float(np.max(np.abs(rA_cal))),
        mean_r_raw=float(np.mean(rA_raw)),
        fit_fixedlam_cal=dict(p=pA, q=qA, lam=lamz),
        fit_fixedlam_raw=dict(p=pAr, q=qAr, lam=lamz),
        note="p и q на k=1.17..1.87 не разделяются (узкий диапазон); честная величина - "
             "сама |r|: весь п.у.+медленный след в остатке <= max|r|")

    # бутстрап SE для одной пары gap=35 (11-19 vs 12-24)
    a35, b35 = R_cal[date(2025, 11, 19)], R_cal[date(2025, 12, 24)]
    rng = np.random.default_rng(0)
    bs = []
    for _ in range(200):
        ii = rng.integers(0, n, n)
        bs.append(cr(a35[ii], b35[ii]))
    rep["A_boot_se_gap35"] = float(np.std(bs))
    print(f"\nA. непересекающиеся (gap>=35): mean r_cal = {np.mean(rA_cal):+.5f}, "
          f"max|r_cal| = {np.max(np.abs(rA_cal)):.5f}, boot SE = {rep['A_boot_se_gap35']:.5f}")
    print(f"   фикс-lam фит: p={pA:+.5f} q={qA:+.5f} (не интерпретировать по отдельности)")

    # таргет-пары на тех же якорях (эталон масштаба)
    tpairs = []
    LYs = {a: np.log1p(np.clip(pl.read_parquet(FEAT / f"anchor={a.isoformat()}.parquet",
                                               columns=["user_id", "target"]).sort("user_id")
                               ["target"].to_numpy().astype(np.float64), 0, None))
           for a in SEQ_ANCHORS}
    for i in range(len(SEQ_ANCHORS)):
        for j in range(i + 1, len(SEQ_ANCHORS)):
            a, b = SEQ_ANCHORS[i], SEQ_ANCHORS[j]
            if (b - a).days >= 35:
                tpairs.append(dict(gap=(b - a).days, r=cr(LYs[a], LYs[b])))
    rep["A_target_pairs_nonoverlap"] = tpairs
    tmean = float(np.mean([p["r"] for p in tpairs]))
    print(f"   таргет на тех же непересекающихся парах: mean r = {tmean:.4f} "
          f"-> остаток стирает {100 * (1 - np.mean(rA_cal) / tmean):.1f}% структуры")

    # ---------- B. глубокий зонд на валидационном якоре ----------
    resid = {}
    lp_seq_val = pl.read_parquet(FEAT / f"anchor={VAL_ANCHOR.isoformat()}.seqoof.parquet") \
        .sort("user_id")["seqoof_pred"].to_numpy().astype(np.float64)
    resid["seqoof_cal"] = decal(lp_seq_val, ly_val - lp_seq_val)
    for c in PACK_COLS:
        lp = pack[c].to_numpy().astype(np.float64)
        resid[c] = ly_val - lp
    solo = {m: float(np.sqrt(np.mean(r ** 2))) for m, r in resid.items()}
    rep["B_solo_scores"] = solo

    past = {}
    for g in DEEP_GAPS:
        a_g = VAL_ANCHOR - timedelta(days=g)          # якорь прошлого окна
        past[g] = win_lp(a_g + timedelta(days=1), a_g + timedelta(days=30))

    kB = [g / 30 for g in DEEP_GAPS]
    target_r = [cr(lv, past[g]) for g in DEEP_GAPS]
    fitB = fit_pql(kB, target_r)
    rep["B_target"] = dict(gaps=DEEP_GAPS, k=kB, r=target_r, fit=fitB)
    print(f"\nB. таргет vs прошлые окна: r({kB[0]:.2f})={target_r[0]:.4f} ... "
          f"r({kB[-1]:.2f})={target_r[-1]:.4f}")
    print(f"   фит: p={fitB['p']:.4f} q={fitB['q']:.4f} lam={fitB['lam']:.4f}")

    rep["B_resid"] = {}
    print(f"\nB. corr(остаток_val, лог-таргет прошлого окна): "
          f"[ближние 35-70 | дальние 84-224]")
    near_idx = [i for i, g in enumerate(DEEP_GAPS) if g <= 70]
    far_idx = [i for i, g in enumerate(DEEP_GAPS) if g >= 84]
    for m, r in resid.items():
        curve = [cr(r, past[g]) for g in DEEP_GAPS]
        pB, qB = fit_pq_fixedlam(kB, curve, fitB["lam"])
        near = float(np.mean([curve[i] for i in near_idx]))
        far = float(np.mean([curve[i] for i in far_idx]))
        cmax = float(np.max(np.abs(curve)))
        rep["B_resid"][m] = dict(cls=CLASS[m], solo=solo[m], curve=curve,
                                 near35_70=near, far84_224=far, max_abs=cmax,
                                 p_fixedlam=pB, q_fixedlam=qB,
                                 gain_bound_single=float(solo[m] * cmax ** 2 / 2))
        print(f"  {m:22s} [{CLASS[m]:5s}] solo={solo[m]:.4f}  near={near:+.4f} "
              f"far={far:+.4f}  p={pB:+.4f} q={qB:+.4f}")

    # ---------- бюджет: сколько медленной компоненты выбрано ----------
    var_y = float(ly_val.var())
    rep["val_target_var"] = var_y
    p_t, q_t = ft["p"], ft["q"]
    rep["budget"] = {}
    print(f"\nбюджет (по таргет-разложению p={p_t:.3f} q={q_t:.3f}, var_y={var_y:.3f}):")
    for m, r in resid.items():
        ev = 1 - float(r.var()) / var_y
        share_s = (ev - p_t) / q_t
        beyond_noise = float(r.var()) / var_y - (1 - p_t - q_t)
        rep["budget"][m] = dict(explained_var=ev, slow_captured_share=share_s,
                                resid_beyond_noise_frac=beyond_noise)
        print(f"  {m:22s} mdl_flint={ev:.4f}  взято медленной={share_s:+.3f}  "
              f"сверх шума={beyond_noise:+.4f}")

    OUT.write_text(json.dumps(rep, ensure_ascii=False, indent=1))
    print(f"\nсохранено {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

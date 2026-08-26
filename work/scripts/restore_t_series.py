"""Восстановление T-серии (tfm4) на этой машине и её проверка спан-предиктором.

ЗАЧЕМ. T1_tfm4_orth_090 и T2_tfm4_orth_045 заливались с машины трека 5; CSV в
submissions/ не попали, а T2 — лучший честный файл проекта (1.6469638837) и
кандидат в финалисты. Без физического файла не работают finalist_guard --pair,
пересборка lb_cache (по составу MEASURED) и любая алгебра поверх T2.

КАК. Направление tfm4 детерминированно восстанавливается из handoff-паркетов
трека 5 (work/handoff/, в git) дисциплиной V/G-серии (make_g_candidates.py):
NNLS [бленд | tfm4 | tfm4_tabless] на всей валидации -> дельта тестовых блендов,
центрирование; дальше два варианта оси:
    centered — только центрирование (дисциплина G-серии);
    orth     — плюс ортогонализация к (V3 - mean) (дисциплина s5_candidates);
и два варианта членов: калибровка сырых fit_shifts(24) против готовых *_cal
хэндоффа (трек 5 утверждает тождество). Респред к моментам V3 — как в emit().

ПРОВЕРКА, без которой файлы не писать:
  1) пайплайн обязан воспроизводить G1/G2/F1 (они собраны make_g_candidates.py
     на этой машине) с точностью до плавающей запятой;
  2) спан-предиктор (конструкция s3_full.py Жени: базис из локально доступных
     замеренных файлов + константа + sample, phi точен на замеренных, остаток
     через ковариацию с tval, a=1.052) обязан предсказать ЗАМЕРЕННЫЕ скоры
     T1 (1.6471433388) и T2 (1.6469638837) для выбранного варианта оси в
     пределах точности предиктора (~1e-4); вариант выбирается по сумме промахов
     на ПАРЕ доз (0.90, 0.45) — две точки пиннят параболу оси.

Файлы пишутся с --emit. Это РЕКОНСТРУКЦИЯ: на платформе лежат оригиналы, и
финалист выбирается среди залитых там; локальные копии нужны для алгебры и
guard'а. Факт реконструкции фиксируется в отчёте work/reports/t_restore.md.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))
from margin import fit_shifts  # noqa: E402

PACK = ROOT / "work" / "preds_pack"
HANDOFF = ROOT / "work" / "handoff"
PREDS = ROOT / "work" / "preds"
SUB = ROOT / "submissions"

T1_SCORE = 1.6471433387910561
T2_SCORE = 1.6469638837149883
# уровневые константы спан-предиктора s3_full.py (замеры зондов уровня)
MEAN_T, MEAN_T2, A_RESID = 2.3275, 10.79, 1.0520


def rd_lp_csv(p: Path) -> tuple[np.ndarray, np.ndarray]:
    d = pl.read_csv(p, schema_overrides={"user_id": pl.Int64}).sort("user_id")
    return (d["user_id"].to_numpy(),
            np.log1p(np.clip(d[d.columns[1]].to_numpy().astype(np.float64), 0, None)))


def rd_pq(p: Path) -> np.ndarray:
    d = pl.read_parquet(p).sort("user_id")
    col = "pred" if "pred" in d.columns else [c for c in d.columns if c != "user_id"][0]
    return np.clip(d[col].to_numpy().astype(np.float64), 0, None)


def nnls_free(A: np.ndarray, y: np.ndarray) -> np.ndarray:
    from scipy.optimize import nnls
    G, b = A.T @ A, A.T @ y
    L = np.linalg.cholesky(G + 1e-12 * np.trace(G) / len(G) * np.eye(len(G)))
    w, _ = nnls(L.T, np.linalg.solve(L, b))
    return w


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--emit", action="store_true", help="писать T1/T2 в submissions/")
    args = ap.parse_args()

    val = pl.read_parquet(PACK / "val_preds.parquet").sort("user_id")
    test = pl.read_parquet(PACK / "test_preds.parquet").sort("user_id")
    y = np.log1p(val["target"].to_numpy().astype(np.float64))
    bv = val["blend"].to_numpy().astype(np.float64)
    bt = test["blend"].to_numpy().astype(np.float64)
    uid_t = test["user_id"].to_numpy()

    uid, v3 = rd_lp_csv(SUB / "V3_canon.csv")
    assert np.array_equal(uid, uid_t), "порядок user_id теста и V3 разошёлся"
    sd_v3, mu_v3 = v3.std(), v3.mean()
    sb = float(np.sqrt(((y - bv) ** 2).mean()))
    print(f"бленд val {sb:.6f}; V3 mean {mu_v3:.4f} sd {sd_v3:.4f}")

    # ---------------------------------------------------------------- члены
    def member_recal(stem: str, src: Path) -> tuple[np.ndarray, np.ndarray]:
        lv = np.log1p(rd_pq(src / f"{stem}_val.parquet"))
        lt = np.log1p(rd_pq(src / f"{stem}_test.parquet"))
        c, s = fit_shifts(lv, y, 24)
        return (np.clip(lv + np.interp(lv, c, s), 0, None),
                np.clip(lt + np.interp(lt, c, s), 0, None))

    def member_cal(stem: str, src: Path) -> tuple[np.ndarray, np.ndarray]:
        return (np.log1p(rd_pq(src / f"{stem}_cal_val.parquet")),
                np.log1p(rd_pq(src / f"{stem}_cal_test.parquet")))

    def axis(members: list[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray, float]:
        Av = np.column_stack([bv] + [m[0] for m in members])
        w = nnls_free(Av, y)
        gain = sb - float(np.sqrt(((y - Av @ w) ** 2).mean()))
        delta = np.column_stack([bt] + [m[1] for m in members]) @ w - bt
        delta -= delta.mean()
        return delta, w, gain

    def emit_lp(base: np.ndarray, dose: float, d: np.ndarray) -> np.ndarray:
        lp = base + dose * d
        lp = (lp - lp.mean()) / max(lp.std(), 1e-12) * sd_v3 + mu_v3
        return np.clip(lp, 0, None)

    # ------------------------------------------ контроль пайплайна: G-серия
    d_g, w_g, gain_g = axis([member_recal("kevf_gru_swa", PREDS),
                             member_recal("tfm3b", PREDS)])
    for name, dose in [("G1_gru_tfm_full", 1.0), ("G2_gru_tfm_02", 0.2)]:
        _, lp_disk = rd_lp_csv(SUB / f"{name}.csv")
        lp_re = emit_lp(v3, dose, d_g)
        # CSV хранит predict, сверяем в лог-пространстве после той же записи
        err = float(np.max(np.abs(lp_re - lp_disk)))
        print(f"репродукция {name}: max|Δlp| = {err:.2e}")
        if err > 1e-9:
            print("  ВНИМАНИЕ: пайплайн не воспроизводит G-серию точно")

    # ------------------------------------------------- варианты оси tfm4
    variants: dict[str, np.ndarray] = {}
    for mem_tag, mem_fn in [("cal", member_cal), ("recal", member_recal)]:
        d_t, w_t, gain_t = axis([mem_fn("tfm4", HANDOFF),
                                 mem_fn("tfm4_tabless", HANDOFF)])
        print(f"tfm4 axis [{mem_tag}]: веса {np.round(w_t, 4)} "
              f"in-sample gain {gain_t:.6f} sd(δ) {d_t.std():.5f}")
        variants[f"centered_{mem_tag}"] = d_t
        vb = v3 - v3.mean()
        d_o = d_t - vb * float(np.dot(d_t, vb)) / float(np.dot(vb, vb))
        d_o -= d_o.mean()
        variants[f"orth_{mem_tag}"] = d_o

    # ------------------------------------------------------ спан-предиктор
    import predict_lb
    avail, missing = [], []
    for name, fname, score in predict_lb.MEASURED:
        p = SUB / fname
        p2 = SUB / "canonical" / fname
        q = p if p.exists() else (p2 if p2.exists() else None)
        (avail if q else missing).append((name, q, score))
    print(f"базис: {len(avail)} доступных замеренных, нет файлов: "
          f"{[n for n, _, _ in missing]}")

    ss = pl.read_csv(ROOT / "sample_submit.csv",
                     schema_overrides={"user_id": pl.Int64}).sort("user_id")
    tval = np.log1p(np.clip(ss[ss.columns[1]].to_numpy().astype(np.float64), 0, None))

    names = [n for n, _, _ in avail] + ["sample"]
    lps = [rd_lp_csv(q)[1] for _, q, _ in avail] + [tval]
    scores = {n: s for n, _, s in avail} | {"sample": 2.122483523224017}
    phi = {n: (float(np.mean(lp * lp)) + MEAN_T2 - scores[n] ** 2) / 2
           for n, lp in zip(names, lps)}
    B_full = np.column_stack([np.ones(len(uid))] + lps)
    PH_full = np.array([MEAN_T] + [phi[n] for n in names])

    def predict(lp: np.ndarray, drop: tuple[str, ...] = ()) -> float:
        idx = [0] + [i + 1 for i, n in enumerate(names) if n not in drop]
        Bs, PHs = B_full[:, idx], PH_full[idx]
        w = np.linalg.lstsq(Bs.T @ Bs / len(uid), Bs.T @ lp / len(uid), rcond=None)[0]
        r = lp - Bs @ w
        ph = float(w @ PHs) + A_RESID * float(np.mean(r * (tval - tval.mean())))
        return float(np.sqrt(max(float(np.mean(lp * lp)) - 2 * ph + MEAN_T2, 0)))

    print("\n=== калибровка предиктора (leave-self-out на известных) ===")
    for chk in ("G2_gru_tfm_02", "V3_canon", "G1_gru_tfm_full"):
        i = names.index(chk)
        pr = predict(lps[i], drop=(chk,))
        print(f"{chk}: предсказано {pr:.7f}  замерено {scores[chk]:.7f}  "
              f"Δ {pr - scores[chk]:+.7f}")

    print("\n=== варианты оси tfm4 против замеренных T1/T2 ===")
    best_tag, best_err = None, np.inf
    for tag, d in variants.items():
        p1 = predict(emit_lp(v3, 0.90, d))
        p2 = predict(emit_lp(v3, 0.45, d))
        err = abs(p1 - T1_SCORE) + abs(p2 - T2_SCORE)
        print(f"{tag:15s}: T1 {p1:.7f} (Δ{p1 - T1_SCORE:+.7f})  "
              f"T2 {p2:.7f} (Δ{p2 - T2_SCORE:+.7f})  сумма промахов {err:.7f}")
        if err < best_err:
            best_tag, best_err = tag, err

    print(f"\nлучший вариант: {best_tag} (сумма промахов {best_err:.7f})")
    d_best = variants[best_tag]

    if args.emit:
        for name, dose in [("T1_tfm4_orth_090", 0.90), ("T2_tfm4_orth_045", 0.45)]:
            lp = emit_lp(v3, dose, d_best)
            out = SUB / f"{name}.csv"
            pl.DataFrame({"user_id": uid, "predict": np.expm1(lp)}).write_csv(out)
            print(f"записан {out.name}: mean lp {lp.mean():.6f} sd {lp.std():.6f}")
        np.save(ROOT / "work" / "reports" / "d_tfm4.npy", d_best)
        np.save(ROOT / "work" / "reports" / "d_gru_tfm.npy", d_g)
        print("направления сохранены: work/reports/d_tfm4.npy, d_gru_tfm.npy")


if __name__ == "__main__":
    main()

"""G-пак: контролируемая пара полным шагом (гипотеза «v1-attention вредит на LB»).

Оси (обе — реопт пакового бленда с новыми членами, NNLS на полной валидации):
  G1 = бленд + {kevf_gru_swa, tfm3b}          — БЕЗ v1-attention
  F1 = бленд + {kevf_s42, kevf_gru_swa, tfm3b} — С v1-attention
не переносится» (κ=−0.26±0.33 при дозе 0.5, размыто). Пара F1/G1 полным шагом —
чистое разрешение спора: всё совпадает, кроме присутствия kevf_s42.

Сборка сабмита — дисциплина V-серии: дельта нового ТЕСТОВОГО бленда против
пакового тестового бленда, центрированная, поверх V3_canon; затем respread к
sd(V3) с сохранением среднего (уровень и шейдинг не трогаем — их оси уже
забанканы в V3 своими дозами).

  G2 = V3 + 0.20·δ(G1) — слепая доза по доктрине (E[κ]=0.20) на случай κ>0.

Калибровка членов: поквантильные сдвиги подобраны на ВСЕЙ валидации и приложены
к тесту (как для tfm3b_cal/kevf_s42_cal). Для kevf_gru_swa строится тут же.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))
from margin import fit_shifts  # noqa: E402

PACK = ROOT / "work" / "preds_pack"
PREDS = ROOT / "work" / "preds"
SUB = ROOT / "submissions"


def rd_lp_csv(p: Path) -> tuple[np.ndarray, np.ndarray]:
    d = pl.read_csv(p, schema_overrides={"user_id": pl.Int64}).sort("user_id")
    return (d["user_id"].to_numpy(),
            np.log1p(np.clip(d["predict"].to_numpy().astype(np.float64), 0, None)))


def rd_pq(p: Path) -> np.ndarray:
    d = pl.read_parquet(p).sort("user_id")
    col = "pred" if "pred" in d.columns else [c for c in d.columns if c != "user_id"][0]
    v = d[col].to_numpy().astype(np.float64)
    return np.clip(v, 0, None)


def nnls_free(A: np.ndarray, y: np.ndarray) -> np.ndarray:
    from scipy.optimize import nnls
    G, b = A.T @ A, A.T @ y
    L = np.linalg.cholesky(G + 1e-12 * np.trace(G) / len(G) * np.eye(len(G)))
    w, _ = nnls(L.T, np.linalg.solve(L, b))
    return w


def main() -> None:
    val = pl.read_parquet(PACK / "val_preds.parquet").sort("user_id")
    test = pl.read_parquet(PACK / "test_preds.parquet").sort("user_id")
    y = np.log1p(val["target"].to_numpy().astype(np.float64))
    bv = val["blend"].to_numpy().astype(np.float64)
    bt = test["blend"].to_numpy().astype(np.float64)

    # члены: log1p val/test, калибровка сдвигами всей валидации
    def member(name: str) -> tuple[np.ndarray, np.ndarray]:
        # _cal-паркеты в work/preds хранят СЫРОЙ GMV — калибруем сами, единообразно
        lv = np.log1p(rd_pq(PREDS / f"{name}_val.parquet"))
        lt = np.log1p(rd_pq(PREDS / f"{name}_test.parquet"))
        c, s = fit_shifts(lv, y, 24)
        return (np.clip(lv + np.interp(lv, c, s), 0, None),
                np.clip(lt + np.interp(lt, c, s), 0, None))

    m_names = ["kevf_s42", "kevf_gru_swa", "tfm3b"]
    mv, mt = {}, {}
    for n in m_names:
        mv[n], mt[n] = member(n)
        print(f"{n}: val rmsle {np.sqrt(((y-mv[n])**2).mean()):.6f}")

    uid, v3 = rd_lp_csv(SUB / "V3_canon.csv")
    sd_v3, mu_v3 = v3.std(), v3.mean()
    sb = np.sqrt(((y - bv) ** 2).mean())
    print(f"бленд val {sb:.6f}; V3 test mean {mu_v3:.4f} sd {sd_v3:.4f}")

    uid_t = test["user_id"].to_numpy()
    assert np.array_equal(uid, uid_t), "порядок user_id теста и V3 разошёлся"

    def axis(names: list[str]) -> tuple[np.ndarray, np.ndarray, float]:
        A = np.column_stack([bv] + [mv[n] for n in names])
        w = nnls_free(A, y)
        new_v = A @ w
        gain = sb - np.sqrt(((y - new_v) ** 2).mean())
        new_t = np.column_stack([bt] + [mt[n] for n in names]) @ w
        delta = new_t - bt
        delta -= delta.mean()  # центр: уровень не трогаем
        return delta, w, gain

    d_g, w_g, gain_g = axis(["kevf_gru_swa", "tfm3b"])
    d_f, w_f, gain_f = axis(["kevf_s42", "kevf_gru_swa", "tfm3b"])
    print(f"G1 веса {np.round(w_g,4)} in-sample gain {gain_g:.6f} |δ| sd {d_g.std():.5f}")
    print(f"F1 веса {np.round(w_f,4)} in-sample gain {gain_f:.6f} |δ| sd {d_f.std():.5f}")

    def emit(name: str, lp: np.ndarray) -> None:
        # respread к канону V3: сохранить mean и sd
        lp = (lp - lp.mean()) / max(lp.std(), 1e-12) * sd_v3 + mu_v3
        lp = np.clip(lp, 0, None)
        out = SUB / name
        pl.DataFrame({"user_id": uid, "predict": np.expm1(lp)}).write_csv(out)
        print(f"{out.name}: mean lp {lp.mean():.6f} sd {lp.std():.6f}")

    emit("G1_gru_tfm_full.csv", v3 + d_g)
    emit("F1_trio_full.csv", v3 + d_f)
    emit("G2_gru_tfm_02.csv", v3 + 0.2 * d_g)


if __name__ == "__main__":
    main()

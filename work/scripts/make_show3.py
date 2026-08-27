"""SHOW3: паблик-арбитраж на выросшем спане (замеренных файла против 67 у SHOW).

Модель (та же алгебра, что в predict_lb): для элемента спана lp = lp_a + Bᵀc
    f²(c) = f_a² + 2·cᵀm + cᵀG c − 2·cᵀψ,   m = B·lp_a/N,  G = BBᵀ/N
Минимум: (G + λR)·c = ψ − m. λ — гребень: он и держит обусловленность
(V2/V3 почти коллинеарны), и контролирует смещение ψ (ψ и G посчитаны на
250k, паблик — 50k; ошибка растёт с ||w||₁ — урок SHOW: расчёт-факт +0.000195).

Смещение калибруем по ДВУМ точкам расчёт-факт (SHOW_maxpub, SHOW2_aggr),
предсказанным базисом БЕЗ обоих SHOW-файлов: miss ≈ γ·||w||₁ + δ0.
Выбор λ* — минимум скорректированного прогноза. Выпускаем два файла:
SHOW3_maxpub (λ*) и SHOW3b_safe (консервативный: ||w||₁ на уровне SHOW,
где закон смещения заякорен, а спан всё равно шире прежнего).

НЕ финалисты. На приват не переносятся. Только по прямому приказу.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))
from predict_lb import ANCHOR, MEAN_T, load_basis  # noqa: E402

SUB = ROOT / "submissions"


def quad_parts(basis: dict, idx: list[int]):
    L, f = basis["L"].astype(np.float64), basis["f"]
    names = list(basis["names"])
    a = names.index(ANCHOR)
    N = L.shape[1]
    lp_a, f_a = L[a], float(f[a])
    qd = (L * L).mean(1)
    psi = ((qd - qd[a]) - (f**2 - f[a] ** 2)) / 2
    D = L[idx] - lp_a
    B = np.vstack([np.ones(N), D])
    G = B @ B.T / N
    m = B @ lp_a / N
    psi_vec = np.concatenate([[MEAN_T], psi[idx]])
    return dict(lp_a=lp_a, f_a=f_a, B=B, G=G, m=m, psi=psi_vec, names=[names[i] for i in idx])


def solve(qp: dict, lam: float):
    k = len(qp["m"])
    R = np.eye(k)
    R[0, 0] = 1e-4  # сдвиг уровня почти свободен: mean_P(t) известна точно
    c = np.linalg.solve(qp["mdl_corund"] + lam * R, qp["psi"] - qp["m"])
    fsq = qp["f_a"] ** 2 + 2 * c @ qp["m"] + c @ qp["mdl_corund"] @ c - 2 * c @ qp["psi"]
    return c, float(np.sqrt(max(fsq, 1e-12))), float(np.abs(c[1:]).sum())


def project(qp: dict, lp: np.ndarray):
    """расчёт для готового файла: проекция на спан (остаток игнорируем — SHOW-файлы в спане)."""
    d = lp - qp["lp_a"]
    N = len(lp)
    c = np.linalg.solve(qp["mdl_corund"] + 1e-9 * np.trace(qp["mdl_corund"]) / len(qp["mdl_corund"]) * np.eye(len(qp["mdl_corund"])),
                        qp["B"] @ d / N)
    q = float((lp * lp).mean())
    q_a = float((qp["lp_a"] ** 2).mean())
    fsq = qp["f_a"] ** 2 + (q - q_a) - 2 * float(c @ qp["psi"])
    return float(np.sqrt(max(fsq, 1e-12))), float(np.abs(c[1:]).sum())


def main() -> None:
    basis = load_basis()
    names = list(basis["names"])
    f = basis["f"]
    show_i = [names.index("SHOW_maxpub"), names.index("SHOW2_aggr")]

    # --- калибровка смещения: базис без SHOW-файлов предсказывает их же
    idx_noshow = [i for i in range(len(names)) if i not in show_i]
    qp0 = quad_parts(basis, idx_noshow)
    pts = []
    for i in show_i:
        pred, wsum = project(qp0, basis["L"][i].astype(np.float64))
        miss = float(f[i]) - pred
        pts.append((wsum, miss))
        print(f"{names[i]}: расчёт {pred:.7f} факт {f[i]:.7f} miss {miss:+.6f} ||w||₁ {wsum:.2f}")
    (w1, m1), (w2, m2) = pts
    if abs(w2 - w1) > 1e-6:
        gam = (m2 - m1) / (w2 - w1)
        d0 = m1 - gam * w1
    else:
        gam, d0 = m1 / max(w1, 1e-9), 0.0
    print(f"закон смещения: miss ≈ {gam:+.2e}·||w||₁ {d0:+.2e}")

    # --- синтез на полном спане
    qp = quad_parts(basis, list(range(len(names))))
    rows = []
    for lam in np.geomspace(3e-7, 3e-2, 40):
        c, fr, wsum = solve(qp, lam)
        corr = fr + max(gam * wsum + d0, 0.0)
        rows.append((lam, fr, wsum, corr, c))
    # трастовая зона: расчёт проверен фактами при ||w||₁ ~ 27 (обе SHOW-точки, |miss| < 1e-4);
    # экстраполяция закона смещения на большие ||w|| неопознаваема (точки почти совпали),
    # поэтому «aggr» — умеренный выход за якорь (×1.7), а не глобальный минимум расчёта
    anchor_w = max(w1, w2)
    in_trust = [r for r in rows if r[2] <= 1.7 * anchor_w]
    best = min(in_trust, key=lambda r: r[3])
    safe_rows = [r for r in rows if r[2] <= anchor_w]
    safe = min(safe_rows, key=lambda r: r[3]) if safe_rows else best
    for tag, r in (("aggr", best), ("safe", safe)):
        lam, fr, wsum, corr, c = r
        print(f"{tag}: λ {lam:.2e} расчёт {fr:.7f} ||w||₁ {wsum:.2f} скорректировано {corr:.7f}")

    uid = basis["uid"]
    for fname, (_, fr, wsum, corr, c) in (("SHOW3_maxpub.csv", best), ("SHOW3b_safe.csv", safe)):
        lp = qp["lp_a"] + qp["B"].T @ c
        nneg = int((lp < 0).sum())
        lp = np.clip(lp, 0, None)
        pl.DataFrame({"user_id": uid, "predict": np.expm1(lp)}).write_csv(SUB / fname)
        print(f"{fname}: mean lp {lp.mean():.4f} sd {lp.std():.4f} clip<0 {nneg} "
              f"расчёт {fr:.7f} → ожидание {corr:.7f}")
    for tag, r in (("aggr", best), ("safe", safe)):
        top = sorted(zip(qp["names"], r[4][1:]), key=lambda t: -abs(t[1]))[:8]
        print(f"топ веса {tag}:", ", ".join(f"{n} {v:+.3f}" for n, v in top))


if __name__ == "__main__":
    main()

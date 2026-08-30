"""Обметание ВСЕХ колонок пакета как осей: живая ось или мёртвая. Пункт A1.

ЗАЧЕМ. `axis_applied.py` считает одну ось за запуск и требует направление либо
parquet-ом (`--dir`), либо моделью из `work/preds/` (`--model`). В этом клоне
`work/preds/` нет, поэтому `--model` не работает. Но 25 модельных колонок пакета
`work/preds_pack/test_preds.parquet` — это готовые направления: колонка минус
`blend`, обе УЖЕ в log1p. Скрипт метёт их все разом одной таблицей.

ЧТО СЧИТАЕТ (конвенция ровно та же, что в `axis_applied.py`, все векторы центрированы):

    d      = колонка − blend                           направление оси
    corr   = lp(отправленный файл) − blend             вся применённая поправка
    q      = mean(d²)                                  квадрат нормы шага
    proj   = mean(corr·d)/q                            доза вдоль оси, В ШАГАХ
    cos    = mean(corr·d)/sqrt(q·mean(corr²))
    сигм   = |proj − mean(случ)|/sd(случ) по 200 случайным направлениям нормы q,
             ровно тот же розыгрыш, что у axis_applied (default_rng(0))

Гейт: `probes.probe_value(q, tau)/NOISE >= 1`, tau = 0.196 — приор ПРЕДЛОЖЕННЫХ
осей (`doctrine.dose.PRIOR_PROPOSED`, он же `PRIOR_MODEL` в солверах линии A).
Модельная разность — предложенная ось, не сегментная и не ось разложения.

ОРТОГОНАЛЬНЫЙ ОСТАТОК. `d_perp = d − D(DᵀD)⁻¹Dᵀd`, где D — подпространство
ДОСТУПНЫХ применённых направлений: поправки `lp(файл) − blend` всех замеренных CSV,
что физически лежат на диске, плюс `work/data/dir_*.parquet`. Настоящая цепочка —
46 направлений, на диске их сильно меньше, поэтому ЛЮБОЙ q_perp отсюда —
ВЕРХНЯЯ ГРАНИЦА новизны, а не оценка.

Запуск:
  .venv/bin/python work/scripts/olya_axis_sweep.py --file F12_ebint
  .venv/bin/python work/scripts/olya_axis_sweep.py --file F13_g0 --json out.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "work"))
from doctrine.probes import probe_value                      # noqa: E402
from doctrine.transfer import NOISE, F0_DEFAULT              # noqa: E402
from doctrine.dose import PRIOR_PROPOSED, PRIOR_DECOMP       # noqa: E402

TAU = PRIOR_PROPOSED[1]          # 0.196 — приор ПРЕДЛОЖЕННЫХ (новых модельных) осей
TAU_DEC = PRIOR_DECOMP[1]        # 0.0148 — приор осей Z-РАЗЛОЖЕНИЯ (члены бленда)
N_RAND = 200
L1 = lambda x: np.log1p(np.clip(np.asarray(x, np.float64), 0, None))


def c(x: np.ndarray) -> np.ndarray:
    """Центрирование — общий сдвиг метрику не двигает, его убирают везде."""
    return x - x.mean()


def read_sub(path: Path, uid: np.ndarray) -> np.ndarray:
    d = pl.read_csv(path, schema_overrides={"user_id": pl.Int64}).sort("user_id")
    col = "predict" if "predict" in d.columns else d.columns[1]
    assert np.array_equal(d["user_id"].to_numpy(), uid), f"{path.name}: другой user_id"
    return L1(d[col].to_numpy())


def rand_dots(corr: np.ndarray, n: int = N_RAND, seed: int = 0) -> np.ndarray:
    """<corr, z_i> для 200 случайных направлений ЕДИНИЧНОЙ нормы (mean(z²)=1).

    Розыгрыш тот же, что в axis_applied.py: default_rng(0), центрирование,
    нормировка. Там z домножается на sqrt(q) и делится на q, значит
    proj_rand = <corr, z_unit>/sqrt(q) — отсюда числа воспроизводятся точно,
    а гонять 250k×200 для каждой из 25 осей не нужно.
    """
    rng = np.random.default_rng(seed)
    out = np.empty(n)
    for i in range(n):
        z = rng.standard_normal(len(corr))
        z -= z.mean()
        z /= np.sqrt(np.mean(z * z))
        out[i] = float(np.mean(corr * z))
    return out


def blend_weights() -> dict[str, float]:
    """Веса действующего бленда. Член бленда -> ось Z-РАЗЛОЖЕНИЯ (tau 0.0148,
    mu = 0 СТРОГО по условию 1-го порядка), не член -> новая модельная ось
    (tau 0.196). Приоры СМЕШИВАТЬ НЕЛЬЗЯ (doctrine/dose.py, K3 §4.2)."""
    d = json.loads((ROOT / "work" / "reports" / "blend_reopt.json").read_text())
    w = d["winner"].get("weights") or d["winner"].get("w") or {}
    return {k: v for k, v in w.items() if abs(v) > 1e-4}


def val_side(pack: str):
    """Валидационная доза оси: kappa_val = <y−blend, d>/<d,d> и соло-скор модели.

    Это НЕ замер лидерборда и не заменяет его: окно другое (15.01–13.02), а
    цепочка дозировалась по паблику. Нужно ровно для одного вопроса — отличить
    «цепочка не тронула ХОРОШЕЕ направление» от «цепочка правильно не тронула
    бесполезное». `probe_value` этого вопроса не задаёт: она растёт с q и не знает,
    куда ось ведёт.
    """
    v = pl.read_parquet(f"{pack}/val_preds.parquet").sort("user_id")
    y = L1(v["target"].to_numpy())
    b = v["blend"].to_numpy().astype(np.float64)
    e = c(y - b)
    out = {}
    for k in v.columns:
        if k in ("user_id", "blend", "target"):
            continue
        m = v[k].to_numpy().astype(np.float64)
        d = c(m - b)
        qv = float(np.mean(d * d))
        kap = float(np.mean(e * d)) / qv
        out[k] = dict(kappa_val=kap, q_val=qv,
                      rmsle_val=float(np.sqrt(np.mean((y - m) ** 2))),
                      gain_val_noise=qv * kap * kap / (2 * F0_DEFAULT) / NOISE)
    out["__blend__"] = float(np.sqrt(np.mean((y - b) ** 2)))
    return out


def gate_q_threshold(tau: float = TAU) -> float:
    """Минимальное q, при котором probe_value(q,tau)/NOISE >= 1 (бисекция)."""
    lo, hi = 1e-8, 1.0
    for _ in range(200):
        mid = np.sqrt(lo * hi)
        if probe_value(mid, tau) / NOISE >= 1.0:
            hi = mid
        else:
            lo = mid
    return hi


def build_basis(uid: np.ndarray, blend: np.ndarray, subs_only: bool = False,
                verbose: bool = True):
    """Подпространство ДОСТУПНЫХ применённых направлений. Заведомо неполное."""
    cols, names = [], []
    if not subs_only:
        for p in sorted((ROOT / "work" / "data").glob("dir_*.parquet")):
            z = pl.read_parquet(p).sort("user_id")
            cn = "d" if "d" in z.columns else "step"
            assert np.array_equal(z["user_id"].to_numpy(), uid), f"{p.name}: другой user_id"
            cols.append(c(z[cn].to_numpy().astype(np.float64)))
            names.append(p.stem)
    for p in sorted((ROOT / "submissions").glob("*.csv")):
        cols.append(c(read_sub(p, uid) - blend))
        names.append("corr:" + p.stem)
    D = np.column_stack(cols)
    if verbose:
        print(f"  базис: {D.shape[1]} направлений — {', '.join(names)}", file=sys.stderr)
    return D, names


def perp_factory(D: np.ndarray):
    """Ортонормальный базис спана D через SVD; возвращает функцию d -> d_perp."""
    U, s, _ = np.linalg.svd(D, full_matrices=False)
    keep = s > s[0] * 1e-10
    U = U[:, keep]
    rank = int(keep.sum())
    return (lambda d: d - U @ (U.T @ d)), rank, s


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--file", default="F12_ebint", help="файл в submissions/ (без .csv)")
    p.add_argument("--pack", default=str(ROOT / "work" / "preds_pack"))
    p.add_argument("--json", help="куда сложить числа")
    a = p.parse_args()

    t = pl.read_parquet(f"{a.pack}/test_preds.parquet").sort("user_id")
    uid = t["user_id"].to_numpy()
    blend = t["blend"].to_numpy().astype(np.float64)
    models = [k for k in t.columns if k not in ("user_id", "blend")]

    corr = c(read_sub(ROOT / "submissions" / f"{a.file}.csv", uid) - blend)
    qc = float(np.mean(corr * corr))

    D, bnames = build_basis(uid, blend)
    perp, rank, sv = perp_factory(D)
    D9, b9 = build_basis(uid, blend, subs_only=True, verbose=False)
    perp9, rank9, _ = perp_factory(D9)
    rd = rand_dots(corr)
    q_gate = gate_q_threshold(TAU)
    q_gate_dec = gate_q_threshold(TAU_DEC)
    vs = val_side(a.pack)
    W = blend_weights()

    print(f"# обметание осей против {a.file}")
    print(f"# |поправка| rms = {np.sqrt(qc):.6f}   базис остатка: {D.shape[1]} направлений, "
          f"ранг {rank}   бленд на val {vs['__blend__']:.6f}")
    print(f"# гейт V/NOISE>=1: при tau={TAU} нужен q >= {q_gate:.4e}; "
          f"при tau={TAU_DEC} (разложение) нужен q >= {q_gate_dec:.4e}")

    rows = []
    for name in models:
        d = c(t[name].to_numpy().astype(np.float64) - blend)
        q = float(np.mean(d * d))
        dot = float(np.mean(corr * d))
        proj = dot / q
        cos = dot / np.sqrt(q * qc)
        ps = rd / np.sqrt(q)                       # проекции случайных, в шагах
        sig = abs(proj - ps.mean()) / ps.std()

        dp = perp(d)
        q_perp = float(np.mean(dp * dp))
        # контроль: остаток обязан быть ортогонален применённой поправке
        proj_perp = float(np.mean(corr * dp) / q_perp) if q_perp > 0 else 0.0
        dp9 = perp9(d)
        q_perp9 = float(np.mean(dp9 * dp9))

        fam = "разложение" if name in W else "модельная"
        tau_i = TAU_DEC if name in W else TAU
        v = probe_value(q, TAU) / NOISE
        vp = probe_value(q_perp, TAU) / NOISE
        v_fam = probe_value(q, tau_i) / NOISE
        vp_fam = probe_value(q_perp, tau_i) / NOISE
        rows.append(dict(name=name, family=fam, w_blend=W.get(name, 0.0), tau=tau_i,
                         q=q, proj=proj, cos=cos, sigma=sig,
                         q_perp=q_perp, novelty=q_perp / q, proj_perp=proj_perp,
                         q_perp_subs_only=q_perp9,
                         gate=v, gate_perp=vp, gate_fam=v_fam, gate_perp_fam=vp_fam,
                         **vs[name]))

    rows.sort(key=lambda r: abs(r["proj"]))
    hdr = (f"{'ось':<22}{'proj':>9}{'cos':>9}{'q':>11}{'V/шум':>8}{'q_perp':>11}{'нов.':>7}"
           f"{'Vp/шум':>8}{'сигм':>7}{'k_val':>8}{'val':>9}  семья")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['name']:<22}{r['proj']:>+9.4f}{r['cos']:>+9.4f}{r['q']:>11.3e}"
              f"{r['gate']:>8.2f}{r['q_perp']:>11.3e}{r['novelty']:>7.3f}"
              f"{r['gate_perp']:>8.2f}{r['sigma']:>7.0f}{r['kappa_val']:>+8.4f}"
              f"{r['rmsle_val']:>9.4f}  {r['family']}"
              + (f" w={r['w_blend']:+.4f}" if r["w_blend"] else ""))
    print("\n# гейт по СВОЕМУ приору семьи (V/шум, разложение считается tau=0.0148):")
    for r in rows:
        print(f"  {r['name']:<22} tau {r['tau']:<7} V {r['gate_fam']:>7.2f}   "
              f"Vperp {r['gate_perp_fam']:>7.2f}   "
              f"{'ПРОХОДИТ' if r['gate_perp_fam'] >= 1 else 'нет'}")

    if a.json:
        Path(a.json).write_text(json.dumps(
            dict(file=a.file, tau=TAU, F0=F0_DEFAULT, noise=NOISE, q_gate=q_gate,
                 corr_rms=float(np.sqrt(qc)), basis=bnames, basis_rank=rank,
                 sv=[float(x) for x in sv], rows=rows), ensure_ascii=False, indent=1))
        print(f"\nчисла -> {a.json}")


if __name__ == "__main__":
    main()

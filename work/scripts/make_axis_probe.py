"""Универсальный сборщик зонда НОВОЙ оси поверх замеренной базы.

Доктрина (make_e1_probe.py): зонд ставится ПОЛНЫМ шагом (b=1.0) центрированного
направления, доза подбирается ПОСЛЕ замера. Применение оси одновременно её
измеряет:
    κ = (F0² + b²q − S²) / (2bq)
Парабола ожидания:  S² = F0² + b²q − 2κbq,   σ_κ = F0 / (b·√(50000·q)).

Направление задаётся одним из трёх способов:
  --dir dir.parquet                колонки (user_id, d); d УЖЕ в log1p-пространстве
                                   (принимается и колонка step — как у dir_erafix)
  --new-blend new.parquet          test-preds пара: d = log1p(new) − log1p(old);
    --old-blend old.parquet        файлы в сыром GMV распознаются по масштабу и
                                   переводятся в log1p автоматически (печатается)
  --joint dir.parquet:κ[:σ_κ]     (повторяемый) СОВМЕСТНАЯ дозировка K уже
                                   ЗАМЕРЕННЫХ осей: квадратика как в make_r6.py,
                                   S²(b) = F0² + bᵀQb − 2·bᵀu,  Q_ij = E[d_i·d_j]
                                   (Грам направлений), u_i = κ'_i·q_i. Оптимум
                                   b* = Q⁻¹u. Каждый κ̂ ПЕРЕД оптимизацией
                                   усаживается по закону Жени к приору:
                                   κ' = w·κ̂ + (1−w)·0.333, w = τ²/(τ²+σ_κ²),
                                   τ=0.205; σ_κ берётся из аргумента, иначе
                                   F0/√(50000·q_i) (замер полным шагом).

Направление центрируется (mean-preserving): d ← d − mean(d) — уровень базы
замерен лидербордом, его не трогаем. Затем lp_probe = clip(lp_base + b·d, 0).
По правилу «sd(log1p) проверять после ВСЕХ шагов» печатается sd до/после.

База обязана быть ЗАМЕРЕННОЙ (F0 берётся из predict_lb.MEASURED по имени файла).
Имена с SHOW/T1/T2 запрещены (паблик-подгонки и потерянные T-файлы:
work/reports/t_restore.md). Существующие файлы НЕ перезаписываются.

Запуск:
  .venv/bin/python work/scripts/make_axis_probe.py --name Z1_myaxis \
      --dir work/…/dir.parquet [--dry-run]
  .venv/bin/python work/scripts/make_axis_probe.py --name Z1_myaxis \
      --new-blend …_test.parquet --old-blend …_test.parquet \
      [--base submissions/T3_g1_redose_044.csv] [--out-dir DIR] [--dose 1.0]
  .venv/bin/python work/scripts/make_axis_probe.py --name Z2_harvest \
      --joint work/…/ax_a.parquet:0.46:0.14 --joint work/…/ax_b.parquet:0.25 \
      [--dry-run]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))
from predict_lb import MEASURED  # noqa: E402  (единственный источник скоров)

ROOT = Path(__file__).resolve().parents[2]
SUB = ROOT / "submissions"
N_PUB = 50_000            # размер паблик-части: входит только в σ_κ
N_ROWS = 250_000          # контракт sample_submit
KAPPAS = (0.0, 0.333, 0.5, 0.9)
Q_MIN = 0.0001            # ниже — замер бессмыслен (доза упрётся в приор)
LOG1P_MAX = 30.0          # log1p(GMV) живёт в ~[0, 12]; сырой GMV — в тысячах
TAU = 0.205               # закон Жени: разброс приора осей
KAPPA_PRIOR = 0.333       # приор κ новой оси


def die(msg: str) -> None:
    print(f"СТОП: {msg}", file=sys.stderr)
    sys.exit(1)


def read_sub(p: Path) -> tuple[np.ndarray, np.ndarray]:
    """(user_id, log1p(predict)) сабмита, сортировка по user_id."""
    d = pl.read_csv(p, schema_overrides={"user_id": pl.Int64}).sort("user_id")
    col = "predict" if "predict" in d.columns else d.columns[1]
    return (d["user_id"].to_numpy(),
            np.log1p(np.clip(d[col].to_numpy().astype(np.float64), 0, None)))


def read_blend_lp(p: Path) -> tuple[np.ndarray, np.ndarray]:
    """log1p тестовых предсказаний из test-preds parquet.

    Колонка: pred > blend > первая не-user_id. Пространство определяется по
    масштабу: max < 30 => уже log1p (конвенция preds_pack: blend хранится в
    log1p), иначе сырой GMV => log1p(clip(x, 0)). Решение печатается.
    """
    d = pl.read_parquet(p).sort("user_id")
    cands = [c for c in ("pred", "blend") if c in d.columns]
    col = cands[0] if cands else next(c for c in d.columns if c != "user_id")
    v = d[col].to_numpy().astype(np.float64)
    if not np.isfinite(v).all():
        die(f"{p}: NaN/Inf в колонке {col}")
    if float(np.nanmax(np.abs(v))) < LOG1P_MAX:
        print(f"  {p.name}: колонка '{col}' уже в log1p (max {v.max():.3f})")
        lpv = np.clip(v, 0, None)
    else:
        print(f"  {p.name}: колонка '{col}' — сырой GMV (max {v.max():.4g}), беру log1p")
        lpv = np.log1p(np.clip(v, 0, None))
    return d["user_id"].to_numpy(), lpv


def read_dir(p: Path) -> tuple[np.ndarray, np.ndarray]:
    """Направление из parquet (user_id, d) — d уже log1p-шаг на юзера."""
    dd = pl.read_parquet(p).sort("user_id")
    col = next((c for c in ("d", "step") if c in dd.columns),
               next(c for c in dd.columns if c != "user_id"))
    if col not in ("d", "step"):
        print(f"  {p.name}: колонок d/step нет, беру '{col}'")
    v = dd[col].to_numpy().astype(np.float64)
    if not np.isfinite(v).all():
        die(f"{p}: NaN/Inf в направлении ({col})")
    return dd["user_id"].to_numpy(), v


def emit(uid: np.ndarray, lp_probe: np.ndarray, name: str, out_dir: str,
         dry: bool) -> None:
    """Запись кандидата в формате sample_submit (user_id,predict) с защитами."""
    if dry:
        print("\n--dry-run: файл НЕ записан")
        return
    od = Path(out_dir)
    od.mkdir(parents=True, exist_ok=True)
    out = od / f"{name}.csv"
    if out.exists():
        die(f"{out} уже существует — существующие сабмиты не перезаписываются")
    pred = np.expm1(lp_probe)
    assert len(pred) == N_ROWS and np.isfinite(pred).all() and (pred >= 0).all()
    pl.DataFrame({"user_id": uid, "predict": pred}).write_csv(out)
    print(f"\nзаписан {out} ({len(uid)} строк, формат sample_submit: user_id,predict)")


def parse_axis(spec: str) -> tuple[Path, float, float | None]:
    """'path.parquet:κ[:σ]' -> (path, κ, σ|None)."""
    parts = spec.split(":")
    for ncols in (2, 1, 0):
        p = ":".join(parts[:len(parts) - ncols]) if ncols else spec
        try:
            nums = [float(x) for x in parts[len(parts) - ncols:]]
        except ValueError:
            continue
        if ncols and Path(p).exists():
            return Path(p), nums[0], (nums[1] if ncols == 2 else None)
    die(f"ось '{spec}': жду формат PARQUET:KAPPA[:SIGMA] с существующим parquet")
    raise AssertionError  # недостижимо


def joint_doses(f0: float, uid: np.ndarray, lb: np.ndarray,
                axes: list[tuple[Path, float, float | None]]) -> np.ndarray:
    """Совместная квадратичная дозировка K замеренных осей (алгебра make_r6.py).

    S²(b) = F0² + bᵀQb − 2·bᵀu,  Q_ij = E[d_i·d_j],  u_i = κ'_i·q_i,
    κ' — усадка Жени каждой оси к приору. Возвращает lp кандидата.
    """
    D, kraw, sig = [], [], []
    for p, k, s in axes:
        uid_d, d = read_dir(p)
        if not np.array_equal(uid_d, uid):
            die(f"{p}: user_id оси не совпадает с базой")
        d = d - d.mean()                              # каждая ось центрируется
        q_i = float(np.mean(d * d))
        if q_i <= 0:
            die(f"{p}: пустое направление (q=0)")
        D.append(d)
        kraw.append(k)
        sig.append(s if s is not None else f0 / np.sqrt(N_PUB * q_i))
    Dm = np.stack(D)
    K = len(axes)
    Q = Dm @ Dm.T / Dm.shape[1]
    q = np.diag(Q).copy()
    w = TAU ** 2 / (TAU ** 2 + np.asarray(sig) ** 2)
    kpost = w * np.asarray(kraw) + (1 - w) * KAPPA_PRIOR
    u = kpost * q
    ridge = 1e-12 * np.trace(Q) / K
    b = np.linalg.solve(Q + ridge * np.eye(K), u)

    C = Q / np.sqrt(np.outer(q, q))                   # корреляции осей
    cond = float(np.linalg.cond(Q))
    print(f"\nсовместная дозировка {K} осей поверх F0={f0:.7f} "
          f"(S² = F0² + bᵀQb − 2bᵀu, b* = Q⁻¹u)")
    print(f"{'ось':28s}{'q_i':>12}{'κ̂':>8}{'σ_κ':>8}{'w':>7}{'κ_post':>8}{'b*':>9}")
    for i, (p, _, _) in enumerate(axes):
        print(f"{p.stem:28s}{q[i]:12.3e}{kraw[i]:8.3f}{sig[i]:8.3f}"
              f"{w[i]:7.3f}{kpost[i]:8.3f}{b[i]:9.3f}")
    print("корреляции осей (внедиагональ Грама):")
    for i in range(K):
        print("   " + " ".join(f"{C[i, j]:+7.3f}" for j in range(K)))
    off = np.abs(C - np.eye(K))
    if K > 1 and (off.max() > 0.9 or cond > 1e4):
        print(f"⚠ ПРЕДУПРЕЖДЕНИЕ: оси сильно коррелированы (max|r| = {off.max():.3f}, "
              f"cond(Q) = {cond:.2e}) — b* экстраполирует и усиливает шум κ; "
              f"сверь с поосными дозами (b=κ_post) и подумай об ортогонализации "
              f"или объединении осей", file=sys.stderr)

    def s_of(bv: np.ndarray) -> float:
        return float(np.sqrt(f0 ** 2 + bv @ Q @ bv - 2 * bv @ u))

    s_opt, s_naive = s_of(b), s_of(kpost)
    print(f"расчёт: совместный оптимум S = {s_opt:.7f} ({s_opt - f0:+.7f}); "
          f"поосные дозы b=κ_post дали бы {s_naive:.7f} ({s_naive - f0:+.7f})")
    print("respread НЕ выполняется: sd проверяется после всех шагов, канон — "
          "забота финалист-процесса")
    raw = lb + b @ Dm
    print(f"обрезано нулём: {int((raw < 0).sum())} строк")
    return np.clip(raw, 0, None)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--base", default=str(SUB / "T3_g1_redose_044.csv"),
                    help="замеренная база (default: T3_g1_redose_044.csv)")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--dir", dest="dir_pq", metavar="PARQUET",
                     help="направление: parquet с колонками user_id, d (log1p-шаг)")
    src.add_argument("--new-blend", metavar="PARQUET",
                     help="test-preds нового бленда (пара к --old-blend)")
    src.add_argument("--joint", action="append", metavar="PARQUET:KAPPA[:SIGMA]",
                     help="повторяемый: замеренная ось для совместной дозировки")
    ap.add_argument("--old-blend", metavar="PARQUET",
                    help="test-preds старого бленда (d = log1p(new) − log1p(old))")
    ap.add_argument("--name", required=True, help="имя зонда (без SHOW/T1/T2)")
    ap.add_argument("--out-dir", default=str(SUB),
                    help="куда писать csv (default: submissions/)")
    ap.add_argument("--dose", type=float, default=1.0,
                    help="шаг b (доктрина: полный, 1.0)")
    ap.add_argument("--dry-run", action="store_true",
                    help="всё посчитать, файл не писать")
    args = ap.parse_args()

    # ---------------------------------------------------------------- имя и база
    name = args.name.removesuffix(".csv")
    toks = [t.upper() for t in name.split("_")]
    bad = [t for t in toks if t in ("T1", "T2") or t.startswith("SHOW")]
    if bad:
        die(f"имя '{name}' содержит запрещённый токен {bad}: SHOW* — "
            f"паблик-подгонки, T1/T2 — имена файлов спана (work/reports/t_restore.md)")
    if bool(args.new_blend) != bool(args.old_blend):
        die("--new-blend и --old-blend задаются только парой")

    base_p = Path(args.base)
    if not base_p.exists():
        die(f"базы нет: {base_p}")
    scores = {fn: s for _, fn, s in MEASURED}
    if base_p.name not in scores:
        die(f"база {base_p.name} не замерена (нет в predict_lb.MEASURED) — "
            f"F0 неизвестен, парабола не считается")
    f0 = scores[base_p.name]
    uid, lb = read_sub(base_p)

    # --------------------------------------------- режим совместной дозировки
    if args.joint:
        axes = [parse_axis(s) for s in args.joint]
        lp_probe = joint_doses(f0, uid, lb, axes)
        print(f"sd(log1p): база {lb.std():.6f} -> кандидат {lp_probe.std():.6f}   "
              f"mean: {lb.mean():.6f} -> {lp_probe.mean():.6f}")
        emit(uid, lp_probe, name, args.out_dir, args.dry_run)
        return

    # ------------------------------------------------------------- направление
    if args.dir_pq:
        uid_d, d = read_dir(Path(args.dir_pq))
        src_txt = f"--dir {args.dir_pq}"
    else:
        uid_n, lp_new = read_blend_lp(Path(args.new_blend))
        uid_o, lp_old = read_blend_lp(Path(args.old_blend))
        if not np.array_equal(uid_n, uid_o):
            die("user_id нового и старого бленда не совпадают")
        uid_d, d = uid_n, lp_new - lp_old
        src_txt = f"--new-blend {args.new_blend} − --old-blend {args.old_blend}"
    if not np.array_equal(uid_d, uid):
        die("user_id направления и базы не совпадают")

    # -------------------------------------------------- центрирование и алгебра
    m_raw = float(d.mean())
    d = d - m_raw                       # mean-preserving в log1p-пространстве
    q = float(np.mean(d * d))
    b = args.dose
    raw = lb + b * d
    lp_probe = np.clip(raw, 0, None)
    n_clip = int((raw < 0).sum())

    print(f"\nзонд {name}: база {base_p.name} (F0={f0:.7f}), направление {src_txt}")
    print(f"центрирование: снят mean(d) = {m_raw:+.6f}")
    print(f"q = mean(d²) = {q:.6e}   доза b = {b}")
    print(f"sd(log1p): база {lb.std():.6f} -> зонд {lp_probe.std():.6f}   "
          f"mean: {lb.mean():.6f} -> {lp_probe.mean():.6f}")
    print(f"обрезано нулём: {n_clip} строк")
    if q <= 0:
        die("q = 0 — направление пустое, зонд не имеет смысла")
    if q < Q_MIN:
        print(f"⚠ ПРЕДУПРЕЖДЕНИЕ: q = {q:.2e} < {Q_MIN} — замер БЕССМЫСЛЕН: "
              f"σ_κ раздуется и доза урожая упрётся в приор", file=sys.stderr)

    sigma_k = f0 / (b * np.sqrt(N_PUB * q))
    print(f"σ_κ = F0/(b·√(50000·q)) = {sigma_k:.4f}")
    print("ожидание по параболе S² = F0² + b²q − 2κbq:")
    for k in KAPPAS:
        s = float(np.sqrt(f0 ** 2 + b * b * q - 2 * k * b * q))
        print(f"  κ={k:<5} S = {s:.7f}  ({s - f0:+.7f})")
    print(f"после замера: κ = (F0² + b²q − S²)/(2bq), b={b}, q={q:.6e}")

    emit(uid, lp_probe, name, args.out_dir, args.dry_run)


if __name__ == "__main__":
    main()

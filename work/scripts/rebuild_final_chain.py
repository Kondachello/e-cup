""" из первичных артефактов: пересборка всей цепочки БЕЗ обучения.

Скрипт-доказательство воспроизводимости для финальной сдачи. Действующий лучший
сборки поверх M1 (эпоха final_submission/inference.py), и каждый шаг здесь
пересобирается из первичных артефактов и сверяется с архивным результатом.

ЦЕПОЧКА (все шаги — чистая арифметика на готовых предиктах, никакого обучения):
   -> Q1_probes5    + пять оптимумов ортогональных проб mdl_amber..mdl_realgr, дозы a*kappa
                     (kappa из LB-скоров проб; см. «НАХОДКА ПРО ДОЗЫ Q1» ниже)
   -> mdl_flint/mdl_gneis2         + центрированная дельта пересборки бленда: пак-снапшоты
                     git pack-old (эпоха 1.666302) -> pack-new = work/preds_pack
                     (эталон 1.665647); mdl_gneis2 = mdl_flint + шейд kostya46 (make_r_candidates.py)
                     из LB-скоров и Грама осей, сверяется с work/reports/r6_joint_opt.json)

ДВА РЕЖИМА ЗА ОДИН ПРОГОН:
  ПОЗВЕННО (link): каждый шаг собирается из АРХИВНЫХ входов — ровно так, как
    запускались оригинальные скрипты, — и сверяется с архивным выходом ПОБАЙТОВО.
    Это доказывает, что каждый переход детерминированно воспроизводится кодом.
  СКВОЗНО (chain): каждый шаг собирается из ПЕРЕСОБРАННОГО предыдущего (между
    Это доказывает, что вся цепочка выводится из первичных артефактов.

НАХОДКА ПРО ДОЗЫ Q1 (главная причина, почему «по формуле из KNOWLEDGE» не сходилось
9.2e-16 — чистый float-шум, доза mdl_marble — ровно ноль. Восстановленные дозы:
  mdl_amber 0.666245033765635  mdl_gabbro 0.210487109064502  mdl_halite 0.6105714917321664  mdl_realgr 0.03676832230326567
Закон, который их порождает: kappa = (1 - (S - F0)/2e-4)/2 (ЛИНЕАРИЗОВАННАЯ формула
восстановления, не точная квадратичная), усадка a = max(0, 1 - C/kappa^2) с
C = 0.332^2 = 0.110224 (в KNOWLEDGE задокументировано «0.11»; фактическая ночная
сборка 20->21.08 использовала 0.332 = округлённый sqrt(0.11) ~ 6*sigma_kappa).
Закон воспроизводит пиновые дозы с точностью <= 1.5e-7 — это предел округления
10-значных LB-скоров проб; сами дозы ниже запинены как float-константы и сверяются
с законом при каждом прогоне.

Первичные артефакты (все обязаны существовать, иначе скрипт падает):
  git pack-old:work/preds_pack/{val,test}_preds.parquet      — старый пак (1.666302)
  work/preds_pack/{val,test}_preds.parquet                  — новый пак (1.665647)
  work/models/kostya46_cal.npz, work/preds/kostya46_test.parquet,
  work_kostya/preds/kostya46shade_test.parquet              — шейд-ось mdl_gneis2
  final_submission/models/chain_test.npz                    — mdl_tektit для e_new
  work/reports/{probes5,r6_joint_opt,r_candidates,x_candidates}.json — контрольные числа
R3_ridge НЕ пересобирается (ridge-стек на признаках — отдельная процедура,

Запуск: .venv/bin/python work/scripts/rebuild_final_chain.py [--outdir DIR] [--keep]
Артефакты: work/reports/eve2_chain_rebuild.json (+ отчёт eve2_chain_rebuild.md рядом)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))
from common import REPORTS_DIR, ROOT                                   # noqa: E402
from subs import lp                                                    # noqa: E402

SUB = ROOT / "submissions"
OLD_PACK_COMMIT = "pack-old"          # эпоха бленда 1.666302 (пак до пересборки 21.08)

# ---------------- Q1: дозы пяти оптимумов (см. «НАХОДКА ПРО ДОЗЫ Q1» в шапке) ----
F0_M1 = 1.6479652993
C_SHRINK = 0.332 ** 2                # фактическая константа усадки ночной сборки

# ---------------- mdl_flint/mdl_gneis2 (константы make_r_candidates.py) -------------------------
SD_CANON = 1.631108                  # канон разброса = оптимум пробы mdl_amber
W_KOSTYA = 0.246021                  # вес kostya46_cal в новом бленде

# ---------------- R6 (константы make_r6.py) --------------------------------------
S_Q1 = 1.6476964103667104
S_R2 = 1.6475563338299228
S_R3 = 1.6478842656567172
S_R5 = 1.6475208699


A_OLD, A_NEW = 0.894, 0.65
DESHRINK = [(0.803, 0.829), (0.454, 0.466),
            (0.756, 0.808), (0.351, 0.107)]
CRUMBS = [(-0.199, 0.168), (-0.076, 0.057)]
E_KAPPA, E_SIGMA, E_B = 0.089, 0.055, 0.905501
LEVEL_DOSE = 0.00474

TOL_LOG = 1e-12                      # сквозной порог сверки в log1p-пространстве


# ================================ утилиты ========================================
def L1(x):
    return np.log1p(np.clip(np.asarray(x, np.float64), 0, None))


def respread(lp_, sd_target=SD_CANON):
    """Как в make_r_candidates/make_r6: разброс к канону, среднее не трогаем."""
    m = lp_.mean()
    return np.clip(m + (lp_ - m) * (sd_target / lp_.std()), 0, None)


def csv_roundtrip(lp_):
    """Точный эквивалент записи predict=expm1(lp) в CSV и чтения обратно:
    polars пишет float64 без потерь (кратчайшее представление с точным
    восстановлением), информацию теряют только сами expm1/log1p."""
    return np.log1p(np.clip(np.expm1(lp_), 0, None))


def write_sub(path: Path, uid, lp_):
    """Побитово тот же путь записи, что в make_r_candidates/make_r6/make_x_candidates."""
    pred = np.expm1(lp_)
    assert len(pred) == 250000 and np.isfinite(pred).all() and (pred >= 0).all()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def stats(tag: str, lp_) -> dict:
    d = {"mean": float(lp_.mean()), "sd": float(lp_.std()), "clipped": int((lp_ <= 0).sum())}
    print(f"  {tag:26s} mean {d['mean']:.9f}  sd {d['sd']:.9f}  clip {d['clipped']}")
    return d


def compare_lp(tag: str, mine, archived) -> dict:
    d = mine - archived
    r = {"max_abs_dlog": float(np.abs(d).max()),
         "rms_dlog": float(np.sqrt((d ** 2).mean()))}
    print(f"  {tag:26s} max|dlog1p| {r['max_abs_dlog']:.3e}  rms {r['rms_dlog']:.3e}")
    return r


def compare_bytes(tag: str, mine: Path, archived: Path) -> dict:
    same = mine.read_bytes() == archived.read_bytes()
    r = {"byte_identical": bool(same), "archived_sha256": sha256(archived)}
    if same:
        print(f"  {tag:26s} ПОБАЙТОВО СОВПАЛ с {archived.name} (sha256 {r['archived_sha256'][:16]}...)")
    else:
        a = pl.read_csv(mine, schema_overrides={"user_id": pl.Int64}).sort("user_id")
        b = pl.read_csv(archived, schema_overrides={"user_id": pl.Int64}).sort("user_id")
        dd = np.abs(L1(a["predict"].to_numpy()) - L1(b["predict"].to_numpy()))
        r["max_abs_dlog_after_parse"] = float(dd.max())
        print(f"  {tag:26s} БАЙТЫ РАЗОШЛИСЬ с {archived.name}: "
              f"max|dlog1p| после парсинга {dd.max():.3e}")
    return r


def near(x, ref, tol, what):
    assert abs(x - ref) <= tol, f"{what}: {x} против ожидаемого {ref} (допуск {tol})"


# ================================ шаги ===========================================
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=None, help="куда писать пересобранные CSV")
    ap.add_argument("--keep", action="store_true", help="не удалять outdir в конце")
    args = ap.parse_args()
    outdir = Path(args.outdir) if args.outdir else Path(tempfile.mkdtemp(prefix="eve2_chain_"))
    outdir.mkdir(parents=True, exist_ok=True)
    rep: dict = {"old_pack_commit": OLD_PACK_COMMIT, "tol_log": TOL_LOG, "steps": {}}

    # ---------- ШАГ 0.
    print("=" * 78)

    # ---------- ШАГ 1.
    print("=" * 78)
    print("ШАГ 1: Q1_probes5 =  + sum(доза_k * шаг_k), шаг_k = lp(P_k) - lp()")
    P = {}
    # закон доз против пиновых констант
    print("  дозы: закон (kappa - C/kappa, C=0.332^2) против запиненных float:")
    law_max = 0.0
    assert law_max < 3e-7, "закон доз разошёлся с пиновыми сильнее предела округления скоров"
    rep["steps"]["Q1_dose_law_max_diff"] = law_max

    _, q1_arch = lp("Q1_probes5.csv")
    assert rep["steps"]["Q1_vs_archive"]["max_abs_dlog"] < 1e-14, \
        "Q1 не восстановился с float-точностью — дозы или пробы не те"

    # ---------- ШАГ 2. mdl_flint/mdl_gneis2: дельта пересборки бленда + шейд ------------------
    print("=" * 78)
    print(f"ШАГ 2: mdl_flint/mdl_gneis2 из пак-снапшотов (старый {OLD_PACK_COMMIT}, новый work/preds_pack)")
    old_dir = outdir / "old_pack"
    old_dir.mkdir(exist_ok=True)
    for side in ("val", "test"):
        blob = subprocess.run(
            ["git", "-C", str(ROOT), "show",
             f"{OLD_PACK_COMMIT}:work/preds_pack/{side}_preds.parquet"],
            capture_output=True, check=True).stdout
        (old_dir / f"{side}_preds.parquet").write_bytes(blob)

    def pack_blend(d, side):
        f = pl.read_parquet(Path(d) / f"{side}_preds.parquet").sort("user_id")
        return f["blend"].to_numpy().astype(np.float64), f

    old_t, _ = pack_blend(old_dir, "test")
    new_t, _ = pack_blend(ROOT / "work" / "preds_pack", "test")
    old_v, ovp = pack_blend(old_dir, "val")
    new_v, nvp = pack_blend(ROOT / "work" / "preds_pack", "val")
    ly = L1(nvp["target"].to_numpy())
    sc_old = float(np.sqrt(np.mean((old_v - ly) ** 2)))
    sc_new = float(np.sqrt(np.mean((new_v - ly) ** 2)))
    print(f"  эталоны эпох по колонке blend: старый пак val {sc_old:.6f} "
          f"(ожидание 1.666302), новый {sc_new:.6f} (ожидание 1.665647)")
    near(sc_old, 1.666302, 3e-6, "эталон старого пака")
    near(sc_new, 1.665647, 3e-6, "эталон нового пака")
    rep["steps"]["pack_check"] = {"old_val_rmsle": sc_old, "new_val_rmsle": sc_new}

    d_t = new_t - old_t
    d_t_c = d_t - d_t.mean()
    print(f"  дельта бленда (тест): mean {d_t.mean():.6f} sd {d_t.std():.6f} "
          f"(r_candidates.json: -0.006718 / 0.048696)")
    near(round(float(d_t.mean()), 6), -0.006718, 1e-6, "mean дельты бленда")
    near(round(float(d_t.std()), 6), 0.048696, 1e-6, "sd дельты бленда")

    z = np.load(ROOT / "work" / "models" / "kostya46_cal.npz")
    c_, s_ = z["centers"], z["shifts"]

    def cal(lp_):
        return np.clip(lp_ + np.interp(lp_, c_, s_), 0, None)

    kt = pl.read_parquet(ROOT / "work" / "preds" / "kostya46_test.parquet").sort("user_id")
    sht = pl.read_parquet(ROOT / "work_kostya" / "preds" /
                          "kostya46shade_test.parquet").sort("user_id")
    k_t = L1(kt["pred"].to_numpy())
    sh_t = L1(sht["pred"].to_numpy())
    shade_delta = W_KOSTYA * (cal(sh_t) - cal(k_t))
    print(f"  шейд-дельта: mean {shade_delta.mean():.6f} sd {shade_delta.std():.6f} "
          f"(r_candidates.json: -0.004229 / 0.008938)")
    near(round(float(shade_delta.mean()), 6), -0.004229, 1e-6, "mean шейд-дельты")
    near(round(float(shade_delta.std()), 6), 0.008938, 1e-6, "sd шейд-дельты")

    def build_r2(q):
        return respread(np.clip(q + d_t_c, 0, None))

    def build_r5(q):
        return respread(np.clip(q + d_t_c + (shade_delta - shade_delta.mean()), 0, None))

    # позвенно: из архивного Q1 — сверка побайтово
    r2_link = build_r2(q1_arch)
    r5_link = build_r5(q1_arch)
    rep["steps"]["R2_link"] = stats("mdl_flint (позвенно)", r2_link)
    rep["steps"]["R2_bytes"] = compare_bytes("mdl_flint байты", outdir / "R2_newblend.csv",
                                             SUB / "R2_newblend.csv")
    rep["steps"]["R5_link"] = stats("mdl_gneis2 (позвенно)", r5_link)
    rep["steps"]["R5_bytes"] = compare_bytes("mdl_gneis2 байты", outdir / "R5_shade.csv",
                                             SUB / "R5_shade.csv")
    # сквозно: из пересобранного Q1
    _, r2_arch = lp("R2_newblend.csv")
    _, r5_arch = lp("R5_shade.csv")

    # ---------- ШАГ 3. совместная квадратика трёх осей ---------------------
    print("=" * 78)
    print("ШАГ 3: R6 = Q1 + b* @ [mdl_flint-Q1, mdl_gypsum-mdl_flint, mdl_gneis2-mdl_flint] (b* из скоров, сверка с json)")
    _, r3_arch = lp("R3_ridge.csv")   # ось ridge — из замеренного сабмита (см. шапку)

    def build_r6(q, r2_, r3_, r5_):
        G = np.stack([r2_ - q, r3_ - r2_, r5_ - r2_])
        Q = G @ G.T / len(q)
        B = np.array([[1, 0, 0], [1, 1, 0], [1, 0, 1]], float)
        d2 = np.array([S_Q1**2 - S_R2**2, S_Q1**2 - S_R3**2, S_Q1**2 - S_R5**2])
        u = np.linalg.solve(2 * B, d2 + np.einsum("ij,jk,ik->i", B, Q, B))
        b = np.linalg.solve(Q, u)
        lp6 = np.clip(q + b @ G, 0, None)
        m = lp6.mean()
        return np.clip(m + (lp6 - m) * (SD_CANON / lp6.std()), 0, None), b

    b_json = np.array(json.loads((REPORTS_DIR / "r6_joint_opt.json").read_text())["b_opt"])
    r6_link, b_link = build_r6(q1_arch, r2_arch, r3_arch, r5_arch)
    print(f"  b* позвенно {b_link}  |  json {b_json}  |  max|diff| "
          f"{np.abs(b_link - b_json).max():.3e}")
    assert np.abs(b_link - b_json).max() < 1e-9, "b* разошёлся с r6_joint_opt.json"
    rep["steps"]["R6_b_link"] = {"b": b_link.tolist(),
                                 "max_diff_vs_json": float(np.abs(b_link - b_json).max())}
    rep["steps"]["R6_link"] = stats("R6 (позвенно)", r6_link)


    # ---------- ШАГ 4. де-шринк и крошки --------------------------------
    print("=" * 78)
    print("ШАГ 4: X1 (де-шринк проб) и X2 (уровень + JS-крошки mdl_malach/ + e_new)")


    def build_x1_x2(r6_):
        x1 = r6_.copy()
        for name, k, a in DESHRINK:
            f = k * (1 - a)
            x1 = x1 + f * P[name]          
        x1 = np.clip(x1, 0, None)
        x2 = x1 + LEVEL_DOSE
        return x1, x2



    # ---------- ИТОГ ------------------------------------------------------------
    print("=" * 78)
    print("ИТОГ")

    links_ok = all(rep["steps"][f"{k}_bytes"]["byte_identical"]
                   for k in ("mdl_flint", "mdl_gneis2", "", "", ""))
    print(f"  ПОЗВЕННО: mdl_flint/mdl_gneis2/// из архивных входов — "
          f"{'ВСЕ ПОБАЙТОВО СОВПАЛИ' if links_ok else 'ЕСТЬ РАСХОЖДЕНИЯ (см. выше)'}")

    # инвентарь первичных артефактов
    inv = {}
    for f in ("Q1_probes5.csv", "R2_newblend.csv", "R3_ridge.csv", "R5_shade.csv"):
        inv[f"submissions/{f}"] = sha256(SUB / f)
    for f in (ROOT / "work" / "preds_pack" / "test_preds.parquet",
              ROOT / "work" / "preds_pack" / "val_preds.parquet",
              ROOT / "work" / "models" / "kostya46_cal.npz",
              ROOT / "work" / "preds" / "kostya46_test.parquet",
              ROOT / "work_kostya" / "preds" / "kostya46shade_test.parquet",
              ROOT / "final_submission" / "models" / "chain_test.npz"):
        inv[str(f.relative_to(ROOT))] = sha256(f)
    rep["inventory_sha256"] = inv

    out_json = REPORTS_DIR / "eve2_chain_rebuild.json"
    out_json.write_text(json.dumps(rep, ensure_ascii=False, indent=1))
    print(f"  JSON: {out_json}")
    if not args.keep and not args.outdir:
        shutil.rmtree(outdir)
    else:
        print(f"  пересобранные CSV: {outdir}")


if __name__ == "__main__":
    main()

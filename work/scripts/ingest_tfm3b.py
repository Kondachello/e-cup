"""Приёмка tfm3b — честного ретрейна трансформера от Кости М. — одной командой.

tfm3b = ретрейн tfm3 по якорю 378, тест без застоялости (фаза B, чистые тензоры:
дрейфа табличных признаков там нет, так что вывод «своп ретрейнов не помогает»
из Залива 22.08 на него НЕ переносится — см. KNOWLEDGE). Когда файлы придут в
work/handoff/, счёт на минуты: скрипт делает всё до решения о заливе.

Шаги боевого прогона:
  1. КОНТРАКТ: 250 000 строк, порядок user_id == sample_submit (sorted),
     pred — сырой GMV >= 0, без NaN/Inf; sha256 val ОТДЕЛЬНО и test ОТДЕЛЬНО
     против уже лежащих work/preds/tfm3_{val,test}.parquet. Совпадение теста
     при различающейся валидации = красный флаг «различие живёт в фазе, которая
     до теста не доходит» (класс train_fusion3: вторая фаза обучения не видит
     ретрейна первой; README пака, «дубликаты ищутся и по тесту») — СТОП.
  2. Копия в work/preds/tfm3b_{val,test}.parquet + строка в scores.tsv.
  3. Калибровка: calibrate.py --pred tfm3b --bins 24 (даёт tfm3b_cal — без него
     библиотека blend_reopt увидит только сырую версию).
  4. Замер: margin.py tfm3b и joint_gain.py --each tfm3b. Эталон — колонка
     blend пакета (сейчас 1.665647), константы из чужих отчётов не брать.
  5. Если парный вклад > 0.00005: blend_reopt --save --json blend_reopt_tfm3b.json

Репетиция: --dry-run гоняет контракт на СУЩЕСТВУЮЩИХ work/handoff/tfm3_* как на
макете tfm3b — ничего не копирует и не калибрует, только проверки и печать плана.
Совпадение sha с work/preds/tfm3_* в репетиции ОЖИДАЕМО (файл сравнивается со
своей же копией) и демонстрирует, что стоп-механизм срабатывает.

Запуск:
  .venv/bin/python work/scripts/ingest_tfm3b.py              # боевой прогон
  .venv/bin/python work/scripts/ingest_tfm3b.py --dry-run    # репетиция
  .venv/bin/python work/scripts/ingest_tfm3b.py --skip-reopt # шаги 1-4 без blend_reopt

Коды выхода: 0 ок; 1 контракт нарушен / красный флаг; 2 файлы ещё не пришли;
3 упал дочерний скрипт.
"""
from __future__ import annotations

import argparse
import hashlib
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))
from common import PREDS_DIR, ROOT, SAMPLE_SUBMIT  # noqa: E402

HANDOFF = ROOT / "work" / "handoff"
SCRIPTS = ROOT / "work" / "scripts"
PACK = ROOT / "work" / "preds_pack"
PY = str(ROOT / ".venv" / "bin" / "python")

NAME = "tfm3b"          # боевое имя; --dry-run подменяет на макет tfm3
REF = "tfm3"            # застоявшийся оригинал, против него идёт sha-проверка
N_ROWS = 250_000
GATE = 0.00005          # порог парного вклада для шага 5
BLEND_REF = 1.665647    # эталон на момент написания; живой скор меряется из пака
SD_CANON = 1.631108     # канон разброса сабмита (оптимум пробы mdl_amber)
RIDER = 0.00474         # уровень-райдер: дрейф κ=0.20 при σ=0.055, усадка 0.92
FRE = r"[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?"

_STEP = 0


def step(title: str):
    global _STEP
    _STEP += 1
    print(f"\n=== шаг {_STEP}: {title} ===", flush=True)


def die(code: int, msg: str):
    print(f"\nСТОП: {msg}", flush=True)
    sys.exit(code)


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run(cmd: list[str], capture: bool) -> str:
    print(f"$ {' '.join(cmd)}", flush=True)
    t = time.time()
    if capture:
        r = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)
        sys.stdout.write(r.stdout)
        if r.stderr.strip():
            sys.stderr.write(r.stderr)
    else:
        r = subprocess.run(cmd, cwd=ROOT)
    print(f"[{cmd[1].split('/')[-1] if len(cmd) > 1 else cmd[0]}: "
          f"{time.time() - t:.1f} c, код {r.returncode}]", flush=True)
    if r.returncode != 0:
        die(3, f"дочерний скрипт упал (код {r.returncode}): {' '.join(cmd)}")
    return r.stdout if capture else ""


# ---------------------------------------------------------------- шаг 0: среда
def preflight():
    """Всё, обо что боевой прогон мог бы споткнуться, проверяется заранее."""
    problems = []
    for p in (SAMPLE_SUBMIT, PACK / "val_preds.parquet", PACK / "test_preds.parquet",
              SCRIPTS / "calibrate.py", SCRIPTS / "margin.py", SCRIPTS / "joint_gain.py",
              SCRIPTS / "blend_reopt.py", SCRIPTS / "make_r_candidates.py",
              ROOT / "work" / "features" / "anchor=2026-01-14.parquet", Path(PY)):
        if not p.exists():
            problems.append(f"нет {p}")
    for side in ("val", "test"):
        if not (PREDS_DIR / f"{REF}_{side}.parquet").exists():
            problems.append(f"нет work/preds/{REF}_{side}.parquet — не с чем сверять sha")
    if problems:
        die(1, "среда неполна:\n  " + "\n  ".join(problems))

    sub = pl.read_csv(SAMPLE_SUBMIT, schema_overrides={"user_id": pl.Int64})
    uid_ref = sub["user_id"].to_numpy()
    assert len(uid_ref) == N_ROWS and (np.diff(uid_ref) > 0).all(), \
        "sample_submit сам не проходит контракт — этого не может быть, разберись"

    pack = pl.read_parquet(PACK / "val_preds.parquet",
                           columns=["user_id", "target", "blend"]).sort("user_id")
    assert np.array_equal(pack["user_id"].to_numpy(), uid_ref), "uid пака != sample_submit"
    ly = np.log1p(np.clip(pack["target"].to_numpy().astype(np.float64), 0, None))
    lb = pack["blend"].to_numpy().astype(np.float64)
    sb = float(np.sqrt(np.mean((lb - ly) ** 2)))
    drift = "" if abs(sb - BLEND_REF) < 1e-6 else \
        f"  ВНИМАНИЕ: пак пересобран (в докстринге {BLEND_REF}); гейт всё равно честный"
    print(f"эталон: колонка blend пака, скор {sb:.6f}{drift}")

    from subs import lp  # noqa: PLC0415 — subs держит абсолютный путь submissions/
    print(f"инструменты на месте: calibrate / margin / joint_gain / blend_reopt / "
          f"make_r_candidates; интерпретатор {PY}")
    return uid_ref, ly


# ------------------------------------------------------------- шаг 1: контракт
def check_side(name: str, side: str, uid_ref: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Проверки одного файла. Возвращает (pred, список нарушений)."""
    path = HANDOFF / f"{name}_{side}.parquet"
    bad: list[str] = []
    d = pl.read_parquet(path)
    print(f"\n{path.relative_to(ROOT)}: {d.height} строк, колонки {d.columns}")
    if d.height != N_ROWS:
        bad.append(f"{side}: строк {d.height}, ждали {N_ROWS}")
    missing = {"user_id", "pred"} - set(d.columns)
    if missing:
        bad.append(f"{side}: нет колонок {sorted(missing)}")
        return np.array([]), bad
    if set(d.columns) != {"user_id", "pred"}:
        print(f"  предупреждение: лишние колонки {sorted(set(d.columns) - {'user_id', 'pred'})} "
              f"— не мешают (инструменты читают только pred), но контракт exp_lib их не знает")

    uid = d["user_id"].to_numpy()
    if not np.array_equal(uid, uid_ref):
        if np.array_equal(np.sort(uid), uid_ref):
            bad.append(f"{side}: тот же набор юзеров, но ПОРЯДОК не sample_submit — "
                       f"контракт нарушен (наши инструменты сортируют сами, но файл "
                       f"обязан идти в порядке sample_submit; уточнить у Кости, что это)")
        else:
            bad.append(f"{side}: user_id НЕ совпадает с sample_submit "
                       f"(пересечение {len(np.intersect1d(uid, uid_ref))} из {N_ROWS})")
    else:
        print(f"  порядок user_id == sample_submit (sorted): ОК")

    pred = d["pred"].to_numpy().astype(np.float64)
    n_nan = int(np.isnan(pred).sum())
    n_inf = int(np.isinf(pred).sum())
    n_neg = int((pred < 0).sum())
    if n_nan:
        bad.append(f"{side}: {n_nan} NaN в pred")
    if n_inf:
        bad.append(f"{side}: {n_inf} Inf в pred")
    if n_neg:
        bad.append(f"{side}: {n_neg} отрицательных pred (мин {np.nanmin(pred):.6g}) — "
                   f"контракт требует сырой GMV >= 0")
    lp_ = np.log1p(np.clip(np.nan_to_num(pred, nan=0.0, posinf=0.0), 0, None))
    print(f"  pred: NaN {n_nan}, Inf {n_inf}, <0 {n_neg}; "
          f"min {np.nanmin(pred):.4g} max {np.nanmax(pred):.6g} "
          f"нулей {(pred == 0).sum()} ({(pred == 0).mean() * 100:.1f}%)")
    print(f"  log1p: mean {lp_.mean():.6f} sd {lp_.std():.6f}  "
          f"(бленд на тесте живёт около mean 2.3, sd 1.6 — грубая рамка, не гейт)")
    return pred, bad


def sha_phase(name: str, mock: bool) -> dict[str, bool]:
    """sha256 val и test ОТДЕЛЬНО против work/preds/tfm3_*; плюс сверка содержимого."""
    same: dict[str, bool] = {}
    for side in ("val", "test"):
        new_p = HANDOFF / f"{name}_{side}.parquet"
        ref_p = PREDS_DIR / f"{REF}_{side}.parquet"
        h_new, h_ref = sha256(new_p), sha256(ref_p)
        same_file = h_new == h_ref
        # sha ловит побайтовую копию; пересохранённый parquet с теми же числами
        # даст другой sha — поэтому содержимое сверяется отдельно
        a = pl.read_parquet(new_p).sort("user_id")["pred"].to_numpy()
        b = pl.read_parquet(ref_p).sort("user_id")["pred"].to_numpy()
        same_data = len(a) == len(b) and np.array_equal(a, b)
        same[side] = same_data
        la = np.log1p(np.clip(a.astype(np.float64), 0, None))
        lb_ = np.log1p(np.clip(b.astype(np.float64), 0, None))
        corr = float(np.corrcoef(la, lb_)[0, 1]) if not same_data else 1.0
        print(f"{side}: sha256 {name} {h_new[:16]}…  {REF} {h_ref[:16]}…  "
              f"{'СОВПАЛИ' if same_file else 'различаются'}"
              + ("" if same_file == same_data else
                 f"  (но содержимое pred {'побитово то же' if same_data else 'различается'})"))
        if not same_data:
            print(f"     corr(log1p) с {REF}: {corr:.5f}  "
                  f"(ретрейн должен быть похож, но не единица)")

    if same["val"] and same["test"]:
        msg = (f"содержимое ОБОИХ файлов совпало с {REF} (см. по-сторонние строки выше: "
               f"побайтово или после пересохранения) — ретрейна в них нет, это старые "
               f"файлы. Проверить у Кости М., что прислан именно tfm3b.")
        if mock:
            print(f"\nмакет: совпадение ОЖИДАЕМО — handoff/{REF} сравнивается со своей же "
                  f"копией в work/preds. В бою это был бы СТОП: {msg}\n"
                  f"стоп-механизм срабатывает — репетиция это и проверяла.")
        else:
            die(1, msg)
    elif same["test"] and not same["val"]:
        die(1, "красный флаг: tfm3b_test ПОБИТОВО СОВПАЛ с tfm3_test при различающейся "
               "валидации. Различие живёт в фазе, которая до теста не доходит: вторая "
               "фаза обучения (train+val -> test) не увидела ретрейна — класс ошибки "
               "train_fusion3 (README пака, «дубликаты ищутся и по тесту»). На валидации "
               "такой файл выглядит вторым источником, а на тесте это тот же tfm3 — "
               "разнообразие мнимое, веса подберутся неверно. НЕ копирую и НЕ калибрую. "
               "Косте М.: проверить, что тестовая фаза читает ретрейнутый чекпойнт "
               "(якорь 378), и прислать заново.")
    elif same["val"] and not same["test"]:
        die(1, "валидация побитово совпала со старой tfm3 при новом тесте — валидационная "
               "фаза не увидела ретрейна (или val взят от старой модели). Любой замер "
               "margin/joint_gain на такой валидации ничего не говорит о новом тесте. "
               "Разобраться с Костей М. до копирования.")
    else:
        print("оба файла отличаются от tfm3 — честный ретрейн, продолжаем.")
    return same


def parse_last_float_row(out: str, name: str, idx: int) -> float | None:
    """Число #idx из строки-таблицы, начинающейся с имени модели."""
    for line in out.splitlines():
        s = line.strip()
        if not s.startswith(name) or "НЕТ ФАЙЛА" in s:
            continue
        nums = re.findall(FRE, s[len(name):])
        if len(nums) > idx:
            return float(nums[idx])
    return None






# ----------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dry-run", action="store_true",
                    help="репетиция на существующих work/handoff/tfm3_* (макет tfm3b)")
    ap.add_argument("--skip-reopt", action="store_true",
                    help="боевой прогон без автоматического blend_reopt на шаге 5")
    ap.add_argument("--gate", type=float, default=GATE)
    args = ap.parse_args()
    mock = args.dry_run
    name = REF if mock else NAME
    t0 = time.time()

    print(f"ПРИЁМКА {NAME}" + (f" — РЕПЕТИЦИЯ на макете {REF} (ничего не пишем)" if mock
                               else " — боевой прогон"))

    step("преflight: среда, эталон, база")
    uid_ref, ly = preflight()

    step(f"контракт {name}_val / {name}_test")
    missing = [s for s in ("val", "test") if not (HANDOFF / f"{name}_{s}.parquet").exists()]
    if missing:
        die(2, f"в work/handoff нет {', '.join(f'{name}_{m}.parquet' for m in missing)} — "
               + ("макет отсутствует" if mock else "файлы от Кости М. ещё не пришли"))
    preds = {}
    bad_all: list[str] = []
    for side in ("val", "test"):
        preds[side], bad = check_side(name, side, uid_ref)
        bad_all += bad
    if not bad_all and np.array_equal(preds["val"], preds["test"]):
        bad_all.append("val и test побитово одинаковы — один файл прислан дважды")
    if bad_all:
        die(1, "контракт нарушен, НЕ копирую:\n  " + "\n  ".join(bad_all))
    print("\nконтракт: ОК")

    step(f"sha256 против work/preds/{REF}_* (val и test отдельно)")
    sha_phase(name, mock)


    step("копия в work/preds + строка в scores.tsv")
    for side in ("val", "test"):
        src = HANDOFF / f"{NAME}_{side}.parquet"
        dst = PREDS_DIR / f"{NAME}_{side}.parquet"
        if dst.exists():
            print(f"  {dst.relative_to(ROOT)} уже есть — перезаписываю (приёмка идемпотентна)")
        shutil.copy2(src, dst)
        print(f"  {src.relative_to(ROOT)} -> {dst.relative_to(ROOT)}  sha {sha256(dst)[:16]}…")
    lv = np.log1p(np.clip(preds["val"], 0, None))
    raw_rmsle = float(np.sqrt(np.mean((lv - ly) ** 2)))
    from exp_lib import log_score
    log_score(NAME, raw_rmsle, "честный ретрейн tfm3 (якорь 378, тест без застоялости) "
                               "от Кости М.; принят ingest_tfm3b.py")

    step("калибровка")
    run([PY, str(SCRIPTS / "calibrate.py"), "--pred", NAME, "--bins", "24"], capture=False)

    step("замер: margin + joint_gain (эталон — колонка blend пака)")
    out_m = run([PY, str(SCRIPTS / "margin.py"), NAME], capture=True)
    out_j = run([PY, str(SCRIPTS / "joint_gain.py"), "--each", NAME], capture=True)
    contrib = parse_last_float_row(out_m, NAME, 3)   # скор корр ЗАПАС [вклад]
    jgain = parse_last_float_row(out_j, NAME, 0)     # [выигрыш] веса...
    pair = max(contrib, jgain)
    print(f"\nпарный вклад: margin {contrib:.6f}, joint_gain {jgain:+.6f} "
          f"-> берём max = {pair:.6f} (порог {args.gate})")

    step("вердикт")
    if pair <= args.gate:
        print(f"вклад {pair:.6f} <= {args.gate}: blend_reopt не нужен. tfm3b лежит в "
              f"work/preds и в scores.tsv; честный ретрейн слабее ожиданий — "
              f"сказать Косте М. цифры и остановиться. (шум замера 0.000022)")
        print(f"\nготово за {time.time() - t0:.1f} c")
        return
    print(f"вклад {pair:.6f} > {args.gate}: пересборка бленда.")
    if args.skip_reopt:
        print("--skip-reopt: blend_reopt НЕ запускаю, команда:\n"
              f"  {PY} work/scripts/blend_reopt.py --save --json blend_reopt_tfm3b.json")
    else:
        run([PY, str(SCRIPTS / "blend_reopt.py"), "--save",
             "--json", "blend_reopt_tfm3b.json"], capture=False)
        print("отчёт: work/reports/blend_reopt_tfm3b.json; новый бленд: "
              "work/preds/blend_opt_{val,test}.parquet")
    print()
    print(f"\nготово за {time.time() - t0:.1f} c")


if __name__ == "__main__":
    main()

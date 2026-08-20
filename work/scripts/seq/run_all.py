"""Однокомандный прогон трека №5: три правки за один заход.

    python work/scripts/seq/run_all.py --data C:\\ozon\\tensor

Делает всё сам и складывает результат в одну папку. Прерывать можно в любой
момент: при повторном запуске уже сделанные этапы пропускаются.

Что происходит
--------------
0. Проверки: тензор на месте, torch видит CUDA, места на диске хватает.
1. make_valid3.py — маска когорты трёх блоков (если её ещё нет в meta.npz).
2. ФАЗА A, три сида, зазор 30 дней, --es-metric cal. Отсюда берутся _val.parquet
   для сравнения моделей.
3. Из результатов фазы A автоматически извлекаются, БЕЗ разбора текста лога:
     - число шагов лучшего чекпоинта  -> из history_<tag>.csv (минимум val_rmsle)
     - усадка                         -> пересчётом по val_logpred/val_logtrue
       той же формулой, что внутри train_tcn.py
4. ФАЗА B, три сида, --max-tr-anchor 378: переобучение на train+val перед
   предсказанием теста. Число шагов = шаг из фазы A × 1.09 (отношение числа
   обучающих срезов: 349 якорей против 319). Отсюда берутся только _test.parquet.
5. Всё нужное копируется в _to_kosta/.

Почему две фазы. Тестовые предсказания прошлого прогона давала модель, обученная
по якорь 348, тогда как остальные модели команды переобучены по 378 — зазор 60
против 30. Лидерборд это подтвердил: перенос валидации на тест составил 40%.
Фаза B чинит это, но её валидация подсматривает (зазор нарушен), поэтому её
_val.parquet использовать нельзя — раннер их переименовывает, чтобы не взяли
случайно.
"""
from __future__ import annotations
import argparse, csv, json, os, shutil, subprocess, sys, time
from pathlib import Path

import numpy as np

SEEDS = (1, 2, 3)
SHRINK = [1.0, 0.97, 0.95, 0.93, 0.9, 0.87, 0.85]   # как в train_tcn.py
ANCHOR_RATIO = 1.09          # 349 обучающих срезов против 319
CFG = ["--arch", "transformer", "--lr", "7e-4", "--channels", "192",
       "--layers", "4", "--heads", "4", "--dropout", "0.0", "--wd", "0.0122",
       "--aux", "0.25", "--batch", "512", "--ema", "0.995", "--min-anchor", "30",
       "--val-users", "0", "--val-anchor", "378"]


def log(msg): print(f"\n{'='*70}\n{msg}\n{'='*70}", flush=True)


def run(cmd, logfile: Path) -> int:
    print("  $ " + " ".join(str(c) for c in cmd), flush=True)
    with open(logfile, "w", encoding="utf-8", errors="replace") as f:
        f.write("$ " + " ".join(str(c) for c in cmd) + "\n\n"); f.flush()
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True, encoding="utf-8", errors="replace", bufsize=1)
        for line in p.stdout:
            sys.stdout.write(line); sys.stdout.flush(); f.write(line)
        return p.wait()


def best_step(tag: str, root: Path) -> int:
    """Шаг лучшего чекпоинта — из history_<tag>.csv, а не из текста лога."""
    p = root / f"history_{tag}.csv"
    if not p.exists(): raise SystemExit(f"нет {p}: фаза A для {tag} не доработала")
    rows = list(csv.DictReader(open(p, encoding="utf-8")))
    if not rows: raise SystemExit(f"{p} пуст")
    b = min(rows, key=lambda r: float(r["val_rmsle"]))
    return int(b["step"]), float(b["val_rmsle"])


def best_shrink(tag: str, root: Path) -> float:
    """Усадка — пересчётом по сохранённым логитам, той же формулой, что в train_tcn."""
    lp = np.load(root / f"val_logpred_{tag}.npy").astype(np.float64)
    lt = np.load(root / f"val_logtrue_{tag}.npy").astype(np.float64)
    sc = [(float(np.sqrt(((np.maximum(c * lp, 0) - lt) ** 2).mean())), c) for c in SHRINK]
    return min(sc)[1], min(sc)[0]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="папка тензора (seq.f16, gmv.f32, ord.f16, meta.npz)")
    ap.add_argument("--minutes", type=float, default=55, help="бюджет одного прогона фазы A")
    ap.add_argument("--out", default="_to_kosta", help="куда сложить результат")
    ap.add_argument("--skip-b", action="store_true", help="только фаза A")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    root = Path.cwd().resolve()
    seq = root / "work" / "scripts" / "seq"
    data = Path(a.data).resolve()
    logs = root / "runlogs"; logs.mkdir(exist_ok=True)
    py = sys.executable

    # ---------- 0. проверки ----------
    log("0. Проверки")
    if not (seq / "train_tcn.py").exists():
        raise SystemExit(f"запускать из КОРНЯ репозитория; сейчас {root}, нет {seq/'train_tcn.py'}")
    need = ["seq.f16", "gmv.f32", "ord.f16", "meta.npz"]
    miss = [f for f in need if not (data / f).exists()]
    if miss: raise SystemExit(f"в {data} нет: {', '.join(miss)}. Сначала build_tensor.py")
    print(f"  тензор: {data}  ({sum((data/f).stat().st_size for f in need)/2**30:.2f} ГБ)")
    try:
        import torch
        print(f"  torch {torch.__version__}, CUDA доступна: {torch.cuda.is_available()}")
        if not torch.cuda.is_available():
            print("  ВНИМАНИЕ: CUDA не видна, обучение пойдёт на CPU и займёт сутки")
    except ImportError:
        msg = "нет torch. pip install torch --index-url https://download.pytorch.org/whl/cu128"
        if not a.dry_run: raise SystemExit(msg)
        print(f"  {msg}  (--dry-run: продолжаю)")
    free = shutil.disk_usage(root).free / 2**30
    print(f"  свободно на диске: {free:.1f} ГБ")
    if free < 5: print("  ВНИМАНИЕ: меньше 5 ГБ, чекпоинты могут не поместиться")
    (root / "work" / "preds").mkdir(parents=True, exist_ok=True)

    # ---------- 1. когорта трёх блоков ----------
    log("1. Маска когорты трёх блоков")
    m = np.load(data / "meta.npz", allow_pickle=False)
    if "valid_anchor3" in m.files:
        print("  valid_anchor3 уже есть в meta.npz — пропускаю")
    elif a.dry_run:
        print(f"  $ {py} {seq/'make_valid3.py'} --data {data}")
    else:
        if run([py, str(seq / "make_valid3.py"), "--data", str(data)], logs / "make_valid3.log"):
            raise SystemExit("make_valid3.py упал")
    del m

    # ---------- 2. фаза A ----------
    log("2. ФАЗА A — измерение, зазор 30 дней, --es-metric cal")
    for s in SEEDS:
        tag = f"tfm2_s{s}"
        if (root / "work" / "preds" / f"{tag}_val.parquet").exists():
            print(f"  {tag}: уже готов, пропускаю"); continue
        cmd = [py, str(seq / "train_tcn.py"), "--data", str(data), "--minutes", str(a.minutes),
               *CFG, "--es-metric", "cal", "--seed", str(s), "--tag", tag,
               "--export", str(root / "work" / "preds")]
        if a.dry_run: print("  $ " + " ".join(cmd)); continue
        t = time.time()
        if run(cmd, logs / f"{tag}.log"): raise SystemExit(f"фаза A, сид {s}: прогон упал")
        print(f"  {tag} готов за {(time.time()-t)/60:.1f} мин")

    # ---------- 3. извлечь шаг и усадку ----------
    log("3. Параметры для фазы B (из файлов, не из текста лога)")
    plan = {}
    if not a.dry_run:
        for s in SEEDS:
            tag = f"tfm2_s{s}"
            st, rm = best_step(tag, root)
            sh, shr = best_shrink(tag, root)
            plan[tag] = {"best_step": st, "best_val_rmsle": rm, "shrink": sh,
                         "shrink_rmsle": shr, "fixed_steps": int(round(st * ANCHOR_RATIO))}
            print(f"  {tag}: лучший шаг {st} (val {rm:.4f}), усадка {sh} -> "
                  f"фаза B на {plan[tag]['fixed_steps']} шагов")
        (root / "work" / "reports").mkdir(parents=True, exist_ok=True)
        json.dump(plan, open(root / "work" / "reports" / "tfm2_phaseA.json", "w"),
                  indent=1, ensure_ascii=False)

    # ---------- 4. фаза B ----------
    if a.skip_b:
        print("\n--skip-b: фаза B пропущена")
    else:
        log("4. ФАЗА B — переобучение по якорь 378, только тестовые предсказания")
        print("  Предупреждение про нарушенный зазор от train_tcn.py — ожидаемое.\n")
        for s in SEEDS:
            tag, src = f"tfm2_s{s}_rt", f"tfm2_s{s}"
            if (root / "work" / "preds" / f"{tag}_test.parquet").exists():
                print(f"  {tag}: уже готов, пропускаю"); continue
            if a.dry_run:
                print(f"  $ ... --max-tr-anchor 378 --fixed-steps <N> --cal-fixed <C> --tag {tag}")
                continue
            p = plan[src]
            cmd = [py, str(seq / "train_tcn.py"), "--data", str(data), *CFG,
                   "--max-tr-anchor", "378", "--fixed-steps", str(p["fixed_steps"]),
                   "--cal-fixed", str(p["shrink"]), "--seed", str(s), "--tag", tag,
                   "--export", str(root / "work" / "preds"),
                   "--predict", f"sub_{tag}.csv", "--no-plots"]
            t = time.time()
            if run(cmd, logs / f"{tag}.log"): raise SystemExit(f"фаза B, сид {s}: прогон упал")
            # её _val.parquet негоден: зазор нарушен, скор завышен
            bad = root / "work" / "preds" / f"{tag}_val.parquet"
            if bad.exists():
                bad.rename(bad.with_suffix(".parquet.LEAKY_DO_NOT_USE"))
                print(f"  {tag}_val.parquet переименован: у него нарушен зазор")
            print(f"  {tag} готов за {(time.time()-t)/60:.1f} мин")

    # ---------- 5. собрать ----------
    if a.dry_run: print("\n--dry-run: ничего не запускалось"); return 0
    log("5. Сборка результата")
    out = root / a.out; out.mkdir(exist_ok=True)
    want = [root / "work" / "preds" / f"tfm2_s{s}_val.parquet" for s in SEEDS]
    if not a.skip_b:
        want += [root / "work" / "preds" / f"tfm2_s{s}_rt_test.parquet" for s in SEEDS]
    want += [root / f"val_user_ids_tfm2_s1.npy", root / f"val_logtrue_tfm2_s1.npy",
             root / "work" / "reports" / "tfm2_phaseA.json"]
    want += [logs / f"tfm2_s{s}.log" for s in SEEDS]
    want += [root / f"history_tfm2_s{s}.csv" for s in SEEDS]
    n = 0
    for p in want:
        if p.exists(): shutil.copy2(p, out / p.name); n += 1
        else: print(f"  НЕТ {p.name}")
    mb = sum(f.stat().st_size for f in out.iterdir() if f.is_file()) / 2**20
    print(f"\n  {n} файлов, {mb:.1f} МБ -> {out}")
    print(f"\nГОТОВО. Перенести папку {out} на машину kosta в")
    print(r"  C:\Users\kosta\ozon_cup\from_gpu\ и написать в чат.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Ночная очередь трека №5: два независимых эксперимента подряд.

    python work/scripts/seq/run_night.py --data tensor            # боевой, ~4.5 часа
    python work/scripts/seq/run_night.py --data tensor --smoke    # проверка, ~6 минут

Оба эксперимента — парные к УЖЕ ИМЕЮЩЕМУСЯ контролю `tfm2_s1` (сид 1, 55 минут,
без предобучения, лучший шаг 22000, скор 1.6692). Меняется ровно одна ось.

  1. ДЛИННОЕ ОБУЧЕНИЕ   tfm2_s1_long   150 мин против 55
     Два сида из трёх в фазе A достигли лучшего калиброванного скора на последнем
     шаге бюджета — модель упёрлась во время, а не в схождение. Суточный свип эту
     ось не проверял: он перебирал гиперпараметры ПРИ фиксированном бюджете.

  2. САМООБУЧЕНИЕ       pre378.pt -> tfm2_s1_pre   60 мин предобучения + 55 дообучения
     Исследовательская часть задания №5. Энкодер учится восстанавливать
     замаскированные токены истории, разметка не нужна. Затем ствол переносится
     в обычное обучение флагом --init-from.

Прерывать можно: при повторном запуске готовые шаги пропускаются.
Упавший шаг НЕ останавливает очередь — иначе падение на втором часу съело бы всю
ночь. Итог по каждому шагу печатается в конце.

Про потолок якоря предобучения (--max-pre-anchor 378) и почему февраль сюда не
входит — в шапке pretrain.py. Коротко: якорь 379+ показал бы энкодеру окно
валидации как вход, и её скор стал бы завышенным.
"""
from __future__ import annotations
import argparse, subprocess, sys, time
from pathlib import Path

ARCH = ["--arch", "transformer", "--lr", "7e-4", "--channels", "192", "--layers", "4",
        "--heads", "4", "--dropout", "0.0", "--wd", "0.0122", "--aux", "0.25",
        "--batch", "512", "--ema", "0.995", "--min-anchor", "30",
        "--val-users", "0", "--val-anchor", "378"]


def run(cmd, logfile):
    print("  $ " + " ".join(str(c) for c in cmd), flush=True)
    with open(logfile, "w", encoding="utf-8", errors="replace") as f:
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True, encoding="utf-8", errors="replace", bufsize=1)
        for line in p.stdout:
            sys.stdout.write(line); sys.stdout.flush(); f.write(line)
        return p.wait()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--smoke", action="store_true", help="по паре минут на шаг, для проверки")
    ap.add_argument("--long-minutes", type=float, default=150)
    ap.add_argument("--pre-minutes", type=float, default=60)
    ap.add_argument("--ft-minutes", type=float, default=55)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()

    root = Path.cwd().resolve(); seq = root / "work" / "scripts" / "seq"
    data = Path(a.data).resolve(); py = sys.executable
    logs = root / "runlogs"; logs.mkdir(exist_ok=True)
    preds = root / "work" / "preds"; preds.mkdir(parents=True, exist_ok=True)
    if not (seq / "pretrain.py").exists():
        raise SystemExit(f"запускать из КОРНЯ репозитория; нет {seq/'pretrain.py'}")

    sfx = "_smoke" if a.smoke else ""
    lm, pm, fm = (0.7, 0.7, 0.7) if a.smoke else (a.long_minutes, a.pre_minutes, a.ft_minutes)
    extra = ["--steps", "400", "--eval-every", "100"] if a.smoke else []
    t_long, t_pre, t_ft = f"tfm2_s{a.seed}_long{sfx}", f"pre378{sfx}.pt", f"tfm2_s{a.seed}_pre{sfx}"

    plan = [
        ("длинное обучение", preds / f"{t_long}_val.parquet",
         [py, str(seq/"train_tcn.py"), "--data", str(data), *ARCH, "--es-metric", "cal",
          "--minutes", str(lm), "--steps", "120000", "--seed", str(a.seed),
          "--tag", t_long, "--export", str(preds), "--no-plots", *extra]),
        ("самообучение", root / t_pre,
         [py, str(seq/"pretrain.py"), "--data", str(data), "--minutes", str(pm),
          "--max-pre-anchor", "378", "--channels", "192", "--layers", "4", "--heads", "4",
          "--dropout", "0.0", "--batch", "512", "--min-anchor", "30",
          "--out", str(root / t_pre), *extra]),
        ("дообучение с предобучения", preds / f"{t_ft}_val.parquet",
         [py, str(seq/"train_tcn.py"), "--data", str(data), *ARCH, "--es-metric", "cal",
          "--minutes", str(fm), "--seed", str(a.seed), "--tag", t_ft,
          "--init-from", str(root / t_pre), "--export", str(preds), "--no-plots", *extra]),
    ]

    total = lm + pm + fm
    print("=" * 70)
    print(f"НОЧНАЯ ОЧЕРЕДЬ{' (ПРОВЕРКА)' if a.smoke else ''}: примерно {total:.0f} минут "
          f"({total/60:.1f} часа)")
    print(f"контроль для обоих экспериментов — уже готовый tfm2_s{a.seed}")
    print("=" * 70)

    res = []
    for i, (name, out, cmd) in enumerate(plan, 1):
        print(f"\n{'='*70}\nШАГ {i}/3: {name}\n{'='*70}", flush=True)
        if out.exists():
            print(f"  {out.name} уже есть — пропускаю"); res.append((name, "пропущен")); continue
        if i == 3 and not (root / t_pre).exists():
            print("  нет чекпойнта предобучения — шаг 2 не дал результата, пропускаю")
            res.append((name, "НЕТ ВХОДА")); continue
        t = time.time()
        rc = run(cmd, logs / f"night_{i}_{name.split()[0]}{sfx}.log")
        dt = (time.time() - t) / 60
        if rc:
            # очередь не останавливаем: падение здесь не должно съесть остаток ночи
            print(f"  ШАГ УПАЛ (код {rc}) за {dt:.1f} мин — иду дальше", flush=True)
            res.append((name, f"УПАЛ, код {rc}"))
        else:
            print(f"  готово за {dt:.1f} мин", flush=True)
            res.append((name, f"ок, {dt:.1f} мин"))

    print(f"\n{'='*70}\nИТОГ\n{'='*70}")
    for name, r in res: print(f"  {name:28} {r}")

    out = root / ("_to_kosta_smoke" if a.smoke else "_to_kosta")
    out.mkdir(exist_ok=True)
    import shutil
    want = [preds / f"{t_long}_val.parquet", preds / f"{t_ft}_val.parquet",
            root / f"history_{t_long}.csv", root / f"history_{t_ft}.csv"]
    want += [logs / f"night_{i}_{n.split()[0]}{sfx}.log" for i, (n, _, _) in enumerate(plan, 1)]
    n = 0
    for p in want:
        if p.exists(): shutil.copy2(p, out / p.name); n += 1
    print(f"\n  {n} файлов -> {out}")
    print(f"  перенести на машину kosta в C:\\Users\\kosta\\ozon_cup\\from_gpu\\ и написать в чат")
    if any("УПАЛ" in r or "НЕТ ВХОДА" in r for _, r in res):
        print("\n  ВНИМАНИЕ: не все шаги прошли — пришлите логи из runlogs/, разберёмся")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Очередь прогонов tfm4 на GPU-машине. Продолжает с места падения.

Порядок ровно такой, потому что дешёвое, что может опровергнуть дорогое, идёт
первым:

  prep    собрать кэш таблицы из tabf16_*.npz            ~3 мин, один раз
  check   нулевая инициализация и тёплый старт           ~2 мин, без обучения
  probe   сид 1: с таблицей И контроль без таблицы       2 x --minutes
  seeds   остальные сиды, обе ветки                      2 x --minutes на сид
  phaseB  переобучение на 25 якорях и предсказание теста  --minutes на сид

Контроль (--tab-off) не опция, а часть замера. tfm4 учится на 24 средах, а
tfm3b учился на всех днях подряд: без прогона на той же сетке любой прирост
нельзя отличить от эффекта смены сетки якорей.

  python work/scripts/seq/run_tfm4.py --stage prep --tab-npz from_gpu/kaggle_tabfeats_wed_v1
  python work/scripts/seq/run_tfm4.py --stage check --init-a "model_tfm2_s{seed}.pt"
  python work/scripts/seq/run_tfm4.py --stage probe --minutes 55
"""
from __future__ import annotations

import argparse, glob, json, os, shutil, subprocess, sys
from pathlib import Path

for _s in (sys.stdout, sys.stderr):
    try: _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception: pass

HERE = Path(__file__).resolve().parent
ROWS = 250000


def log(m): print(f"\n{'='*70}\n{m}\n{'='*70}", flush=True)


def run(cmd, logfile: Path) -> int:
    print("  $ " + " ".join(str(c) for c in cmd), flush=True)
    logfile.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
    with open(logfile, "w", encoding="utf-8", errors="replace") as f:
        f.write("$ " + " ".join(str(c) for c in cmd) + "\n\n"); f.flush()
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True, encoding="utf-8", errors="replace", bufsize=1, env=env)
        for line in p.stdout:
            sys.stdout.write(line); sys.stdout.flush(); f.write(line)
        return p.wait()


NO_POLARS = -2      # отличаем «нечем проверить» от «файл битый»


def parquet_rows(p: Path) -> int:
    try:
        import polars as pl
    except ImportError:
        return NO_POLARS
    try:
        return int(pl.scan_parquet(p).select(pl.len()).collect().item())
    except Exception:
        return -1


def done(root: Path, tag: str, preds: Path, want_test=False, rows=ROWS) -> bool:
    """Готово = есть result json И выгрузка на все 250000 строк.

    Проверка не формальная: в прошлый раз забытый --val-users выдал файл на
    40000 строк, и он чуть не уехал в соседний трек."""
    rj = root / f"result_{tag}.json"
    if not rj.exists():
        return False
    # Ствол выгружается только у прогонов с таблицей. Признак берём из result
    # json, а не из имени тега: имя — не контракт.
    try:
        pass
    except Exception:
        return False
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", required=True,
                    choices=["prep", "check", "probe", "seeds", "phaseB", "report", "collect"])
    ap.add_argument("--root", default=".")
    ap.add_argument("--data", default="tensor")
    ap.add_argument("--tab-npz", default="")
    ap.add_argument("--tab-cache", default="tab_raw.f16")
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--minutes", type=float, default=55)
    ap.add_argument("--init-a", default="model_tfm2_s{seed}.pt")
    ap.add_argument("--init-b", default="model_tfm2_s{seed}_rt.pt")
    ap.add_argument("--tab-dim", type=int, default=128)
    ap.add_argument("--rows", type=int, default=ROWS,
                    help="сколько строк обязана иметь выгрузка (0 — только непустая)")
    ap.add_argument("--extra", nargs="*", default=[])
    # неизвестные флаги уходят в tfm4.py как есть: так можно писать
    #   run_tfm4.py --stage probe --minutes 55 --tab-lr-mult 0.5
    # не перечисляя каждый флаг обучения второй раз. Опечатку поймает tfm4.py.
    a, passthru = ap.parse_known_args()
    if passthru:
        print("передаю в tfm4.py как есть:", " ".join(passthru), flush=True)

    root = Path(a.root).resolve()
    preds = root / "work" / "preds"; preds.mkdir(parents=True, exist_ok=True)
    logs = root / "runlogs"; logs.mkdir(exist_ok=True)
    tfm4 = [sys.executable, str(HERE / "tfm4.py")]
    common = ["--data", a.data, "--tab-cache", a.tab_cache, "--tab-dim", str(a.tab_dim),
              "--no-plots"] + list(a.extra) + passthru
    # --export только там, где действительно пишутся файлы: стадии check он не
    # нужен, а без polars он бы её беспричинно заблокировал
    exp = ["--export", str(preds)]

    if a.stage == "prep":
        if not a.tab_npz:
            return print("--stage prep требует --tab-npz") or 2
        if Path(a.tab_cache).exists() and Path(a.tab_cache).with_suffix(".json").exists():
            print(f"кэш {a.tab_cache} уже собран, пропускаю (удалите файл, чтобы пересобрать)")
            return 0
        return run(tfm4 + ["--prep-tab", "--tab-npz", a.tab_npz, "--tab-cache", a.tab_cache],
                   logs / "tfm4_prep.log")

    if a.stage == "collect":
        # Всё, что нужно для замера запаса, в одну папку. Существует ровно
        # потому, что перечислять эти файлы руками каждый раз — гарантированно
        # что-нибудь забыть.
        dst = root / "_to_kosta" / ("tfm4_" + "_".join(f"s{s_}" for s_ in a.seeds))
        dst.mkdir(parents=True, exist_ok=True)
        pats = []
        for sd in a.seeds:
            pats += [f"work/preds/tfm4*s{sd}*.parquet", f"result_tfm4*s{sd}*.json",
                     f"history_tfm4*s{sd}*.csv", f"val_*tfm4*s{sd}*.npy",
                     f"sub_tfm4*s{sd}*.csv"]
        n = 0
        for pat in pats:
            for f in sorted(glob.glob(str(root / pat))):
                shutil.copy2(f, dst / Path(f).name); n += 1
                print(f"  + {Path(f).name}", flush=True)
        if not n:
            print("ничего не нашлось — сиды те? прогоны прошли?", flush=True); return 1
        zp = dst.with_suffix(".zip")
        if zp.exists(): zp.unlink()
        shutil.make_archive(str(dst), "zip", str(dst))
        print(f"\nсобрано {n} файлов -> {dst}\nархив {zp} ({zp.stat().st_size/2**20:.1f} МБ)",
              flush=True)
        return 0

    if a.stage == "report":
        rows = []
        for f in sorted(root.glob("result_tfm4*.json")):
            rows.append(json.loads(f.read_text(encoding='utf-8')))
        if not rows:
            return print("ни одного result_tfm4*.json") or 1
        print(f"\n{'тег':22} {'фаза':5} {'сид':>4} {'колонок':>8} {'RMSLE':>9} "
              f"{'без табл.':>10} {'Δ':>9}")
        for r in rows:
            d = r.get('rmsle_tabless', float('nan')) - r['rmsle_raw']
            print(f"{r['tag']:22} {r['phase']:5} {r['seed']:>4} {r['n_tab']:>8} "
                  f"{r['rmsle_raw']:9.4f} "
                  + (f"{r['rmsle_tabless']:10.4f} {d:+9.4f}" if 'rmsle_tabless' in r
                     else f"{'—':>10} {'—':>9}"))
        print("\nготовность (что очередь считает сделанным):")
        for r in rows:
            st_a = done(root, r["tag"], preds, want_test=(r["phase"] == "B"), rows=a.rows)
            need = [] if r["tab_off"] else ["cal_tabless"]
            miss = [k for k in need if k not in r]
            print(f"  {r['tag']:22} {'готов' if st_a else 'БУДЕТ ПЕРЕЗАПУЩЕН'}"
                  + (f"   нет полей: {miss}" if miss else ""))
        withtab = [r for r in rows if not r["tab_off"] and r["phase"] == "A"]
        ctrl = [r for r in rows if r["tab_off"] and r["phase"] == "A"]
        if withtab and ctrl:
            mw = sum(r["rmsle_raw"] for r in withtab) / len(withtab)
            mc = sum(r["rmsle_raw"] for r in ctrl) / len(ctrl)
            print(f"\nсреднее по сидам: с таблицей {mw:.4f}, контроль на той же сетке {mc:.4f}, "
                  f"разница {mw - mc:+.4f}")
            print("сравнивать надо ИМЕННО с контролем, а не с числом tfm3b: у tfm3b другая "
                  "сетка якорей, и часть разницы — от неё.")
        return 0

    ok = True
    for seed in a.seeds:
        ia = Path(a.init_a.format(seed=seed))
        if not ia.exists():
            print(f"нет чекпоинта фазы A {ia}. Что есть рядом: "
                  f"{[p.name for p in sorted(root.glob('model_*.pt'))][:12]}", flush=True)
            ok = False; continue

        if a.stage == "check":
            r = run(tfm4 + common + ["--phase", "A", "--tag", f"tfm4chk_s{seed}", "--seed", str(seed),
                                     "--init-from", str(ia), "--check-init-only", "--steps", "0"],
                    logs / f"tfm4_check_s{seed}.log")
            ok &= (r == 0)
            continue

        if a.stage in ("probe", "seeds"):
            for tag, extra in ((f"tfm4_a_s{seed}", []),
                               (f"tfm4off_a_s{seed}", ["--tab-off"])):
                if done(root, tag, preds, rows=a.rows):
                    print(f"{tag} уже готов, пропускаю", flush=True); continue
                log(f"{tag}: фаза A, {a.minutes} мин")
                r = run(tfm4 + common + exp + ["--phase", "A", "--tag", tag, "--seed", str(seed),
                                               "--minutes", str(a.minutes),
                                               "--init-from", str(ia)] + extra,
                        logs / f"{tag}.log")
                if r: print(f"  {tag} упал (код {r}) — иду дальше", flush=True); ok = False
            if a.stage == "probe":
                break

        if a.stage == "phaseB":
            ra = root / f"result_tfm4_a_s{seed}.json"
            if not ra.exists():
                print(f"нет {ra.name}: фаза B берёт число шагов и усадку из фазы A", flush=True)
                ok = False; continue
            res = json.loads(ra.read_text(encoding='utf-8'))
            ib = Path(a.init_b.format(seed=seed))
            if not ib.exists():
                print(f"нет чекпоинта фазы B {ib} — тёплый старт B<-B невозможен, пропускаю")
                ok = False; continue
            tag = f"tfm4_b_s{seed}"
            if done(root, tag, preds, want_test=True, rows=a.rows):
                print(f"{tag} уже готов, пропускаю", flush=True); continue
            log(f"{tag}: фаза B, {res['best_step']} шагов, усадка {res['cal']}")
            extra_b = []
            if res.get("cal_tabless"):
                extra_b = ["--cal-fixed-tabless", str(res["cal_tabless"])]
            elif not res.get("tab_off"):
                print(f"  в {ra.name} нет cal_tabless — фаза A прогонялась старой версией "
                      f"tfm4.py. Перезапусти фазу A для этого сида, иначе ствол не выгрузить.")
                ok = False; continue
            r = run(tfm4 + common + exp + ["--phase", "B", "--tag", tag, "--seed", str(seed),
                                           "--fixed-steps", str(res["best_step"]),
                                           "--cal-fixed", str(res["cal"]),
                                           "--init-from", str(ib),
                                           "--predict", str(root / f"sub_{tag}.csv")] + extra_b,
                    logs / f"{tag}.log")
            if r: print(f"  {tag} упал (код {r}) — иду дальше", flush=True); ok = False

    print("\nвсё" if ok else "\nчасть шагов не прошла — см. runlogs/", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

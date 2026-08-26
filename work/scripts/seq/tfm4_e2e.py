"""Сквозная проверка очереди run_tfm4.py на синтетике: prep -> check -> probe
-> report -> phaseB. Ловит ошибки склейки аргументов, которых не видно в
самопроверке отдельных функций."""
import datetime, json, os, shutil, subprocess, sys, tempfile
from pathlib import Path

import torch

try:
    import polars
    HAVE_POLARS = polars is not None
except ImportError:
    HAVE_POLARS = False


# Windows: при перенаправлении вывода в файл Python берёт кодировку локали
# (cp866/cp1251), и первая же кириллица роняет скрипт UnicodeEncodeError.
# Тот же приём, что в run_all.py.
for _s in (sys.stdout, sys.stderr):
    try: _s.reconfigure(encoding='utf-8', errors='replace')
    except Exception: pass

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import tfm4_selftest as S
from train_tcn import Transformer

TINY = ["--eval-every", "10", "--channels", "16", "--layers", "1",
        "--heads", "2", "--batch", "32", "--workers", "1", "--val-users", "200",
        "--no-val-all", "--steps", "20"]
ROWS0 = ["--rows", "0"]


def sh(args, cwd):
    # encoding='utf-8' обязателен: дочерний процесс пишет UTF-8, а родитель на
    # Windows по умолчанию декодирует в кодировке локали — кириллица превращается
    # в кашу, и проверки вроде «уже готов, пропускаю» молча перестают срабатывать
    r = subprocess.run([sys.executable, str(HERE / "run_tfm4.py")] + args, cwd=cwd,
                       capture_output=True, text=True, encoding="utf-8", errors="replace",
                       env=dict(os.environ, PYTHONIOENCODING="utf-8"))
    return r.returncode, r.stdout + r.stderr


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="tfm4_e2e_"))
    bad = []
    try:
        d0 = datetime.date(2025, 1, 1)
        dates = [(d0 + datetime.timedelta(days=182 + 7 * k)).isoformat() for k in range(8)] \
            + ["2026-01-14", "2026-02-13"]
        S.make_tensor(tmp / "tensor"); S.make_npz(tmp / "npz", dates, n_train=8)
        torch.manual_seed(7)
        for n in ("model_tfm2_s1.pt", "model_tfm2_s1_rt.pt"):
            torch.save(Transformer(S.N_CH + 3, 16, 1, 0.0, 2).state_dict(), tmp / n)

        stages = (
            ("prep",   ["--stage", "prep", "--tab-npz", str(tmp / "npz")]),
            ("check",  ["--stage", "check", "--data", str(tmp / "tensor"), "--seeds", "1"] + ROWS0 + TINY),
            ("probe",  ["--stage", "probe", "--data", str(tmp / "tensor"), "--seeds", "1",
                        "--minutes", "0.05"] + ROWS0 + TINY),
            ("report", ["--stage", "report"]),
            ("phaseB", ["--stage", "phaseB", "--data", str(tmp / "tensor"), "--seeds", "1"] + ROWS0 + TINY),
        )
        if not HAVE_POLARS:
            print("polars не установлен: стадии с выгрузкой parquet пропускаю, "
                  "проверяю prep и check")
            stages = stages[:2]
        for name, args in stages:
            rc, out = sh(args, tmp)
            tail = "\n".join(out.strip().splitlines()[-6:])
            print(f"\n--- {name}: код {rc}\n{tail}", flush=True)
            if rc != 0: bad.append(f"{name} вернул {rc}")

        if not HAVE_POLARS:
            print("\n" + "=" * 60)
            print("prep и check пройдены; остальное требует polars (pip install polars)")
            return 0
        for f in ("result_tfm4_a_s1.json", "result_tfm4off_a_s1.json", "result_tfm4_b_s1.json"):
            if not (tmp / f).exists(): bad.append(f"нет {f}")
        if (tmp / "result_tfm4_a_s1.json").exists():
            r = json.loads((tmp / "result_tfm4_a_s1.json").read_text())
            if "rmsle_tabless" not in r: bad.append("в result нет rmsle_tabless")
            if r["tab_off"]: bad.append("основной прогон помечен tab_off")
        if (tmp / "result_tfm4off_a_s1.json").exists():
            if not json.loads((tmp / "result_tfm4off_a_s1.json").read_text())["tab_off"]:
                bad.append("контроль не помечен tab_off")
        if (tmp / "result_tfm4_b_s1.json").exists():
            rb = json.loads((tmp / "result_tfm4_b_s1.json").read_text())
            if len(rb["grid"]) != 9: bad.append(f"в фазе B {len(rb['grid'])} якорей, ждали 9")

        # повторный запуск обязан пропустить готовое
        rc, out = sh(["--stage", "probe", "--data", str(tmp / "tensor"), "--seeds", "1",
                      "--minutes", "0.05"] + ROWS0 + TINY, tmp)
        if "уже готов, пропускаю" not in out: bad.append("повторный запуск не пропустил готовое")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("\n" + "=" * 60)
    if bad:
        print("ПРОВАЛЕНО:"); [print("  -", b) for b in bad]; return 1
    print("сквозная проверка пройдена"); return 0


if __name__ == "__main__":
    raise SystemExit(main())

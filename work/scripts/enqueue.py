"""Helper to enqueue training jobs. Usage examples in __main__."""
from __future__ import annotations

import json
import sys
from pathlib import Path

Q = Path("/Users/alexanderkondakov/ozon-cup/work/queue")
Q.mkdir(exist_ok=True)


def enqueue(prio: int, name: str, cmd: str, env: dict | None = None):
    spec = {"name": name, "cmd": cmd, "env": env or {}}
    p = Q / f"{prio:03d}_{name}.json"
    p.write_text(json.dumps(spec, ensure_ascii=False, indent=1))
    print(f"enqueued {p.name}")


def gbdt(prio, name, args, use_v3=False):
    env = {"USE_V2": "1", "OMP_NUM_THREADS": "6"}
    if use_v3:
        env["USE_V3"] = "1"
    cmd = f".venv/bin/python work/scripts/train_gbdt.py --name {name} --threads 6 --gap-days 30 {args}"
    enqueue(prio, name, cmd, env)


if __name__ == "__main__":
    for line in sys.stdin:
        line = line.strip()
        if line:
            exec(line)


def snapshot_submissions():
    """Копирует канонические сабмиты в submissions/canonical, чтобы переименование
    при заливке не ломало последующие расчёты (см. историю: файлы переименовывали 4 раза)."""
    import shutil
    from pathlib import Path
    src = Path("/Users/alexanderkondakov/ozon-cup/submissions")
    dst = src / "canonical"; dst.mkdir(exist_ok=True)
    n = 0
    for p in src.glob("*.csv"):
        if p.parent.name == "canonical":
            continue
        t = dst / p.name
        if not t.exists():
            shutil.copy2(p, t); n += 1
    return n

"""Crash-safe sequential job runner. Executes work/queue/*.json jobs FIFO (by filename).

Job spec: {"name": str, "cmd": str (shell command), "env": {optional extra env}}
Moves finished specs to work/queue/done/<name>.json with exit code.
Safety: before each job, waits until system free memory >= 25% and disk >= 12GB.
Stops when work/queue/STOP exists and queue is empty (or immediately if STOP_NOW).
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from pathlib import Path

WORK = Path("/Users/alexanderkondakov/ozon-cup/work")
Q = WORK / "queue"
DONE = Q / "done"
Q.mkdir(exist_ok=True)
DONE.mkdir(exist_ok=True)
LOG = WORK / "reports" / "queue.log"


def log(msg: str):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def free_mem_pct() -> int:
    try:
        out = subprocess.run(["memory_pressure", "-Q"], capture_output=True, text=True, timeout=10).stdout
        for tok in out.split():
            if tok.endswith("%"):
                return int(tok.rstrip("%"))
    except Exception:
        pass
    return 50


def disk_free_gb() -> float:
    st = os.statvfs("/")
    return st.f_bavail * st.f_frsize / 1e9


def wait_safe():
    while True:
        m, d = free_mem_pct(), disk_free_gb()
        if m >= 25 and d >= 12:
            return
        log(f"WAIT resources: mem_free={m}% disk={d:.1f}GB")
        time.sleep(30)


def main():
    log("queue runner started")
    idle = 0
    while True:
        if (Q / "STOP_NOW").exists():
            log("STOP_NOW — exiting")
            return
        jobs = sorted(p for p in Q.glob("*.json"))
        if not jobs:
            if (Q / "STOP").exists():
                log("queue empty + STOP — exiting")
                return
            idle += 1
            if idle % 30 == 1:
                log("queue empty, waiting")
            time.sleep(20)
            continue
        idle = 0
        spec_p = jobs[0]
        try:
            spec = json.loads(spec_p.read_text())
        except Exception as e:
            log(f"BAD SPEC {spec_p.name}: {e}")
            shutil.move(str(spec_p), DONE / (spec_p.name + ".bad"))
            continue
        wait_safe()
        name = spec.get("name", spec_p.stem)
        env = os.environ.copy()
        env.update({k: str(v) for k, v in spec.get("env", {}).items()})
        log(f"START {name}")
        t0 = time.time()
        r = subprocess.run(["/bin/zsh", "-c", spec["cmd"]], env=env,
                           cwd="/Users/alexanderkondakov/ozon-cup",
                           capture_output=True, text=True)
        dt = time.time() - t0
        tail = (r.stdout + r.stderr)[-600:]
        log(f"END {name} exit={r.returncode} {dt:.0f}s | tail: {tail.splitlines()[-1] if tail.splitlines() else ''}")
        spec["exit"] = r.returncode
        spec["seconds"] = round(dt)
        spec["tail"] = tail
        (DONE / spec_p.name).write_text(json.dumps(spec, ensure_ascii=False, indent=1))
        spec_p.unlink()


if __name__ == "__main__":
    main()

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

ROOT = Path(os.environ.get("OZON_ROOT", str(Path(__file__).resolve().parents[2])))
WORK = ROOT / "work"
# Job shell: zsh on the mac this was written for, bash elsewhere. $SHELL wins if set.
SHELL = os.environ.get("QUEUE_SHELL") or (shutil.which("zsh") or shutil.which("bash") or "/bin/sh")
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
    # Linux: MemAvailable/MemTotal from /proc. macOS: memory_pressure -Q.
    try:
        meminfo = Path("/proc/meminfo")
        if meminfo.exists():
            vals = {}
            for line in meminfo.read_text().splitlines():
                k, _, rest = line.partition(":")
                vals[k] = float(rest.strip().split()[0])
            if vals.get("MemTotal"):
                return int(100 * vals.get("MemAvailable", 0) / vals["MemTotal"])
        out = subprocess.run(["memory_pressure", "-Q"], capture_output=True, text=True, timeout=10).stdout
        for tok in out.split():
            if tok.endswith("%"):
                return int(tok.rstrip("%"))
    except Exception:
        pass
    return 50


def disk_free_gb() -> float:
    st = os.statvfs(str(ROOT))
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
        # Atomic claim. Two runners were started by accident and both picked jobs[0],
        # ran the same training twice and doubled peak memory until the OOM killer took
        # them (20.08, lagdir_smoke). os.rename is atomic on POSIX: exactly one runner
        # can win the claim, the loser sees FileNotFoundError and moves to the next job.
        claimed = jobs[0].with_suffix(".json.running")
        try:
            os.rename(jobs[0], claimed)
        except FileNotFoundError:
            continue
        spec_p = claimed
        try:
            spec = json.loads(spec_p.read_text())
        except Exception as e:
            log(f"BAD SPEC {spec_p.name}: {e}")
            shutil.move(str(spec_p), DONE / (spec_p.name + ".bad"))
            continue
        wait_safe()
        name = spec.get("name", spec_p.stem.removesuffix(".json"))
        env = os.environ.copy()
        env.update({k: str(v) for k, v in spec.get("env", {}).items()})
        log(f"START {name}")
        t0 = time.time()
        job_log = WORK / "reports" / f"job_{name}.log"
        with open(job_log, "w") as lf:
            r = subprocess.run([SHELL, "-c", spec["cmd"]], env=env,
                               cwd=str(ROOT),
                               stdout=lf, stderr=subprocess.STDOUT, text=True)
        dt = time.time() - t0
        tail = open(job_log).read()[-600:]
        log(f"END {name} exit={r.returncode} {dt:.0f}s | tail: {tail.splitlines()[-1] if tail.splitlines() else ''}")
        spec["exit"] = r.returncode
        spec["seconds"] = round(dt)
        spec["tail"] = tail
        (DONE / spec_p.name.removesuffix(".running")).write_text(json.dumps(spec, ensure_ascii=False, indent=1))
        # задание могли снять из очереди уже во время выполнения — это не ошибка
        spec_p.unlink(missing_ok=True)


if __name__ == "__main__":
    main()

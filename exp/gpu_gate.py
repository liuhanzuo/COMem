"""Windows single-GPU admission and allocator cap used for local measurements.

Checks nvidia-smi residency and Python compute processes, coordinates local
benchmark processes through a PID lock, and applies a process CUDA budget.
Run GPU workers serially; this helper is specific to Windows/tasklist.
"""

from __future__ import annotations

import atexit
import json
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path

LOCK = Path(__file__).resolve().parent / "results" / ".gpu.lock"
HARD_IDLE_CEILING_GIB = 5.0


def _nvidia_smi():
    """(used_gb, total_gb, [python compute process names]) straight from nvidia-smi."""
    q = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True).stdout.strip().splitlines()[0]
    used, total = (float(x) / 1024.0 for x in q.split(","))
    apps = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader"],
        capture_output=True, text=True, check=False).stdout.strip().splitlines()
    procs = []
    for line in apps:
        line = line.strip()
        if not line:
            continue
        pid, _, name = line.partition(",")
        pid, name = pid.strip(), name.strip()
        if pid.isdigit() and int(pid) == os.getpid():
            continue
        # "[Insufficient Permissions]" rows are system/desktop processes that are present
        # even on an idle card; they are covered by the memory.used threshold, not here.
        if "python" in name.lower():
            procs.append(f"{pid}:{name}")
    return used, total, procs


def _pid_alive(pid: int) -> bool:
    try:
        # tasklist uses the Windows output code page, which can differ from
        # PYTHONUTF8. Inspect the ASCII PID column in raw CSV bytes: a localized
        # no-match message must not raise a decoding exception and pin a dead
        # process's GPU lock forever. Execution failures remain conservative.
        result = subprocess.run(["tasklist", "/FI", f"PID eq {int(pid)}", "/FO", "CSV", "/NH"],
                                capture_output=True, text=False)
        if result.returncode:
            return True
        pattern = rb'^"[^"\r\n]*","' + str(int(pid)).encode("ascii") + rb'"(?:,|$)'
        return any(re.match(pattern, line.strip()) for line in result.stdout.splitlines())
    except Exception:
        return True  # be conservative: treat as alive


def _read_lock():
    try:
        return json.loads(LOCK.read_text(encoding="utf-8"))
    except Exception:
        return None


def _release_lock():
    info = _read_lock()
    if info and info.get("pid") == os.getpid():
        try:
            LOCK.unlink()
        except FileNotFoundError:
            pass


def acquire_gpu(need_gb: float, cap_gb: float | None = None, idle_slack_gb: float = 3.0,
                poll: int = 30, max_wait: int = 6 * 3600, tag: str = "") -> dict:
    """Block until (a) no other gate holds the lock, (b) nvidia-smi shows the card idle,
    then take the lock and (if cap_gb) cap this process's torch allocation."""
    if not math.isfinite(idle_slack_gb) or idle_slack_gb <= 0:
        raise ValueError("GPU idle threshold must be finite and positive")
    requested_idle_slack_gb = idle_slack_gb
    idle_slack_gb = min(idle_slack_gb, HARD_IDLE_CEILING_GIB)
    if requested_idle_slack_gb > idle_slack_gb:
        print(f"[gpu_gate] requested idle threshold {requested_idle_slack_gb:g} GiB; "
              f"hard ceiling enforces used < {idle_slack_gb:g} GiB", flush=True)
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    waited, last_msg = 0, -1
    while True:
        # --- guard 2: lock ---
        info = _read_lock()
        holder = None
        if info is not None:
            if _pid_alive(int(info.get("pid", -1))):
                holder = info
            else:
                print(f"[gpu_gate] removing stale lock held by dead pid {info.get('pid')}",
                      flush=True)
                LOCK.unlink(missing_ok=True)
        # --- guard 1: idle per nvidia-smi ---
        used, total, procs = _nvidia_smi()
        free = total - used
        idle = used < idle_slack_gb and not procs and free >= need_gb
        if holder is None and idle:
            # take the lock, then re-check once: another gate may have won the race
            LOCK.write_text(json.dumps({"pid": os.getpid(), "cmd": tag or " ".join(sys.argv),
                                        "t": time.strftime("%Y-%m-%d %H:%M:%S")}),
                            encoding="utf-8")
            time.sleep(2)
            info = _read_lock()
            used2, total2, procs2 = _nvidia_smi()
            if (info and info.get("pid") == os.getpid() and used2 < idle_slack_gb
                    and not procs2 and total2-used2 >= need_gb):
                admission = {"requested_idle_slack_gib": requested_idle_slack_gb,
                             "effective_idle_slack_gib": idle_slack_gb,
                             "comparison": "strictly_less_than", "source": "nvidia-smi MiB / 1024",
                             "initial_used_gib": used, "recheck_used_gib": used2,
                             "other_python_compute_processes": procs2,
                             "admitted_at": time.strftime("%Y-%m-%d %H:%M:%S")}
                info["admission"] = admission
                LOCK.write_text(json.dumps(info), encoding="utf-8")
                break
            _release_lock()
        if waited >= max_wait:
            raise SystemExit(f"[gpu_gate] still blocked after {waited}s: used {used:.1f} GB, "
                             f"procs {procs}, lock {holder} -- not starting")
        if waited // 300 != last_msg:
            last_msg = waited // 300
            why = f"lock held by {holder}" if holder else f"{used:.1f} GB in use, procs {procs}"
            print(f"[gpu_gate] waiting ({waited}s): {why}", flush=True)
        time.sleep(poll)
        waited += poll
    atexit.register(_release_lock)
    if cap_gb is not None:
        import torch
        tot = torch.cuda.get_device_properties(0).total_memory / 1e9
        torch.cuda.set_per_process_memory_fraction(min(cap_gb / tot, 1.0))
    print(f"[gpu_gate] GPU idle per nvidia-smi (used {used2:.3f} < {idle_slack_gb:g} GiB; "
          f"{total2-used2:.1f}/{total2:.1f} GiB free); lock taken"
          + (f"; capped at {cap_gb:.1f} GB" if cap_gb is not None else ""), flush=True)
    return admission


def main():
    import argparse
    ap = argparse.ArgumentParser(description="hold the GPU gate+lock around a child command")
    ap.add_argument("--need-gb", type=float, default=20.0)
    ap.add_argument("--idle-slack-gb", type=float, default=3.0)
    ap.add_argument("--poll", type=int, default=30)
    ap.add_argument("cmd", nargs=argparse.REMAINDER, help="-- command to run")
    args = ap.parse_args()
    cmd = args.cmd[1:] if args.cmd and args.cmd[0] == "--" else args.cmd
    if not cmd:
        raise SystemExit("usage: gpu_gate.py [--need-gb N] -- <command ...>")
    acquire_gpu(args.need_gb, None, args.idle_slack_gb, args.poll, tag=" ".join(cmd))
    rc = subprocess.call(cmd)
    _release_lock()
    sys.exit(rc)


if __name__ == "__main__":
    main()

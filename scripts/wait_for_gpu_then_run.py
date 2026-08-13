#!/usr/bin/env python3
"""Wait for a GPU to remain idle, then run one command under a local lock."""

from __future__ import annotations

import argparse
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--max-used-memory-mib", type=int, default=1024)
    parser.add_argument("--max-utilization", type=int, default=10)
    parser.add_argument("--stable-checks", type=int, default=3)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--lock-dir", default="/tmp")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    if not args.command:
        parser.error("a command is required after --")
    if args.stable_checks < 1 or args.poll_seconds <= 0:
        parser.error("stable-checks and poll-seconds must be positive")
    return args


def gpu_state(index: int) -> tuple[int, int]:
    output = subprocess.check_output(
        [
            "nvidia-smi",
            f"--id={index}",
            "--query-gpu=memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip()
    used_memory, utilization = (int(value.strip()) for value in output.split(","))
    return used_memory, utilization


def main() -> int:
    args = parse_args()
    lock_path = Path(args.lock_dir) / f"cswae-gpu-{args.gpu}.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w") as lock:
        print(f"Waiting for local GPU lock: {lock_path}", flush=True)
        fcntl.flock(lock, fcntl.LOCK_EX)
        print(f"Acquired local GPU lock for cuda:{args.gpu}", flush=True)
        consecutive = 0
        while consecutive < args.stable_checks:
            try:
                used_memory, utilization = gpu_state(args.gpu)
            except (OSError, subprocess.CalledProcessError, ValueError) as error:
                consecutive = 0
                print(f"GPU query failed ({error}); retrying", flush=True)
            else:
                idle = (
                    used_memory <= args.max_used_memory_mib
                    and utilization <= args.max_utilization
                )
                consecutive = consecutive + 1 if idle else 0
                print(
                    f"cuda:{args.gpu} memory={used_memory} MiB util={utilization}% "
                    f"idle_check={consecutive}/{args.stable_checks}",
                    flush=True,
                )
            if consecutive < args.stable_checks:
                time.sleep(args.poll_seconds)
        print(f"Launching: {' '.join(args.command)}", flush=True)
        completed = subprocess.run(args.command, check=False)
        return int(completed.returncode)


if __name__ == "__main__":
    sys.exit(main())

"""
run_delta_sweep.py

Orchestrates the F-CS-WAE per-class-style-MMD (delta) sweep: trains new
checkpoints at intermediate delta values (the paper currently only has
delta in {0.0, 1.0}) and runs compute_leakage_diagnostics.py against every
checkpoint, old and new, so plot_delta_pareto.py has a consistent set of
JSON files to merge.

Jobs are ordered delta-major (all datasets at a given delta, then move to
the next delta) so an interrupted sweep still covers both datasets across
the delta range instead of finishing one dataset and nothing on the other.
Diagnostics for a finished checkpoint are launched on --diag-device (CPU by
default) and allowed to run concurrently with the *next* GPU training job,
since they cost nothing on the GPU.

delta=0.0 and delta=1.0 map to the existing runs_f/{dataset}/seed_{seed} and
seed_{seed}_pcmmd checkpoints (never retrained, only re-diagnosed with the
now-fixed aux-classifier default in compute_leakage_diagnostics.py); every
other delta value trains a fresh runs_f/{dataset}/seed_{seed}_delta{delta}
checkpoint via train_f_cs_wae.py --delta-final (already fully wired, no new
training-side code needed).

Usage
-----
# Inspect the plan without spending any compute
python scripts/run_delta_sweep.py --dry-run

# Run the agreed grid: new deltas {0.03, 0.1, 0.3, 3.0} x {mnist, cifar10},
# plus re-diagnosing the existing 0.0/1.0 checkpoints
python scripts/run_delta_sweep.py --device cuda:0
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PYTHON = sys.executable


def resolve_run_dir(dataset: str, seed: int, delta_str: str) -> Path:
    """Map a delta value to its checkpoint directory (existing or new)."""
    if float(delta_str) == 0.0:
        return ROOT / "runs_f" / dataset / f"seed_{seed}"
    if float(delta_str) == 1.0:
        return ROOT / "runs_f" / dataset / f"seed_{seed}_pcmmd"
    return ROOT / "runs_f" / dataset / f"seed_{seed}_delta{delta_str}"


def train_command(dataset: str, seed: int, delta_str: str, device: str, epochs, out_dir: Path) -> list[str]:
    cmd = [
        PYTHON, str(ROOT / "train_f_cs_wae.py"),
        "--dataset", dataset,
        "--seed", str(seed),
        "--delta-final", delta_str,
        "--device", device,
        "--output-dir", str(out_dir),
    ]
    if epochs is not None:
        cmd += ["--epochs", str(epochs)]
    return cmd


def diag_command(
    checkpoint: Path, dataset: str, diag_device: str, n_samples: int,
    gen_per_class: int, seed: int, out_json: Path,
) -> list[str]:
    return [
        PYTHON, str(ROOT / "scripts" / "compute_leakage_diagnostics.py"),
        "--checkpoint", str(checkpoint),
        "--dataset", dataset,
        "--device", diag_device,
        "--n-samples", str(n_samples),
        "--gen-per-class", str(gen_per_class),
        "--seed", str(seed),
        "--out", str(out_json),
    ]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--deltas", nargs="+", default=["0.03", "0.1", "0.3", "3.0"],
                   help="New delta values to train (0.0 and 1.0 are always included "
                        "automatically, mapped to the existing baseline/pcmmd checkpoints)")
    p.add_argument("--datasets", nargs="+", default=["mnist", "cifar10"], choices=["mnist", "cifar10"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda:0", help="Device for training")
    p.add_argument("--diag-device", default="cpu", help="Device for compute_leakage_diagnostics.py")
    p.add_argument("--epochs", type=int, default=None, help="Override total epochs (default: full 300)")
    p.add_argument("--n-samples", type=int, default=5000)
    p.add_argument("--gen-per-class", type=int, default=100)
    p.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True,
                   help="Skip training if the checkpoint already exists (default: on)")
    p.add_argument("--dry-run", action="store_true", help="Print the planned jobs and exit")
    return p.parse_args()


def main():
    args = parse_args()
    sweep_dir = ROOT / "runs_diag" / "delta_sweep"
    sweep_dir.mkdir(parents=True, exist_ok=True)
    sweep_log_path = sweep_dir / "sweep_log.jsonl"

    all_deltas = sorted(set(["0.0", "1.0"] + list(args.deltas)), key=float)
    jobs = [(delta, dataset) for delta in all_deltas for dataset in args.datasets]

    print(f"Planned {len(jobs)} (delta, dataset) jobs, delta-major order:")
    for delta, dataset in jobs:
        run_dir = resolve_run_dir(dataset, args.seed, delta)
        ckpt = run_dir / "f_cs_wae_model.pth"
        status = "[exists, will skip training]" if ckpt.exists() else "[will train]"
        print(f"  delta={delta:6s} dataset={dataset:10s} run_dir={run_dir}  {status}")

    if args.dry_run:
        print("\n--dry-run: exiting without training or diagnosing anything.")
        return

    pending_diag: subprocess.Popen | None = None
    pending_diag_desc = ""

    for delta, dataset in jobs:
        run_dir = resolve_run_dir(dataset, args.seed, delta)
        run_dir.parent.mkdir(parents=True, exist_ok=True)
        ckpt_path = run_dir / "f_cs_wae_model.pth"

        if args.skip_existing and ckpt_path.exists():
            print(f"\n[skip-train] {dataset} delta={delta}: checkpoint already exists at {ckpt_path}")
        else:
            cmd = train_command(dataset, args.seed, delta, args.device, args.epochs, run_dir)
            log_file = Path(str(run_dir) + ".log")
            print(f"\n[train] {dataset} delta={delta}")
            print(f"        cmd: {' '.join(cmd)}")
            print(f"        log: {log_file}")
            t0 = time.time()
            with open(log_file, "w") as lf:
                proc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, cwd=str(ROOT))
            elapsed = time.time() - t0
            record = {
                "dataset": dataset, "delta": delta, "returncode": proc.returncode,
                "elapsed_sec": elapsed, "log": str(log_file),
                "timestamp": datetime.now().isoformat(),
            }
            with open(sweep_log_path, "a") as lf2:
                lf2.write(json.dumps(record) + "\n")
            if proc.returncode != 0:
                print(f"[FAILED] {dataset} delta={delta} exited with code {proc.returncode} "
                      f"(see {log_file}) — continuing sweep, this point will be missing.")
                continue
            print(f"        done in {elapsed / 3600:.2f}h")

        if not ckpt_path.exists():
            print(f"[skip-diag] {dataset} delta={delta}: no checkpoint at {ckpt_path}, skipping diagnostics.")
            continue

        # Wait for the previous (CPU) diagnostics job before launching the next
        # GPU training job's diagnostics, keeping at most one diag process live.
        if pending_diag is not None:
            print(f"[wait-diag] waiting for previous diagnostics job ({pending_diag_desc}) to finish...")
            pending_diag.wait()

        diag_out = sweep_dir / f"{dataset}_delta{delta}.json"
        cmd = diag_command(
            ckpt_path, dataset, args.diag_device, args.n_samples, args.gen_per_class, args.seed, diag_out
        )
        diag_log = sweep_dir / f"{dataset}_delta{delta}.diag.log"
        print(f"[diag] {dataset} delta={delta} -> {diag_out} (backgrounded on {args.diag_device})")
        diag_log_fh = open(diag_log, "w")
        pending_diag = subprocess.Popen(cmd, stdout=diag_log_fh, stderr=subprocess.STDOUT, cwd=str(ROOT))
        pending_diag_desc = f"{dataset} delta={delta}"

    if pending_diag is not None:
        print(f"\n[wait-diag] waiting for final diagnostics job ({pending_diag_desc}) to finish...")
        pending_diag.wait()

    print("\nDelta sweep complete.")
    print(f"Per-job training log: {sweep_log_path}")
    print(f"Diagnostics JSONs under: {sweep_dir}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Audit the repaired FACT candidate on independent evaluation seeds 1 and 2."""

from __future__ import annotations

import argparse
import concurrent.futures
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import statistics
import subprocess


ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv" / "bin" / "python"
OUTPUT = (
    ROOT / "runs_diag" / "fact" / "estimator_repair_mnist_40e" /
    "eval_seed_robustness"
)
LOGS = ROOT / "logs" / "fact_estimator_repair_eval_seeds"
CHECKPOINTS = {
    "cap_control": ROOT / "runs_fact" / "development_mnist_40e" /
    "cap_control" / "seed_0" / "f_cs_wae_model.pth",
    "repair_lr01": ROOT / "runs_fact" / "estimator_repair_mnist_40e" /
    "repair_lr01" / "seed_0" / "f_cs_wae_model.pth",
}


def run_logged(command: list[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", buffering=1) as log:
        log.write(
            f"\n[{datetime.now(timezone.utc).isoformat()}] "
            f"COMMAND: {' '.join(command)}\n"
        )
        completed = subprocess.run(
            command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=False
        )
        log.write(f"EXIT: {completed.returncode}\n")
    if completed.returncode:
        raise subprocess.CalledProcessError(completed.returncode, command)


def audit_seed(seed: int, gpu: int) -> int:
    seed_dir = OUTPUT / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    joint = [
        str(PYTHON), str(ROOT / "scripts" / "audit_mnist_joint_contract.py"),
        "--device", f"cuda:{gpu}", "--samples-per-class", "200",
        "--permutations", "199", "--seed", str(seed),
        "--output", str((seed_dir / f"joint_seed{seed}.json").relative_to(ROOT)),
    ]
    for name, checkpoint in CHECKPOINTS.items():
        joint.extend(["--checkpoint", f"{name}={checkpoint}"])
    run_logged(joint, LOGS / f"joint_seed{seed}.log")
    for name, checkpoint in CHECKPOINTS.items():
        leakage = [
            str(PYTHON),
            str(ROOT / "scripts" / "compute_leakage_diagnostics.py"),
            "--checkpoint", str(checkpoint), "--dataset", "mnist",
            "--device", f"cuda:{gpu}", "--n-samples", "2048",
            "--probe-epochs", "300", "--hsic-permutations", "199",
            "--seed", str(seed),
            "--out", str(seed_dir / f"leakage_{name}_seed{seed}.json"),
        ]
        run_logged(leakage, LOGS / f"leakage_{name}_seed{seed}.log")
    return seed


def mean_sd(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.mean(values),
        "sample_sd": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def summarize(seeds: list[int]) -> dict:
    rows = []
    for seed in seeds:
        seed_dir = OUTPUT / f"seed_{seed}"
        joint_payload = json.loads((seed_dir / f"joint_seed{seed}.json").read_text())
        joint = {row["name"]: row for row in joint_payload["results"]}
        for name in CHECKPOINTS:
            leakage = json.loads(
                (seed_dir / f"leakage_{name}_seed{seed}.json").read_text()
            )["results"]
            sampling = leakage["sampling_contract"]
            probes = sampling["probe_suite_z_s_sample"]["models"]
            within = leakage["within_class_dependence"][
                "classwise_conditional_hsic_mu_c_mu_s"
            ]
            rows.append(
                {
                    "evaluation_seed": seed,
                    "name": name,
                    "semantic_accuracy": joint[name]["semantic_head_accuracy"],
                    "reconstruction_mae": joint[name]["reconstruction_mae"],
                    "content_mmd": joint[name]["content"]["mmd2_u"],
                    "style_mmd": joint[name]["style"]["mmd2_u"],
                    "joint_mmd": joint[name]["joint"]["mmd2_u"],
                    "conditional_style_mmd": sampling[
                        "conditional_mmd2_u_z_s_sample"
                    ]["mean_mmd2_u"],
                    "sample_logistic_accuracy": probes["logistic"]["test"]["accuracy"],
                    "sample_mlp_accuracy": probes["mlp"]["test"]["accuracy"],
                    "label_hsic": sampling["hsic_z_s_sample"]["statistic"],
                    "conditional_hsic": within["statistic"],
                }
            )
    metric_names = [
        key for key in rows[0]
        if key not in {"evaluation_seed", "name"}
    ]
    aggregate = {}
    for name in CHECKPOINTS:
        selected = [row for row in rows if row["name"] == name]
        aggregate[name] = {
            metric: mean_sd([float(row[metric]) for row in selected])
            for metric in metric_names
        }
    return {
        "protocol": "fact-estimator-repair-eval-seed-robustness-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_seed": 0,
        "evaluation_seeds": seeds,
        "rows": rows,
        "aggregate": aggregate,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2])
    parser.add_argument("--gpus", nargs="+", type=int, default=[0, 1])
    args = parser.parse_args()
    if len(args.seeds) > len(args.gpus):
        raise ValueError("this runner requires one GPU per evaluation seed")
    for checkpoint in CHECKPOINTS.values():
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(args.seeds)) as pool:
        futures = [
            pool.submit(audit_seed, seed, gpu)
            for seed, gpu in zip(args.seeds, args.gpus)
        ]
        for future in futures:
            future.result()
    payload = summarize(args.seeds)
    target = OUTPUT / "eval_seed_summary.json"
    temporary = target.with_name(f".{target.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, target)


if __name__ == "__main__":
    main()

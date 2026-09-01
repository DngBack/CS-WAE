#!/usr/bin/env python3
"""Audit label-adv=0.10 on independent evaluation seeds before model seeds."""

from __future__ import annotations

import argparse
import concurrent.futures
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess


ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv" / "bin" / "python"
OUTPUT = (
    ROOT / "runs_diag" / "fact" / "label_adversary_mnist_40e"
    / "eval_seed_robustness"
)
LOGS = ROOT / "logs" / "fact_label_adversary_eval_seeds"
CHECKPOINTS = {
    "cap_control": ROOT / "runs_fact" / "development_mnist_40e"
    / "cap_control" / "seed_0" / "f_cs_wae_model.pth",
    "label_adv_w01": ROOT / "runs_fact" / "label_adversary_mnist_40e"
    / "label_adv_w01" / "seed_0" / "f_cs_wae_model.pth",
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_logged(command: list[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", buffering=1) as log:
        log.write(f"\n[{now()}] COMMAND: {' '.join(command)}\n")
        completed = subprocess.run(
            command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=False
        )
        log.write(f"[{now()}] EXIT: {completed.returncode}\n")
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


def read_row(seed: int, name: str, joint: dict) -> dict:
    seed_dir = OUTPUT / f"seed_{seed}"
    leakage = json.loads(
        (seed_dir / f"leakage_{name}_seed{seed}.json").read_text()
    )["results"]
    sampling = leakage["sampling_contract"]
    probes = sampling["probe_suite_z_s_sample"]["models"]
    within = leakage["within_class_dependence"][
        "classwise_conditional_hsic_mu_c_mu_s"
    ]
    return {
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
        "sample_rbf_svm_accuracy": probes["rbf_svm"]["test"]["accuracy"],
        "label_hsic": sampling["hsic_z_s_sample"]["statistic"],
        "label_hsic_p": sampling["hsic_z_s_sample"]["p_value"],
        "conditional_hsic": within["statistic"],
        "conditional_hsic_p": within["p_value"],
    }


def summarize(seeds: list[int]) -> dict:
    rows = []
    for seed in seeds:
        seed_dir = OUTPUT / f"seed_{seed}"
        joint_payload = json.loads((seed_dir / f"joint_seed{seed}.json").read_text())
        joint = {row["name"]: row for row in joint_payload["results"]}
        seed_rows = {name: read_row(seed, name, joint) for name in CHECKPOINTS}
        control = seed_rows["cap_control"]
        candidate = seed_rows["label_adv_w01"]
        for row in seed_rows.values():
            row["joint_reduction_vs_control"] = (
                control["joint_mmd"] - row["joint_mmd"]
            ) / control["joint_mmd"]
            row["conditional_mmd_reduction_vs_control"] = (
                control["conditional_style_mmd"]
                - row["conditional_style_mmd"]
            ) / control["conditional_style_mmd"]
            row["conditional_hsic_ratio_vs_control"] = (
                row["conditional_hsic"] / control["conditional_hsic"]
            )
            rows.append(row)
        gates = {
            "semantic_accuracy_ge_0_985": candidate["semantic_accuracy"] >= 0.985,
            "reconstruction_within_10pct_control": candidate[
                "reconstruction_mae"
            ] <= 1.10 * control["reconstruction_mae"],
            "joint_reduction_ge_20pct": candidate[
                "joint_reduction_vs_control"
            ] >= 0.20,
            "conditional_mmd_reduction_ge_20pct": candidate[
                "conditional_mmd_reduction_vs_control"
            ] >= 0.20,
            "sample_mlp_le_0_40": candidate["sample_mlp_accuracy"] <= 0.40,
            "sample_rbf_svm_le_0_40": candidate[
                "sample_rbf_svm_accuracy"
            ] <= 0.40,
            "conditional_hsic_ratio_le_1_15": candidate[
                "conditional_hsic_ratio_vs_control"
            ] <= 1.15,
        }
        candidate["stability_gates"] = gates
        candidate["stability_pass"] = all(gates.values())
    metric_names = [
        key for key, value in rows[0].items()
        if key not in {"evaluation_seed", "name", "stability_gates", "stability_pass"}
        and isinstance(value, (int, float))
    ]
    aggregate = {}
    for name in CHECKPOINTS:
        selected = [row for row in rows if row["name"] == name]
        aggregate[name] = {
            metric: mean_sd([float(row[metric]) for row in selected])
            for metric in metric_names
        }
    candidate_rows = [row for row in rows if row["name"] == "label_adv_w01"]
    return {
        "protocol": "fact-label-adversary-eval-seed-robustness-v1",
        "created_at_utc": now(), "model_seed": 0,
        "evaluation_seeds": seeds,
        "checkpoint_sha256": {
            name: sha256(path) for name, path in CHECKPOINTS.items()
        },
        "stability_allowed": all(row["stability_pass"] for row in candidate_rows),
        "rows": rows, "aggregate": aggregate,
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
    atomic_json(OUTPUT / "eval_seed_summary.json", summarize(args.seeds))


if __name__ == "__main__":
    main()

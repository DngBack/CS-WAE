#!/usr/bin/env python3
"""Run the frozen MNIST FACT development sweep and its held-out audits.

This is deliberately a development-only, model-seed-0 sweep.  It must not be
reported as multi-seed confirmation.  Runs are assigned to one worker per GPU,
are crash-safe through ``--auto-resume``, and are audited only at epoch 40.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone


ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv" / "bin" / "python"
RUN_ROOT = ROOT / "runs_fact" / "development_mnist_40e"
AUDIT_ROOT = ROOT / "runs_diag" / "fact" / "development_mnist_40e"
LOG_ROOT = ROOT / "logs" / "fact_development_mnist_40e"
MANIFEST = RUN_ROOT / "sweep_manifest.json"

COMMON = [
    "--dataset", "mnist",
    "--seed", "0",
    "--epochs", "40",
    "--batch-size", "320",
    "--phase-a-end", "5",
    "--phase-b-end", "12",
    "--phase-c-end", "26",
    "--phase-d-end", "40",
    "--phase-weight-freeze-epoch", "34",
    "--delta-final", "1",
    "--joint-contract-final", "0",
    "--checkpoint-every", "1",
    "--skip-eval",
    "--auto-resume",
]

RUNS = [
    {"name": "cap_control", "fact": False, "dual_lr": None, "dual_init": None},
    {"name": "fact_lr003_init025", "fact": True, "dual_lr": 0.03, "dual_init": 0.25},
    {"name": "fact_lr003_init1", "fact": True, "dual_lr": 0.03, "dual_init": 1.0},
    {"name": "fact_lr01_init025", "fact": True, "dual_lr": 0.1, "dual_init": 0.25},
    {"name": "fact_lr01_init1", "fact": True, "dual_lr": 0.1, "dual_init": 1.0},
    {"name": "fact_lr03_init025", "fact": True, "dual_lr": 0.3, "dual_init": 0.25},
    {"name": "fact_lr03_init1", "fact": True, "dual_lr": 0.3, "dual_init": 1.0},
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def is_complete(run: dict) -> bool:
    run_dir = RUN_ROOT / run["name"] / "seed_0"
    model = run_dir / "f_cs_wae_model.pth"
    history = run_dir / "training_history.json"
    if not model.is_file() or not history.is_file():
        return False
    try:
        return len(json.loads(history.read_text())) == 40
    except (json.JSONDecodeError, OSError):
        return False


def train_command(run: dict, gpu: int) -> list[str]:
    run_dir = RUN_ROOT / run["name"] / "seed_0"
    command = [str(PYTHON), str(ROOT / "train_f_cs_wae.py"), *COMMON]
    command.extend(["--device", f"cuda:{gpu}", "--output-dir", str(run_dir)])
    if run["fact"]:
        command.extend(
            [
                "--fact",
                "--fact-dual-lr", str(run["dual_lr"]),
                "--fact-dual-init", str(run["dual_init"]),
            ]
        )
    return command


def run_logged(command: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", buffering=1) as log:
        log.write(f"\n[{utc_now()}] COMMAND: {' '.join(command)}\n")
        completed = subprocess.run(
            command,
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
        log.write(f"[{utc_now()}] EXIT: {completed.returncode}\n")
    if completed.returncode:
        raise subprocess.CalledProcessError(completed.returncode, command)


def train_worker(gpu: int, runs: list[dict]) -> list[str]:
    completed = []
    for run in runs:
        if not is_complete(run):
            run_logged(train_command(run, gpu), LOG_ROOT / f"{run['name']}.log")
        if not is_complete(run):
            raise RuntimeError(f"run did not produce a complete epoch-40 model: {run['name']}")
        completed.append(run["name"])
    return completed


def checkpoint(run: dict) -> Path:
    return RUN_ROOT / run["name"] / "seed_0" / "f_cs_wae_model.pth"


def audit_joint(gpu: int) -> None:
    output = AUDIT_ROOT / "joint_seed0.json"
    command = [
        str(PYTHON), str(ROOT / "scripts" / "audit_mnist_joint_contract.py"),
        "--device", f"cuda:{gpu}",
        "--samples-per-class", "200",
        "--permutations", "199",
        "--seed", "0",
        "--output", str(output.relative_to(ROOT)),
    ]
    for run in RUNS:
        command.extend(["--checkpoint", f"{run['name']}={checkpoint(run)}"])
    run_logged(command, LOG_ROOT / "audit_joint.log")


def audit_leakage(run: dict, gpu: int) -> str:
    output = AUDIT_ROOT / f"leakage_{run['name']}_seed0.json"
    command = [
        str(PYTHON), str(ROOT / "scripts" / "compute_leakage_diagnostics.py"),
        "--checkpoint", str(checkpoint(run)),
        "--dataset", "mnist",
        "--device", f"cuda:{gpu}",
        "--n-samples", "2048",
        "--probe-epochs", "300",
        "--hsic-permutations", "199",
        "--seed", "0",
        "--out", str(output),
    ]
    run_logged(command, LOG_ROOT / f"audit_leakage_{run['name']}.log")
    return run["name"]


def audit_worker(gpu: int, runs: list[dict]) -> list[str]:
    return [audit_leakage(run, gpu) for run in runs]


def build_summary() -> dict:
    joint_payload = json.loads((AUDIT_ROOT / "joint_seed0.json").read_text())
    joint = {row["name"]: row for row in joint_payload["results"]}
    rows = []
    for run in RUNS:
        name = run["name"]
        leakage_payload = json.loads(
            (AUDIT_ROOT / f"leakage_{name}_seed0.json").read_text()
        )
        leakage = leakage_payload["results"]
        sampling = leakage["sampling_contract"]
        probes = sampling["probe_suite_z_s_sample"]["models"]
        conditional_hsic = leakage["within_class_dependence"][
            "classwise_conditional_hsic_mu_c_mu_s"
        ]
        row = {
            "name": name,
            "fact": run["fact"],
            "dual_lr": run["dual_lr"],
            "dual_init": run["dual_init"],
            "checkpoint": str(checkpoint(run).relative_to(ROOT)),
            "checkpoint_sha256": sha256(checkpoint(run)),
            "reconstruction_mae": joint[name]["reconstruction_mae"],
            "semantic_accuracy": joint[name]["semantic_head_accuracy"],
            "content_mmd": joint[name]["content"]["mmd2_u"],
            "style_mmd": joint[name]["style"]["mmd2_u"],
            "joint_mmd": joint[name]["joint"]["mmd2_u"],
            "global_style_mmd": leakage["global_mmd"],
            "conditional_style_mmd": sampling[
                "conditional_mmd2_u_z_s_sample"
            ]["mean_mmd2_u"],
            "sample_logistic_accuracy": probes["logistic"]["test"]["accuracy"],
            "sample_mlp_accuracy": probes["mlp"]["test"]["accuracy"],
            "conditional_hsic": conditional_hsic["statistic"],
            "conditional_hsic_p": conditional_hsic["p_value"],
        }
        rows.append(row)
    control = next(row for row in rows if row["name"] == "cap_control")
    for row in rows:
        row["joint_reduction_vs_control"] = (
            (control["joint_mmd"] - row["joint_mmd"]) / control["joint_mmd"]
        )
        row["conditional_mmd_reduction_vs_control"] = (
            (control["conditional_style_mmd"] - row["conditional_style_mmd"])
            / control["conditional_style_mmd"]
        )
        row["development_gates"] = {
            "semantic_accuracy_ge_0_985": row["semantic_accuracy"] >= 0.985,
            "reconstruction_within_10pct_control": row["reconstruction_mae"]
            <= 1.10 * control["reconstruction_mae"],
            "joint_reduction_ge_20pct": row["joint_reduction_vs_control"] >= 0.20,
            "conditional_mmd_reduction_ge_20pct": row[
                "conditional_mmd_reduction_vs_control"
            ] >= 0.20,
            "sample_mlp_le_0_40": row["sample_mlp_accuracy"] <= 0.40,
        }
        row["n_gates_passed"] = sum(row["development_gates"].values())
    eligible = [row for row in rows if row["fact"]]
    eligible.sort(
        key=lambda row: (
            -row["n_gates_passed"],
            row["sample_mlp_accuracy"],
            row["joint_mmd"],
            row["conditional_style_mmd"],
        )
    )
    return {
        "protocol": "fact-development-mnist-40e-v1",
        "created_at_utc": utc_now(),
        "selection_is_development_only": True,
        "control": "cap_control",
        "ranking_rule": [
            "most development gates passed",
            "lowest sample-view MLP accuracy",
            "lowest direct joint MMD",
            "lowest conditional style MMD",
        ],
        "recommended_candidate": eligible[0]["name"],
        "estimator_repair_required": eligible[0]["sample_mlp_accuracy"] > 0.40,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--train-only", action="store_true")
    args = parser.parse_args()
    if not PYTHON.is_file():
        raise FileNotFoundError(PYTHON)
    if not args.gpus:
        raise ValueError("at least one GPU is required")
    for directory in (RUN_ROOT, AUDIT_ROOT, LOG_ROOT):
        directory.mkdir(parents=True, exist_ok=True)
    manifest = {
        "protocol": "fact-development-mnist-40e-v1",
        "created_at_utc": utc_now(),
        "status": "training",
        "model_seed": 0,
        "evaluation_seed": 0,
        "gpus": args.gpus,
        "runs": RUNS,
        "common_arguments": COMMON,
        "repository_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
    }
    atomic_json(MANIFEST, manifest)
    assignments = [RUNS[index :: len(args.gpus)] for index in range(len(args.gpus))]
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(args.gpus)) as pool:
            futures = [
                pool.submit(train_worker, gpu, assigned)
                for gpu, assigned in zip(args.gpus, assignments)
            ]
            for future in futures:
                future.result()
        manifest["training_completed_at_utc"] = utc_now()
        manifest["status"] = "trained" if args.train_only else "auditing"
        atomic_json(MANIFEST, manifest)
        if args.train_only:
            return
        audit_joint(args.gpus[0])
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(args.gpus)) as pool:
            futures = [
                pool.submit(audit_worker, gpu, assigned)
                for gpu, assigned in zip(args.gpus, assignments)
            ]
            for future in futures:
                future.result()
        summary = build_summary()
        atomic_json(AUDIT_ROOT / "development_summary.json", summary)
        manifest["status"] = "complete"
        manifest["completed_at_utc"] = utc_now()
        manifest["recommended_candidate"] = summary["recommended_candidate"]
        manifest["estimator_repair_required"] = summary["estimator_repair_required"]
        atomic_json(MANIFEST, manifest)
    except BaseException as error:
        manifest["status"] = "failed"
        manifest["failed_at_utc"] = utc_now()
        manifest["error"] = repr(error)
        atomic_json(MANIFEST, manifest)
        raise


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)

#!/usr/bin/env python3
"""Run the frozen MNIST FACT estimator/dual-scaling repair sweep."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import subprocess
from datetime import datetime, timezone


ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv" / "bin" / "python"
RUN_ROOT = ROOT / "runs_fact" / "estimator_repair_mnist_40e"
AUDIT_ROOT = ROOT / "runs_diag" / "fact" / "estimator_repair_mnist_40e"
LOG_ROOT = ROOT / "logs" / "fact_estimator_repair_mnist_40e"
MANIFEST = RUN_ROOT / "sweep_manifest.json"
CONTROL = {
    "name": "cap_control",
    "checkpoint": ROOT / "runs_fact" / "development_mnist_40e" /
    "cap_control" / "seed_0" / "f_cs_wae_model.pth",
}
RUNS = [
    {"name": "repair_lr003", "dual_lr": 0.03},
    {"name": "repair_lr01", "dual_lr": 0.1},
    {"name": "repair_lr03", "dual_lr": 0.3},
]
COMMON = [
    "--dataset", "mnist", "--seed", "0", "--epochs", "40",
    "--batch-size", "320",
    "--phase-a-end", "5", "--phase-b-end", "12",
    "--phase-c-end", "26", "--phase-d-end", "40",
    "--phase-weight-freeze-epoch", "34",
    "--delta-final", "1", "--joint-contract-final", "0",
    "--fact", "--fact-dual-init", "1", "--fact-dual-max", "10",
    "--fact-null-draws", "4",
    "--fact-dual-reference-style", "0.002",
    "--fact-dual-reference-content", "0.0015",
    "--fact-dual-reference-dependence", "0.000025",
    "--fact-dual-step-max", "0.25",
    "--checkpoint-every", "1", "--skip-eval", "--auto-resume",
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


def run_dir(run: dict) -> Path:
    return RUN_ROOT / run["name"] / "seed_0"


def checkpoint(run: dict) -> Path:
    return run_dir(run) / "f_cs_wae_model.pth"


def is_complete(run: dict) -> bool:
    history_path = run_dir(run) / "training_history.json"
    if not checkpoint(run).is_file() or not history_path.is_file():
        return False
    try:
        return len(json.loads(history_path.read_text())) == 40
    except (OSError, json.JSONDecodeError):
        return False


def run_logged(command: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", buffering=1) as log:
        log.write(f"\n[{utc_now()}] COMMAND: {' '.join(command)}\n")
        completed = subprocess.run(
            command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=False
        )
        log.write(f"[{utc_now()}] EXIT: {completed.returncode}\n")
    if completed.returncode:
        raise subprocess.CalledProcessError(completed.returncode, command)


def train_worker(gpu: int, assigned: list[dict]) -> list[str]:
    completed = []
    for run in assigned:
        if not is_complete(run):
            command = [
                str(PYTHON), str(ROOT / "train_f_cs_wae.py"), *COMMON,
                "--fact-dual-lr", str(run["dual_lr"]),
                "--device", f"cuda:{gpu}",
                "--output-dir", str(run_dir(run)),
            ]
            run_logged(command, LOG_ROOT / f"{run['name']}.log")
        if not is_complete(run):
            raise RuntimeError(f"incomplete epoch-40 model: {run['name']}")
        completed.append(run["name"])
    return completed


def audit_joint(gpu: int) -> None:
    command = [
        str(PYTHON), str(ROOT / "scripts" / "audit_mnist_joint_contract.py"),
        "--device", f"cuda:{gpu}", "--samples-per-class", "200",
        "--permutations", "199", "--seed", "0",
        "--output", str((AUDIT_ROOT / "joint_seed0.json").relative_to(ROOT)),
        "--checkpoint", f"{CONTROL['name']}={CONTROL['checkpoint']}",
    ]
    for run in RUNS:
        command.extend(["--checkpoint", f"{run['name']}={checkpoint(run)}"])
    run_logged(command, LOG_ROOT / "audit_joint.log")


def audit_leakage(name: str, model_path: Path, gpu: int) -> str:
    output = AUDIT_ROOT / f"leakage_{name}_seed0.json"
    command = [
        str(PYTHON), str(ROOT / "scripts" / "compute_leakage_diagnostics.py"),
        "--checkpoint", str(model_path), "--dataset", "mnist",
        "--device", f"cuda:{gpu}", "--n-samples", "2048",
        "--probe-epochs", "300", "--hsic-permutations", "199",
        "--seed", "0", "--out", str(output),
    ]
    run_logged(command, LOG_ROOT / f"audit_leakage_{name}.log")
    return name


def audit_worker(gpu: int, assigned: list[tuple[str, Path]]) -> list[str]:
    return [audit_leakage(name, path, gpu) for name, path in assigned]


def metric_row(name: str, model_path: Path, joint: dict, run: dict | None) -> dict:
    leakage = json.loads(
        (AUDIT_ROOT / f"leakage_{name}_seed0.json").read_text()
    )["results"]
    sampling = leakage["sampling_contract"]
    probes = sampling["probe_suite_z_s_sample"]["models"]
    dependence = leakage["within_class_dependence"][
        "classwise_conditional_hsic_mu_c_mu_s"
    ]
    return {
        "name": name,
        "dual_lr": None if run is None else run["dual_lr"],
        "checkpoint": str(model_path.relative_to(ROOT)),
        "checkpoint_sha256": sha256(model_path),
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
        "conditional_hsic": dependence["statistic"],
        "conditional_hsic_p": dependence["p_value"],
    }


def build_summary() -> dict:
    payload = json.loads((AUDIT_ROOT / "joint_seed0.json").read_text())
    joint = {row["name"]: row for row in payload["results"]}
    rows = [metric_row(CONTROL["name"], CONTROL["checkpoint"], joint, None)]
    rows.extend(metric_row(run["name"], checkpoint(run), joint, run) for run in RUNS)
    control = rows[0]
    for row in rows:
        row["joint_reduction_vs_control"] = (
            control["joint_mmd"] - row["joint_mmd"]
        ) / control["joint_mmd"]
        row["conditional_mmd_reduction_vs_control"] = (
            control["conditional_style_mmd"] - row["conditional_style_mmd"]
        ) / control["conditional_style_mmd"]
        gates = {
            "semantic_accuracy_ge_0_985": row["semantic_accuracy"] >= 0.985,
            "reconstruction_within_10pct_control": row["reconstruction_mae"]
            <= 1.10 * control["reconstruction_mae"],
            "joint_reduction_ge_20pct": row["joint_reduction_vs_control"] >= 0.20,
            "conditional_mmd_reduction_ge_20pct": row[
                "conditional_mmd_reduction_vs_control"
            ] >= 0.20,
            "sample_mlp_le_0_40": row["sample_mlp_accuracy"] <= 0.40,
        }
        row["development_gates"] = gates
        row["n_gates_passed"] = sum(gates.values())
    candidates = sorted(
        rows[1:],
        key=lambda row: (
            -row["n_gates_passed"], row["sample_mlp_accuracy"],
            row["joint_mmd"], row["conditional_style_mmd"],
        ),
    )
    best = candidates[0]
    promotion = best["n_gates_passed"] == len(best["development_gates"])
    return {
        "protocol": "fact-estimator-repair-mnist-40e-v1",
        "created_at_utc": utc_now(),
        "selection_is_development_only": True,
        "control": CONTROL["name"],
        "recommended_candidate": best["name"],
        "promotion_allowed": promotion,
        "estimator_repair_required": not promotion,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", nargs="+", type=int, default=[0, 1])
    args = parser.parse_args()
    if not CONTROL["checkpoint"].is_file():
        raise FileNotFoundError(CONTROL["checkpoint"])
    if not args.gpus:
        raise ValueError("at least one GPU is required")
    for directory in (RUN_ROOT, AUDIT_ROOT, LOG_ROOT):
        directory.mkdir(parents=True, exist_ok=True)
    source_paths = [
        ROOT / "train_f_cs_wae.py",
        ROOT / "src" / "config_f_cs_wae.py",
        ROOT / "src" / "trainers" / "trainer_f_cs_wae.py",
        ROOT / "src" / "utils" / "loss_f_cs_wae.py",
        Path(__file__).resolve(),
    ]
    manifest = {
        "protocol": "fact-estimator-repair-mnist-40e-v1",
        "created_at_utc": utc_now(), "status": "training",
        "model_seed": 0, "evaluation_seed": 0, "gpus": args.gpus,
        "control_checkpoint": str(CONTROL["checkpoint"].relative_to(ROOT)),
        "control_checkpoint_sha256": sha256(CONTROL["checkpoint"]),
        "runs": RUNS, "common_arguments": COMMON,
        "source_sha256": {
            str(path.relative_to(ROOT)): sha256(path) for path in source_paths
        },
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
        manifest["status"] = "auditing"
        manifest["training_completed_at_utc"] = utc_now()
        atomic_json(MANIFEST, manifest)
        audit_joint(args.gpus[0])
        audit_items = [(CONTROL["name"], CONTROL["checkpoint"])] + [
            (run["name"], checkpoint(run)) for run in RUNS
        ]
        audit_assignments = [
            audit_items[index :: len(args.gpus)] for index in range(len(args.gpus))
        ]
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(args.gpus)) as pool:
            futures = [
                pool.submit(audit_worker, gpu, assigned)
                for gpu, assigned in zip(args.gpus, audit_assignments)
            ]
            for future in futures:
                future.result()
        summary = build_summary()
        atomic_json(AUDIT_ROOT / "repair_summary.json", summary)
        manifest.update(
            status="complete", completed_at_utc=utc_now(),
            recommended_candidate=summary["recommended_candidate"],
            promotion_allowed=summary["promotion_allowed"],
        )
        atomic_json(MANIFEST, manifest)
    except BaseException as error:
        manifest.update(status="failed", failed_at_utc=utc_now(), error=repr(error))
        atomic_json(MANIFEST, manifest)
        raise


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Run the frozen three-strength MNIST style-label HSIC sweep."""

from __future__ import annotations

import argparse
import concurrent.futures
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv" / "bin" / "python"
RUN_ROOT = ROOT / "runs_fact" / "label_hsic_mnist_40e"
AUDIT_ROOT = ROOT / "runs_diag" / "fact" / "label_hsic_mnist_40e"
LOG_ROOT = ROOT / "logs" / "fact_label_hsic_mnist_40e"
MANIFEST = RUN_ROOT / "sweep_manifest.json"
REFERENCES = {
    "cap_control": ROOT / "runs_fact" / "development_mnist_40e" /
    "cap_control" / "seed_0" / "f_cs_wae_model.pth",
    "repair_lr01": ROOT / "runs_fact" / "estimator_repair_mnist_40e" /
    "repair_lr01" / "seed_0" / "f_cs_wae_model.pth",
}
RUNS = [
    {"name": "label_hsic_w10", "weight": 10.0},
    {"name": "label_hsic_w30", "weight": 30.0},
    {"name": "label_hsic_w100", "weight": 100.0},
]
COMMON = [
    "--dataset", "mnist", "--seed", "0", "--epochs", "40",
    "--batch-size", "320",
    "--phase-a-end", "5", "--phase-b-end", "12",
    "--phase-c-end", "26", "--phase-d-end", "40",
    "--phase-weight-freeze-epoch", "34",
    "--delta-final", "1", "--joint-contract-final", "0",
    "--fact", "--fact-dual-init", "1", "--fact-dual-max", "10",
    "--fact-dual-lr", "0.1", "--fact-null-draws", "4",
    "--fact-dual-reference-style", "0.002",
    "--fact-dual-reference-content", "0.0015",
    "--fact-dual-reference-dependence", "0.000025",
    "--fact-dual-step-max", "0.25",
    "--checkpoint-every", "1", "--skip-eval", "--auto-resume",
]


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


def run_logged(command: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", buffering=1) as log:
        log.write(f"\n[{now()}] COMMAND: {' '.join(command)}\n")
        completed = subprocess.run(
            command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=False
        )
        log.write(f"[{now()}] EXIT: {completed.returncode}\n")
    if completed.returncode:
        raise subprocess.CalledProcessError(completed.returncode, command)


def directory(run: dict) -> Path:
    return RUN_ROOT / run["name"] / "seed_0"


def checkpoint(run: dict) -> Path:
    return directory(run) / "f_cs_wae_model.pth"


def complete(run: dict) -> bool:
    history = directory(run) / "training_history.json"
    if not checkpoint(run).is_file() or not history.is_file():
        return False
    try:
        return len(json.loads(history.read_text())) == 40
    except (OSError, json.JSONDecodeError):
        return False


def train_worker(gpu: int, runs: list[dict]) -> None:
    for run in runs:
        if not complete(run):
            command = [
                str(PYTHON), str(ROOT / "train_f_cs_wae.py"), *COMMON,
                "--fact-label-hsic-weight", str(run["weight"]),
                "--device", f"cuda:{gpu}", "--output-dir", str(directory(run)),
            ]
            run_logged(command, LOG_ROOT / f"{run['name']}.log")
        if not complete(run):
            raise RuntimeError(f"incomplete training result: {run['name']}")


def all_checkpoints() -> dict[str, Path]:
    values = dict(REFERENCES)
    values.update({run["name"]: checkpoint(run) for run in RUNS})
    return values


def audit_joint(gpu: int) -> None:
    command = [
        str(PYTHON), str(ROOT / "scripts" / "audit_mnist_joint_contract.py"),
        "--device", f"cuda:{gpu}", "--samples-per-class", "200",
        "--permutations", "199", "--seed", "0",
        "--output", str((AUDIT_ROOT / "joint_seed0.json").relative_to(ROOT)),
    ]
    for name, path in all_checkpoints().items():
        command.extend(["--checkpoint", f"{name}={path}"])
    run_logged(command, LOG_ROOT / "audit_joint.log")


def audit_one(name: str, path: Path, gpu: int) -> None:
    output = AUDIT_ROOT / f"leakage_{name}_seed0.json"
    command = [
        str(PYTHON), str(ROOT / "scripts" / "compute_leakage_diagnostics.py"),
        "--checkpoint", str(path), "--dataset", "mnist",
        "--device", f"cuda:{gpu}", "--n-samples", "2048",
        "--probe-epochs", "300", "--hsic-permutations", "199",
        "--seed", "0", "--out", str(output),
    ]
    run_logged(command, LOG_ROOT / f"audit_leakage_{name}.log")


def audit_worker(gpu: int, items: list[tuple[str, Path]]) -> None:
    for name, path in items:
        audit_one(name, path, gpu)


def summary_row(name: str, path: Path, joint: dict) -> dict:
    leakage = json.loads(
        (AUDIT_ROOT / f"leakage_{name}_seed0.json").read_text()
    )["results"]
    sampling = leakage["sampling_contract"]
    probes = sampling["probe_suite_z_s_sample"]["models"]
    within = leakage["within_class_dependence"][
        "classwise_conditional_hsic_mu_c_mu_s"
    ]
    return {
        "name": name, "checkpoint": str(path.relative_to(ROOT)),
        "checkpoint_sha256": sha256(path),
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
        "label_hsic_p": sampling["hsic_z_s_sample"]["p_value"],
        "conditional_hsic": within["statistic"],
        "conditional_hsic_p": within["p_value"],
    }


def build_summary() -> dict:
    joint_payload = json.loads((AUDIT_ROOT / "joint_seed0.json").read_text())
    joint = {row["name"]: row for row in joint_payload["results"]}
    rows = [summary_row(name, path, joint) for name, path in all_checkpoints().items()]
    control = next(row for row in rows if row["name"] == "cap_control")
    base = next(row for row in rows if row["name"] == "repair_lr01")
    for row in rows:
        row["joint_reduction_vs_control"] = (
            control["joint_mmd"] - row["joint_mmd"]
        ) / control["joint_mmd"]
        row["conditional_mmd_reduction_vs_control"] = (
            control["conditional_style_mmd"] - row["conditional_style_mmd"]
        ) / control["conditional_style_mmd"]
        row["label_hsic_reduction_vs_repair_base"] = (
            base["label_hsic"] - row["label_hsic"]
        ) / base["label_hsic"]
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
        row["conditional_hsic_no_worse_than_control"] = (
            row["conditional_hsic"] <= control["conditional_hsic"]
        )
    candidates = sorted(
        [row for row in rows if row["name"].startswith("label_hsic_")],
        key=lambda row: (
            -row["n_gates_passed"], row["sample_mlp_accuracy"],
            row["joint_mmd"], row["reconstruction_mae"],
        ),
    )
    best = candidates[0]
    promotion = (
        best["n_gates_passed"] == len(best["development_gates"])
        and best["conditional_hsic_no_worse_than_control"]
    )
    return {
        "protocol": "fact-style-label-hsic-mnist-40e-v1",
        "created_at_utc": now(), "model_seed": 0, "evaluation_seed": 0,
        "recommended_candidate": best["name"],
        "promotion_allowed": promotion,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", nargs="+", type=int, default=[0, 1])
    args = parser.parse_args()
    if not args.gpus:
        raise ValueError("at least one GPU is required")
    for path in REFERENCES.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    for target in (RUN_ROOT, AUDIT_ROOT, LOG_ROOT):
        target.mkdir(parents=True, exist_ok=True)
    sources = [
        ROOT / "train_f_cs_wae.py", ROOT / "src" / "config_f_cs_wae.py",
        ROOT / "src" / "trainers" / "trainer_f_cs_wae.py",
        ROOT / "src" / "utils" / "loss_f_cs_wae.py", Path(__file__).resolve(),
    ]
    manifest = {
        "protocol": "fact-style-label-hsic-mnist-40e-v1",
        "created_at_utc": now(), "status": "training", "gpus": args.gpus,
        "model_seed": 0, "evaluation_seed": 0,
        "runs": RUNS, "common_arguments": COMMON,
        "reference_sha256": {name: sha256(path) for name, path in REFERENCES.items()},
        "source_sha256": {
            str(path.relative_to(ROOT)): sha256(path) for path in sources
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
        manifest.update(status="auditing", training_completed_at_utc=now())
        atomic_json(MANIFEST, manifest)
        audit_joint(args.gpus[0])
        items = list(all_checkpoints().items())
        audit_assignments = [items[index :: len(args.gpus)] for index in range(len(args.gpus))]
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(args.gpus)) as pool:
            futures = [
                pool.submit(audit_worker, gpu, assigned)
                for gpu, assigned in zip(args.gpus, audit_assignments)
            ]
            for future in futures:
                future.result()
        summary = build_summary()
        atomic_json(AUDIT_ROOT / "label_hsic_summary.json", summary)
        manifest.update(
            status="complete", completed_at_utc=now(),
            recommended_candidate=summary["recommended_candidate"],
            promotion_allowed=summary["promotion_allowed"],
        )
        atomic_json(MANIFEST, manifest)
    except BaseException as error:
        manifest.update(status="failed", failed_at_utc=now(), error=repr(error))
        atomic_json(MANIFEST, manifest)
        raise


if __name__ == "__main__":
    main()

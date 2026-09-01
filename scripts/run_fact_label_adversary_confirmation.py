#!/usr/bin/env python3
"""Gate, train, audit, and aggregate label-adv=0.10 model seeds 0--2."""

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
FAMILY_ROOT = ROOT / "runs_fact" / "label_adversary_mnist_40e" / "label_adv_w01"
AUDIT_ROOT = (
    ROOT / "runs_diag" / "fact" / "label_adversary_mnist_40e"
    / "model_seed_confirmation"
)
EVAL_STABILITY = (
    ROOT / "runs_diag" / "fact" / "label_adversary_mnist_40e"
    / "eval_seed_robustness" / "eval_seed_summary.json"
)
LOG_ROOT = ROOT / "logs" / "fact_label_adversary_confirmation"
MANIFEST = FAMILY_ROOT / "confirmatory_manifest.json"
CONTROL = (
    ROOT / "runs_fact" / "development_mnist_40e" / "cap_control"
    / "seed_0" / "f_cs_wae_model.pth"
)
MODEL_SEEDS = [0, 1, 2]
TRAIN_SEEDS = [1, 2]
COMMON = [
    "--dataset", "mnist", "--epochs", "40", "--batch-size", "320",
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
    "--fact-label-hsic-weight", "100",
    "--fact-label-adversary-weight", "0.1",
    "--fact-label-adversary-lr", "0.001",
    "--fact-label-adversary-steps", "1",
    "--fact-label-adversary-weight-decay", "0.0001",
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


def model_dir(seed: int) -> Path:
    return FAMILY_ROOT / f"seed_{seed}"


def checkpoint(seed: int) -> Path:
    return model_dir(seed) / "f_cs_wae_model.pth"


def training_complete(seed: int) -> bool:
    history = model_dir(seed) / "training_history.json"
    if not history.is_file() or not checkpoint(seed).is_file():
        return False
    try:
        return len(json.loads(history.read_text())) == 40
    except (OSError, json.JSONDecodeError):
        return False


def train_seed(seed: int, gpu: int) -> int:
    if not training_complete(seed):
        command = [
            str(PYTHON), str(ROOT / "train_f_cs_wae.py"), *COMMON,
            "--seed", str(seed), "--device", f"cuda:{gpu}",
            "--output-dir", str(model_dir(seed)),
        ]
        run_logged(command, LOG_ROOT / f"train_seed{seed}.log")
    if not training_complete(seed):
        raise RuntimeError(f"incomplete model seed: {seed}")
    return seed


def named_checkpoints() -> dict[str, Path]:
    values = {"cap_control": CONTROL}
    values.update({f"label_adv_w01_seed{seed}": checkpoint(seed) for seed in MODEL_SEEDS})
    return values


def audit_joint(gpu: int) -> None:
    command = [
        str(PYTHON), str(ROOT / "scripts" / "audit_mnist_joint_contract.py"),
        "--device", f"cuda:{gpu}", "--samples-per-class", "200",
        "--permutations", "199", "--seed", "0",
        "--output", str((AUDIT_ROOT / "joint_eval_seed0.json").relative_to(ROOT)),
    ]
    for name, path in named_checkpoints().items():
        command.extend(["--checkpoint", f"{name}={path}"])
    run_logged(command, LOG_ROOT / "audit_joint.log")


def audit_leakage(name: str, path: Path, gpu: int) -> None:
    command = [
        str(PYTHON), str(ROOT / "scripts" / "compute_leakage_diagnostics.py"),
        "--checkpoint", str(path), "--dataset", "mnist",
        "--device", f"cuda:{gpu}", "--n-samples", "2048",
        "--probe-epochs", "300", "--hsic-permutations", "199",
        "--seed", "0", "--out", str(AUDIT_ROOT / f"leakage_{name}.json"),
    ]
    run_logged(command, LOG_ROOT / f"audit_leakage_{name}.log")


def audit_worker(gpu: int, items: list[tuple[str, Path]]) -> None:
    for name, path in items:
        audit_leakage(name, path, gpu)


def mean_sd(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.mean(values),
        "sample_sd": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values), "max": max(values),
    }


def read_row(name: str, path: Path, joint: dict) -> dict:
    leakage = json.loads((AUDIT_ROOT / f"leakage_{name}.json").read_text())[
        "results"
    ]
    sampling = leakage["sampling_contract"]
    probes = sampling["probe_suite_z_s_sample"]["models"]
    within = leakage["within_class_dependence"][
        "classwise_conditional_hsic_mu_c_mu_s"
    ]
    row = {
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
        "sample_rbf_svm_accuracy": probes["rbf_svm"]["test"]["accuracy"],
        "label_hsic": sampling["hsic_z_s_sample"]["statistic"],
        "conditional_hsic": within["statistic"],
    }
    if name.startswith("label_adv_w01_seed"):
        row["model_seed"] = int(name.rsplit("seed", 1)[1])
    return row


def build_summary() -> dict:
    payload = json.loads((AUDIT_ROOT / "joint_eval_seed0.json").read_text())
    joint = {row["name"]: row for row in payload["results"]}
    rows = [read_row(name, path, joint) for name, path in named_checkpoints().items()]
    control = next(row for row in rows if row["name"] == "cap_control")
    candidate_rows = [row for row in rows if row["name"].startswith("label_adv_")]
    for row in rows:
        row["joint_reduction_vs_control"] = (
            control["joint_mmd"] - row["joint_mmd"]
        ) / control["joint_mmd"]
        row["conditional_mmd_reduction_vs_control"] = (
            control["conditional_style_mmd"] - row["conditional_style_mmd"]
        ) / control["conditional_style_mmd"]
        row["conditional_hsic_ratio_vs_control"] = (
            row["conditional_hsic"] / control["conditional_hsic"]
        )
        gates = {
            "semantic_accuracy_ge_0_985": row["semantic_accuracy"] >= 0.985,
            "reconstruction_within_10pct_control": row["reconstruction_mae"]
            <= 1.10 * control["reconstruction_mae"],
            "joint_reduction_ge_20pct": row["joint_reduction_vs_control"] >= 0.20,
            "conditional_mmd_reduction_ge_20pct": row[
                "conditional_mmd_reduction_vs_control"
            ] >= 0.20,
            "sample_mlp_le_0_40": row["sample_mlp_accuracy"] <= 0.40,
            "sample_rbf_svm_le_0_40": row["sample_rbf_svm_accuracy"] <= 0.40,
        }
        row["confirmation_gates"] = gates
        row["primary_confirmation_pass"] = all(gates.values())
        row["strict_conditional_hsic_safeguard"] = (
            row["conditional_hsic"] <= control["conditional_hsic"]
        )
    metric_names = [
        key for key, value in candidate_rows[0].items()
        if key not in {
            "name", "checkpoint", "checkpoint_sha256", "model_seed",
            "confirmation_gates", "primary_confirmation_pass",
            "strict_conditional_hsic_safeguard",
        } and isinstance(value, (int, float))
    ]
    aggregate = {
        metric: mean_sd([float(row[metric]) for row in candidate_rows])
        for metric in metric_names
    }
    primary_pass = all(row["primary_confirmation_pass"] for row in candidate_rows)
    strict_pass = all(
        row["strict_conditional_hsic_safeguard"] for row in candidate_rows
    )
    return {
        "protocol": "fact-label-adversary-model-seed-confirmation-v1",
        "created_at_utc": now(), "evaluation_seed": 0,
        "model_seeds": MODEL_SEEDS,
        "primary_confirmation_pass": primary_pass,
        "strict_conditional_hsic_pass": strict_pass,
        "promotion_allowed": primary_pass and strict_pass,
        "rows": rows, "aggregate": aggregate,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", nargs="+", type=int, default=[0, 1])
    args = parser.parse_args()
    if len(args.gpus) < 2:
        raise ValueError("confirmation requires two GPUs")
    for path in (CONTROL, checkpoint(0)):
        if not path.is_file():
            raise FileNotFoundError(path)
    for target in (FAMILY_ROOT, AUDIT_ROOT, LOG_ROOT):
        target.mkdir(parents=True, exist_ok=True)
    sources = [
        ROOT / "train_f_cs_wae.py", ROOT / "src" / "config_f_cs_wae.py",
        ROOT / "src" / "trainers" / "trainer_f_cs_wae.py",
        ROOT / "src" / "utils" / "loss_f_cs_wae.py",
        ROOT / "scripts" / "audit_fact_label_adversary_eval_seeds.py",
        Path(__file__).resolve(),
    ]
    manifest = {
        "protocol": "fact-label-adversary-confirmation-pipeline-v1",
        "created_at_utc": now(), "status": "eval_seed_auditing",
        "gpus": args.gpus[:2], "train_model_seeds": TRAIN_SEEDS,
        "report_model_seeds": MODEL_SEEDS, "common_arguments": COMMON,
        "control_sha256": sha256(CONTROL),
        "development_candidate_sha256": sha256(checkpoint(0)),
        "source_sha256": {
            str(path.relative_to(ROOT)): sha256(path) for path in sources
        },
    }
    atomic_json(MANIFEST, manifest)
    try:
        eval_command = [
            str(PYTHON),
            str(ROOT / "scripts" / "audit_fact_label_adversary_eval_seeds.py"),
            "--seeds", "1", "2", "--gpus", *map(str, args.gpus[:2]),
        ]
        run_logged(eval_command, LOG_ROOT / "eval_seed_gate.log")
        stability = json.loads(EVAL_STABILITY.read_text())
        manifest.update(
            eval_seed_audit_completed_at_utc=now(),
            stability_allowed=stability["stability_allowed"],
            stability_summary=str(EVAL_STABILITY.relative_to(ROOT)),
        )
        if not stability["stability_allowed"]:
            manifest.update(status="stability_gate_failed", completed_at_utc=now())
            atomic_json(MANIFEST, manifest)
            return
        manifest.update(status="training", training_started_at_utc=now())
        atomic_json(MANIFEST, manifest)
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            futures = [
                pool.submit(train_seed, seed, gpu)
                for seed, gpu in zip(TRAIN_SEEDS, args.gpus[:2])
            ]
            for future in futures:
                future.result()
        manifest.update(status="auditing", training_completed_at_utc=now())
        atomic_json(MANIFEST, manifest)
        audit_joint(args.gpus[0])
        items = list(named_checkpoints().items())
        assignments = [items[index::2] for index in range(2)]
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            futures = [
                pool.submit(audit_worker, gpu, assigned)
                for gpu, assigned in zip(args.gpus[:2], assignments)
            ]
            for future in futures:
                future.result()
        summary = build_summary()
        atomic_json(AUDIT_ROOT / "model_seed_summary.json", summary)
        manifest.update(
            status="complete", completed_at_utc=now(),
            primary_confirmation_pass=summary["primary_confirmation_pass"],
            promotion_allowed=summary["promotion_allowed"],
        )
        atomic_json(MANIFEST, manifest)
    except BaseException as error:
        manifest.update(status="failed", failed_at_utc=now(), error=repr(error))
        atomic_json(MANIFEST, manifest)
        raise


if __name__ == "__main__":
    main()

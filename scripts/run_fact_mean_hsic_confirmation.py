#!/usr/bin/env python3
"""Train and audit frozen mean-HSIC=4 MNIST model seeds 0--2."""

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
FAMILY_ROOT = ROOT / "runs_fact" / "mean_hsic_mnist_40e" / "mean_hsic_w4"
AUDIT_ROOT = (
    ROOT / "runs_diag" / "fact" / "mean_hsic_mnist_40e"
    / "model_seed_confirmation_w4"
)
LOG_ROOT = ROOT / "logs" / "fact_mean_hsic_mnist_40e" / "confirmation_w4"
MANIFEST = FAMILY_ROOT / "confirmatory_manifest.json"
CONTROL = (
    ROOT / "runs_fact" / "development_mnist_40e" / "cap_control"
    / "seed_0" / "f_cs_wae_model.pth"
)
MODEL_SEEDS = [0, 1, 2]
TRAIN_SEEDS = [1, 2]
REUSED_LEAKAGE = {
    "cap_control": (
        ROOT / "runs_diag" / "fact" / "label_adversary_mnist_40e"
        / "leakage_cap_control_seed0.json"
    ),
    "mean_hsic_w4_seed0": (
        ROOT / "runs_diag" / "fact" / "mean_hsic_mnist_40e" / "bridge_w4"
        / "eval_seed_0" / "leakage_mean_hsic_w4.json"
    ),
}
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
    "--fact-mean-hsic-weight", "4",
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


def train_seed(seed: int, gpu: int) -> None:
    if not training_complete(seed):
        run_logged(
            [
                str(PYTHON), str(ROOT / "train_f_cs_wae.py"), *COMMON,
                "--seed", str(seed), "--device", f"cuda:{gpu}",
                "--output-dir", str(model_dir(seed)),
            ],
            LOG_ROOT / f"train_seed{seed}.log",
        )
    if not training_complete(seed):
        raise RuntimeError(f"incomplete model seed: {seed}")


def named_checkpoints() -> dict[str, Path]:
    values = {"cap_control": CONTROL}
    values.update(
        {f"mean_hsic_w4_seed{seed}": checkpoint(seed) for seed in MODEL_SEEDS}
    )
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


def leakage_path(name: str) -> Path:
    return REUSED_LEAKAGE.get(name, AUDIT_ROOT / f"leakage_{name}.json")


def audit_leakage(name: str, path: Path, gpu: int) -> None:
    if name in REUSED_LEAKAGE:
        return
    run_logged(
        [
            str(PYTHON),
            str(ROOT / "scripts" / "compute_leakage_diagnostics.py"),
            "--checkpoint", str(path), "--dataset", "mnist",
            "--device", f"cuda:{gpu}", "--n-samples", "2048",
            "--probe-epochs", "300", "--hsic-permutations", "199",
            "--seed", "0", "--out", str(leakage_path(name)),
        ],
        LOG_ROOT / f"audit_leakage_{name}.log",
    )


def mean_sd(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.mean(values),
        "sample_sd": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values), "max": max(values),
    }


def read_row(name: str, path: Path, joint: dict) -> dict:
    leakage = json.loads(leakage_path(name).read_text())["results"]
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
    if name.startswith("mean_hsic_w4_seed"):
        row["model_seed"] = int(name.rsplit("seed", 1)[1])
    return row


def build_summary() -> dict:
    joint_payload = json.loads((AUDIT_ROOT / "joint_eval_seed0.json").read_text())
    joint = {row["name"]: row for row in joint_payload["results"]}
    rows = [
        read_row(name, path, joint) for name, path in named_checkpoints().items()
    ]
    control = next(row for row in rows if row["name"] == "cap_control")
    candidates = [row for row in rows if row["name"].startswith("mean_hsic_w4_")]
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
            "conditional_hsic_ratio_le_1_15": row[
                "conditional_hsic_ratio_vs_control"
            ] <= 1.15,
        }
        row["confirmation_gates"] = gates
        row["confirmation_pass"] = all(gates.values())
        row["strict_conditional_hsic_below_control"] = (
            row["conditional_hsic_ratio_vs_control"] <= 1.0
        )
    excluded = {
        "name", "checkpoint", "checkpoint_sha256", "model_seed",
        "confirmation_gates", "confirmation_pass",
        "strict_conditional_hsic_below_control",
    }
    metric_names = [
        key for key, value in candidates[0].items()
        if key not in excluded and isinstance(value, (int, float))
        and not isinstance(value, bool)
    ]
    aggregate = {
        metric: mean_sd([float(row[metric]) for row in candidates])
        for metric in metric_names
    }
    confirmation_pass = all(row["confirmation_pass"] for row in candidates)
    strict_pass = all(
        row["strict_conditional_hsic_below_control"] for row in candidates
    )
    return {
        "protocol": "fact-mean-hsic-w4-model-seed-confirmation-v1",
        "created_at_utc": now(), "evaluation_seed": 0,
        "model_seeds": MODEL_SEEDS,
        "confirmation_pass": confirmation_pass,
        "strict_conditional_hsic_pass": strict_pass,
        "promotion_allowed": confirmation_pass and strict_pass,
        "rows": rows, "aggregate": aggregate,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", nargs="+", type=int, default=[0, 1])
    args = parser.parse_args()
    if len(args.gpus) < 2:
        raise ValueError("confirmation requires two GPUs")
    required = [CONTROL, checkpoint(0), *REUSED_LEAKAGE.values()]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    for target in (FAMILY_ROOT, AUDIT_ROOT, LOG_ROOT):
        target.mkdir(parents=True, exist_ok=True)
    sources = [
        ROOT / "train_f_cs_wae.py", ROOT / "src" / "config_f_cs_wae.py",
        ROOT / "src" / "trainers" / "trainer_f_cs_wae.py",
        ROOT / "src" / "utils" / "loss_f_cs_wae.py", Path(__file__).resolve(),
    ]
    manifest = {
        "protocol": "fact-mean-hsic-w4-confirmation-pipeline-v1",
        "created_at_utc": now(), "status": "training",
        "gpus": args.gpus[:2], "train_model_seeds": TRAIN_SEEDS,
        "report_model_seeds": MODEL_SEEDS, "evaluation_seed": 0,
        "common_arguments": COMMON,
        "control_sha256": sha256(CONTROL),
        "development_candidate_sha256": sha256(checkpoint(0)),
        "reused_leakage_sha256": {
            name: sha256(path) for name, path in REUSED_LEAKAGE.items()
        },
        "source_sha256": {
            str(path.relative_to(ROOT)): sha256(path) for path in sources
        },
    }
    atomic_json(MANIFEST, manifest)
    try:
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
        audit_items = [
            (name, path) for name, path in named_checkpoints().items()
            if name not in REUSED_LEAKAGE
        ]
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            futures = [
                pool.submit(audit_leakage, name, path, gpu)
                for (name, path), gpu in zip(audit_items, args.gpus[:2])
            ]
            for future in futures:
                future.result()
        summary = build_summary()
        atomic_json(AUDIT_ROOT / "model_seed_summary.json", summary)
        manifest.update(
            status="complete", completed_at_utc=now(),
            confirmation_pass=summary["confirmation_pass"],
            promotion_allowed=summary["promotion_allowed"],
        )
        atomic_json(MANIFEST, manifest)
    except BaseException as error:
        manifest.update(status="failed", failed_at_utc=now(), error=repr(error))
        atomic_json(MANIFEST, manifest)
        raise


if __name__ == "__main__":
    main()

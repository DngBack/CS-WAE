#!/usr/bin/env python3
"""Train mean-HSIC strengths and audit each on evaluation seeds 0--2."""

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
RUN_ROOT = ROOT / "runs_fact" / "mean_hsic_mnist_40e"
AUDIT_ROOT = ROOT / "runs_diag" / "fact" / "mean_hsic_mnist_40e"
LOG_ROOT = ROOT / "logs" / "fact_mean_hsic_mnist_40e"
MANIFEST = RUN_ROOT / "sweep_manifest.json"
CALIBRATION = (
    ROOT / "runs_diag" / "fact" / "label_adversary_mnist_40e"
    / "mean_hsic_calibration.json"
)
REFERENCES = {
    "cap_control": ROOT / "runs_fact" / "development_mnist_40e"
    / "cap_control" / "seed_0" / "f_cs_wae_model.pth",
    "label_adv_w01": ROOT / "runs_fact" / "label_adversary_mnist_40e"
    / "label_adv_w01" / "seed_0" / "f_cs_wae_model.pth",
}
REFERENCE_LEAKAGE = {
    0: {
        "cap_control": ROOT / "runs_diag" / "fact" / "label_adversary_mnist_40e"
        / "leakage_cap_control_seed0.json",
        "label_adv_w01": ROOT / "runs_diag" / "fact" / "label_adversary_mnist_40e"
        / "leakage_label_adv_w01_seed0.json",
    },
    1: {
        name: ROOT / "runs_diag" / "fact" / "label_adversary_mnist_40e"
        / "eval_seed_robustness" / "seed_1" / f"leakage_{name}_seed1.json"
        for name in REFERENCES
    },
    2: {
        name: ROOT / "runs_diag" / "fact" / "label_adversary_mnist_40e"
        / "eval_seed_robustness" / "seed_2" / f"leakage_{name}_seed2.json"
        for name in REFERENCES
    },
}
RUNS = [
    {"name": "mean_hsic_w2", "weight": 2.0},
    {"name": "mean_hsic_w5", "weight": 5.0},
    {"name": "mean_hsic_w20", "weight": 20.0},
]
EVALUATION_SEEDS = [0, 1, 2]
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


def run_dir(run: dict) -> Path:
    return RUN_ROOT / run["name"] / "seed_0"


def checkpoint(run: dict) -> Path:
    return run_dir(run) / "f_cs_wae_model.pth"


def complete(run: dict) -> bool:
    history = run_dir(run) / "training_history.json"
    if not history.is_file() or not checkpoint(run).is_file():
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
                "--fact-mean-hsic-weight", str(run["weight"]),
                "--device", f"cuda:{gpu}", "--output-dir", str(run_dir(run)),
            ]
            run_logged(command, LOG_ROOT / f"train_{run['name']}.log")
        if not complete(run):
            raise RuntimeError(f"incomplete training result: {run['name']}")


def all_checkpoints() -> dict[str, Path]:
    values = dict(REFERENCES)
    values.update({run["name"]: checkpoint(run) for run in RUNS})
    return values


def audit_joint(seed: int, gpu: int) -> None:
    seed_dir = AUDIT_ROOT / f"eval_seed_{seed}"
    command = [
        str(PYTHON), str(ROOT / "scripts" / "audit_mnist_joint_contract.py"),
        "--device", f"cuda:{gpu}", "--samples-per-class", "200",
        "--permutations", "199", "--seed", str(seed),
        "--output", str((seed_dir / "joint.json").relative_to(ROOT)),
    ]
    for name, path in all_checkpoints().items():
        command.extend(["--checkpoint", f"{name}={path}"])
    run_logged(command, LOG_ROOT / f"audit_joint_seed{seed}.log")


def audit_new_models(seed: int, gpu: int) -> None:
    audit_joint(seed, gpu)
    seed_dir = AUDIT_ROOT / f"eval_seed_{seed}"
    for run in RUNS:
        command = [
            str(PYTHON),
            str(ROOT / "scripts" / "compute_leakage_diagnostics.py"),
            "--checkpoint", str(checkpoint(run)), "--dataset", "mnist",
            "--device", f"cuda:{gpu}", "--n-samples", "2048",
            "--probe-epochs", "300", "--hsic-permutations", "199",
            "--seed", str(seed),
            "--out", str(seed_dir / f"leakage_{run['name']}.json"),
        ]
        run_logged(command, LOG_ROOT / f"audit_{run['name']}_seed{seed}.log")


def leakage_path(seed: int, name: str) -> Path:
    if name in REFERENCES:
        return REFERENCE_LEAKAGE[seed][name]
    return AUDIT_ROOT / f"eval_seed_{seed}" / f"leakage_{name}.json"


def read_row(seed: int, name: str, path: Path, joint: dict) -> dict:
    leakage = json.loads(leakage_path(seed, name).read_text())["results"]
    sampling = leakage["sampling_contract"]
    probes = sampling["probe_suite_z_s_sample"]["models"]
    within = leakage["within_class_dependence"][
        "classwise_conditional_hsic_mu_c_mu_s"
    ]
    return {
        "evaluation_seed": seed, "name": name,
        "checkpoint": str(path.relative_to(ROOT)),
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


def mean_sd(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.mean(values),
        "sample_sd": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values), "max": max(values),
    }


def summarize() -> dict:
    rows = []
    paths = all_checkpoints()
    for seed in EVALUATION_SEEDS:
        payload = json.loads((AUDIT_ROOT / f"eval_seed_{seed}" / "joint.json").read_text())
        joint = {row["name"]: row for row in payload["results"]}
        seed_rows = {name: read_row(seed, name, path, joint) for name, path in paths.items()}
        control = seed_rows["cap_control"]
        for row in seed_rows.values():
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
            row["stability_gates"] = gates
            row["stability_pass"] = all(gates.values())
            rows.append(row)
    numeric_metrics = [
        key for key, value in rows[0].items()
        if key not in {
            "evaluation_seed", "name", "checkpoint", "stability_pass"
        }
        and isinstance(value, (int, float)) and not isinstance(value, bool)
    ]
    aggregate = {}
    for name in paths:
        selected = [row for row in rows if row["name"] == name]
        aggregate[name] = {
            metric: mean_sd([float(row[metric]) for row in selected])
            for metric in numeric_metrics
        }
        aggregate[name]["stability_pass_count"] = sum(
            bool(row["stability_pass"]) for row in selected
        )
    candidates = sorted(
        RUNS,
        key=lambda run: (
            -aggregate[run["name"]]["stability_pass_count"],
            aggregate[run["name"]]["conditional_hsic_ratio_vs_control"]["mean"],
            aggregate[run["name"]]["sample_mlp_accuracy"]["mean"],
            aggregate[run["name"]]["joint_mmd"]["mean"],
        ),
    )
    best = candidates[0]["name"]
    return {
        "protocol": "fact-audit-aligned-mean-hsic-eval-seeds-v1",
        "created_at_utc": now(), "model_seed": 0,
        "evaluation_seeds": EVALUATION_SEEDS,
        "calibration": str(CALIBRATION.relative_to(ROOT)),
        "recommended_candidate": best,
        "stability_allowed": aggregate[best]["stability_pass_count"] == 3,
        "rows": rows, "aggregate": aggregate,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", nargs="+", type=int, default=[0, 1])
    args = parser.parse_args()
    if len(args.gpus) < 2:
        raise ValueError("this sweep requires two GPUs")
    required = [*REFERENCES.values(), CALIBRATION]
    required.extend(path for values in REFERENCE_LEAKAGE.values() for path in values.values())
    for path in required:
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
        "protocol": "fact-audit-aligned-mean-hsic-sweep-v1",
        "created_at_utc": now(), "status": "training",
        "gpus": args.gpus[:2], "model_seed": 0,
        "evaluation_seeds": EVALUATION_SEEDS,
        "runs": RUNS, "common_arguments": COMMON,
        "calibration_sha256": sha256(CALIBRATION),
        "reference_sha256": {
            name: sha256(path) for name, path in REFERENCES.items()
        },
        "source_sha256": {
            str(path.relative_to(ROOT)): sha256(path) for path in sources
        },
    }
    atomic_json(MANIFEST, manifest)
    assignments = [RUNS[index::2] for index in range(2)]
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            futures = [
                pool.submit(train_worker, gpu, assigned)
                for gpu, assigned in zip(args.gpus[:2], assignments)
            ]
            for future in futures:
                future.result()
        manifest.update(status="auditing", training_completed_at_utc=now())
        atomic_json(MANIFEST, manifest)
        seed_assignments = [EVALUATION_SEEDS[index::2] for index in range(2)]
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            futures = [
                pool.submit(
                    lambda seeds, gpu: [audit_new_models(seed, gpu) for seed in seeds],
                    seeds, gpu,
                )
                for seeds, gpu in zip(seed_assignments, args.gpus[:2])
            ]
            for future in futures:
                future.result()
        summary = summarize()
        atomic_json(AUDIT_ROOT / "mean_hsic_summary.json", summary)
        manifest.update(
            status="complete", completed_at_utc=now(),
            recommended_candidate=summary["recommended_candidate"],
            stability_allowed=summary["stability_allowed"],
        )
        atomic_json(MANIFEST, manifest)
    except BaseException as error:
        manifest.update(status="failed", failed_at_utc=now(), error=repr(error))
        atomic_json(MANIFEST, manifest)
        raise


if __name__ == "__main__":
    main()

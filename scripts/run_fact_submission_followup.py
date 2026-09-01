#!/usr/bin/env python3
"""Run the frozen FACT submission follow-up on two GPUs.

The protocol has three pre-specified parts:

1. train matched cap-control model seeds 1--2 and report paired deltas for
   training seeds 0--2;
2. run independent-pixel-classifier generation and latent-swap interventions
   for the same control/FACT checkpoint pairs;
3. train post-freeze FACT model seeds 3--5 and audit them without changing the
   configuration selected on development seed 0.

All training is crash-safe through ``--auto-resume``.  The manifest and final
summary are written atomically, and adverse seeds are retained.
"""

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
from typing import Callable, Iterable


ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv" / "bin" / "python"
CONTROL_ROOT = ROOT / "runs_fact" / "development_mnist_40e" / "cap_control"
FACT_ROOT = ROOT / "runs_fact" / "mean_hsic_mnist_40e" / "mean_hsic_w4"
AUDIT_ROOT = (
    ROOT / "runs_diag" / "fact" / "mean_hsic_mnist_40e"
    / "submission_followup_v1"
)
DECODER_ROOT = AUDIT_ROOT / "decoder"
LOG_ROOT = ROOT / "logs" / "fact_submission_followup_v1"
MANIFEST = AUDIT_ROOT / "manifest.json"
SUMMARY = AUDIT_ROOT / "summary.json"
EXTERNAL_CLASSIFIER = (
    ROOT / "external_classifiers" / "mnist" / "grayscale_cnn_4conv_seed0.pth"
)

MATCHED_SEEDS = [0, 1, 2]
NEW_CONTROL_SEEDS = [1, 2]
POSTFREEZE_FACT_SEEDS = [3, 4, 5]

TRAIN_COMMON = [
    "--dataset", "mnist", "--epochs", "40", "--batch-size", "320",
    "--phase-a-end", "5", "--phase-b-end", "12",
    "--phase-c-end", "26", "--phase-d-end", "40",
    "--phase-weight-freeze-epoch", "34",
    "--delta-final", "1", "--joint-contract-final", "0",
    "--checkpoint-every", "1", "--skip-eval", "--auto-resume",
]
FACT_ARGUMENTS = [
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


def checkpoint(kind: str, seed: int) -> Path:
    family = CONTROL_ROOT if kind == "control" else FACT_ROOT
    return family / f"seed_{seed}" / "f_cs_wae_model.pth"


def run_dir(kind: str, seed: int) -> Path:
    return checkpoint(kind, seed).parent


def training_complete(kind: str, seed: int) -> bool:
    history = run_dir(kind, seed) / "training_history.json"
    if not checkpoint(kind, seed).is_file() or not history.is_file():
        return False
    try:
        return len(json.loads(history.read_text())) == 40
    except (json.JSONDecodeError, OSError):
        return False


def validate_frozen_config(kind: str, seed: int) -> None:
    config = json.loads((run_dir(kind, seed) / "run_config.json").read_text())
    expected = {
        "dataset": "mnist", "epochs": 40, "batch_size": 320,
        "seed": seed, "phase_a_end": 5, "phase_b_end": 12,
        "phase_c_end": 26, "phase_d_end": 40,
        "phase_weight_freeze_epoch": 34, "delta_final": 1.0,
        "joint_contract_final": 0.0, "fact_enabled": kind == "fact",
    }
    if kind == "fact":
        expected.update({
            "fact_dual_init": 1.0, "fact_dual_max": 10.0,
            "fact_dual_lr": 0.1, "fact_null_draws": 4,
            "fact_dual_reference_style": 0.002,
            "fact_dual_reference_content": 0.0015,
            "fact_dual_reference_dependence": 0.000025,
            "fact_dual_step_max": 0.25, "fact_label_hsic_weight": 100.0,
            "fact_label_adversary_weight": 0.1,
            "fact_label_adversary_lr": 0.001,
            "fact_label_adversary_steps": 1,
            "fact_label_adversary_weight_decay": 0.0001,
            "fact_mean_hsic_weight": 4.0,
        })
    mismatches = {
        key: {"expected": value, "observed": config.get(key)}
        for key, value in expected.items() if config.get(key) != value
    }
    if mismatches:
        raise RuntimeError(
            f"frozen configuration mismatch for {kind} seed {seed}: {mismatches}"
        )


def train(kind: str, seed: int, gpu: int) -> str:
    name = f"{kind}_seed{seed}"
    if not training_complete(kind, seed):
        command = [
            str(PYTHON), str(ROOT / "train_f_cs_wae.py"), *TRAIN_COMMON,
            "--seed", str(seed), "--device", f"cuda:{gpu}",
            "--output-dir", str(run_dir(kind, seed)),
        ]
        if kind == "fact":
            command.extend(FACT_ARGUMENTS)
        run_logged(command, LOG_ROOT / f"train_{name}.log")
    if not training_complete(kind, seed):
        raise RuntimeError(f"incomplete epoch-40 training run: {name}")
    validate_frozen_config(kind, seed)
    return name


def gpu_worker(gpu: int, jobs: Iterable[tuple[str, int]]) -> list[str]:
    return [train(kind, seed, gpu) for kind, seed in jobs]


def run_gpu_workers(
    gpus: list[int], jobs_by_gpu: list[list], worker: Callable
) -> list:
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        futures = [
            pool.submit(worker, gpu, jobs)
            for gpu, jobs in zip(gpus, jobs_by_gpu)
        ]
        for future in futures:
            results.extend(future.result())
    return results


def decoder_dir(kind: str, seed: int, intervention: str) -> Path:
    return DECODER_ROOT / f"{kind}_seed{seed}" / intervention


def decoder_complete(kind: str, seed: int, intervention: str) -> bool:
    output = decoder_dir(kind, seed, intervention) / "results.json"
    if not output.is_file():
        return False
    try:
        payload = json.loads(output.read_text())
        return payload["manifest"]["checkpoint"]["sha256"] == sha256(
            checkpoint(kind, seed)
        )
    except (KeyError, json.JSONDecodeError, OSError):
        return False


def decoder_interventions(kind: str, seed: int, gpu: int) -> str:
    name = f"{kind}_seed{seed}"
    model = checkpoint(kind, seed)
    if not model.is_file():
        raise FileNotFoundError(model)
    if not decoder_complete(kind, seed, "latent_swap"):
        run_logged(
            [
                str(PYTHON), str(ROOT / "scripts" / "latent_swap_diagnostics.py"),
                "--checkpoint", str(model), "--dataset", "mnist",
                "--tag", name, "--device", f"cuda:{gpu}",
                "--n-samples", "2048", "--n-per-pair", "100",
                "--seed", "0", "--external-classifier-checkpoint",
                str(EXTERNAL_CLASSIFIER), "--output-dir",
                str(decoder_dir(kind, seed, "latent_swap")),
            ],
            LOG_ROOT / f"decoder_swap_{name}.log",
        )
    if not decoder_complete(kind, seed, "generation_accuracy"):
        run_logged(
            [
                str(PYTHON),
                str(ROOT / "scripts" / "generation_accuracy_diagnostics.py"),
                "--checkpoint", str(model),
                "--external-classifier-checkpoint", str(EXTERNAL_CLASSIFIER),
                "--dataset", "mnist", "--tag", name,
                "--device", f"cuda:{gpu}", "--n-per-class", "1000",
                "--style-bank-size", "12000", "--seed", "0",
                "--output-dir",
                str(decoder_dir(kind, seed, "generation_accuracy")),
            ],
            LOG_ROOT / f"decoder_genacc_{name}.log",
        )
    return name


def decoder_worker(gpu: int, jobs: Iterable[tuple[str, int]]) -> list[str]:
    return [decoder_interventions(kind, seed, gpu) for kind, seed in jobs]


def leakage_path(kind: str, seed: int) -> Path:
    return AUDIT_ROOT / "leakage" / f"{kind}_seed{seed}.json"


def audit_leakage(kind: str, seed: int, gpu: int) -> str:
    output = leakage_path(kind, seed)
    run_logged(
        [
            str(PYTHON), str(ROOT / "scripts" / "compute_leakage_diagnostics.py"),
            "--checkpoint", str(checkpoint(kind, seed)),
            "--dataset", "mnist", "--device", f"cuda:{gpu}",
            "--n-samples", "2048", "--probe-epochs", "300",
            "--hsic-permutations", "199", "--seed", "0",
            "--out", str(output),
        ],
        LOG_ROOT / f"audit_leakage_{kind}_seed{seed}.log",
    )
    return f"{kind}_seed{seed}"


def leakage_worker(gpu: int, jobs: Iterable[tuple[str, int]]) -> list[str]:
    return [audit_leakage(kind, seed, gpu) for kind, seed in jobs]


def audit_joint(models: list[tuple[str, int]], output: Path, gpu: int) -> None:
    command = [
        str(PYTHON), str(ROOT / "scripts" / "audit_mnist_joint_contract.py"),
        "--device", f"cuda:{gpu}", "--samples-per-class", "200",
        "--permutations", "199", "--seed", "0",
        "--output", str(output.relative_to(ROOT)),
    ]
    for kind, seed in models:
        command.extend([
            "--checkpoint", f"{kind}_seed{seed}={checkpoint(kind, seed)}"
        ])
    run_logged(command, LOG_ROOT / f"audit_joint_{output.stem}.log")


def extract_row(kind: str, seed: int, joint: dict[str, dict]) -> dict:
    name = f"{kind}_seed{seed}"
    leakage = json.loads(leakage_path(kind, seed).read_text())["results"]
    sampling = leakage["sampling_contract"]
    probes = sampling["probe_suite_z_s_sample"]["models"]
    conditional_hsic = leakage["within_class_dependence"][
        "classwise_conditional_hsic_mu_c_mu_s"
    ]
    return {
        "kind": kind, "training_seed": seed,
        "checkpoint": str(checkpoint(kind, seed).relative_to(ROOT)),
        "checkpoint_sha256": sha256(checkpoint(kind, seed)),
        "semantic_accuracy": joint[name]["semantic_head_accuracy"],
        "reconstruction_mae": joint[name]["reconstruction_mae"],
        "content_mmd": joint[name]["content"]["mmd2_u"],
        "style_mmd": joint[name]["style"]["mmd2_u"],
        "joint_mmd": joint[name]["joint"]["mmd2_u"],
        "conditional_style_mmd": sampling[
            "conditional_mmd2_u_z_s_sample"
        ]["mean_mmd2_u"],
        "sample_mlp_accuracy": probes["mlp"]["test"]["accuracy"],
        "sample_rbf_svm_accuracy": probes["rbf_svm"]["test"]["accuracy"],
        "label_hsic": sampling["hsic_z_s_sample"]["statistic"],
        "conditional_hsic": conditional_hsic["statistic"],
    }


def decoder_metrics(kind: str, seed: int) -> dict:
    swap = json.loads(
        (decoder_dir(kind, seed, "latent_swap") / "results.json").read_text()
    )["results"]["summary"]
    generation = json.loads(
        (
            decoder_dir(kind, seed, "generation_accuracy") / "results.json"
        ).read_text()
    )["results"]["generation"]
    return {
        "swap_content_following": swap["mean_offdiag_content_rate"],
        "swap_style_following": swap["mean_offdiag_style_rate"],
        "swap_neither_following": swap["mean_offdiag_neither_rate"],
        "global_gen_accuracy": generation["global_gaussian"][
            "macro_gen_accuracy"
        ],
        "conditional_gen_accuracy": generation["class_diag_t025"][
            "macro_gen_accuracy"
        ],
    }


def mean_sd(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.mean(values),
        "sample_sd": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values), "max": max(values),
    }


def numeric_aggregate(rows: list[dict], excluded: set[str]) -> dict:
    metrics = [
        key for key, value in rows[0].items()
        if key not in excluded and isinstance(value, (int, float))
        and not isinstance(value, bool)
    ]
    return {
        metric: mean_sd([float(row[metric]) for row in rows])
        for metric in metrics
    }


def build_summary() -> dict:
    matched_joint_payload = json.loads(
        (AUDIT_ROOT / "joint_matched_seed0.json").read_text()
    )
    matched_joint = {
        row["name"]: row for row in matched_joint_payload["results"]
    }
    paired = []
    matched_rows = []
    for seed in MATCHED_SEEDS:
        control = extract_row("control", seed, matched_joint)
        fact = extract_row("fact", seed, matched_joint)
        control.update(decoder_metrics("control", seed))
        fact.update(decoder_metrics("fact", seed))
        matched_rows.extend([control, fact])
        paired.append({
            "training_seed": seed,
            "semantic_accuracy_delta": (
                fact["semantic_accuracy"] - control["semantic_accuracy"]
            ),
            "reconstruction_relative_change": (
                fact["reconstruction_mae"] / control["reconstruction_mae"] - 1
            ),
            "joint_mmd_reduction": (
                1 - fact["joint_mmd"] / control["joint_mmd"]
            ),
            "conditional_mmd_reduction": (
                1 - fact["conditional_style_mmd"]
                / control["conditional_style_mmd"]
            ),
            "conditional_hsic_reduction": (
                1 - fact["conditional_hsic"] / control["conditional_hsic"]
            ),
            "sample_mlp_accuracy_delta": (
                fact["sample_mlp_accuracy"] - control["sample_mlp_accuracy"]
            ),
            "swap_content_following_delta": (
                fact["swap_content_following"]
                - control["swap_content_following"]
            ),
            "swap_style_following_delta": (
                fact["swap_style_following"] - control["swap_style_following"]
            ),
            "global_gen_accuracy_delta": (
                fact["global_gen_accuracy"] - control["global_gen_accuracy"]
            ),
            "conditional_gen_accuracy_delta": (
                fact["conditional_gen_accuracy"]
                - control["conditional_gen_accuracy"]
            ),
        })

    post_joint_payload = json.loads(
        (AUDIT_ROOT / "joint_postfreeze_seed0.json").read_text()
    )
    post_joint = {row["name"]: row for row in post_joint_payload["results"]}
    reference = extract_row("control", 0, post_joint)
    postfreeze = []
    for seed in POSTFREEZE_FACT_SEEDS:
        row = extract_row("fact", seed, post_joint)
        row["joint_mmd_reduction_vs_frozen_control_seed0"] = (
            1 - row["joint_mmd"] / reference["joint_mmd"]
        )
        row["conditional_mmd_reduction_vs_frozen_control_seed0"] = (
            1 - row["conditional_style_mmd"]
            / reference["conditional_style_mmd"]
        )
        row["conditional_hsic_ratio_vs_frozen_control_seed0"] = (
            row["conditional_hsic"] / reference["conditional_hsic"]
        )
        row["confirmation_gates"] = {
            "semantic_accuracy_ge_0_985": row["semantic_accuracy"] >= 0.985,
            "reconstruction_within_10pct_control": row["reconstruction_mae"]
            <= 1.10 * reference["reconstruction_mae"],
            "joint_reduction_ge_20pct": row[
                "joint_mmd_reduction_vs_frozen_control_seed0"
            ] >= 0.20,
            "conditional_mmd_reduction_ge_20pct": row[
                "conditional_mmd_reduction_vs_frozen_control_seed0"
            ] >= 0.20,
            "sample_mlp_le_0_40": row["sample_mlp_accuracy"] <= 0.40,
            "sample_rbf_svm_le_0_40": row[
                "sample_rbf_svm_accuracy"
            ] <= 0.40,
            "conditional_hsic_ratio_le_1_15": row[
                "conditional_hsic_ratio_vs_frozen_control_seed0"
            ] <= 1.15,
        }
        row["confirmation_pass"] = all(row["confirmation_gates"].values())
        postfreeze.append(row)

    return {
        "protocol": "fact-submission-followup-v1",
        "created_at_utc": now(),
        "development_seed": 0,
        "initial_confirmation_seeds": [1, 2],
        "postfreeze_confirmation_seeds": POSTFREEZE_FACT_SEEDS,
        "evaluation_seed": 0,
        "matched_rows": matched_rows,
        "paired_deltas": paired,
        "paired_delta_aggregate": numeric_aggregate(
            paired, {"training_seed"}
        ),
        "postfreeze_reference": reference,
        "postfreeze_rows": postfreeze,
        "postfreeze_aggregate": numeric_aggregate(
            postfreeze,
            {
                "kind", "training_seed", "checkpoint", "checkpoint_sha256",
                "confirmation_gates", "confirmation_pass",
            },
        ),
        "postfreeze_pass_count": sum(
            row["confirmation_pass"] for row in postfreeze
        ),
        "all_postfreeze_seeds_pass": all(
            row["confirmation_pass"] for row in postfreeze
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--skip-decoder", action="store_true")
    parser.add_argument("--train-only", action="store_true")
    args = parser.parse_args()
    if len(args.gpus) != 2:
        raise ValueError("the frozen schedule requires exactly two GPUs")
    required = [PYTHON, EXTERNAL_CLASSIFIER, checkpoint("control", 0)]
    required.extend(checkpoint("fact", seed) for seed in MATCHED_SEEDS)
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    for directory in (AUDIT_ROOT, DECODER_ROOT, LOG_ROOT):
        directory.mkdir(parents=True, exist_ok=True)

    repository_head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    manifest = {
        "protocol": "fact-submission-followup-v1", "created_at_utc": now(),
        "status": "decoder_preflight", "gpus": args.gpus,
        "matched_seeds": MATCHED_SEEDS,
        "new_control_seeds": NEW_CONTROL_SEEDS,
        "postfreeze_fact_seeds": POSTFREEZE_FACT_SEEDS,
        "train_common": TRAIN_COMMON, "fact_arguments": FACT_ARGUMENTS,
        "repository_head": repository_head,
        "repository_dirty_at_start": bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=ROOT, text=True
            ).strip()
        ),
        "source_sha256": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in [
                Path(__file__).resolve(), ROOT / "train_f_cs_wae.py",
                ROOT / "src" / "trainers" / "trainer_f_cs_wae.py",
                ROOT / "src" / "utils" / "loss_f_cs_wae.py",
                ROOT / "scripts" / "latent_swap_diagnostics.py",
                ROOT / "scripts" / "generation_accuracy_diagnostics.py",
            ]
        },
    }
    atomic_json(MANIFEST, manifest)
    try:
        if not args.skip_decoder:
            decoder_jobs = [
                [("control", 0), ("fact", 1)],
                [("fact", 0), ("fact", 2)],
            ]
            run_gpu_workers(args.gpus, decoder_jobs, decoder_worker)
        manifest.update(status="training", decoder_preflight_completed_at_utc=now())
        atomic_json(MANIFEST, manifest)

        training_jobs = [
            [("control", 1), ("fact", 3), ("fact", 5)],
            [("control", 2), ("fact", 4)],
        ]
        trained = run_gpu_workers(args.gpus, training_jobs, gpu_worker)
        manifest.update(
            status="trained" if args.train_only else "decoder_controls",
            training_completed_at_utc=now(), trained=trained,
        )
        atomic_json(MANIFEST, manifest)
        if args.train_only:
            return

        if not args.skip_decoder:
            run_gpu_workers(
                args.gpus,
                [[("control", 1)], [("control", 2)]],
                decoder_worker,
            )
        manifest.update(status="auditing", decoder_completed_at_utc=now())
        atomic_json(MANIFEST, manifest)

        matched_models = [
            (kind, seed) for seed in MATCHED_SEEDS
            for kind in ("control", "fact")
        ]
        postfreeze_models = [("control", 0)] + [
            ("fact", seed) for seed in POSTFREEZE_FACT_SEEDS
        ]
        audit_joint(
            matched_models, AUDIT_ROOT / "joint_matched_seed0.json", args.gpus[0]
        )
        audit_joint(
            postfreeze_models,
            AUDIT_ROOT / "joint_postfreeze_seed0.json", args.gpus[1],
        )
        leakage_jobs = [
            matched_models[::2] + [("fact", 3), ("fact", 5)],
            matched_models[1::2] + [("fact", 4)],
        ]
        run_gpu_workers(args.gpus, leakage_jobs, leakage_worker)
        summary = build_summary()
        atomic_json(SUMMARY, summary)
        manifest.update(
            status="complete", completed_at_utc=now(),
            summary=str(SUMMARY.relative_to(ROOT)),
            postfreeze_pass_count=summary["postfreeze_pass_count"],
            all_postfreeze_seeds_pass=summary["all_postfreeze_seeds_pass"],
        )
        atomic_json(MANIFEST, manifest)
    except BaseException as error:
        manifest.update(status="failed", failed_at_utc=now(), error=repr(error))
        atomic_json(MANIFEST, manifest)
        raise


if __name__ == "__main__":
    main()

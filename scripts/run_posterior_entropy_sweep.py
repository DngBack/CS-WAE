#!/usr/bin/env python3
"""Train, audit, aggregate, and gate the controlled posterior-entropy sweep."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable

# These levels are frozen before looking at sweep outcomes.
MNIST_LEVELS = (0.0, 0.025, 0.05, 0.15, 0.35)
CIFAR_CONFIRMATION_LEVELS = (0.0, 0.15, 0.35)
PROMOTION_ENTROPY_GAIN_NATS_PER_DIM = 0.75
PROMOTION_PROBE_ACCURACY = 0.20
PROMOTION_MAX_RECONSTRUCTION_RATIO = 2.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("mnist", "cifar-if-needed", "cifar"),
        default="mnist",
        help=(
            "mnist runs the full MNIST sweep; cifar-if-needed obeys the frozen "
            "promotion gate; cifar unconditionally runs the frozen CIFAR "
            "confirmation levels after the MNIST summary exists"
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--devices", nargs="+", default=["cuda:0", "cuda:1"])
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--delta-final", type=float, default=0.0)
    parser.add_argument("--output-root", default="runs_f/entropy_sweep")
    parser.add_argument("--n-audit-samples", type=int, default=2048)
    parser.add_argument("--probe-epochs", type=int, default=300)
    parser.add_argument("--hsic-permutations", type=int, default=200)
    parser.add_argument("--gen-per-class", type=int, default=200)
    parser.add_argument("--checkpoint-every", type=int, default=5)
    parser.add_argument("--with-quality-eval", action="store_true")
    parser.add_argument(
        "--retrain-baseline",
        action="store_true",
        help="Retrain sigma=0 instead of auditing the existing paper checkpoint.",
    )
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def level_name(level: float) -> str:
    return f"sigma_{level:g}"


def classifier_checkpoint(dataset: str) -> Path:
    if dataset == "mnist":
        return ROOT / "external_classifiers/mnist/grayscale_cnn_4conv_seed0.pth"
    return ROOT / "external_classifiers/cifar10/wide_resnet_28_10_seed0.pth"


def paper_baseline_checkpoint(dataset: str, seed: int) -> Path:
    return ROOT / "runs_f" / dataset / f"seed_{seed}" / "f_cs_wae_model.pth"


def run_command(command: list[str], log_path: Path, dry_run: bool) -> None:
    rendered = " ".join(command)
    print(f"[{log_path.parent.name}] {rendered}", flush=True)
    if dry_run:
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as log:
        log.write(f"\n$ {rendered}\n")
        log.flush()
        subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)


def run_level(
    *,
    dataset: str,
    level: float,
    device: str,
    args: argparse.Namespace,
    force_audit: bool = False,
) -> Path:
    run_dir = ROOT / args.output_root / dataset / level_name(level) / f"seed_{args.seed}"
    local_checkpoint = run_dir / "f_cs_wae_model.pth"
    historical_baseline = paper_baseline_checkpoint(dataset, args.seed)
    reuse_baseline = level == 0.0 and not args.retrain_baseline and historical_baseline.exists()
    checkpoint = historical_baseline if reuse_baseline else local_checkpoint
    audit_path = run_dir / "audit_protocol_1_2.json"
    log_path = run_dir / "pipeline.log"
    if reuse_baseline:
        print(f"[{level_name(level)}] audit paper baseline {checkpoint}", flush=True)
    elif not (args.skip_existing and checkpoint.exists()):
        command = [
            PYTHON,
            "train_f_cs_wae.py",
            "--dataset", dataset,
            "--seed", str(args.seed),
            "--device", device,
            "--epochs", str(args.epochs),
            "--delta-final", str(args.delta_final),
            "--style-sigma-floor", str(level),
            "--output-dir", str(run_dir),
            "--checkpoint-every", str(args.checkpoint_every),
            "--auto-resume",
        ]
        if not args.with_quality_eval:
            command.append("--skip-eval")
        run_command(command, log_path, args.dry_run)
    else:
        print(f"[{level_name(level)}] reuse checkpoint {checkpoint}", flush=True)

    if force_audit or not (args.skip_existing and audit_is_current(audit_path)):
        command = [
            PYTHON,
            "scripts/compute_leakage_diagnostics.py",
            "--model-type", "fcswae",
            "--checkpoint", str(checkpoint),
            "--dataset", dataset,
            "--device", device,
            "--n-samples", str(args.n_audit_samples),
            "--probe-epochs", str(args.probe_epochs),
            "--hsic-permutations", str(args.hsic_permutations),
            "--gen-per-class", str(args.gen_per_class),
            "--external-classifier-checkpoint", str(classifier_checkpoint(dataset)),
            "--seed", str(args.seed),
            "--out", str(audit_path),
        ]
        run_command(command, log_path, args.dry_run)
    else:
        print(f"[{level_name(level)}] reuse audit {audit_path}", flush=True)
    return audit_path


def audit_is_current(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        with path.open() as handle:
            payload = json.load(handle)
        results = payload["results"]
        return (
            payload["manifest"]["audit_protocol_version"] == "stage0-1.2.0"
            and results["protocol"].get("factorized_adapter") is not None
            and "effective_noise_rms_norm_mean"
            in results["sampling_contract"]["style_posterior"]
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False


def refresh_stale_audits(
    dataset: str,
    levels: Iterable[float],
    args: argparse.Namespace,
) -> None:
    for index, level in enumerate(levels):
        audit_path = (
            ROOT / args.output_root / dataset / level_name(level)
            / f"seed_{args.seed}" / "audit_protocol_1_2.json"
        )
        if not audit_is_current(audit_path):
            run_level(
                dataset=dataset,
                level=level,
                device=args.devices[index % len(args.devices)],
                args=args,
                force_audit=True,
            )


def run_levels(
    dataset: str,
    levels: Iterable[float],
    args: argparse.Namespace,
) -> list[Path]:
    levels = tuple(levels)
    if not args.devices:
        raise ValueError("At least one device is required")
    queues = [levels[index::len(args.devices)] for index in range(len(args.devices))]

    def device_worker(device: str, queue: tuple[float, ...]) -> list[Path]:
        return [
            run_level(dataset=dataset, level=level, device=device, args=args)
            for level in queue
        ]

    paths: list[Path] = []
    with ThreadPoolExecutor(max_workers=len(args.devices)) as executor:
        futures = [
            executor.submit(device_worker, device, queue)
            for device, queue in zip(args.devices, queues)
            if queue
        ]
        for future in futures:
            paths.extend(future.result())
    return sorted(paths)


def aggregate(dataset: str, levels: Iterable[float], args: argparse.Namespace) -> list[dict]:
    rows = []
    for level in levels:
        run_dir = ROOT / args.output_root / dataset / level_name(level) / f"seed_{args.seed}"
        audit_path = run_dir / "audit_protocol_1_2.json"
        with audit_path.open() as handle:
            payload = json.load(handle)
        audit = payload["results"] if "results" in payload else payload
        checkpoint_path = Path(payload["manifest"]["checkpoint"]["path"])
        sampling = audit["sampling_contract"]
        style = sampling["style_posterior"]
        stochastic_probe = sampling["probe_suite_z_s_sample"]["models"]["logistic"]["test"]
        external = audit["generation"]["external"]
        global_generation = None if external is None else external["global_gaussian"]["macro_gen_accuracy"]
        row = {
            "dataset": dataset,
            "seed": args.seed,
            "sigma_floor": float(level),
            "entropy_nats_per_dimension": style["entropy_nats_per_dimension_mean"],
            "effective_std_median": style["effective_std_median"],
            "raw_floor_hit_fraction": style["raw_logvar_at_or_below_minus10_fraction"],
            "probe_accuracy": stochastic_probe["accuracy"],
            "information_lower_bound_nats": stochastic_probe["information_lower_bound_nats"],
            "hsic": sampling["hsic_z_s_sample"]["statistic"],
            "hsic_p_value": sampling["hsic_z_s_sample"]["p_value"],
            "global_mmd2_u": sampling["global_mmd2_u_z_s_sample"],
            "conditional_mmd2_u": sampling["conditional_mmd2_u_z_s_sample"]["mean_mmd2_u"],
            "external_global_gen_accuracy": global_generation,
            "audit_path": str(audit_path),
        }
        history_path = checkpoint_path.parent / "training_history.json"
        if history_path.exists():
            with history_path.open() as handle:
                history = json.load(handle)
            row["final_training_reconstruction"] = float(history[-1]["rec"])
        else:
            row["final_training_reconstruction"] = None
        metrics_path = run_dir / "metrics.json"
        if metrics_path.exists():
            with metrics_path.open() as handle:
                row["quality_metrics"] = json.load(handle)
        rows.append(row)
    rows.sort(key=lambda item: item["sigma_floor"])
    output_dir = ROOT / args.output_root / dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "entropy_sweep_summary.json").open("w") as handle:
        json.dump({"predeclared_levels": list(levels), "rows": rows}, handle, indent=2)
    plot_summary(rows, output_dir / "entropy_sweep.png")
    return rows


def plot_summary(rows: list[dict], output_path: Path) -> None:
    entropy = np.asarray([row["entropy_nats_per_dimension"] for row in rows])
    order = np.argsort(entropy)
    entropy = entropy[order]

    def column(name: str) -> np.ndarray:
        return np.asarray([rows[index][name] for index in order], dtype=np.float64)

    fig, axes = plt.subplots(1, 3, figsize=(12.8, 3.7))
    axes[0].plot(entropy, column("information_lower_bound_nats"), marker="o", label=r"$I_{LB}$")
    accuracy_axis = axes[0].twinx()
    accuracy_axis.plot(entropy, column("probe_accuracy"), marker="s", color="tab:orange", label="probe ACC")
    axes[0].set(ylabel=r"$I_{LB}(z_s;y)$ [nats]", xlabel="posterior entropy [nats/dim]")
    accuracy_axis.set(ylabel="probe accuracy", ylim=(0, 1.02))
    axes[0].set_title("Label leakage")

    axes[1].plot(entropy, column("global_mmd2_u"), marker="o", label="global")
    axes[1].plot(entropy, column("conditional_mmd2_u"), marker="s", label="conditional mean")
    axes[1].axhline(0.0, color="black", linewidth=0.6)
    axes[1].set(xlabel="posterior entropy [nats/dim]", ylabel=r"MMD$^2_U$", title="Prior matching")
    axes[1].legend(frameon=False)

    generation = column("external_global_gen_accuracy")
    axes[2].plot(entropy, generation, marker="o")
    axes[2].set(xlabel="posterior entropy [nats/dim]", ylabel="external Gen-ACC", title="Generation competence", ylim=(0, 1.02))
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def cifar_promotion_decision(rows: list[dict]) -> dict:
    by_level = {row["sigma_floor"]: row for row in rows}
    missing = [level for level in CIFAR_CONFIRMATION_LEVELS if level not in by_level]
    if missing:
        raise ValueError(f"MNIST promotion decision is missing predeclared levels: {missing}")
    baseline = by_level[0.0]
    candidates = []
    for level in CIFAR_CONFIRMATION_LEVELS[1:]:
        row = by_level[level]
        checks = {
            "entropy_gain": row["entropy_nats_per_dimension"]
            - baseline["entropy_nats_per_dimension"]
            >= PROMOTION_ENTROPY_GAIN_NATS_PER_DIM,
            "probe_above_chance": row["probe_accuracy"] >= PROMOTION_PROBE_ACCURACY,
            "hsic_rejects": row["hsic_p_value"] <= 0.05,
            "conditional_exceeds_global": row["conditional_mmd2_u"]
            > row["global_mmd2_u"],
            "reconstruction_stable": row["final_training_reconstruction"] is not None
            and baseline["final_training_reconstruction"] is not None
            and row["final_training_reconstruction"]
            <= PROMOTION_MAX_RECONSTRUCTION_RATIO
            * baseline["final_training_reconstruction"],
        }
        candidates.append({"sigma_floor": level, "checks": checks, "passes": all(checks.values())})
    return {
        "promote_to_cifar": any(candidate["passes"] for candidate in candidates),
        "fixed_cifar_levels": list(CIFAR_CONFIRMATION_LEVELS),
        "thresholds": {
            "entropy_gain_nats_per_dimension": PROMOTION_ENTROPY_GAIN_NATS_PER_DIM,
            "probe_accuracy": PROMOTION_PROBE_ACCURACY,
            "hsic_p_value": 0.05,
            "max_reconstruction_ratio_to_baseline": PROMOTION_MAX_RECONSTRUCTION_RATIO,
        },
        "candidates": candidates,
    }


def main() -> None:
    args = parse_args()
    if args.stage == "mnist":
        levels = MNIST_LEVELS
        run_levels("mnist", levels, args)
        if not args.dry_run:
            refresh_stale_audits("mnist", levels, args)
            rows = aggregate("mnist", levels, args)
            decision = cifar_promotion_decision(rows)
            decision_path = ROOT / args.output_root / "mnist/cifar_promotion_decision.json"
            with decision_path.open("w") as handle:
                json.dump(decision, handle, indent=2)
            print(f"CIFAR promotion decision: {decision['promote_to_cifar']} ({decision_path})")
        return

    summary_path = ROOT / args.output_root / "mnist/entropy_sweep_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError("Run the complete MNIST stage before CIFAR confirmation")

    if args.stage == "cifar-if-needed":
        with summary_path.open() as handle:
            rows = json.load(handle)["rows"]
        decision = cifar_promotion_decision(rows)
        if not decision["promote_to_cifar"]:
            print("CIFAR confirmation is not needed: the frozen MNIST gate did not pass.")
            return
    else:
        print(
            "Running the complete frozen CIFAR confirmation suite "
            f"unconditionally: {CIFAR_CONFIRMATION_LEVELS}",
            flush=True,
        )

    run_levels("cifar10", CIFAR_CONFIRMATION_LEVELS, args)
    if not args.dry_run:
        refresh_stale_audits("cifar10", CIFAR_CONFIRMATION_LEVELS, args)
        aggregate("cifar10", CIFAR_CONFIRMATION_LEVELS, args)


if __name__ == "__main__":
    main()

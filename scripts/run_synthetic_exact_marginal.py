#!/usr/bin/env python3
"""Exact-marginal synthetic audit for Proposition 1.

The latent samples are always drawn from N(0, I).  Labels are then sampled
from a smooth softmax partition of the first K coordinates.  Consequently the
population marginal is unchanged for every kappa, while q(z | y) becomes
increasingly class-informative.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import multiprocessing
import os
from pathlib import Path
import platform
import subprocess
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.metrics.audit_protocol import (  # noqa: E402
    AUDIT_PROTOCOL_VERSION,
    conditional_mmd_to_standard_normal,
    hsic_permutation_test,
    mmd2_permutation_test,
    run_probe_suite,
)
from src.utils.provenance import sha256_file  # noqa: E402


QUICK_OUTPUT_DIR = "runs_diag/synthetic_exact_marginal"
PAPER_OUTPUT_DIR = "runs_diag/synthetic_exact_marginal_paper"
PARTIAL_SCHEMA_VERSION = "synthetic-exact-marginal-partial-1.0.0"
FROZEN_ACCEPTANCE = {
    "schema_version": "synthetic-exact-marginal-acceptance-1.0.0",
    "locked_before_paper_run": True,
    "required_replications": 50,
    "nominal_global_mmd_alpha": 0.05,
    "global_mmd_rejection_count_interval": [0, 6],
    "global_mmd_rejection_interval_basis": (
        "central 95% prediction interval for Binomial(n=50, p=0.05)"
    ),
    "global_mmd_within_replication_kappa_range_tolerance": 1e-12,
    "large_kappas": [4.0, 8.0, 16.0],
    "minimum_mean_probe_accuracy_at_each_large_kappa": 0.90,
    "trend_kappas": [0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0],
    "minimum_spearman_trend": 0.70,
    "require_high_kappa_hsic_above_null": True,
    "require_high_kappa_conditional_mmd_above_null": True,
    "interpretation": (
        "Acceptance checks estimator calibration and pedagogical visibility; "
        "it is not a test of the analytic identity q(z)=N(0,I), which holds "
        "by construction."
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=QUICK_OUTPUT_DIR)
    parser.add_argument("--dimension", type=int, default=32)
    parser.add_argument("--n-classes", type=int, default=2)
    parser.add_argument("--n-samples", type=int, default=768)
    parser.add_argument("--replications", type=int, default=5)
    parser.add_argument("--kappas", type=float, nargs="+", default=[0, 0.5, 1, 2, 4, 8, 16])
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--mmd-permutations", type=int, default=99)
    parser.add_argument("--hsic-permutations", type=int, default=99)
    parser.add_argument("--probe-epochs", type=int, default=200)
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Independent replication workers (paper default: up to 8).",
    )
    parser.add_argument(
        "--threads-per-worker",
        type=int,
        default=None,
        help="Torch CPU threads per replication worker (paper default: up to 8).",
    )
    parser.add_argument(
        "--hsic-permutation-batch-size",
        type=int,
        default=None,
        help="Number of HSIC null permutations evaluated per matrix multiply.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Reject an existing partial checkpoint instead of resuming it.",
    )
    parser.add_argument(
        "--paper",
        action="store_true",
        help="Use the predeclared paper setting: n=2048, 50 repetitions, 499 permutations.",
    )
    return parser.parse_args()


def configure_args(args: argparse.Namespace) -> argparse.Namespace:
    """Apply predeclared paper settings and execution-only defaults."""

    available_cpus = max(1, os.cpu_count() or 1)
    if args.paper:
        args.dimension = 128
        args.n_samples = 2048
        args.replications = 50
        args.mmd_permutations = 499
        args.hsic_permutations = 499
        args.probe_epochs = 300
        if args.output_dir == QUICK_OUTPUT_DIR:
            args.output_dir = PAPER_OUTPUT_DIR
    if args.workers is None:
        args.workers = min(8, max(1, available_cpus // 8)) if args.paper else 1
    if args.threads_per_worker is None:
        args.threads_per_worker = min(
            8, max(1, available_cpus // max(1, args.workers))
        )
    if args.hsic_permutation_batch_size is None:
        args.hsic_permutation_batch_size = 64 if args.paper else 32
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    if args.threads_per_worker < 1:
        raise ValueError("--threads-per-worker must be positive")
    if args.hsic_permutation_batch_size < 1:
        raise ValueError("--hsic-permutation-batch-size must be positive")
    return args


def _git_metadata() -> dict:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=ROOT, text=True
            ).strip()
        )
        return {"commit": commit, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def generate_problem(
    n_samples: int,
    dimension: int,
    n_classes: int,
    kappa: float,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Draw exact Gaussian z and smooth region labels p(y|z)."""

    if dimension < n_classes:
        raise ValueError("dimension must be at least n_classes")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    z = torch.randn((n_samples, dimension), generator=generator)
    probabilities = label_probabilities(z, n_classes, kappa)
    labels = torch.multinomial(probabilities, 1, generator=generator).squeeze(1)
    return z, labels


def label_probabilities(
    z: torch.Tensor,
    n_classes: int,
    kappa: float,
) -> torch.Tensor:
    """Smooth balanced label regions; binary mode matches Proposition 1."""

    if n_classes == 2:
        logits = torch.stack([-float(kappa) * z[:, 0], float(kappa) * z[:, 0]], dim=1)
    else:
        logits = float(kappa) * z[:, :n_classes]
    return torch.softmax(logits, dim=1)


def scientific_config(args: argparse.Namespace) -> dict:
    """Return the result-affecting configuration used to validate resume."""

    return {
        "audit_protocol_version": AUDIT_PROTOCOL_VERSION,
        "construction": (
            "z~N(0,I); binary y|z~Categorical(softmax([-kappa*z1,kappa*z1])); "
            "multiclass uses softmax(kappa*z[:K])"
        ),
        "dimension": int(args.dimension),
        "n_classes": int(args.n_classes),
        "n_samples": int(args.n_samples),
        "replications": int(args.replications),
        "kappas": [float(value) for value in args.kappas],
        "seed": int(args.seed),
        "mmd_permutations": int(args.mmd_permutations),
        "hsic_permutations": int(args.hsic_permutations),
        "probe_epochs": int(args.probe_epochs),
    }


def _atomic_json_dump(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_text_write(path: Path, contents: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w") as handle:
            handle.write(contents)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_result_save(path: Path, results: dict, manifest: dict) -> str:
    _atomic_json_dump(path, {"manifest": manifest, "results": results})
    digest = sha256_file(path)
    _atomic_text_write(
        path.with_suffix(path.suffix + ".sha256"),
        f"{digest}  {path.name}\n",
    )
    return digest


def _records_by_completed_replication(
    records: list[dict], kappas: list[float]
) -> set[int]:
    expected = set(kappas)
    observed: dict[int, set[float]] = {}
    for record in records:
        replication = int(record["replication"])
        observed.setdefault(replication, set()).add(float(record["kappa"]))
    invalid = {
        replication: values
        for replication, values in observed.items()
        if values != expected
    }
    if invalid:
        raise ValueError(
            "Partial checkpoint contains incomplete replication records: "
            f"{invalid}"
        )
    return set(observed)


def _load_partial_records(
    partial_path: Path,
    expected_config: dict,
    resume: bool,
) -> list[dict]:
    if not partial_path.exists():
        return []
    if not resume:
        raise FileExistsError(
            f"Partial checkpoint exists at {partial_path}; omit --no-resume "
            "or choose a new --output-dir."
        )
    with partial_path.open() as handle:
        payload = json.load(handle)
    if payload.get("schema_version") != PARTIAL_SCHEMA_VERSION:
        raise ValueError(f"Unsupported partial checkpoint schema: {partial_path}")
    if payload.get("scientific_config") != expected_config:
        raise ValueError(
            "Resume configuration mismatch. Use the original paper settings "
            "or choose a new --output-dir."
        )
    records = payload.get("records")
    if not isinstance(records, list):
        raise ValueError(f"Partial checkpoint has no records list: {partial_path}")
    _records_by_completed_replication(records, expected_config["kappas"])
    return records


def _save_partial_records(
    partial_path: Path,
    config: dict,
    records: list[dict],
) -> None:
    completed = sorted(_records_by_completed_replication(records, config["kappas"]))
    _atomic_json_dump(
        partial_path,
        {
            "schema_version": PARTIAL_SCHEMA_VERSION,
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            "scientific_config": config,
            "completed_replications": completed,
            "records": sorted(
                records,
                key=lambda row: (int(row["replication"]), float(row["kappa"])),
            ),
        },
    )


def run_replication(replication: int, config: dict, execution: dict) -> list[dict]:
    """Run one deterministic replication in a worker process."""

    torch.set_num_threads(int(execution["threads_per_worker"]))
    replication_seed = int(config["seed"]) + 100_000 * replication
    generator = torch.Generator(device="cpu").manual_seed(replication_seed)
    z = torch.randn(
        (int(config["n_samples"]), int(config["dimension"])),
        generator=generator,
    )
    reference = torch.randn(z.shape, generator=generator)
    global_mmd = mmd2_permutation_test(
        z,
        reference,
        seed=replication_seed + 1,
        n_permutations=int(config["mmd_permutations"]),
    )

    records = []
    for kappa_index, kappa in enumerate(config["kappas"]):
        label_generator = torch.Generator(device="cpu").manual_seed(
            replication_seed + 10_000 + kappa_index
        )
        probabilities = label_probabilities(z, int(config["n_classes"]), kappa)
        labels = torch.multinomial(
            probabilities, 1, generator=label_generator
        ).squeeze(1)
        if torch.unique(labels).numel() != int(config["n_classes"]):
            raise RuntimeError(
                "A class is absent; increase --n-samples or change the seed."
            )
        probe = run_probe_suite(
            z,
            labels,
            seed=replication_seed + kappa_index,
            probe_names=("logistic",),
            epochs=int(config["probe_epochs"]),
        )
        hsic = hsic_permutation_test(
            z,
            labels,
            seed=replication_seed + 20_000 + kappa_index,
            n_permutations=int(config["hsic_permutations"]),
            permutation_batch_size=int(execution["hsic_permutation_batch_size"]),
        )
        conditional = conditional_mmd_to_standard_normal(
            z,
            labels,
            int(config["n_classes"]),
            seed=replication_seed + 30_000 + kappa_index,
        )
        logistic = probe["models"]["logistic"]["test"]
        records.append(
            {
                "replication": replication,
                "seed": replication_seed,
                "kappa": float(kappa),
                "global_mmd2_u": global_mmd["statistic"],
                "global_mmd_p_value": global_mmd["p_value"],
                "global_mmd_null_q95": global_mmd["null_quantiles"]["q95"],
                "conditional_mmd2_u_mean": conditional["mean_mmd2_u"],
                "probe_accuracy": logistic["accuracy"],
                "probe_cross_entropy_nats": logistic["cross_entropy_nats"],
                "probe_information_lower_bound_nats": logistic[
                    "information_lower_bound_nats"
                ],
                "hsic": hsic["statistic"],
                "hsic_p_value": hsic["p_value"],
                "class_counts": conditional["class_counts"],
            }
        )
    return records


def summarize_records(records: list[dict], args: argparse.Namespace) -> dict:
    summaries = []
    for kappa in args.kappas:
        group = [record for record in records if record["kappa"] == float(kappa)]
        summary = {"kappa": float(kappa), "n_replications": len(group)}
        for key in (
            "global_mmd2_u",
            "global_mmd_p_value",
            "global_mmd_null_q95",
            "conditional_mmd2_u_mean",
            "probe_accuracy",
            "probe_information_lower_bound_nats",
            "hsic",
        ):
            values = np.asarray([record[key] for record in group], dtype=np.float64)
            summary[f"{key}_mean"] = float(values.mean())
            summary[f"{key}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        summary["global_mmd_rejection_rate_005"] = float(
            np.mean([record["global_mmd_p_value"] <= 0.05 for record in group])
        )
        summary["hsic_rejection_rate_005"] = float(
            np.mean([record["hsic_p_value"] <= 0.05 for record in group])
        )
        summaries.append(summary)
    return {
        "records": sorted(
            records,
            key=lambda row: (int(row["replication"]), float(row["kappa"])),
        ),
        "summary_by_kappa": summaries,
    }


def run_experiment(args: argparse.Namespace, output_dir: Path) -> dict:
    config = scientific_config(args)
    partial_path = output_dir / "partial_results.json"
    records = _load_partial_records(
        partial_path,
        expected_config=config,
        resume=not args.no_resume,
    )
    completed = _records_by_completed_replication(records, config["kappas"])
    pending = [
        replication
        for replication in range(args.replications)
        if replication not in completed
    ]
    if completed:
        print(
            f"Resuming {len(completed)}/{args.replications} completed replications "
            f"from {partial_path}",
            flush=True,
        )

    execution = {
        "threads_per_worker": int(args.threads_per_worker),
        "hsic_permutation_batch_size": int(args.hsic_permutation_batch_size),
    }

    def checkpoint(replication_records: list[dict]) -> None:
        records.extend(replication_records)
        _save_partial_records(partial_path, config, records)
        completed_count = len(
            _records_by_completed_replication(records, config["kappas"])
        )
        print(
            f"Completed and checkpointed replication "
            f"{completed_count}/{args.replications}",
            flush=True,
        )

    if args.workers == 1:
        for replication in pending:
            checkpoint(run_replication(replication, config, execution))
    elif pending:
        # PyTorch's optimizer performs a CUDA graph health check even for a
        # CPU model.  Forking after the parent imported CUDA-capable PyTorch
        # can therefore inherit an invalid CUDA runtime.  Spawn gives every
        # CPU worker a clean interpreter and is required for durable service
        # execution on GPU hosts.
        with ProcessPoolExecutor(
            max_workers=args.workers,
            mp_context=multiprocessing.get_context("spawn"),
        ) as executor:
            futures = {
                executor.submit(run_replication, replication, config, execution): replication
                for replication in pending
            }
            for future in as_completed(futures):
                replication = futures[future]
                try:
                    checkpoint(future.result())
                except Exception as error:
                    raise RuntimeError(
                        f"Synthetic replication {replication} failed"
                    ) from error

    completed = _records_by_completed_replication(records, config["kappas"])
    if completed != set(range(args.replications)):
        raise RuntimeError(
            f"Expected {args.replications} completed replications, found {len(completed)}"
        )
    return summarize_records(records, args)


def plot_results(results: dict, output_path: Path, n_classes: int) -> None:
    summary = results["summary_by_kappa"]
    kappas = np.asarray([row["kappa"] for row in summary])

    def values(name: str) -> tuple[np.ndarray, np.ndarray]:
        return (
            np.asarray([row[f"{name}_mean"] for row in summary]),
            np.asarray([row[f"{name}_std"] for row in summary]),
        )

    mmd, mmd_std = values("global_mmd2_u")
    null_q95, _ = values("global_mmd_null_q95")
    accuracy, accuracy_std = values("probe_accuracy")
    conditional, conditional_std = values("conditional_mmd2_u_mean")

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.6))
    axes[0].errorbar(kappas, mmd, yerr=mmd_std, marker="o", capsize=3)
    axes[0].plot(kappas, null_q95, linestyle="--", color="gray", label="null 95% quantile")
    axes[0].axhline(0.0, color="black", linewidth=0.7)
    rejection_rate = summary[0]["global_mmd_rejection_rate_005"]
    axes[0].set(
        title=r"By construction: $q(z)=\mathcal{N}(0,I)$",
        xlabel=r"partition strength $\kappa$",
        ylabel=r"global MMD$^2_U$",
    )
    axes[0].text(
        0.03,
        0.05,
        f"MMD rejection rate: {rejection_rate:.1%}",
        transform=axes[0].transAxes,
        fontsize=8,
    )
    axes[0].legend(frameon=False, fontsize=8)

    axes[1].errorbar(kappas, accuracy, yerr=accuracy_std, marker="o", capsize=3)
    axes[1].axhline(1.0 / n_classes, linestyle="--", color="gray", label="chance")
    axes[1].set(title="Recoverable label information", xlabel=r"partition strength $\kappa$", ylabel="linear probe accuracy", ylim=(0, 1.02))
    axes[1].legend(frameon=False, fontsize=8)

    axes[2].errorbar(kappas, conditional, yerr=conditional_std, marker="o", capsize=3)
    axes[2].axhline(0.0, color="black", linewidth=0.7)
    axes[2].set(title="Conditional mismatch", xlabel=r"partition strength $\kappa$", ylabel=r"mean conditional MMD$^2_U$")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _write_json_with_checksum(path: Path, payload: dict) -> str:
    _atomic_json_dump(path, payload)
    digest = sha256_file(path)
    _atomic_text_write(
        path.with_suffix(path.suffix + ".sha256"),
        f"{digest}  {path.name}\n",
    )
    return digest


def write_frozen_acceptance(output_dir: Path) -> Path:
    """Persist the acceptance contract before any paper result is evaluated."""

    path = output_dir / "acceptance_criteria.json"
    if path.exists():
        with path.open() as handle:
            existing = json.load(handle)
        if existing != FROZEN_ACCEPTANCE:
            raise ValueError(
                f"Frozen acceptance criteria mismatch at {path}; use a new output directory."
            )
    else:
        _write_json_with_checksum(path, FROZEN_ACCEPTANCE)
    return path


def _rankdata(values: np.ndarray) -> np.ndarray:
    """Average ranks for a small vector, including tied values."""

    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return ranks


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    ranked_x = _rankdata(np.asarray(x, dtype=np.float64))
    ranked_y = _rankdata(np.asarray(y, dtype=np.float64))
    if np.std(ranked_x) == 0.0 or np.std(ranked_y) == 0.0:
        return 0.0
    return float(np.corrcoef(ranked_x, ranked_y)[0, 1])


def evaluate_frozen_acceptance(results: dict, args: argparse.Namespace) -> dict:
    """Evaluate the predeclared paper-scale success and calibration checks."""

    summary = {
        float(row["kappa"]): row for row in results["summary_by_kappa"]
    }
    trend_kappas = [float(value) for value in FROZEN_ACCEPTANCE["trend_kappas"]]
    missing = [value for value in trend_kappas if value not in summary]
    if missing:
        raise ValueError(f"Acceptance evaluation is missing kappas: {missing}")

    hsic_means = np.asarray([summary[value]["hsic_mean"] for value in trend_kappas])
    conditional_means = np.asarray(
        [summary[value]["conditional_mmd2_u_mean_mean"] for value in trend_kappas]
    )
    hsic_spearman = _spearman(np.asarray(trend_kappas), hsic_means)
    conditional_spearman = _spearman(
        np.asarray(trend_kappas), conditional_means
    )

    large_kappas = [float(value) for value in FROZEN_ACCEPTANCE["large_kappas"]]
    large_probe_means = {
        str(value): float(summary[value]["probe_accuracy_mean"])
        for value in large_kappas
    }
    high_hsic_mean = float(np.mean([summary[value]["hsic_mean"] for value in large_kappas]))
    high_conditional_mean = float(
        np.mean(
            [summary[value]["conditional_mmd2_u_mean_mean"] for value in large_kappas]
        )
    )

    records_by_replication: dict[int, list[dict]] = {}
    for record in results["records"]:
        records_by_replication.setdefault(int(record["replication"]), []).append(record)
    global_ranges = {
        str(replication): float(
            np.ptp([float(row["global_mmd2_u"]) for row in records])
        )
        for replication, records in records_by_replication.items()
    }
    global_p_values = {}
    for replication, records in records_by_replication.items():
        values = {float(row["global_mmd_p_value"]) for row in records}
        if len(values) != 1:
            raise ValueError(
                f"Replication {replication} has kappa-dependent global MMD p-values"
            )
        global_p_values[replication] = values.pop()
    rejection_count = sum(
        value <= FROZEN_ACCEPTANCE["nominal_global_mmd_alpha"]
        for value in global_p_values.values()
    )
    rejection_lower, rejection_upper = FROZEN_ACCEPTANCE[
        "global_mmd_rejection_count_interval"
    ]

    checks = {
        "paper_replication_count": len(records_by_replication)
        == FROZEN_ACCEPTANCE["required_replications"],
        "probe_above_90pct_at_all_large_kappas": all(
            value
            > FROZEN_ACCEPTANCE["minimum_mean_probe_accuracy_at_each_large_kappa"]
            for value in large_probe_means.values()
        ),
        "hsic_positive_trend": hsic_spearman
        >= FROZEN_ACCEPTANCE["minimum_spearman_trend"],
        "conditional_mmd_positive_trend": conditional_spearman
        >= FROZEN_ACCEPTANCE["minimum_spearman_trend"],
        "high_kappa_hsic_above_null": high_hsic_mean > float(hsic_means[0]),
        "high_kappa_conditional_mmd_above_null": high_conditional_mean
        > float(conditional_means[0]),
        "global_mmd_identical_across_kappa_within_replication": max(
            global_ranges.values(), default=float("inf")
        )
        <= FROZEN_ACCEPTANCE[
            "global_mmd_within_replication_kappa_range_tolerance"
        ],
        "global_mmd_rejection_count_binomial_compatible": rejection_lower
        <= rejection_count
        <= rejection_upper,
    }
    return {
        "schema_version": "synthetic-exact-marginal-acceptance-result-1.0.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "criteria": FROZEN_ACCEPTANCE,
        "observed": {
            "n_replications": len(records_by_replication),
            "large_kappa_probe_accuracy_means": large_probe_means,
            "hsic_spearman_vs_kappa": hsic_spearman,
            "conditional_mmd_spearman_vs_kappa": conditional_spearman,
            "null_kappa_hsic_mean": float(hsic_means[0]),
            "large_kappa_hsic_mean": high_hsic_mean,
            "null_kappa_conditional_mmd_mean": float(conditional_means[0]),
            "large_kappa_conditional_mmd_mean": high_conditional_mean,
            "maximum_within_replication_global_mmd_range": max(
                global_ranges.values(), default=None
            ),
            "global_mmd_rejection_count": int(rejection_count),
            "global_mmd_rejection_rate": float(
                rejection_count / max(1, len(global_p_values))
            ),
        },
        "checks": checks,
        "passes_all": all(checks.values()),
        "interpretation": (
            "A failed empirical gate must be reported as a result; it does not "
            "invalidate the analytic marginal identity."
        ),
    }


def load_completed_result(path: Path, expected_config: dict) -> dict | None:
    if not path.exists():
        return None
    checksum_path = path.with_suffix(path.suffix + ".sha256")
    if not checksum_path.exists():
        raise ValueError(f"Completed result has no checksum: {path}")
    recorded_digest = checksum_path.read_text().split()[0]
    actual_digest = sha256_file(path)
    if recorded_digest != actual_digest:
        raise ValueError(f"Completed result checksum mismatch: {path}")
    with path.open() as handle:
        payload = json.load(handle)
    previous_config = payload.get("manifest", {}).get("evaluation_config", {}).get(
        "scientific_config"
    )
    if previous_config != expected_config:
        raise ValueError(
            f"Completed result configuration mismatch at {path}; use a new output directory."
        )
    completed = _records_by_completed_replication(
        payload["results"]["records"], expected_config["kappas"]
    )
    if completed != set(range(expected_config["replications"])):
        raise ValueError(f"Completed result is missing replications: {path}")
    return payload["results"]


def main() -> None:
    args = configure_args(parse_args())
    torch.set_num_threads(1)
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    criteria_path = write_frozen_acceptance(output_dir)
    config = scientific_config(args)
    json_path = output_dir / "results.json"
    results = load_completed_result(json_path, config)
    if results is None:
        results = run_experiment(args, output_dir)
    else:
        print(f"Reusing completed, checksum-verified result: {json_path}")

    evaluation_config = {
        "scientific_config": config,
        "execution": {
            "workers": args.workers,
            "threads_per_worker": args.threads_per_worker,
            "hsic_permutation_batch_size": args.hsic_permutation_batch_size,
            "resume_enabled": not args.no_resume,
        },
        "paper": bool(args.paper),
        "output_dir": str(args.output_dir),
        "acceptance_criteria_path": str(criteria_path),
    }
    manifest = {
        "schema_version": "audit-result-1.0.0",
        "audit_protocol_version": AUDIT_PROTOCOL_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": "synthetic_exact_marginal",
        "seed": args.seed,
        "evaluation_config": evaluation_config,
        "repository": _git_metadata(),
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": "cpu",
            "available_cpus": os.cpu_count(),
        },
    }
    digest = _atomic_result_save(json_path, results, manifest)
    figure_path = output_dir / "exact_marginal_vs_leakage.png"
    plot_results(results, figure_path, args.n_classes)
    acceptance_path = output_dir / "acceptance_result.json"
    if args.paper:
        acceptance = evaluate_frozen_acceptance(results, args)
    else:
        acceptance = {
            "schema_version": "synthetic-exact-marginal-acceptance-result-1.0.0",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "criteria": FROZEN_ACCEPTANCE,
            "status": "not_evaluated",
            "passes_all": None,
            "reason": "Frozen acceptance is evaluated only for --paper runs.",
        }
    _write_json_with_checksum(acceptance_path, acceptance)
    print(f"Results: {json_path}")
    print(f"Figure:  {figure_path}")
    print(f"Acceptance: {acceptance_path} (passes_all={acceptance['passes_all']})")
    print(f"SHA-256: {digest}")


if __name__ == "__main__":
    main()

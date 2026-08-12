#!/usr/bin/env python3
"""Exact-marginal synthetic audit for Proposition 1.

The latent samples are always drawn from N(0, I).  Labels are then sampled
from a smooth softmax partition of the first K coordinates.  Consequently the
population marginal is unchanged for every kappa, while q(z | y) becomes
increasingly class-informative.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
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
from src.utils.provenance import save_result_with_manifest  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="runs_diag/synthetic_exact_marginal")
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
        "--paper",
        action="store_true",
        help="Use the predeclared paper setting: n=2048, 50 repetitions, 499 permutations.",
    )
    return parser.parse_args()


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


def run_experiment(args: argparse.Namespace) -> dict:
    records = []
    for replication in range(args.replications):
        replication_seed = args.seed + 100_000 * replication
        generator = torch.Generator(device="cpu").manual_seed(replication_seed)
        z = torch.randn((args.n_samples, args.dimension), generator=generator)
        reference = torch.randn((args.n_samples, args.dimension), generator=generator)
        global_mmd = mmd2_permutation_test(
            z,
            reference,
            seed=replication_seed + 1,
            n_permutations=args.mmd_permutations,
        )

        for kappa_index, kappa in enumerate(args.kappas):
            label_generator = torch.Generator(device="cpu").manual_seed(
                replication_seed + 10_000 + kappa_index
            )
            probabilities = label_probabilities(z, args.n_classes, kappa)
            labels = torch.multinomial(
                probabilities, 1, generator=label_generator
            ).squeeze(1)
            if torch.unique(labels).numel() != args.n_classes:
                raise RuntimeError(
                    "A class is absent; increase --n-samples or change the seed."
                )
            probe = run_probe_suite(
                z,
                labels,
                seed=replication_seed + kappa_index,
                probe_names=("logistic",),
                epochs=args.probe_epochs,
            )
            hsic = hsic_permutation_test(
                z,
                labels,
                seed=replication_seed + 20_000 + kappa_index,
                n_permutations=args.hsic_permutations,
            )
            conditional = conditional_mmd_to_standard_normal(
                z,
                labels,
                args.n_classes,
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
        print(f"Completed replication {replication + 1}/{args.replications}")

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
    return {"records": records, "summary_by_kappa": summaries}


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


def main() -> None:
    args = parse_args()
    if args.paper:
        args.dimension = 128
        args.n_samples = 2048
        args.replications = 50
        args.mmd_permutations = 499
        args.hsic_permutations = 499
        args.probe_epochs = 300
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    results = run_experiment(args)
    config = vars(args).copy()
    config["construction"] = (
        "z~N(0,I); binary y|z~Categorical(softmax([-kappa*z1,kappa*z1])); "
        "multiclass uses softmax(kappa*z[:K])"
    )
    manifest = {
        "schema_version": "audit-result-1.0.0",
        "audit_protocol_version": AUDIT_PROTOCOL_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": "synthetic_exact_marginal",
        "seed": args.seed,
        "evaluation_config": config,
        "repository": _git_metadata(),
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": "cpu",
        },
    }
    json_path = output_dir / "results.json"
    digest = save_result_with_manifest(json_path, results, manifest)
    figure_path = output_dir / "exact_marginal_vs_leakage.png"
    plot_results(results, figure_path, args.n_classes)
    print(f"Results: {json_path}")
    print(f"Figure:  {figure_path}")
    print(f"SHA-256: {digest}")


if __name__ == "__main__":
    main()

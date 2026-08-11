"""Aggregate independent-evaluator latent-swap runs into a paper figure."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_result(path: Path) -> dict:
    payload = json.loads(path.read_text())
    result = payload.get("results", payload)
    if result.get("evaluator") != "external_pixel_classifier":
        raise ValueError(f"{path} is not an independent-evaluator swap result")
    return result


def paired_metrics(result: dict) -> np.ndarray:
    content = np.asarray(result["content_rate"], dtype=np.float64)
    style = np.asarray(result["style_rate"], dtype=np.float64)
    neither = np.asarray(result["neither_rate"], dtype=np.float64)
    mask = ~np.eye(content.shape[0], dtype=bool)
    return np.stack([content[mask], style[mask], neither[mask]], axis=1)


def bootstrap_summary(values: np.ndarray, seed: int, repetitions: int = 5000) -> dict:
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, values.shape[0], size=(repetitions, values.shape[0]))
    bootstrap = values[indices].mean(axis=1)
    metric_names = ("content_following", "style_following", "neither")
    return {
        name: {
            "mean": float(values[:, index].mean()),
            "pair_bootstrap_ci95": [
                float(np.quantile(bootstrap[:, index], 0.025)),
                float(np.quantile(bootstrap[:, index], 0.975)),
            ],
        }
        for index, name in enumerate(metric_names)
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mnist-baseline", required=True, type=Path)
    parser.add_argument("--mnist-remedy", required=True, type=Path)
    parser.add_argument("--cifar-baseline", required=True, type=Path)
    parser.add_argument("--cifar-remedy", required=True, type=Path)
    parser.add_argument("--figure-out", required=True, type=Path)
    parser.add_argument("--json-out", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    paths = {
        "MNIST": {"global MMD": args.mnist_baseline, "per-class MMD": args.mnist_remedy},
        "CIFAR-10": {"global MMD": args.cifar_baseline, "per-class MMD": args.cifar_remedy},
    }
    results = {
        dataset: {setting: load_result(path) for setting, path in settings.items()}
        for dataset, settings in paths.items()
    }
    summaries = {
        dataset: {
            setting: bootstrap_summary(paired_metrics(result), args.seed)
            for setting, result in settings.items()
        }
        for dataset, settings in results.items()
    }

    colors = {
        "content_following": "#4c78a8",
        "style_following": "#e45756",
        "neither": "#9d9d9d",
    }
    display_labels = {
        "content_following": "follows semantic donor",
        "style_following": "follows style donor",
        "neither": "neither",
    }
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.05), sharey=True)
    for ax, dataset in zip(axes, ("MNIST", "CIFAR-10")):
        x_positions = np.arange(2)
        width = 0.22
        for metric_index, metric in enumerate(
            ("content_following", "style_following", "neither")
        ):
            means = [
                summaries[dataset][setting][metric]["mean"]
                for setting in ("global MMD", "per-class MMD")
            ]
            intervals = [
                summaries[dataset][setting][metric]["pair_bootstrap_ci95"]
                for setting in ("global MMD", "per-class MMD")
            ]
            lower = np.asarray(means) - np.asarray([item[0] for item in intervals])
            upper = np.asarray([item[1] for item in intervals]) - np.asarray(means)
            ax.bar(
                x_positions + (metric_index - 1) * width,
                means,
                width,
                color=colors[metric],
                label=display_labels[metric],
                yerr=np.vstack([lower, upper]),
                capsize=2,
                error_kw={"linewidth": 0.8},
            )
        ax.set_xticks(x_positions, [r"$\delta=0$", r"$\delta=1$"])
        ax.set_ylim(0.0, 1.05)
        ax.set_title(dataset)
        ax.set_xlabel("style objective")
        ax.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("off-diagonal swap rate")
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, legend_labels, loc="upper center", ncol=3, frameon=False,
        bbox_to_anchor=(0.5, 1.04), fontsize=8,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    args.figure_out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.figure_out, dpi=300, bbox_inches="tight")
    plt.close(fig)

    output = {
        "protocol": (
            "Deterministic posterior-mean swaps; independent pixel classifier; "
            "95% percentile bootstrap over 90 ordered off-diagonal class pairs."
        ),
        "inputs": {
            dataset: {setting: str(path) for setting, path in settings.items()}
            for dataset, settings in paths.items()
        },
        "summary": summaries,
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summaries, indent=2))
    print(f"Saved {args.figure_out} and {args.json_out}")


if __name__ == "__main__":
    main()

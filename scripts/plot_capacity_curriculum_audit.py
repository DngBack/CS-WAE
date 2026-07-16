"""
plot_capacity_curriculum_audit.py

Grouped bar chart for Experiment 7 (Section~\\ref{sec:exp7}): linear-probe
accuracy LP(z_s -> y) across the baseline plus 6 single-variable
perturbations (capacity x2, curriculum, prior geometry x2, supervision),
MNIST vs CIFAR-10 side by side. Visualizes the paper's central finding that
leakage survives every perturbation on both datasets, but capacity-related
perturbations (smaller style, balanced dims) have a much larger effect on
CIFAR-10 than on MNIST.

Reads runs_diag/capacity_curriculum_audit/*.json and
runs_diag/joint_mmd/{mnist,cifar10}_delta0.json (baseline).

Usage: python scripts/plot_capacity_curriculum_audit.py
"""
from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update(
    {
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.fontsize": 10,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    }
)

ROOT = Path(__file__).resolve().parent.parent
DIAG = ROOT / "runs_diag" / "capacity_curriculum_audit"
JOINT = ROOT / "runs_diag" / "joint_mmd"
OUT = ROOT / "paper-f-cs-wae" / "figures" / "capacity_curriculum_audit_lp.png"

PERTURBATIONS = [
    ("Baseline\n(none)", None),
    ("Balanced dims\n($d_c{=}d_s{=}64$)", "balanced_dc64_ds64"),
    ("Smaller style\n($d_s{=}32$)", "smallstyle_dc64_ds32"),
    ("No warmup", "nowarmup"),
    ("Gaussian\nprior", "gaussian_class_prior"),
    ("vMF\nprior", "vmf_class_prior"),
    ("No\nclassifier", "no_classifier"),
]


def load_lp(dataset: str, tag: str | None) -> float:
    if tag is None:
        path = JOINT / f"{dataset}_delta0.json"
    else:
        path = DIAG / f"{dataset}_{tag}_delta0.json"
    with path.open() as f:
        return json.load(f)["lp_accuracy"] * 100.0


def main() -> None:
    labels = [p[0] for p in PERTURBATIONS]
    mnist_lp = [load_lp("mnist", p[1]) for p in PERTURBATIONS]
    cifar_lp = [load_lp("cifar10", p[1]) for p in PERTURBATIONS]

    x = np.arange(len(labels))
    width = 0.36

    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    b1 = ax.bar(x - width / 2, mnist_lp, width, label="MNIST", color="#4C72B0")
    b2 = ax.bar(x + width / 2, cifar_lp, width, label="CIFAR-10", color="#DD8452")

    ax.axhline(10, color="gray", linestyle="--", linewidth=1, label="Chance (10%)")

    for bars in (b1, b2):
        for rect in bars:
            h = rect.get_height()
            ax.annotate(f"{h:.0f}", xy=(rect.get_x() + rect.get_width() / 2, h),
                        xytext=(0, 2), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8.5)

    ax.set_ylabel("Linear probe accuracy LP($z_s \\to y$)  [%]")
    ax.set_ylim(0, 112)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.32), ncol=3, frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title("Experiment 7: leakage (LP) survives every perturbation on both datasets,\n"
                 "but capacity-related perturbations affect CIFAR-10 far more than MNIST")

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.28)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT)
    print(f"Saved: {OUT}")


if __name__ == "__main__":
    main()

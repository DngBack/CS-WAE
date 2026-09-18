#!/usr/bin/env python3
"""Render the v26 FACT trade-off panel from the archived matched-pair summary."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "runs_diag/fact/mean_hsic_mnist_40e/submission_followup_v1/summary.json"
OUTPUT = ROOT / "fcswae/figures/v26_fact_tradeoff.pdf"


def mean_sd(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    return float(array.mean()), float(array.std(ddof=1))


def main() -> None:
    payload = json.loads(SOURCE.read_text(encoding="utf-8"))
    rows = payload["matched_rows"]
    controls = sorted((r for r in rows if r["kind"] == "control"), key=lambda r: r["training_seed"])
    facts = sorted((r for r in rows if r["kind"] == "fact"), key=lambda r: r["training_seed"])
    assert [r["training_seed"] for r in controls] == [r["training_seed"] for r in facts]

    categories = ["content donor", "style donor", "neither"]
    keys = ["swap_content_following", "swap_style_following", "swap_neither_following"]
    ctrl_stats = [mean_sd([100 * row[key] for row in controls]) for key in keys]
    fact_stats = [mean_sd([100 * row[key] for row in facts]) for key in keys]

    paired = payload["paired_delta_aggregate"]
    effects = [
        ("joint MMD\nreduction (%)", 100 * paired["joint_mmd_reduction"]["mean"], 100 * paired["joint_mmd_reduction"]["sample_sd"]),
        ("conditional MMD\nreduction (%)", 100 * paired["conditional_mmd_reduction"]["mean"], 100 * paired["conditional_mmd_reduction"]["sample_sd"]),
        ("content-donor\nfollowing", 100 * paired["swap_content_following_delta"]["mean"], 100 * paired["swap_content_following_delta"]["sample_sd"]),
        ("style-donor\nfollowing", 100 * paired["swap_style_following_delta"]["mean"], 100 * paired["swap_style_following_delta"]["sample_sd"]),
        ("generation\naccuracy", 100 * paired["global_gen_accuracy_delta"]["mean"], 100 * paired["global_gen_accuracy_delta"]["sample_sd"]),
        ("reconstruction\n(relative %)", -100 * paired["reconstruction_relative_change"]["mean"], 100 * paired["reconstruction_relative_change"]["sample_sd"]),
    ]

    plt.rcParams.update({
        "font.size": 8.2,
        "axes.titlesize": 9.3,
        "axes.labelsize": 8.5,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7.8,
        "pdf.fonttype": 42,
    })
    fig, axes = plt.subplots(1, 2, figsize=(7.05, 1.82), gridspec_kw={"width_ratios": [0.86, 1.55]})

    x = np.arange(len(categories))
    width = 0.36
    axes[0].bar(x - width / 2, [m for m, _ in ctrl_stats], width,
                yerr=[s for _, s in ctrl_stats], capsize=2, label="Control", color="#6b7f99")
    axes[0].bar(x + width / 2, [m for m, _ in fact_stats], width,
                yerr=[s for _, s in fact_stats], capsize=2, label="FACT", color="#d88746")
    axes[0].set_title("Posterior-swap attribution")
    axes[0].set_ylabel("off-diagonal swaps (%)")
    axes[0].set_xticks(x, categories, rotation=17, ha="right")
    axes[0].set_ylim(0, 80)
    axes[0].legend(frameon=False, ncol=2, loc="upper center")
    axes[0].grid(axis="y", alpha=0.2, linewidth=0.6)

    names = [item[0].replace("\n", " ") for item in effects]
    means = np.asarray([item[1] for item in effects])
    sds = np.asarray([item[2] for item in effects])
    colors = ["#3a8f6b" if value >= 0 else "#c75b5b" for value in means]
    positions = np.arange(len(effects))
    axes[1].barh(positions, means, xerr=sds, capsize=2, color=colors)
    axes[1].axvline(0, color="#333333", linewidth=0.8)
    axes[1].set_title("Matched FACT $-$ control (three training seeds)")
    axes[1].set_xlabel("signed effect; pp unless marked (%)")
    axes[1].set_yticks(positions, names)
    axes[1].invert_yaxis()
    axes[1].set_xlim(-43, 50)
    axes[1].grid(axis="x", alpha=0.2, linewidth=0.6)
    for index, value in enumerate(means):
        axes[1].text(value + (1.5 if value >= 0 else -1.5), index, f"{value:+.1f}",
                     ha="left" if value >= 0 else "right", va="center", fontsize=7.2)

    fig.tight_layout(w_pad=1.2)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, bbox_inches="tight")
    print(OUTPUT)


if __name__ == "__main__":
    main()

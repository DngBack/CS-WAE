"""
wrong_style_rate_diagnostics.py

Wrong-style-rate / off-manifold sampling diagnostic for F-CS-WAE.

A pure latent-space diagnostic, run BEFORE any image is decoded: train a
linear probe on real encoded z_s -> y, then for a requested class k, draw
z_s from a given sampling strategy (naive global_gaussian, or one of the
post-hoc class-conditional strategies already used for generation) and ask
the probe what class that z_s "looks like". This turns "the sampler may be
drawing an incompatible style code" into a concrete confusion matrix and a
per-class wrong-style rate, independent of decoder/classifier quality.

Note on the naive global_gaussian strategy: it draws z_s ~ N(0,I)
independently of the requested class k, so every row k of its confusion
matrix is drawn from the same distribution -- rows will look near-identical
up to sampling noise. That is the expected, self-explanatory result (it
reflects only how the probe carves up R^{d_s}, not any real class
conditioning), not a bug; it is why the class-conditional strategies
(class_mean, class_diag_t*, class_empirical) are compared alongside it.

Usage
-----
python scripts/wrong_style_rate_diagnostics.py \\
    --checkpoint runs_f/mnist/seed_0/f_cs_wae_model.pth \\
    --dataset mnist --tag mnist_delta0 --device cpu \\
    --strategies global_gaussian class_mean class_diag_t0.25 class_empirical
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

plt.rcParams.update(
    {
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    }
)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.compute_leakage_diagnostics import (
    load_fcswae_checkpoint,
    get_eval_subset_loader,
    extract_full_encodings,
)
from scripts.analyze_fcswae_sampling_strategies import collect_style_stats, sample_zs
from src.metrics.audit_protocol import DEFAULT_EVAL_SAMPLES, fit_logistic_probe, stratified_probe_split
from src.utils.provenance import build_manifest, save_result_with_manifest
from src.utils.seed import set_seed


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def train_probe_on_real_zs(model, dataset, device, n_samples, seed, split="test"):
    """Fit a mean-view probe with a stratified 60/20/20 split."""
    loader = get_eval_subset_loader(dataset, n_samples, batch_size=256, seed=seed, split=split)
    _mu_c, mu_s, labels = extract_full_encodings(model, loader, device)
    indices = stratified_probe_split(labels, seed=seed)
    validation_accuracy, probe = fit_logistic_probe(
        mu_s[indices.train], labels[indices.train],
        mu_s[indices.validation], labels[indices.validation],
        seed=seed,
    )
    probe = probe.to(device)
    with torch.no_grad():
        predictions = probe(mu_s[indices.test].to(device)).argmax(1).cpu()
        test_accuracy = float((predictions == labels[indices.test]).float().mean().item())
    return probe, validation_accuracy, test_accuracy, indices.metadata()


@torch.no_grad()
def evaluate_strategy(model, stats, strategy, probe, n_classes, n_probe_draws, device):
    """Confusion matrix between requested class k (rows) and probe-predicted style class (cols)."""
    confusion = np.zeros((n_classes, n_classes))
    for k in range(n_classes):
        z_s_k = sample_zs(strategy, k, n_probe_draws, stats, device)
        pred_k = probe(z_s_k).argmax(dim=1).cpu().numpy()
        for c in range(n_classes):
            confusion[k, c] = float((pred_k == c).mean())
    style_compatible_rate = np.diag(confusion)
    wrong_style_rate = 1.0 - style_compatible_rate
    return confusion, style_compatible_rate, wrong_style_rate


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_confusion(confusion, title, out_path):
    n_classes = confusion.shape[0]
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(confusion, cmap="viridis", vmin=0.0, vmax=1.0)
    ax.set_xlabel("probe-predicted style class")
    ax.set_ylabel("requested class k")
    ax.set_title(title, fontsize=10)
    ax.set_xticks(range(n_classes))
    ax.set_yticks(range(n_classes))
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_grouped_bar(per_strategy, n_classes, title, out_path):
    strategies = list(per_strategy.keys())
    n_strat = len(strategies)
    width = 0.8 / max(n_strat, 1)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(n_classes)
    for i, strat in enumerate(strategies):
        vals = per_strategy[strat]["wrong_style_rate"]
        ax.bar(x + i * width - 0.4 + width / 2, vals, width=width, label=strat)
    ax.set_xlabel("requested class k")
    ax.set_ylabel("wrong-style rate")
    ax.set_title(title, fontsize=10)
    ax.set_xticks(x)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Wrong-style-rate / off-manifold sampling diagnostic")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--dataset", required=True, choices=["mnist", "fashion_mnist", "cifar10"])
    p.add_argument("--tag", required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--n-samples", type=int, default=DEFAULT_EVAL_SAMPLES,
                   help=f"Real mu_s samples used by the probe protocol (default: {DEFAULT_EVAL_SAMPLES})")
    p.add_argument("--n-probe-draws", type=int, default=2000, help="Synthetic z_s draws per class per strategy")
    p.add_argument("--style-bank-max", type=int, default=12000)
    p.add_argument(
        "--strategies", nargs="+", default=["global_gaussian"],
        help="Any of: global_gaussian, class_mean, class_diag_t0.25, class_diag_t0.50, "
             "class_diag_t1.00, class_empirical",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-dir", default=None, help="Defaults to runs_diag/wrong_style_rate/<tag>/")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    set_seed(args.seed)

    out_dir = Path(args.output_dir) if args.output_dir else ROOT / "runs_diag" / "wrong_style_rate" / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model from {args.checkpoint} ...")
    model = load_fcswae_checkpoint(args.checkpoint, args.dataset, device)
    n_classes = model.n_classes

    print(f"Training linear probe on real z_s -> y (n={args.n_samples}, seed={args.seed}) ...")
    probe, val_acc, test_acc, probe_split = train_probe_on_real_zs(
        model, args.dataset, device, args.n_samples, args.seed
    )
    print(f"Probe validation/test accuracy on real mu_s: {val_acc:.4f}/{test_acc:.4f}")

    print(f"Collecting style stats for class-conditional strategies (max={args.style_bank_max}) ...")
    stats = collect_style_stats(model, args.dataset, device, args.style_bank_max, batch_size=256)

    per_strategy = {}
    for strategy in args.strategies:
        print(f"Evaluating strategy: {strategy} ...")
        confusion, compat_rate, wrong_rate = evaluate_strategy(
            model, stats, strategy, probe, n_classes, args.n_probe_draws, device
        )
        per_strategy[strategy] = {
            "confusion_matrix": confusion.tolist(),
            "style_compatible_rate": compat_rate.tolist(),
            "wrong_style_rate": wrong_rate.tolist(),
            "mean_style_compatible_rate": float(compat_rate.mean()),
            "mean_wrong_style_rate": float(wrong_rate.mean()),
        }
        plot_confusion(
            confusion,
            f"Requested class vs probe-implied style class ({strategy}) — {args.tag}",
            out_dir / f"confusion_{strategy}.png",
        )
        print(f"  mean style-compatible rate = {compat_rate.mean():.4f}")

    results = {
        "checkpoint": str(args.checkpoint),
        "dataset": args.dataset,
        "tag": args.tag,
        "n_classes": n_classes,
        "latent_view": "mu_s",
        "probe_validation_accuracy_on_real_mu_s": val_acc,
        "probe_test_accuracy_on_real_mu_s": test_acc,
        "probe_split": probe_split,
        "strategies": per_strategy,
    }
    manifest = build_manifest(
        repository_root=ROOT,
        checkpoint_path=args.checkpoint,
        dataset=args.dataset,
        seed=args.seed,
        evaluation_config={
            "n_samples": args.n_samples,
            "n_probe_draws": args.n_probe_draws,
            "style_bank_max": args.style_bank_max,
            "strategies": args.strategies,
            "latent_view": "mu_s",
            "probe_split": [0.60, 0.20, 0.20],
        },
        model_config={
            "semantic_dim": model.semantic_dim,
            "style_dim": model.style_dim,
            "n_classes": n_classes,
        },
    )
    save_result_with_manifest(out_dir / "results.json", results, manifest)

    with (out_dir / "results.csv").open("w", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["strategy", "class", "style_compatible_rate", "wrong_style_rate"])
        for strategy, row in per_strategy.items():
            for k in range(n_classes):
                writer.writerow(
                    [strategy, k, row["style_compatible_rate"][k], row["wrong_style_rate"][k]]
                )

    plot_grouped_bar(
        per_strategy, n_classes,
        f"Wrong-style rate by requested class and sampling strategy — {args.tag}",
        out_dir / "wrong_style_rate_grouped_bar.png",
    )

    print(f"Saved outputs to {out_dir}")


if __name__ == "__main__":
    main()

"""
conditional_mmd_diagnostics.py

Conditional-MMD / pairwise mismatch diagnostic for F-CS-WAE's style latent
z_s. Global MMD (already reported elsewhere) only checks the marginal
q(z_s) against N(0,I); this script checks the conditionals directly:

  1. Per-class MMD(q(z_s|y=k), N(0,I))       -- should be >> global MMD if
                                                 the marginal masks per-class
                                                 structure (the paper's core
                                                 claim).
  2. Per-class MMD(q(z_s|y=k), q(z_s))       -- distance of each class from
                                                 the overall mixture.
  3. Pairwise MMD(q(z_s|y=i), q(z_s|y=j))    -- K x K matrix; large
                                                 off-diagonal values mean the
                                                 classes occupy visibly
                                                 different regions of z_s.
  4. A 1D projection: PCA on the K x d_s matrix of class means (the same
     signal Delta_inter already summarizes into one scalar), used to plot
     per-class overlaid histograms of z_s projected onto the top direction.

Uses sampled z_s (via extract_style_latents, with reparameterization noise),
not the deterministic mu_s, so global_mmd/delta_inter computed here are a
built-in cross-check against the already-committed
runs_diag/fcswae_*_baseline/metrics.json values (same computation).

Usage
-----
python scripts/conditional_mmd_diagnostics.py \\
    --checkpoint runs_f/mnist/seed_0/f_cs_wae_model.pth \\
    --dataset mnist --tag mnist_delta0 --device cpu
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
    extract_latent_views,
    mmd2_rbf,
    compute_global_mmd,
    compute_delta_inter,
)
from src.metrics.audit_protocol import (
    DEFAULT_EVAL_SAMPLES,
    conditional_mmd_to_standard_normal,
)
from src.utils.provenance import build_manifest, save_result_with_manifest
from src.utils.seed import set_seed


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def compute_conditional_mmd(
    z_s: torch.Tensor, labels: torch.Tensor, n_classes: int, seed: int = 0
):
    """Per-class MMD to prior/marginal, and pairwise conditional MMD."""
    mmd_to_prior = np.zeros(n_classes)
    mmd_to_marginal = np.zeros(n_classes)
    class_z = {}

    canonical_prior = conditional_mmd_to_standard_normal(
        z_s,
        labels,
        n_classes,
        seed=seed + 11_000,
    )
    for class_key, value in canonical_prior["per_class_mmd2_u"].items():
        mmd_to_prior[int(class_key)] = value

    for k in range(n_classes):
        mask = labels == k
        z_k = z_s[mask]
        class_z[k] = z_k
        if z_k.shape[0] < 2:
            continue
        mmd_to_marginal[k] = mmd2_rbf(z_k, z_s)

    pairwise_mmd = np.zeros((n_classes, n_classes))
    for i in range(n_classes):
        for j in range(i + 1, n_classes):
            if class_z[i].shape[0] < 2 or class_z[j].shape[0] < 2:
                continue
            val = mmd2_rbf(class_z[i], class_z[j])
            pairwise_mmd[i, j] = val
            pairwise_mmd[j, i] = val

    return mmd_to_prior, mmd_to_marginal, pairwise_mmd


def compute_class_mean_projection(z_s: torch.Tensor, labels: torch.Tensor, n_classes: int):
    """Top PCA direction of the K x d_s class-mean matrix, and per-class projections.

    Chosen over a full Fisher-LDA direction because it needs no within-class
    covariance inversion (ill-conditioned here: ~n_samples/n_classes examples
    per class vs. d_s dimensions) and is directly consistent with how
    Delta_inter itself is defined (mean pairwise distance between class
    means) -- this answers "what single direction best captures the same
    signal Delta_inter already summarizes into one scalar."
    """
    class_means = torch.stack([
        z_s[labels == k].mean(dim=0) if (labels == k).sum() > 0
        else torch.zeros(z_s.shape[1])
        for k in range(n_classes)
    ])
    centered = class_means - class_means.mean(dim=0, keepdim=True)
    U, S, Vt = torch.linalg.svd(centered, full_matrices=False)
    w = Vt[0]
    explained_ratio = (S[0] ** 2 / (S ** 2).sum()).item() if (S ** 2).sum() > 0 else 0.0

    proj = (z_s @ w).detach().cpu().numpy()
    return w.detach().cpu().numpy(), explained_ratio, proj


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_pairwise_heatmap(matrix, title, out_path):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(matrix, cmap="magma")
    ax.set_xlabel("class j")
    ax.set_ylabel("class i")
    ax.set_title(title, fontsize=10)
    ax.set_xticks(range(matrix.shape[1]))
    ax.set_yticks(range(matrix.shape[0]))
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_per_class_bar(mmd_to_prior, global_mmd, title, out_path):
    n_classes = len(mmd_to_prior)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(range(n_classes), mmd_to_prior, color="tab:blue", label="MMD(q(z_s|y=k), N(0,I))")
    ax.axhline(global_mmd, color="tab:red", linestyle="--", label=f"Global MMD = {global_mmd:.4f}")
    ax.set_xlabel("class k")
    ax.set_ylabel("MMD²")
    ax.set_title(title, fontsize=10)
    ax.set_xticks(range(n_classes))
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_projection_histogram(proj, labels, explained_ratio, title, out_path):
    n_classes = int(labels.max().item()) + 1
    fig, ax = plt.subplots(figsize=(7, 4))
    labels_np = labels.detach().cpu().numpy()
    cmap = plt.get_cmap("tab10")
    for k in range(n_classes):
        vals = proj[labels_np == k]
        if vals.size == 0:
            continue
        ax.hist(vals, bins=40, alpha=0.5, color=cmap(k % 10), label=str(k), density=True)
    ax.set_xlabel(f"projection onto top class-mean PCA direction (explained ratio={explained_ratio:.2f})")
    ax.set_ylabel("density")
    ax.set_title(title, fontsize=10)
    ax.legend(title="class", ncol=2, fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Conditional-MMD / pairwise mismatch diagnostic for F-CS-WAE")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--dataset", required=True, choices=["mnist", "fashion_mnist", "cifar10"])
    p.add_argument("--tag", required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--split", default="test", choices=["train", "test"])
    p.add_argument("--n-samples", type=int, default=DEFAULT_EVAL_SAMPLES,
                   help=f"Stage-0 evaluation subset size (default: {DEFAULT_EVAL_SAMPLES})")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-dir", default=None, help="Defaults to runs_diag/conditional_mmd/<tag>/")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    set_seed(args.seed)

    out_dir = Path(args.output_dir) if args.output_dir else ROOT / "runs_diag" / "conditional_mmd" / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model from {args.checkpoint} ...")
    model = load_fcswae_checkpoint(args.checkpoint, args.dataset, device)
    n_classes = model.n_classes

    print(f"Loading {args.dataset} {args.split} subset (n={args.n_samples}, seed={args.seed}) ...")
    loader = get_eval_subset_loader(
        args.dataset, args.n_samples, batch_size=256, seed=args.seed, split=args.split
    )
    views = extract_latent_views(model, loader, device, model_type="fcswae", seed=args.seed)
    z_s = views["z_s_sample"].to(device)
    labels = views["labels"].to(device)

    print("Computing global MMD / Delta_inter (cross-check vs committed baseline) ...")
    global_mmd_reference_seed = args.seed + 10_000
    global_mmd = compute_global_mmd(z_s, seed=global_mmd_reference_seed)
    delta_inter = compute_delta_inter(z_s, labels, n_classes)

    print("Computing per-class and pairwise conditional MMD ...")
    mmd_to_prior, mmd_to_marginal, pairwise_mmd = compute_conditional_mmd(
        z_s, labels, n_classes, seed=args.seed
    )

    print("Computing class-mean PCA projection ...")
    w, explained_ratio, proj = compute_class_mean_projection(z_s, labels, n_classes)

    results = {
        "checkpoint": str(args.checkpoint),
        "dataset": args.dataset,
        "tag": args.tag,
        "n_classes": n_classes,
        "latent_view": "z_s_sample",
        "mmd_estimator": "unbiased U-statistic; negative finite-sample estimates retained",
        "global_mmd_reference_seed": global_mmd_reference_seed,
        "global_mmd": global_mmd,
        "delta_inter": delta_inter,
        "mmd_to_prior": mmd_to_prior.tolist(),
        "mmd_to_marginal": mmd_to_marginal.tolist(),
        "pairwise_mmd": pairwise_mmd.tolist(),
        "projection_explained_ratio": explained_ratio,
        "summary": {
            "mean_per_class_mmd_to_prior": float(mmd_to_prior.mean()),
            "mean_pairwise_mmd": float(pairwise_mmd[~np.eye(n_classes, dtype=bool)].mean()),
        },
    }
    manifest = build_manifest(
        repository_root=ROOT,
        checkpoint_path=args.checkpoint,
        dataset=args.dataset,
        seed=args.seed,
        evaluation_config={
            "split": args.split,
            "n_samples": args.n_samples,
            "latent_view": "z_s_sample",
            "mmd_estimator": "unbiased U-statistic",
        },
        model_config={
            "semantic_dim": model.semantic_dim,
            "style_dim": model.style_dim,
            "n_classes": n_classes,
        },
    )
    save_result_with_manifest(out_dir / "results.json", results, manifest)

    with (out_dir / "pairwise_mmd.csv").open("w", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["class_i", "class_j", "mmd"])
        for i in range(n_classes):
            for j in range(n_classes):
                writer.writerow([i, j, pairwise_mmd[i, j]])

    plot_pairwise_heatmap(
        pairwise_mmd, f"Pairwise conditional MMD(q(z_s|i), q(z_s|j)) — {args.tag}",
        out_dir / "pairwise_mmd_heatmap.png",
    )
    plot_per_class_bar(
        mmd_to_prior, global_mmd,
        f"Per-class MMD(q(z_s|k), N(0,I)) vs global MMD — {args.tag}",
        out_dir / "per_class_mmd_bar.png",
    )
    plot_projection_histogram(
        proj, labels, explained_ratio,
        f"Per-class z_s projection onto top class-mean direction — {args.tag}",
        out_dir / "projection_histogram.png",
    )

    print(json.dumps(results["summary"], indent=2))
    print(f"global_mmd={global_mmd:.6f}  delta_inter={delta_inter:.4f}  "
          f"(cross-check against runs_diag/fcswae_*_baseline metrics.json)")
    print(f"Saved outputs to {out_dir}")


if __name__ == "__main__":
    main()

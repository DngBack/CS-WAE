"""
latent_swap_diagnostics.py

Cross-class latent-swap intervention for F-CS-WAE: causal evidence that the
decoder actually USES leaked class information carried in z_s, rather than
just proving (as the linear probe does) that the information is present.

For every ordered class pair (a, b):
    z_c_a = mu_c of a real image from class a   (content donor)
    z_s_b = mu_s of a real image from class b   (style donor)
    x_hat = model.decoder(cat(z_c_a, z_s_b))
    external pixel classifier(x_hat) -> prediction

    content_rate[a, b] = P(prediction == a)   "identity follows z_c"
    style_rate[a, b]   = P(prediction == b)   "identity follows leaked z_s"
    neither_rate[a, b] = 1 - content_rate - style_rate   (off-diagonal only)

Uses deterministic mu_c/mu_s (not resampled z_c/z_s) so the intervention
isolates "which point in latent space" from reparameterization noise.
If no external-classifier checkpoint is supplied, the historical internal
re-encoding evaluator is used and the result is marked robustness-only.

Usage
-----
python scripts/latent_swap_diagnostics.py \\
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
    extract_full_encodings,
)
from scripts.analyze_fcswae_sampling_strategies import save_class_grid
from src.metrics.audit_protocol import DEFAULT_EVAL_SAMPLES
from src.models.external_classifiers import load_external_classifier_checkpoint
from src.utils.provenance import build_manifest, save_result_with_manifest
from src.utils.seed import set_seed


# ---------------------------------------------------------------------------
# Core intervention
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_swap(model, mu_c, mu_s, labels, n_classes, n_per_pair, device, seed,
             external_classifier=None):
    """Compute content/style/neither-following rate matrices over all (a, b)."""
    generator = torch.Generator().manual_seed(seed)
    class_idx = [torch.nonzero(labels == k, as_tuple=True)[0] for k in range(n_classes)]

    content_rate = np.zeros((n_classes, n_classes))
    style_rate = np.zeros((n_classes, n_classes))
    neither_rate = np.full((n_classes, n_classes), np.nan)

    for a in range(n_classes):
        pool_a = class_idx[a]
        if pool_a.numel() == 0:
            continue
        for b in range(n_classes):
            pool_b = class_idx[b]
            if pool_b.numel() == 0:
                continue

            idx_a = pool_a[torch.randint(0, pool_a.numel(), (n_per_pair,), generator=generator)]
            idx_b = pool_b[torch.randint(0, pool_b.numel(), (n_per_pair,), generator=generator)]

            z_c = mu_c[idx_a].to(device)
            z_s = mu_s[idx_b].to(device)
            x_hat = model.decoder(torch.cat([z_c, z_s], dim=1)).clamp(0, 1)
            if external_classifier is not None:
                pred = external_classifier(x_hat).argmax(dim=1).cpu()
            else:
                mu_c_hat, _ = model.encode_to_distribution(x_hat)
                pred = model.classifier(mu_c_hat).argmax(dim=1).cpu()

            c_rate = (pred == a).float().mean().item()
            s_rate = (pred == b).float().mean().item()
            content_rate[a, b] = c_rate
            style_rate[a, b] = s_rate
            if a != b:
                neither_rate[a, b] = max(0.0, 1.0 - c_rate - s_rate)

    return content_rate, style_rate, neither_rate


@torch.no_grad()
def build_swap_grid(model, mu_c, mu_s, labels, n_classes, device, donor_index):
    """One canonical donor image per class -> K x K grid.

    Row a = fixed z_c (content) class; column b = swapped-in z_s (style)
    class. Returned as a flat (K*K, C, H, W) tensor, row-major (a*K + b),
    matching save_class_grid's expected layout when cols=n_classes.
    """
    donor_mu_c, donor_mu_s = [], []
    for k in range(n_classes):
        idx_k = torch.nonzero(labels == k, as_tuple=True)[0]
        if idx_k.numel() == 0:
            raise ValueError(f"No examples of class {k} in the eval subset — increase --n-samples")
        chosen = idx_k[donor_index % idx_k.numel()]
        donor_mu_c.append(mu_c[chosen])
        donor_mu_s.append(mu_s[chosen])
    donor_mu_c = torch.stack(donor_mu_c).to(device)
    donor_mu_s = torch.stack(donor_mu_s).to(device)

    images = []
    for a in range(n_classes):
        z_c_a = donor_mu_c[a].unsqueeze(0).expand(n_classes, -1)
        x_hat = model.decoder(torch.cat([z_c_a, donor_mu_s], dim=1)).clamp(0, 1)
        images.append(x_hat.cpu())
    return torch.cat(images, dim=0)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_heatmap(matrix, title, out_path, vmin=0.0, vmax=1.0):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(matrix, cmap="viridis", vmin=vmin, vmax=vmax)
    ax.set_xlabel("class of z_s (style donor, b)")
    ax.set_ylabel("class of z_c (content donor, a)")
    ax.set_title(title, fontsize=10)
    ax.set_xticks(range(matrix.shape[1]))
    ax.set_yticks(range(matrix.shape[0]))
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Cross-class latent-swap intervention for F-CS-WAE")
    p.add_argument("--checkpoint", required=True, help="Path to F-CS-WAE checkpoint (.pth)")
    p.add_argument("--dataset", required=True, choices=["mnist", "fashion_mnist", "cifar10"])
    p.add_argument("--tag", required=True, help="Run identifier, used for the output subdir and plot titles")
    p.add_argument("--device", default="cpu")
    p.add_argument("--split", default="test", choices=["train", "test"])
    p.add_argument("--n-samples", type=int, default=DEFAULT_EVAL_SAMPLES,
                   help=f"Eval subset size used to draw donors (Stage-0 default: {DEFAULT_EVAL_SAMPLES})")
    p.add_argument("--n-per-pair", type=int, default=100, help="Donor draws per (a, b) class pair")
    p.add_argument("--grid-donor-index", type=int, default=0,
                   help="Which occurrence of each class (in the eval subset) to use for the qualitative grid")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--external-classifier-checkpoint", default=None,
                   help="Independent pixel classifier. Without it, scores are robustness-only internal scores.")
    p.add_argument("--output-dir", default=None,
                   help="Defaults to runs_diag/latent_swap/<tag>/")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    set_seed(args.seed)

    out_dir = Path(args.output_dir) if args.output_dir else ROOT / "runs_diag" / "latent_swap" / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model from {args.checkpoint} ...")
    model = load_fcswae_checkpoint(args.checkpoint, args.dataset, device)
    n_classes = model.n_classes
    external_classifier = None
    external_metadata = None
    if args.external_classifier_checkpoint:
        external_classifier, external_metadata = load_external_classifier_checkpoint(
            args.external_classifier_checkpoint, args.dataset, device
        )

    print(f"Loading {args.dataset} {args.split} subset (n={args.n_samples}, seed={args.seed}) ...")
    loader = get_eval_subset_loader(
        args.dataset, args.n_samples, batch_size=256, seed=args.seed, split=args.split
    )
    mu_c, mu_s, labels = extract_full_encodings(model, loader, device)

    print(f"Running latent-swap intervention over all {n_classes}x{n_classes} class pairs "
          f"({args.n_per_pair} draws/pair) ...")
    content_rate, style_rate, neither_rate = run_swap(
        model, mu_c, mu_s, labels, n_classes, args.n_per_pair, device, args.seed,
        external_classifier=external_classifier,
    )

    offdiag_mask = ~np.eye(n_classes, dtype=bool)
    summary = {
        "mean_diag_content_rate": float(np.diag(content_rate).mean()),
        "mean_offdiag_content_rate": float(content_rate[offdiag_mask].mean()),
        "mean_offdiag_style_rate": float(style_rate[offdiag_mask].mean()),
        "mean_offdiag_neither_rate": float(np.nanmean(neither_rate[offdiag_mask])),
    }
    results = {
        "checkpoint": str(args.checkpoint),
        "dataset": args.dataset,
        "tag": args.tag,
        "n_classes": n_classes,
        "n_per_pair": args.n_per_pair,
        "latent_view": "deterministic posterior means mu_c and mu_s",
        "evaluator": "external_pixel_classifier" if external_classifier is not None else "internal_reencode_robustness_only",
        "external_classifier_metadata": external_metadata,
        "content_rate": content_rate.tolist(),
        "style_rate": style_rate.tolist(),
        "neither_rate": neither_rate.tolist(),
        "summary": summary,
    }
    manifest = build_manifest(
        repository_root=ROOT,
        checkpoint_path=args.checkpoint,
        dataset=args.dataset,
        seed=args.seed,
        evaluation_config={
            "split": args.split,
            "n_samples": args.n_samples,
            "n_per_pair": args.n_per_pair,
            "latent_view": "mu_c_mu_s",
            "evaluator": results["evaluator"],
        },
        model_config={
            "semantic_dim": model.semantic_dim,
            "style_dim": model.style_dim,
            "n_classes": n_classes,
        },
        external_classifier_path=args.external_classifier_checkpoint,
    )
    save_result_with_manifest(out_dir / "results.json", results, manifest)

    with (out_dir / "results.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["a", "b", "content_rate", "style_rate", "neither_rate", "n"])
        for a in range(n_classes):
            for b in range(n_classes):
                writer.writerow(
                    [a, b, content_rate[a, b], style_rate[a, b], neither_rate[a, b], args.n_per_pair]
                )

    plot_heatmap(
        content_rate, f"Content-following rate P(pred=a) — {args.tag}",
        out_dir / "content_following_heatmap.png",
    )
    plot_heatmap(
        style_rate, f"Style-following rate P(pred=b) — {args.tag}",
        out_dir / "style_following_heatmap.png",
    )

    print("Building qualitative swap grid ...")
    grid_images = build_swap_grid(model, mu_c, mu_s, labels, n_classes, device, args.grid_donor_index)
    save_class_grid(
        grid_images,
        out_dir / "swap_grid.png",
        title=f"Latent swap: row = z_c class (content), col = z_s class (style) — {args.tag}",
        n_classes=n_classes,
        cols=n_classes,
    )

    print(json.dumps(summary, indent=2))
    print(f"Saved outputs to {out_dir}")


if __name__ == "__main__":
    main()

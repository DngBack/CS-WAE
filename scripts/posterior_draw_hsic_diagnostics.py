"""Mean-view and repeated posterior-sample conditional-HSIC audit.

This directly checks term 4, z_c independent of z_s conditional on y, on the
stochastic variables used by the sampling contract rather than only on
encoder means.
"""

from __future__ import annotations

import argparse
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
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.compute_leakage_diagnostics import get_eval_subset_loader, load_fcswae_checkpoint
from src.metrics.audit_protocol import (
    DEFAULT_EVAL_SAMPLES,
    DEFAULT_HSIC_PERMUTATIONS,
    classwise_conditional_hsic_permutation_test,
)
from src.utils.provenance import build_manifest, save_result_with_manifest
from src.utils.seed import set_seed
from src.utils.utils import mobius_reparam


@torch.no_grad()
def extract_posterior_parameters(model, loader, device):
    values = {key: [] for key in ("mu_c", "rho_c", "mu_s", "logvar_s", "labels")}
    model.eval()
    for images, labels in loader:
        mu_c, rho_c, mu_s, logvar_s = model.encode(images.to(device))
        values["mu_c"].append(mu_c.cpu())
        values["rho_c"].append(rho_c.cpu())
        values["mu_s"].append(mu_s.cpu())
        values["logvar_s"].append(logvar_s.cpu())
        values["labels"].append(labels.cpu())
    return {key: torch.cat(parts) for key, parts in values.items()}


def draw_latents(model, parameters, seed, device):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    mu_c = parameters["mu_c"]
    epsilon_c = F.normalize(
        torch.randn(mu_c.shape, generator=generator, dtype=mu_c.dtype), p=2, dim=1
    )
    z_c = mobius_reparam(
        epsilon_c.to(device), mu_c.to(device), parameters["rho_c"].to(device)
    )
    z_s, _std_s = model.sample_style(
        parameters["mu_s"].to(device),
        parameters["logvar_s"].to(device),
        generator=generator,
    )
    return z_c, z_s


def parse_args():
    parser = argparse.ArgumentParser(description="Repeated-draw conditional HSIC for F-CS-WAE")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", required=True, choices=["mnist", "fashion_mnist", "cifar10"])
    parser.add_argument("--tag", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--n-samples", type=int, default=DEFAULT_EVAL_SAMPLES)
    parser.add_argument("--draws", type=int, default=10)
    parser.add_argument("--hsic-permutations", type=int, default=DEFAULT_HSIC_PERMUTATIONS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    set_seed(args.seed)
    out_dir = Path(args.output_dir) if args.output_dir else ROOT / "runs_diag" / "posterior_draw_hsic" / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    model = load_fcswae_checkpoint(args.checkpoint, args.dataset, device)
    loader = get_eval_subset_loader(
        args.dataset, args.n_samples, batch_size=256, seed=args.seed, split="test"
    )
    parameters = extract_posterior_parameters(model, loader, device)
    labels = parameters["labels"].to(device)
    mean_view = classwise_conditional_hsic_permutation_test(
        parameters["mu_c"].to(device), parameters["mu_s"].to(device), labels,
        model.n_classes, seed=args.seed, n_permutations=args.hsic_permutations,
    )

    draw_results = []
    for draw_index in range(args.draws):
        draw_seed = args.seed + draw_index
        z_c, z_s = draw_latents(model, parameters, draw_seed, device)
        result = classwise_conditional_hsic_permutation_test(
            z_c, z_s, labels, model.n_classes, seed=draw_seed,
            n_permutations=args.hsic_permutations,
        )
        result["draw_index"] = draw_index
        result["draw_seed"] = draw_seed
        draw_results.append(result)

    statistics = np.asarray([item["statistic"] for item in draw_results])
    p_values = np.asarray([item["p_value"] for item in draw_results])
    q95 = np.asarray([item["null_quantiles"]["q95"] for item in draw_results])
    raw_logvar = parameters["logvar_s"].numpy()
    summary = {
        "sample_statistic_mean": float(statistics.mean()),
        "sample_statistic_std": float(statistics.std(ddof=1)) if args.draws > 1 else 0.0,
        "sample_statistic_min": float(statistics.min()),
        "sample_statistic_max": float(statistics.max()),
        "all_draws_reject_at_0.05": bool(np.all(p_values <= 0.05)),
        "p_values": p_values.tolist(),
        "fraction_raw_logvar_at_or_below_minus10": float((raw_logvar <= -10.0).mean()),
        "configured_sigma_floor": float(model.style_sigma_floor),
        "effective_std_median": float(
            np.median(
                np.sqrt(
                    np.exp(np.clip(raw_logvar, -10, 10))
                    + model.style_sigma_floor**2
                )
            )
        ),
    }

    fig, ax = plt.subplots(figsize=(5.4, 3.3))
    x_positions = np.arange(args.draws)
    ax.plot(x_positions, statistics, marker="o", color="#1f77b4", label="posterior draws")
    ax.plot(x_positions, q95, linestyle="--", color="#7f7f7f", label="draw-specific null 95%")
    ax.axhline(mean_view["statistic"], color="#d62728", linewidth=1.6, label="mean view")
    ax.set_xlabel("posterior draw")
    ax.set_ylabel("classwise conditional HSIC")
    ax.set_title(f"Stochastic term-4 audit: {args.tag}")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "posterior_draw_hsic.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    results = {
        "checkpoint": args.checkpoint,
        "dataset": args.dataset,
        "tag": args.tag,
        "mean_view": mean_view,
        "posterior_draws": draw_results,
        "summary": summary,
    }
    manifest = build_manifest(
        repository_root=ROOT,
        checkpoint_path=args.checkpoint,
        dataset=args.dataset,
        seed=args.seed,
        evaluation_config={
            "n_samples": int(labels.shape[0]),
            "posterior_draws": args.draws,
            "hsic_permutations_per_draw": args.hsic_permutations,
            "test": "classwise conditional HSIC with within-class permutations",
        },
        model_config={
            "semantic_dim": model.semantic_dim,
            "style_dim": model.style_dim,
            "n_classes": model.n_classes,
        },
    )
    save_result_with_manifest(out_dir / "results.json", results, manifest)
    print(json.dumps(summary, indent=2))
    print(f"Saved outputs to {out_dir}")


if __name__ == "__main__":
    main()

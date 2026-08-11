"""Permutation calibration for the global style-prior MMD diagnostic.

The output separates numerical magnitude from a test of exact equality.  A
small MMD can still be statistically distinguishable from zero at n=2048;
that outcome does not invalidate the marginal-versus-conditional gap.
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
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.compute_leakage_diagnostics import (
    extract_latent_views,
    get_eval_subset_loader,
    load_fcswae_checkpoint,
)
from src.metrics.audit_protocol import DEFAULT_EVAL_SAMPLES, mmd2_permutation_test
from src.utils.provenance import build_manifest, save_result_with_manifest
from src.utils.seed import set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Calibrate global MMD with its finite-sample null")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", required=True, choices=["mnist", "fashion_mnist", "cifar10"])
    parser.add_argument("--tag", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--n-samples", type=int, default=DEFAULT_EVAL_SAMPLES)
    parser.add_argument("--n-permutations", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    set_seed(args.seed)
    out_dir = Path(args.output_dir) if args.output_dir else ROOT / "runs_diag" / "mmd_null" / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    model = load_fcswae_checkpoint(args.checkpoint, args.dataset, device)
    loader = get_eval_subset_loader(
        args.dataset, args.n_samples, batch_size=256, seed=args.seed, split="test"
    )
    views = extract_latent_views(model, loader, device, model_type="fcswae", seed=args.seed)
    z_s = views["z_s_sample"].to(device)
    reference_seed = args.seed + 10_000
    permutation_seed = args.seed + 20_000
    generator = torch.Generator(device="cpu").manual_seed(reference_seed)
    reference = torch.randn(z_s.shape, generator=generator, dtype=z_s.dtype).to(device)
    calibration = mmd2_permutation_test(
        z_s, reference, seed=permutation_seed, n_permutations=args.n_permutations
    )

    figure_path = out_dir / "global_mmd_null.png"
    fig, ax = plt.subplots(figsize=(5.2, 3.3))
    ax.hist(calibration["null_values"], bins=35, color="#8da0cb", alpha=0.85,
            label="permutation null")
    ax.axvline(calibration["statistic"], color="#d62728", linewidth=2.0,
               label=f'observed = {calibration["statistic"]:.6f}')
    ax.axvline(calibration["null_quantiles"]["q95"], color="black", linestyle="--",
               linewidth=1.2, label="null 95th percentile")
    ax.set_xlabel(r"unbiased global $\mathrm{MMD}^2$")
    ax.set_ylabel("permutations")
    ax.set_title(f"Global MMD finite-sample calibration: {args.tag}")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(figure_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    results = {
        "checkpoint": args.checkpoint,
        "dataset": args.dataset,
        "tag": args.tag,
        "latent_view": "seeded posterior sample z_s",
        "posterior_draw_seed": args.seed,
        "reference_seed": reference_seed,
        "permutation_seed": permutation_seed,
        "calibration": calibration,
    }
    manifest = build_manifest(
        repository_root=ROOT,
        checkpoint_path=args.checkpoint,
        dataset=args.dataset,
        seed=args.seed,
        evaluation_config={
            "n_samples": int(z_s.shape[0]),
            "n_permutations": args.n_permutations,
            "latent_view": "z_s_sample",
            "null": "balanced pooled-label permutation",
            "posterior_draw_seed": args.seed,
            "reference_seed": reference_seed,
            "permutation_seed": permutation_seed,
        },
        model_config={
            "semantic_dim": model.semantic_dim,
            "style_dim": model.style_dim,
            "n_classes": model.n_classes,
        },
    )
    save_result_with_manifest(out_dir / "results.json", results, manifest)
    print(json.dumps({key: value for key, value in calibration.items() if key != "null_values"}, indent=2))
    print(f"Saved outputs to {out_dir}")


if __name__ == "__main__":
    main()

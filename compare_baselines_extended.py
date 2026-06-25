"""
Extended baseline comparison for F-CS-WAE.

Three comparison groups, all using ResNet-18 backbone:
    Group 1 (unsupervised):  ResNetAE
    Group 2 (label-guided):  AEWithCE, AEWithSupCon, AEWithCenterLoss, AEWithTriplet
    Group 3 (conditional):   ConditionalVAE, ConditionalWAE_MMD, GaussianClassPriorWAE

Usage
-----
    # All groups + F-CS-WAE
    python compare_baselines_extended.py --dataset cifar10 --device cuda:0

    # One group only
    python compare_baselines_extended.py --dataset cifar10 --device cuda:0 --group label_guided

    # Skip F-CS-WAE (compare baselines only)
    python compare_baselines_extended.py --dataset cifar10 --device cuda:0 --skip-f-cs-wae

    # Load from a pre-trained F-CS-WAE checkpoint
    python compare_baselines_extended.py --dataset cifar10 --device cuda:0 \\
        --f-cs-wae-ckpt runs_f/cifar10/seed_0/f_cs_wae_model.pth

Output
------
    <output-dir>/extended_comparison_results.csv
    <output-dir>/extended_comparison_results.json
"""

from __future__ import annotations

import argparse
import json
import os
import warnings

import pandas as pd
import torch

warnings.filterwarnings("ignore")

from src.config_f_cs_wae import f_cs_wae_config as cfg
from src.models.f_cs_wae import FCSWAE
from src.models.baselines_extended import (
    ResNetAE,
    AEWithCE,
    AEWithSupCon,
    AEWithCenterLoss,
    AEWithTriplet,
    ConditionalVAE,
    ConditionalWAE_MMD,
    GaussianClassPriorWAE,
    MODEL_META,
)
from src.datasets.loaders import (
    get_loaders,
    get_dataset_info,
    get_default_runs_dir,
    SUPPORTED_DATASETS,
)
from src.utils.dataset_config import apply_dataset_config
from src.trainers.trainer_baselines_extended import ExtendedBaselineTrainer
from src.trainers.trainer_f_cs_wae import FCSWAETrainer
from src.metrics.evaluation import ModelEvaluator
from src.utils.seed import set_seed
from src.utils.device import set_device
from src.utils.run_io import save_metrics

GROUPS = {
    "unsupervised": ["ResNetAE"],
    "label_guided": ["AEWithCE", "AEWithSupCon", "AEWithCenterLoss", "AEWithTriplet"],
    "conditional":  ["ConditionalVAE", "ConditionalWAE_MMD", "GaussianClassPriorWAE"],
}

MODEL_CONSTRUCTORS = {
    "ResNetAE":              lambda n_cls, ic, isz: ResNetAE(in_channels=ic, image_size=isz),
    "AEWithCE":              lambda n_cls, ic, isz: AEWithCE(n_classes=n_cls, in_channels=ic, image_size=isz),
    "AEWithSupCon":          lambda n_cls, ic, isz: AEWithSupCon(in_channels=ic, image_size=isz),
    "AEWithCenterLoss":      lambda n_cls, ic, isz: AEWithCenterLoss(n_classes=n_cls, in_channels=ic, image_size=isz),
    "AEWithTriplet":         lambda n_cls, ic, isz: AEWithTriplet(in_channels=ic, image_size=isz),
    "ConditionalVAE":        lambda n_cls, ic, isz: ConditionalVAE(n_classes=n_cls, in_channels=ic, image_size=isz),
    "ConditionalWAE_MMD":    lambda n_cls, ic, isz: ConditionalWAE_MMD(n_classes=n_cls, in_channels=ic, image_size=isz),
    "GaussianClassPriorWAE": lambda n_cls, ic, isz: GaussianClassPriorWAE(n_classes=n_cls, in_channels=ic, image_size=isz),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extended baseline comparison for F-CS-WAE")
    p.add_argument("--dataset",       type=str,  default="cifar10",
                   choices=list(SUPPORTED_DATASETS))
    p.add_argument("--seed",          type=int,  default=0)
    p.add_argument("--device",        type=str,  default=None)
    p.add_argument("--epochs",        type=int,  default=None,
                   help="Training epochs for baselines (default: cfg.total_epochs)")
    p.add_argument("--group",         type=str,  default="all",
                   choices=["all", "unsupervised", "label_guided", "conditional"])
    p.add_argument("--output-dir",    type=str,  default=None)
    p.add_argument("--skip-f-cs-wae", action="store_true",
                   help="Skip training/evaluating F-CS-WAE (compare baselines only)")
    p.add_argument("--f-cs-wae-ckpt", type=str,  default=None,
                   help="Path to pre-trained F-CS-WAE weights (skips training)")
    p.add_argument("--models",        type=str,  nargs="+", default=None,
                   help="Subset of model names to run (overrides --group)")
    p.add_argument("--summarize-only", action="store_true",
                   help="Load existing metrics.json files and print summary table only")
    return p.parse_args()


def collect_metrics(results_dir: str) -> dict[str, dict]:
    """Load per-model metrics.json files from subdirs of results_dir."""
    all_results: dict[str, dict] = {}
    for entry in os.listdir(results_dir):
        model_dir = os.path.join(results_dir, entry)
        metrics_path = os.path.join(model_dir, "metrics.json")
        if os.path.isdir(model_dir) and os.path.exists(metrics_path):
            with open(metrics_path) as f:
                all_results[entry] = json.load(f)
    return all_results


def print_comparison_table(all_results: dict[str, dict], output_dir: str) -> None:
    COLS = ["ACC", "NMI", "ARI", "FID", "LPIPS", "SSIM", "PSNR"]
    rows = []
    for model_name, metrics in sorted(all_results.items()):
        meta = MODEL_META.get(model_name, {"uses_labels": "?", "generative_prior": "?"})
        row = {
            "Method":           model_name,
            "UsesLabels":       meta["uses_labels"],
            "GenerativePrior":  meta["generative_prior"],
        }
        for col in COLS:
            row[col] = round(metrics.get(col, float("nan")), 4)
        rows.append(row)

    df = pd.DataFrame(rows).set_index("Method")
    print("\n" + "=" * 90)
    print(" EXTENDED BASELINE COMPARISON ".center(90, "="))
    print("=" * 90)
    print(df.to_markdown(floatfmt=".4f"))
    print("=" * 90)

    csv_path = os.path.join(output_dir, "extended_comparison_results.csv")
    df.to_csv(csv_path)
    print(f"\nSaved: {csv_path}")

    json_path = os.path.join(output_dir, "extended_comparison_results.json")
    with open(json_path, "w") as f:
        json.dump({k: dict(v) for k, v in df.iterrows()}, f, indent=2, default=str)
    print(f"Saved: {json_path}")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = set_device(args.device)

    dataset_info = apply_dataset_config(args.dataset, get_dataset_info, backbone="resnet18")
    n_classes    = dataset_info["n_classes"]
    in_channels  = dataset_info.get("in_channels", 3)
    image_size   = dataset_info.get("image_size",  32)

    output_dir = args.output_dir or os.path.join(
        f"runs_f/{args.dataset}", "baselines_extended", f"seed_{args.seed}"
    )
    os.makedirs(output_dir, exist_ok=True)
    epochs = args.epochs or cfg.total_epochs

    # -- Summarize only --
    if args.summarize_only:
        all_results = collect_metrics(output_dir)
        if not all_results:
            print(f"No metrics found in {output_dir}")
            return
        print_comparison_table(all_results, output_dir)
        return

    # -- Select models to run --
    if args.models:
        model_names = args.models
    elif args.group == "all":
        model_names = list(MODEL_CONSTRUCTORS.keys())
    else:
        model_names = GROUPS[args.group]

    print("=" * 60)
    print("Extended Baseline Comparison")
    print("=" * 60)
    print(f"  Dataset : {args.dataset}   Seed : {args.seed}   Device : {device}")
    print(f"  Models  : {model_names}")
    print(f"  Output  : {output_dir}")

    # Load data (shared across all models)
    train_loader, test_loader = get_loaders(
        dataset=args.dataset,
        seed=args.seed,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
    )
    evaluator = ModelEvaluator(device=device, dataset=args.dataset)

    all_results: dict[str, dict] = {}

    # ------------------------------------------------------------------ #
    # Train + evaluate each baseline
    # ------------------------------------------------------------------ #
    for model_name in model_names:
        print(f"\n{'=' * 60}")
        print(f"  [{model_name}]")
        print(f"{'=' * 60}")

        model_dir = os.path.join(output_dir, model_name)
        os.makedirs(model_dir, exist_ok=True)

        model = MODEL_CONSTRUCTORS[model_name](n_classes, in_channels, image_size).to(device)

        trainer = ExtendedBaselineTrainer(
            model, model_name, train_loader, device=device, epochs=epochs
        )
        trainer.train()

        # Save weights
        torch.save(model.state_dict(), os.path.join(model_dir, "model.pth"))

        # Evaluate
        metrics = evaluator.comprehensive_evaluation(model, model_name, test_loader, model_dir)
        save_metrics(model_dir, metrics)
        all_results[model_name] = metrics

        print(f"  ACC={metrics.get('ACC', 'N/A'):.4f}  FID={metrics.get('FID', 'N/A'):.2f}")

    # ------------------------------------------------------------------ #
    # F-CS-WAE
    # ------------------------------------------------------------------ #
    if not args.skip_f_cs_wae:
        print(f"\n{'=' * 60}\n  [F-CS-WAE]\n{'=' * 60}")
        f_model = FCSWAE(
            semantic_dim = cfg.semantic_dim,
            style_dim    = cfg.style_dim,
            n_classes    = n_classes,
            in_channels  = in_channels,
            image_size   = image_size,
        ).to(device)

        if args.f_cs_wae_ckpt:
            print(f"Loading F-CS-WAE from {args.f_cs_wae_ckpt}")
            f_model.load_state_dict(torch.load(args.f_cs_wae_ckpt, map_location=device))
        else:
            f_trainer = FCSWAETrainer(f_model, train_loader, device=device)
            f_trainer.train(epochs=epochs)

        f_dir = os.path.join(output_dir, "F-CS-WAE")
        os.makedirs(f_dir, exist_ok=True)
        torch.save(f_model.state_dict(), os.path.join(f_dir, "model.pth"))
        metrics = evaluator.comprehensive_evaluation(f_model, "F-CS-WAE", test_loader, f_dir)
        save_metrics(f_dir, metrics)
        all_results["F-CS-WAE"] = metrics

    # ------------------------------------------------------------------ #
    # Summary table
    # ------------------------------------------------------------------ #
    print_comparison_table(all_results, output_dir)
    print("\nExtended baseline comparison complete.")


if __name__ == "__main__":
    main()

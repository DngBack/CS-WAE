"""
Baseline comparison script for evaluating CS-WAE against other methods
"""
import argparse
import json
import os
import warnings

import torch

warnings.filterwarnings("ignore")

from src.config import config
from src.models import VAE, WAE_MMD, S_VAE, VaDE, SphericalWAE_Supervised
from src.datasets.loaders import (
    get_loaders,
    get_dataset_info,
    get_default_runs_dir,
    SUPPORTED_DATASETS,
)
from src.utils.dataset_config import apply_dataset_config
from src.trainers.trainer import BaselineTrainer, CSWAETrainer
from src.metrics.evaluation import ModelEvaluator, create_comparison_table
from src.utils.seed import set_seed
from src.utils.device import set_device
from src.utils.run_io import config_to_dict, save_run_metadata, save_metrics


def collect_baseline_metrics(results_dir: str) -> dict:
    """Load per-model metrics.json from baseline output subdirectories."""
    all_results = {}
    for entry in os.listdir(results_dir):
        model_dir = os.path.join(results_dir, entry)
        metrics_path = os.path.join(model_dir, "metrics.json")
        if os.path.isdir(model_dir) and os.path.exists(metrics_path):
            with open(metrics_path) as f:
                all_results[entry] = json.load(f)
    return all_results


def summarize_baseline_results(results_dir: str) -> None:
    all_results = collect_baseline_metrics(results_dir)
    if not all_results:
        print(f"No baseline metrics found under {results_dir}")
        return

    comparison_df = create_comparison_table(all_results)
    csv_path = os.path.join(results_dir, "comparison_results.csv")
    comparison_df.to_csv(csv_path)
    print("\nBASELINE COMPARISON (summarized):")
    print(comparison_df.to_string())
    print(f"\nSaved: {csv_path}")


def build_models(
    n_classes: int,
    in_channels: int = 1,
    image_size: int = 28,
    backbone: str | None = None,
) -> dict:
    """Build baseline models, skipping optional ones that fail to import."""
    backbone = backbone or config.backbone
    cnn = {
        "in_channels": in_channels,
        "image_size": image_size,
        "backbone": backbone,
    }
    models = {
        "VAE": VAE(config.latent_dim, **cnn),
        "WAE-MMD": WAE_MMD(config.latent_dim, **cnn),
        "VaDE": VaDE(config.latent_dim, n_classes, **cnn),
        "CS-WAE": SphericalWAE_Supervised(config.latent_dim, n_classes, **cnn),
    }

    try:
        models["S-VAE"] = S_VAE(config.latent_dim)
    except ImportError as e:
        print(f"Skipping S-VAE: {e}")

    return models


def parse_args():
    parser = argparse.ArgumentParser(description="Compare CS-WAE with baselines")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: runs/<dataset>/baselines/seed_<seed>)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="mnist",
        choices=list(SUPPORTED_DATASETS),
        help="Dataset name",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Training epochs for each baseline (default: 50)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Subset of models to run (default: all available)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device, e.g. cuda:0 or cuda:1 (default: cuda:0)",
    )
    parser.add_argument(
        "--skip-summary",
        action="store_true",
        help="Skip final comparison CSV (for parallel workers)",
    )
    parser.add_argument(
        "--summarize-only",
        action="store_true",
        help="Build comparison CSV from existing metrics.json files",
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default=None,
        choices=["cnn", "resnet18"],
        help="Encoder backbone (default: cnn; use resnet18 for CIFAR-10)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    results_dir = args.output_dir
    if args.summarize_only:
        if not results_dir:
            results_dir = f"{get_default_runs_dir(args.dataset, args.backbone)}/baselines/seed_{args.seed}"
        os.makedirs(results_dir, exist_ok=True)
        summarize_baseline_results(results_dir)
        return

    set_seed(args.seed)
    device = set_device(args.device)

    dataset_info = apply_dataset_config(args.dataset, get_dataset_info, backbone=args.backbone)
    results_dir = args.output_dir or f"{get_default_runs_dir(args.dataset, config.backbone)}/baselines/seed_{args.seed}"
    os.makedirs(results_dir, exist_ok=True)

    print("=" * 80)
    print("CS-WAE vs Baselines Comparison")
    print("=" * 80)
    print(f"Seed: {args.seed}")
    print(f"Device: {device}")
    print(f"Dataset: {args.dataset}")
    print(f"Backbone: {config.backbone}")
    print(f"Epochs: {args.epochs}")
    print(f"Output: {results_dir}")

    save_run_metadata(
        results_dir,
        config_to_dict(config),
        args.seed,
        extra={"dataset": args.dataset, "epochs": args.epochs, "backbone": config.backbone},
    )

    print("Loading dataset...")
    train_loader, test_loader = get_loaders(
        dataset=args.dataset,
        seed=args.seed,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
    )

    models_to_run = build_models(
        dataset_info["n_classes"],
        in_channels=dataset_info.get("in_channels", 1),
        image_size=dataset_info.get("image_size", 28),
        backbone=config.backbone,
    )
    if args.models:
        models_to_run = {k: v for k, v in models_to_run.items() if k in args.models}
        missing = set(args.models) - set(models_to_run.keys())
        if missing:
            print(f"Warning: requested models not available: {missing}")

    all_results = {}
    evaluator = ModelEvaluator(device=device, dataset=args.dataset)

    for model_name, model in models_to_run.items():
        print(f"\n{'=' * 30}")
        print(f"Training {model_name}")
        print(f"{'=' * 30}")

        model.to(config.device)
        model_dir = os.path.join(results_dir, model_name.replace("/", "-"))
        os.makedirs(model_dir, exist_ok=True)

        if model_name == "CS-WAE":
            trainer = CSWAETrainer(model, train_loader)
            trainer.train(epochs=args.epochs)
        else:
            trainer = BaselineTrainer(model, model_name, train_loader)
            trainer.train(epochs=args.epochs)

        model_path = os.path.join(model_dir, f"{model_name.replace('/', '-')}.pth")
        torch.save(model.state_dict(), model_path)

        print(f"Evaluating {model_name}...")
        metrics = evaluator.comprehensive_evaluation(
            model, model_name, test_loader, model_dir
        )
        all_results[model_name] = metrics
        save_metrics(model_dir, metrics)

        print(f"Completed {model_name}")

    if not args.skip_summary:
        print("\n" + "=" * 80)
        print("FINAL COMPARISON")
        print("=" * 80)

        comparison_df = create_comparison_table(all_results)
        csv_path = os.path.join(results_dir, "comparison_results.csv")
        comparison_df.to_csv(csv_path)
        print(comparison_df.to_string())
        print(f"\nResults saved to: {csv_path}")

    print("Comparison completed successfully!")


if __name__ == "__main__":
    main()

"""
Main script for running CS-WAE ablation studies
"""

import argparse
import json
import os
import torch
import warnings
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

warnings.filterwarnings("ignore")

from src.config_ablation import ablation_config
from src.models.cs_wae_ablation import create_ablation_model
from src.datasets.loaders import get_loaders, get_default_runs_dir, SUPPORTED_DATASETS
from src.trainers.trainer import CSWAETrainer
from src.trainers.trainer_ablation import AblationTrainer, NoiseRobustnessEvaluator
from src.metrics.evaluation import ModelEvaluator
from src.visualization.plots import plot_results
from src.utils.seed import set_seed
from src.utils.device import set_device
from src.utils.run_io import config_to_dict, save_run_metadata


def uses_main_trainer(variant_name: str) -> bool:
    """Baseline variant trains with the same stack as train_cs_wae.py."""
    return variant_name == "baseline"


def normalize_training_history(history, from_main_trainer: bool):
    """Convert CSWAETrainer tuple history to ablation dict format."""
    if not from_main_trainer:
        return history
    return [
        {
            "total": h[0],
            "recon": h[1],
            "sup_mmd": h[2],
            "unsup_mmd": h[3],
            "kld": 0.0,
        }
        for h in history
    ]


def model_reconstruct(model, data):
    """Return reconstructions for main or ablation model forward signatures."""
    return model(data)[0]


def get_model_flags(model):
    """Read ablation flags; main model defaults to full CS-WAE."""
    return (
        getattr(model, "use_supervised_mmd", True),
        getattr(model, "use_spherical_space", True),
    )


def run_single_ablation(variant_name, train_loader, test_loader, results_dir, dataset="mnist"):
    """Run training and evaluation for a single ablation variant"""

    print(f"\n{'=' * 60}")
    print(f"RUNNING ABLATION: {variant_name.upper()}")
    print(f"{'=' * 60}")

    # Create model
    model = create_ablation_model(variant_name)
    model.to(ablation_config.device)

    variant_config = ablation_config.ablation_variants[variant_name]
    print(f"Description: {variant_config['description']}")
    print(f"Configuration: {variant_config}")

    # Create variant-specific directory
    variant_dir = os.path.join(results_dir, variant_name)
    os.makedirs(variant_dir, exist_ok=True)

    # Train model — baseline uses main trainer for identical training dynamics
    if uses_main_trainer(variant_name):
        print("Using main CSWAETrainer (same as train_cs_wae.py)")
        trainer = CSWAETrainer(model, train_loader)
        history = normalize_training_history(trainer.train(), from_main_trainer=True)
    else:
        trainer = AblationTrainer(model, train_loader, variant_config["name"])
        history = trainer.train()

    use_supervised_mmd, use_spherical_space = get_model_flags(model)

    # Save model
    model_path = os.path.join(variant_dir, f"{variant_name}_model.pth")
    torch.save(model.state_dict(), model_path)

    # Generate visualizations (adapted for ablation models)
    print(f"Generating visualizations for {variant_name}...")
    try:
        # Save training history
        plt.figure(figsize=(12, 8))
        epochs = range(1, len(history) + 1)

        plt.plot(epochs, [h["total"] for h in history], label="Total Loss", linewidth=2)
        plt.plot(
            epochs,
            [h["recon"] for h in history],
            label="Reconstruction Loss",
            linewidth=2,
        )

        if use_supervised_mmd:
            plt.plot(
                epochs,
                [h["sup_mmd"] for h in history],
                label="Supervised MMD Loss",
                linewidth=2,
            )

        plt.plot(
            epochs,
            [h["unsup_mmd"] for h in history],
            label="Unsupervised MMD Loss",
            linewidth=2,
        )

        if not use_spherical_space:
            plt.plot(epochs, [h["kld"] for h in history], label="KLD Loss", linewidth=2)

        plt.title(f"Training History - {variant_config['name']}")
        plt.xlabel("Epoch")
        plt.ylabel("Loss Value")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(
            os.path.join(variant_dir, "loss_history.png"), dpi=300, bbox_inches="tight"
        )
        plt.close()

        # Generate sample reconstructions
        model.eval()
        with torch.no_grad():
            data, _ = next(iter(test_loader))
            data = data.to(ablation_config.device)
            x_hat = model_reconstruct(model, data)

            fig, axes = plt.subplots(2, 10, figsize=(20, 4))
            for i in range(10):
                # Original images
                axes[0, i].imshow(data[i].cpu().squeeze(), cmap="gray")
                axes[0, i].set_title("Original")
                axes[0, i].axis("off")

                # Reconstructed images
                axes[1, i].imshow(x_hat[i].cpu().squeeze(), cmap="gray")
                axes[1, i].set_title("Reconstructed")
                axes[1, i].axis("off")

            plt.suptitle(f"Reconstructions - {variant_config['name']}")
            plt.savefig(
                os.path.join(variant_dir, "reconstructions.png"),
                dpi=300,
                bbox_inches="tight",
            )
            plt.close()

    except Exception as e:
        print(f"Warning: Could not generate all visualizations for {variant_name}: {e}")

    # Evaluate model
    print(f"Evaluating {variant_name}...")
    evaluator = ModelEvaluator(dataset=dataset)

    # Adapt evaluation for ablation models
    try:
        metrics = evaluator.comprehensive_evaluation(
            model, variant_config["name"], test_loader, variant_dir
        )
    except Exception as e:
        print(f"Warning: Full evaluation failed for {variant_name}: {e}")
        # Fallback to basic metrics
        metrics = {
            "ACC": 0.0,
            "NMI": 0.0,
            "ARI": 0.0,
            "FID": float("inf"),
            "LPIPS": 0.0,
            "SSIM": 0.0,
            "PSNR": 0.0,
        }

    print(f"Completed evaluation for {variant_name}")
    print(
        f"  Metrics: ACC={metrics.get('ACC', 0):.4f}, NMI={metrics.get('NMI', 0):.4f}, "
        f"FID={metrics.get('FID', float('inf')):.4f}, SSIM={metrics.get('SSIM', 0):.4f}"
    )

    metrics_path = os.path.join(variant_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics, history, model


def collect_metrics_from_disk(results_dir):
    """Load metrics.json written by each ablation variant subdirectory."""
    all_results = {}
    for variant_key, variant_config in ablation_config.ablation_variants.items():
        metrics_path = os.path.join(results_dir, variant_key, "metrics.json")
        if os.path.exists(metrics_path):
            with open(metrics_path) as f:
                all_results[variant_config["name"]] = json.load(f)
    return all_results


def plot_ablation_comparison(df, results_dir):
    """Save bar-chart comparison figure from ablation metrics table."""
    if df is None or df.empty:
        return

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    metrics_to_plot = ["ACC", "NMI", "ARI", "FID", "LPIPS", "SSIM"]

    for i, metric in enumerate(metrics_to_plot[:6]):
        if metric in df.columns:
            ax = axes[i]
            values = df[metric].sort_values(ascending=(metric != "FID"))
            bars = ax.bar(range(len(values)), values.values)
            ax.set_xticks(range(len(values)))
            ax.set_xticklabels(values.index, rotation=45, ha="right")
            ax.set_title(f"{metric} Comparison")
            ax.set_ylabel(metric)
            if metric == "FID":
                best_idx, worst_idx = values.argmin(), values.argmax()
            else:
                best_idx, worst_idx = values.argmax(), values.argmin()
            bars[best_idx].set_color("green")
            bars[worst_idx].set_color("red")
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        os.path.join(results_dir, "ablation_comparison.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def summarize_ablation_results(results_dir):
    """Build comparison table and plots from saved variant metrics."""
    all_results = collect_metrics_from_disk(results_dir)
    if not all_results:
        print(f"No ablation metrics found under {results_dir}")
        return None

    print(f"\n{'=' * 80}")
    print("ABLATION STUDY RESULTS (summarized)")
    print(f"{'=' * 80}")

    df = save_results_table(all_results, results_dir)
    plot_ablation_comparison(df, results_dir)
    return df


def save_results_table(all_results, results_dir):
    """Save and print ablation comparison table."""
    if not all_results:
        print("No results to save.")
        return None

    df = pd.DataFrame(all_results).T
    column_order = ["ACC", "NMI", "ARI", "FID", "LPIPS", "SSIM", "PSNR"]
    existing_cols = [col for col in column_order if col in df.columns]
    df = df[existing_cols]

    csv_path = os.path.join(results_dir, "ablation_results.csv")
    df.to_csv(csv_path)

    print("\nPERFORMANCE COMPARISON:")
    print(df.to_markdown(floatfmt=".4f"))
    print(f"\nSaved: {csv_path}")
    return df


def evaluate_saved_variant(variant_name, test_loader, results_dir, dataset="mnist"):
    """Load a saved checkpoint and run evaluation only."""
    variant_dir = os.path.join(results_dir, variant_name)
    model_path = os.path.join(variant_dir, f"{variant_name}_model.pth")
    if not os.path.exists(model_path):
        print(f"Skip {variant_name}: checkpoint not found at {model_path}")
        return None

    variant_config = ablation_config.ablation_variants[variant_name]
    model = create_ablation_model(variant_name)
    model.load_state_dict(torch.load(model_path, map_location=ablation_config.device))
    model.to(ablation_config.device)

    print(f"\nEvaluating saved checkpoint: {variant_name}")
    evaluator = ModelEvaluator(dataset=dataset)
    metrics = evaluator.comprehensive_evaluation(
        model, variant_config["name"], test_loader, variant_dir
    )
    print(
        f"  Metrics: ACC={metrics.get('ACC', 0):.4f}, NMI={metrics.get('NMI', 0):.4f}, "
        f"FID={metrics.get('FID', float('inf')):.4f}, SSIM={metrics.get('SSIM', 0):.4f}"
    )

    with open(os.path.join(variant_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    return ablation_config.ablation_variants[variant_name]["name"], metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Run CS-WAE ablation study")
    parser.add_argument(
        "--run-noise",
        action="store_true",
        help="Run noise robustness evaluation after training (slow)",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["baseline", "no_sup_mmd", "euclidean", "vmf_prior", "minimal"],
        choices=list(ablation_config.ablation_variants.keys()),
        help="Ablation variants to run",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help="Existing results directory (for --eval-only or resume)",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Skip training; evaluate saved checkpoints and build comparison table",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for training and data shuffling",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device, e.g. cuda:0 or cuda:1 (default: cuda:0)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="mnist",
        choices=list(SUPPORTED_DATASETS),
        help="Dataset name",
    )
    parser.add_argument(
        "--skip-aggregate",
        action="store_true",
        help="Skip final CSV/plots (for parallel workers sharing one results dir)",
    )
    parser.add_argument(
        "--summarize-only",
        action="store_true",
        help="Build comparison table/plots from existing metrics.json files",
    )
    return parser.parse_args()


def main():
    """Main function for ablation study"""
    args = parse_args()
    set_seed(args.seed)
    device = set_device(args.device)

    if args.results_dir:
        results_dir = args.results_dir
        os.makedirs(results_dir, exist_ok=True)
        timestamp = os.path.basename(results_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = f"{get_default_runs_dir(args.dataset)}/ablation_{timestamp}"
        os.makedirs(results_dir, exist_ok=True)

    if args.summarize_only:
        summarize_ablation_results(results_dir)
        return {}, results_dir

    print("=" * 80)
    print("CS-WAE ABLATION STUDY")
    print("=" * 80)
    print(f"Seed: {args.seed}")
    print(f"Device: {device}")
    print(f"Dataset: {args.dataset}")
    print("This study systematically evaluates the contribution of each component:")
    print("1. Supervised MMD Loss")
    print("2. Spherical vs Euclidean Space")
    print("3. Spherical Cauchy vs von Mises-Fisher Prior")
    print("4. Noise Robustness")
    print("=" * 80)

    # Load data
    variants_to_run = args.variants
    print(f"Loading {args.dataset} dataset...")
    train_loader, test_loader = get_loaders(dataset=args.dataset, seed=args.seed)

    save_run_metadata(
        results_dir,
        config_to_dict(ablation_config),
        args.seed,
        extra={"variants": variants_to_run, "dataset": args.dataset},
    )
    print(f"Variants to run: {variants_to_run}")
    print(f"Epochs per variant: {ablation_config.epochs}")
    print(f"Results directory: {results_dir}")

    all_results = {}
    all_histories = {}
    trained_models = {}

    if args.eval_only:
        print("\n--- EVAL-ONLY MODE: loading saved checkpoints ---")
        for variant in variants_to_run:
            try:
                result = evaluate_saved_variant(
                    variant, test_loader, results_dir, dataset=args.dataset
                )
                if result:
                    name, metrics = result
                    all_results[name] = metrics
            except Exception as e:
                print(f"ERROR: Failed to evaluate {variant}: {e}")
    else:
        for variant in variants_to_run:
            try:
                metrics, history, model = run_single_ablation(
                    variant, train_loader, test_loader, results_dir, dataset=args.dataset
                )
                all_results[ablation_config.ablation_variants[variant]["name"]] = metrics
                all_histories[variant] = history
                trained_models[variant] = model
                if not args.skip_aggregate:
                    save_results_table(all_results, results_dir)
            except Exception as e:
                print(f"ERROR: Failed to run ablation {variant}: {e}")
                continue

        # Training loss comparison (full run only)
        if all_histories:
            plt.figure(figsize=(15, 10))

            # Plot total loss for each variant
            for variant, history in all_histories.items():
                variant_name = ablation_config.ablation_variants[variant]["name"]
                epochs = range(1, len(history) + 1)
                total_losses = [h["total"] for h in history]
                plt.plot(epochs, total_losses, label=variant_name, linewidth=2)

            plt.title("Training Loss Comparison Across Ablation Variants")
            plt.xlabel("Epoch")
            plt.ylabel("Total Loss")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(
                os.path.join(results_dir, "training_comparison.png"),
                dpi=300,
                bbox_inches="tight",
            )
            plt.close()

    if not args.skip_aggregate:
        print(f"\n{'=' * 80}")
        print("ABLATION STUDY RESULTS")
        print(f"{'=' * 80}")

        if all_results:
            df = save_results_table(all_results, results_dir)
            plot_ablation_comparison(df, results_dir)
        else:
            summarize_ablation_results(results_dir)

    # Noise robustness evaluation (optional - can be time-consuming)
    run_noise_evaluation = args.run_noise

    if run_noise_evaluation and trained_models and not args.eval_only:
        print("\n" + "=" * 60)
        print("NOISE ROBUSTNESS EVALUATION")
        print("=" * 60)

        noise_evaluator = NoiseRobustnessEvaluator()
        noise_results = {}

        for variant_name, model in trained_models.items():
            variant_display_name = ablation_config.ablation_variants[variant_name][
                "name"
            ]
            print(f"\nEvaluating noise robustness for {variant_display_name}...")

            try:
                noise_result = noise_evaluator.evaluate_noise_robustness(
                    model, test_loader, variant_display_name
                )
                noise_results[variant_display_name] = noise_result
            except Exception as e:
                print(f"Error in noise evaluation for {variant_name}: {e}")

        if noise_results:
            # Compare noise robustness
            comparison_df = noise_evaluator.compare_noise_robustness(noise_results)
            comparison_df.to_csv(
                os.path.join(results_dir, "noise_robustness_results.csv")
            )
            plt.savefig(
                os.path.join(results_dir, "noise_robustness_comparison.png"),
                dpi=300,
                bbox_inches="tight",
            )
            plt.close()

    # Save configuration for reproducibility
    config_path = os.path.join(results_dir, "experiment_config.txt")
    with open(config_path, "w") as f:
        f.write("CS-WAE Ablation Study Configuration\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Seed: {args.seed}\n")
        f.write(f"Device: {ablation_config.device}\n")
        f.write(f"Epochs: {ablation_config.epochs}\n")
        f.write(f"Batch Size: {ablation_config.batch_size}\n")
        f.write(f"Learning Rate: {ablation_config.lr}\n")
        f.write(f"Latent Dimension: {ablation_config.latent_dim}\n\n")

        f.write("Ablation Variants:\n")
        for variant, config in ablation_config.ablation_variants.items():
            if variant in variants_to_run:
                f.write(f"\n{variant}:\n")
                for key, value in config.items():
                    f.write(f"  {key}: {value}\n")

    print(f"\n{'=' * 80}")
    print("ABLATION STUDY COMPLETED!")
    print(f"Results saved to: {results_dir}")
    print(f"{'=' * 80}")

    return all_results, results_dir


if __name__ == "__main__":
    main()

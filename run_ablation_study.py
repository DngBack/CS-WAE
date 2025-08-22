"""
Main script for running CS-WAE ablation studies
"""

import os
import torch
import warnings
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

warnings.filterwarnings("ignore")

from src.config_ablation import ablation_config
from src.models.cs_wae_ablation import create_ablation_model
from src.datasets.mnist import get_mnist_loaders
from src.trainers.trainer_ablation import AblationTrainer, NoiseRobustnessEvaluator
from src.metrics.evaluation import ModelEvaluator
from src.visualization.plots import plot_results


def run_single_ablation(variant_name, train_loader, test_loader, results_dir):
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

    # Train model
    trainer = AblationTrainer(model, train_loader, variant_config["name"])
    history = trainer.train()

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

        if model.use_supervised_mmd:
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

        if not model.use_spherical_space:
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
            x_hat, _, _, _ = model(data)

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
    evaluator = ModelEvaluator()

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
    return metrics, history, model


def main():
    """Main function for ablation study"""

    print("=" * 80)
    print("CS-WAE ABLATION STUDY")
    print("=" * 80)
    print("This study systematically evaluates the contribution of each component:")
    print("1. Supervised MMD Loss")
    print("2. Spherical vs Euclidean Space")
    print("3. Spherical Cauchy vs von Mises-Fisher Prior")
    print("4. Noise Robustness")
    print("=" * 80)

    # Create results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"ablation_results/ablation_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)

    # Load data
    print("Loading MNIST dataset...")
    train_loader, test_loader = get_mnist_loaders()

    # Define which variants to run (you can comment out some for faster testing)
    variants_to_run = [
        # "baseline",  # Full CS-WAE
        "no_sup_mmd",  # Without supervised MMD
        "euclidean",  # Euclidean space
        "vmf_prior",  # von Mises-Fisher prior
        "minimal",  # Minimal variant
    ]

    # Run ablation experiments
    all_results = {}
    all_histories = {}
    trained_models = {}

    for variant in variants_to_run:
        try:
            metrics, history, model = run_single_ablation(
                variant, train_loader, test_loader, results_dir
            )
            all_results[ablation_config.ablation_variants[variant]["name"]] = metrics
            all_histories[variant] = history
            trained_models[variant] = model
        except Exception as e:
            print(f"ERROR: Failed to run ablation {variant}: {e}")
            continue

    # Create comprehensive comparison
    print(f"\n{'=' * 80}")
    print("ABLATION STUDY RESULTS")
    print(f"{'=' * 80}")

    if all_results:
        # Create comparison table
        df = pd.DataFrame(all_results).T

        # Reorder columns for better readability
        column_order = ["ACC", "NMI", "ARI", "FID", "LPIPS", "SSIM", "PSNR"]
        existing_cols = [col for col in column_order if col in df.columns]
        df = df[existing_cols]

        print("\nPERFORMANCE COMPARISON:")
        print(df.to_markdown(floatfmt=".4f"))

        # Save results
        df.to_csv(os.path.join(results_dir, "ablation_results.csv"))

        # Create comparison plots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()

        metrics_to_plot = ["ACC", "NMI", "ARI", "FID", "LPIPS", "SSIM"]

        for i, metric in enumerate(metrics_to_plot[:6]):
            if metric in df.columns:
                ax = axes[i]
                values = df[metric].sort_values(
                    ascending=(metric != "FID")
                )  # FID: lower is better

                bars = ax.bar(range(len(values)), values.values)
                ax.set_xticks(range(len(values)))
                ax.set_xticklabels(values.index, rotation=45, ha="right")
                ax.set_title(f"{metric} Comparison")
                ax.set_ylabel(metric)

                # Color bars: green for best, red for worst
                if metric == "FID":  # Lower is better
                    best_idx = values.argmin()
                    worst_idx = values.argmax()
                else:  # Higher is better
                    best_idx = values.argmax()
                    worst_idx = values.argmin()

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

        # Training loss comparison
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

    # Noise robustness evaluation (optional - can be time-consuming)
    run_noise_evaluation = (
        input("\nRun noise robustness evaluation? (y/n): ").lower().strip() == "y"
    )

    if run_noise_evaluation and trained_models:
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

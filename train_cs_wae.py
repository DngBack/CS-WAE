"""
Main training and evaluation script for CS-WAE
"""
import argparse
import json
import os

import torch

from src.config import config
from src.models import SphericalWAE_Supervised
from src.datasets.loaders import get_loaders, get_dataset_info, get_default_runs_dir, SUPPORTED_DATASETS
from src.utils.dataset_config import apply_dataset_config
from src.trainers.trainer import CSWAETrainer
from src.visualization.plots import plot_results, plot_slerp
from src.metrics.evaluation import ModelEvaluator
from src.utils.seed import set_seed
from src.utils.device import set_device
from src.utils.run_io import config_to_dict, save_run_metadata, save_metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Train and evaluate CS-WAE")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: runs/mnist/seed_<seed>)",
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
        default=None,
        help="Override number of training epochs",
    )
    parser.add_argument(
        "--skip-viz",
        action="store_true",
        help="Skip basic visualizations (loss, UMAP, slerp)",
    )
    parser.add_argument(
        "--skip-advanced-viz",
        action="store_true",
        help="Skip advanced visual tests",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device, e.g. cuda:0 or cuda:1 (default: cuda:0)",
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
    set_seed(args.seed)
    device = set_device(args.device)

    dataset_info = apply_dataset_config(args.dataset, get_dataset_info, backbone=args.backbone)

    save_dir = args.output_dir or f"{get_default_runs_dir(args.dataset, config.backbone)}/seed_{args.seed}"
    os.makedirs(save_dir, exist_ok=True)

    epochs = args.epochs or config.epochs

    print("=" * 60)
    print("CS-WAE Training and Evaluation")
    print("=" * 60)
    print(f"Seed: {args.seed}")
    print(f"Device: {device}")
    print(f"Dataset: {args.dataset}")
    print(f"Backbone: {config.backbone}")
    print(f"Output: {save_dir}")
    print(f"Epochs: {epochs}")

    save_run_metadata(
        save_dir,
        config_to_dict(config),
        args.seed,
        extra={"dataset": args.dataset, "epochs": epochs, "backbone": config.backbone},
    )

    print("Loading dataset...")
    train_loader, test_loader = get_loaders(
        dataset=args.dataset,
        seed=args.seed,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
    )

    print(f"Initializing CS-WAE model on device: {config.device}")
    model = SphericalWAE_Supervised(
        latent_dim=config.latent_dim,
        n_classes=dataset_info["n_classes"],
        in_channels=dataset_info.get("in_channels", 1),
        image_size=dataset_info.get("image_size", 28),
        backbone=config.backbone,
    ).to(config.device)

    trainer = CSWAETrainer(model, train_loader)

    print("Starting training...")
    history = trainer.train(epochs=epochs)

    history_path = os.path.join(save_dir, "training_history.json")
    with open(history_path, "w") as f:
        json.dump(
            [
                {
                    "total": h[0],
                    "recon": h[1],
                    "sup_mmd": h[2],
                    "unsup_mmd": h[3],
                }
                for h in history
            ],
            f,
            indent=2,
        )

    model_path = os.path.join(save_dir, "cs_wae_model.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to: {model_path}")

    if not args.skip_viz:
        print("Generating visualizations...")
        plot_results(history, model, test_loader, save_dir=save_dir, device=device)
        plot_slerp(model, save_dir=save_dir, device=device)

    if not args.skip_viz and not args.skip_advanced_viz:
        print("Generating advanced visual analysis...")
        from src.visualization.advanced_tests import run_all_advanced_tests

        run_all_advanced_tests(
            model=model,
            test_dataset=test_loader.dataset,
            test_loader=test_loader,
            save_dir=save_dir,
            device=device,
        )

    print("Starting comprehensive evaluation...")
    evaluator = ModelEvaluator(device=device, dataset=args.dataset)
    metrics = evaluator.comprehensive_evaluation(
        model, "CS-WAE", test_loader, save_dir
    )
    save_metrics(save_dir, metrics)

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    for metric, value in metrics.items():
        if isinstance(value, (int, float)):
            print(f"{metric:<15}: {value:.4f}")
        else:
            print(f"{metric:<15}: {value}")
    print("=" * 60)

    print(f"\nAll results saved to: {save_dir}")
    print("Training and evaluation completed successfully!")


if __name__ == "__main__":
    main()

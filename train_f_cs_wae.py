"""
Training and evaluation entry point for F-CS-WAE.

Usage examples:
    # CIFAR-10 (default: ResNet-18, 300 epochs)
    python train_f_cs_wae.py --dataset cifar10 --seed 0 --device cuda:0

    # Smoke test (10 epochs)
    python train_f_cs_wae.py --dataset cifar10 --seed 0 --device cuda:0 --epochs 10

    # MNIST with CNN-style settings (override dims)
    python train_f_cs_wae.py --dataset mnist --seed 0 --device cuda:0
"""

import argparse
import json
import os

import torch

from src.config_f_cs_wae import f_cs_wae_config as cfg
from src.models.f_cs_wae import FCSWAE
from src.datasets.loaders import (
    get_loaders,
    get_dataset_info,
    get_default_runs_dir,
    SUPPORTED_DATASETS,
)
from src.utils.dataset_config import apply_dataset_config
from src.trainers.trainer_f_cs_wae import FCSWAETrainer
from src.metrics.evaluation import ModelEvaluator
from src.utils.seed import set_seed
from src.utils.device import set_device
from src.utils.run_io import config_to_dict, save_run_metadata, save_metrics


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train and evaluate F-CS-WAE")
    p.add_argument("--seed",        type=int,   default=0)
    p.add_argument("--dataset",     type=str,   default="cifar10",
                   choices=list(SUPPORTED_DATASETS))
    p.add_argument("--device",      type=str,   default=None,
                   help="e.g. cuda:0 or cpu")
    p.add_argument("--epochs",      type=int,   default=None,
                   help="Override total training epochs")
    p.add_argument("--n-centers",   type=int,   default=None,
                   help="Centers per class (default: cfg.n_centers = 1)")
    p.add_argument("--output-dir",  type=str,   default=None,
                   help="Output directory (default: runs_f/<dataset>/seed_<N>)")
    p.add_argument("--skip-eval",   action="store_true",
                   help="Skip comprehensive evaluation (useful for quick training runs)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = set_device(args.device)

    # Sync global config with dataset
    dataset_info = apply_dataset_config(args.dataset, get_dataset_info, backbone="resnet18")

    # Resolve output directory
    runs_root = f"runs_f/{args.dataset}"
    save_dir = args.output_dir or os.path.join(runs_root, f"seed_{args.seed}")
    os.makedirs(save_dir, exist_ok=True)

    # Apply CLI overrides to config
    epochs    = args.epochs    or cfg.total_epochs
    n_centers = args.n_centers or cfg.n_centers

    print("=" * 60)
    print("F-CS-WAE Training")
    print("=" * 60)
    print(f"  Seed       : {args.seed}")
    print(f"  Device     : {device}")
    print(f"  Dataset    : {args.dataset}")
    print(f"  Epochs     : {epochs}")
    print(f"  n_centers  : {n_centers}")
    print(f"  Output     : {save_dir}")
    print(f"  semantic_dim: {cfg.semantic_dim}  style_dim: {cfg.style_dim}")

    # Save run metadata
    run_meta = config_to_dict(cfg)
    run_meta.update({
        "dataset":    args.dataset,
        "epochs":     epochs,
        "n_centers":  n_centers,
        "seed":       args.seed,
        "backbone":   "resnet18",
    })
    save_run_metadata(save_dir, run_meta, args.seed, extra=run_meta)

    # Data
    print("Loading dataset ...")
    train_loader, test_loader = get_loaders(
        dataset=args.dataset,
        seed=args.seed,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
    )

    # Model
    print("Building F-CS-WAE model ...")
    model = FCSWAE(
        semantic_dim  = cfg.semantic_dim,
        style_dim     = cfg.style_dim,
        n_classes     = dataset_info["n_classes"],
        in_channels   = dataset_info.get("in_channels", 3),
        image_size    = dataset_info.get("image_size",  32),
        n_centers     = n_centers,
        rho_prior     = cfg.rho_prior,
        ema_momentum  = cfg.ema_momentum,
    ).to(device)

    # Trainer
    trainer = FCSWAETrainer(model, train_loader, device=device)

    # Train
    print("Starting training ...")
    history = trainer.train(epochs=epochs)

    # Save history
    history_path = os.path.join(save_dir, "training_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"Training history → {history_path}")

    # Save model weights
    model_path = os.path.join(save_dir, "f_cs_wae_model.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved → {model_path}")

    # Evaluation
    if not args.skip_eval:
        print("Evaluating ...")
        evaluator = ModelEvaluator(device=device, dataset=args.dataset)
        metrics = evaluator.comprehensive_evaluation(
            model, "F-CS-WAE", test_loader, save_dir
        )
        save_metrics(save_dir, metrics)

        print("\n" + "=" * 60)
        print("EVALUATION RESULTS")
        print("=" * 60)
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                print(f"  {k:<15}: {v:.4f}")
            else:
                print(f"  {k:<15}: {v}")
        print("=" * 60)

    print(f"\nAll results saved to: {save_dir}")
    print("F-CS-WAE training and evaluation complete.")


if __name__ == "__main__":
    main()

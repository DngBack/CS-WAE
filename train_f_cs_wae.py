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
from src.trainers.trainer_f_cs_wae import FCSWAETrainer, atomic_torch_save
from src.metrics.evaluation import ModelEvaluator
from src.metrics.audit_protocol import AUDIT_PROTOCOL_VERSION
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
    p.add_argument("--delta-final", type=float, default=None,
                   help="Per-class style MMD weight (0=disabled, default: cfg.delta_final=1.0). "
                        "Set 0 to reproduce baseline without the proposed fix.")
    p.add_argument("--dc",          type=int,   default=None,
                   help="Override semantic_dim d_c (default: cfg.semantic_dim=64)")
    p.add_argument("--ds",          type=int,   default=None,
                   help="Override style_dim d_s (default: cfg.style_dim=128)")
    p.add_argument("--phase-a-end", type=int,   default=None,
                   help="Override cfg.phase_a_end (default 50). Set 0 to skip the "
                        "reconstruction-only warmup and start style/class regularization "
                        "from epoch 0.")
    p.add_argument("--style-sigma-floor", type=float, default=None,
                   help="Lower bound on posterior style standard deviation. The "
                        "effective variance is exp(clamp(logvar,-10,10)) + floor^2.")
    p.add_argument("--output-dir",  type=str,   default=None,
                   help="Output directory (default: runs_f/<dataset>/seed_<N>)")
    p.add_argument("--skip-eval",   action="store_true",
                   help="Skip comprehensive evaluation (useful for quick training runs)")
    p.add_argument("--resume-from", type=str, default=None,
                   help="Resume from an explicit crash-safe training checkpoint")
    p.add_argument("--auto-resume", action="store_true",
                   help="Resume from <output-dir>/training_checkpoint.pt when it exists")
    p.add_argument("--checkpoint-every", type=int, default=5,
                   help="Save crash-safe state every N completed epochs (default: 5)")
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
    if args.delta_final is not None:
        cfg.delta_final = args.delta_final
    cfg.semantic_dim = args.dc if args.dc is not None else cfg.semantic_dim
    cfg.style_dim    = args.ds if args.ds is not None else cfg.style_dim
    if args.phase_a_end is not None:
        cfg.phase_a_end = args.phase_a_end
        assert cfg.phase_a_end < cfg.phase_b_end, \
            f"--phase-a-end ({cfg.phase_a_end}) must be < phase_b_end ({cfg.phase_b_end})"
    if args.style_sigma_floor is not None:
        if args.style_sigma_floor < 0.0:
            raise ValueError("--style-sigma-floor must be non-negative")
        cfg.style_sigma_floor = args.style_sigma_floor
    if args.checkpoint_every < 1:
        raise ValueError("--checkpoint-every must be positive")

    print("=" * 60)
    print("F-CS-WAE Training")
    print("=" * 60)
    print(f"  Seed       : {args.seed}")
    print(f"  Device     : {device}")
    print(f"  Dataset    : {args.dataset}")
    print(f"  Epochs     : {epochs}")
    print(f"  n_centers  : {n_centers}")
    print(f"  delta_final: {cfg.delta_final}  (per-class style MMD; 0=disabled)")
    print(f"  Output     : {save_dir}")
    print(f"  semantic_dim: {cfg.semantic_dim}  style_dim: {cfg.style_dim}")
    print(f"  phase_a_end: {cfg.phase_a_end}  (0 = skip reconstruction-only warmup)")
    print(f"  style sigma floor: {cfg.style_sigma_floor}")

    # Save run metadata
    run_meta = config_to_dict(cfg)
    run_meta.update({
        "dataset":    args.dataset,
        "epochs":     epochs,
        "n_centers":  n_centers,
        "seed":       args.seed,
        "backbone":   "resnet18",
        "semantic_dim": cfg.semantic_dim,
        "style_dim": cfg.style_dim,
        "delta_final": cfg.delta_final,
        "style_sigma_floor": cfg.style_sigma_floor,
        "phase_a_end": cfg.phase_a_end,
        "phase_b_end": cfg.phase_b_end,
        "audit_protocol_version": AUDIT_PROTOCOL_VERSION,
        "checkpoint_every": args.checkpoint_every,
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
        style_sigma_floor = cfg.style_sigma_floor,
    ).to(device)

    # Trainer
    trainer = FCSWAETrainer(model, train_loader, device=device)

    checkpoint_path = os.path.join(save_dir, "training_checkpoint.pt")
    resume_path = args.resume_from
    if resume_path is None and args.auto_resume and os.path.exists(checkpoint_path):
        resume_path = checkpoint_path
    start_epoch = 0
    history = []
    if resume_path is not None:
        if not os.path.exists(resume_path):
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        start_epoch, history = trainer.load_training_checkpoint(
            resume_path,
            expected_run_config=run_meta,
        )
        print(
            f"Resumed full training state from {resume_path} "
            f"after {start_epoch}/{epochs} epochs"
        )

    # Train
    print("Starting training ...")
    history = trainer.train(
        epochs=epochs,
        start_epoch=start_epoch,
        history=history,
        checkpoint_path=checkpoint_path,
        checkpoint_every=args.checkpoint_every,
        run_config=run_meta,
    )

    # Save history
    history_path = os.path.join(save_dir, "training_history.json")
    history_tmp_path = f"{history_path}.tmp"
    with open(history_tmp_path, "w") as f:
        json.dump(history, f, indent=2)
    os.replace(history_tmp_path, history_path)
    print(f"Training history → {history_path}")

    # Save model weights
    model_path = os.path.join(save_dir, "f_cs_wae_model.pth")
    atomic_torch_save(model.state_dict(), model_path)
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

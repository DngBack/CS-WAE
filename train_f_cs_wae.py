"""
Training and evaluation entry point for F-CS-WAE.

Usage examples:
    # CIFAR-10 (default: ResNet-18, 300 epochs)
    python train_f_cs_wae.py --dataset cifar10 --seed 0 --device cuda:0

    # Smoke test (10 epochs)
    python train_f_cs_wae.py --dataset cifar10 --seed 0 --device cuda:0 --epochs 10

    # MNIST with CNN-style settings (override dims)
    python train_f_cs_wae.py --dataset mnist --seed 0 --device cuda:0

    # FACT smoke test (two batches per epoch; never use the cap for real runs)
    python train_f_cs_wae.py --dataset mnist --device cuda:0 --epochs 2 \
        --phase-a-end 0 --phase-b-end 1 --phase-c-end 2 --phase-d-end 3 \
        --fact --max-train-batches 2 --skip-eval
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
    p.add_argument("--joint-contract-final", type=float, default=None,
                   help="Weight lambda_J for per-class product-kernel joint contract MMD "
                        "(default: 0, disabled).")
    p.add_argument("--fact", action="store_true",
                   help="Enable null-calibrated clause-aligned FACT training.")
    p.add_argument("--fact-start-epoch", type=int, default=None,
                   help="Zero-indexed epoch where FACT replaces legacy class/per-class-style "
                        "penalties (default: phase B end).")
    p.add_argument("--fact-dual-lr", type=float, default=None,
                   help="Projected dual-ascent learning rate (default: 0.1).")
    p.add_argument("--fact-dual-init", type=float, default=None,
                   help="Initial multiplier for each FACT constraint (default: 1.0).")
    p.add_argument("--fact-dual-max", type=float, default=None,
                   help="Maximum FACT multiplier (default: 10.0).")
    p.add_argument("--fact-null-draws", type=int, default=None,
                   help="Independent matched-null estimates averaged per class.")
    p.add_argument("--fact-dual-reference-style", type=float, default=None,
                   help="Positive reference magnitude for normalized style dual updates.")
    p.add_argument("--fact-dual-reference-content", type=float, default=None,
                   help="Positive reference magnitude for normalized content dual updates.")
    p.add_argument("--fact-dual-reference-dependence", type=float, default=None,
                   help="Positive reference magnitude for normalized dependence dual updates.")
    p.add_argument("--fact-dual-step-max", type=float, default=None,
                   help="Optional positive cap on each epoch's dual multiplier update.")
    p.add_argument("--fact-label-hsic-weight", type=float, default=None,
                   help="Fixed non-negative weight for matched-null style-label HSIC.")
    p.add_argument("--fact-label-adversary-weight", type=float, default=None,
                   help="Non-negative encoder weight for nonlinear style-label confusion.")
    p.add_argument("--fact-label-adversary-lr", type=float, default=None,
                   help="Positive learning rate for the alternating label adversary.")
    p.add_argument("--fact-label-adversary-steps", type=int, default=None,
                   help="Positive adversary updates per encoder minibatch.")
    p.add_argument("--fact-label-adversary-weight-decay", type=float, default=None,
                   help="Non-negative adversary Adam weight decay.")
    p.add_argument("--fact-mean-hsic-weight", type=float, default=None,
                   help="Fixed non-negative weight for audit-aligned mean conditional HSIC.")
    p.add_argument("--fact-style-tolerance", type=float, default=None)
    p.add_argument("--fact-content-tolerance", type=float, default=None)
    p.add_argument("--fact-dependence-tolerance", type=float, default=None)
    p.add_argument("--fact-update-content-from-hsic", action="store_true",
                   help="Allow conditional HSIC gradients into content. By default FACT "
                        "detaches content and repairs dependence through style only.")
    p.add_argument("--dc",          type=int,   default=None,
                   help="Override semantic_dim d_c (default: cfg.semantic_dim=64)")
    p.add_argument("--ds",          type=int,   default=None,
                   help="Override style_dim d_s (default: cfg.style_dim=128)")
    p.add_argument("--phase-a-end", type=int,   default=None,
                   help="Override cfg.phase_a_end (default 50). Set 0 to skip the "
                        "reconstruction-only warmup and start style/class regularization "
                        "from epoch 0.")
    p.add_argument("--phase-b-end", type=int, default=None,
                   help="Override cfg.phase_b_end (default 100).")
    p.add_argument("--phase-c-end", type=int, default=None,
                   help="Override cfg.phase_c_end (default 200).")
    p.add_argument("--phase-d-end", type=int, default=None,
                   help="Override cfg.phase_d_end (default 300).")
    p.add_argument(
        "--phase-weight-freeze-epoch",
        type=int,
        default=None,
        help=(
            "Zero-indexed epoch whose scheduled alpha/beta/gamma/delta/eta "
            "weights are reused for all later epochs; FACT remains active."
        ),
    )
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
    p.add_argument("--batch-size", type=int, default=None,
                   help="Override training/evaluation batch size. FACT pilots should use "
                        "a larger batch (e.g. 320 on MNIST) for more samples per class.")
    p.add_argument("--max-train-batches", type=int, default=None,
                   help="Debug/smoke-only cap on batches per epoch. Omit for real runs.")
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
    if args.joint_contract_final is not None:
        if args.joint_contract_final < 0.0:
            raise ValueError("--joint-contract-final must be non-negative")
        cfg.joint_contract_final = args.joint_contract_final
    cfg.fact_enabled = bool(args.fact)
    cfg.semantic_dim = args.dc if args.dc is not None else cfg.semantic_dim
    cfg.style_dim    = args.ds if args.ds is not None else cfg.style_dim
    if args.phase_a_end is not None:
        cfg.phase_a_end = args.phase_a_end
    if args.phase_b_end is not None:
        cfg.phase_b_end = args.phase_b_end
    if args.phase_c_end is not None:
        cfg.phase_c_end = args.phase_c_end
    if args.phase_d_end is not None:
        cfg.phase_d_end = args.phase_d_end
    if args.phase_weight_freeze_epoch is not None:
        cfg.phase_weight_freeze_epoch = args.phase_weight_freeze_epoch
    if not (0 <= cfg.phase_a_end < cfg.phase_b_end < cfg.phase_c_end < cfg.phase_d_end):
        raise ValueError(
            "phase boundaries must satisfy 0 <= A < B < C < D; got "
            f"{cfg.phase_a_end}, {cfg.phase_b_end}, {cfg.phase_c_end}, {cfg.phase_d_end}"
        )
    if epochs > cfg.phase_d_end:
        raise ValueError(
            f"epochs ({epochs}) cannot exceed phase_d_end ({cfg.phase_d_end})"
        )
    if cfg.phase_weight_freeze_epoch is not None and not (
        cfg.phase_c_end <= cfg.phase_weight_freeze_epoch < epochs
    ):
        raise ValueError(
            "phase weight freeze must satisfy phase_c_end <= freeze < epochs; "
            f"got freeze={cfg.phase_weight_freeze_epoch}, "
            f"phase_c_end={cfg.phase_c_end}, epochs={epochs}"
        )
    if cfg.fact_enabled and cfg.joint_contract_final > 0.0:
        raise ValueError(
            "--fact and --joint-contract-final > 0 are mutually exclusive; "
            "the direct joint MMD is an evaluation metric in FACT runs"
        )
    if args.fact_start_epoch is not None and not cfg.fact_enabled:
        raise ValueError("--fact-start-epoch requires --fact")
    if cfg.fact_enabled:
        cfg.fact_start_epoch = (
            args.fact_start_epoch
            if args.fact_start_epoch is not None
            else cfg.phase_b_end
        )
        if not (0 <= cfg.fact_start_epoch < epochs):
            raise ValueError(
                "FACT start must satisfy 0 <= start < epochs; got "
                f"{cfg.fact_start_epoch} for {epochs} epochs"
            )
    cfg.fact_style_only_dependence = not args.fact_update_content_from_hsic
    fact_overrides = {
        "fact_dual_lr": args.fact_dual_lr,
        "fact_dual_init": args.fact_dual_init,
        "fact_dual_max": args.fact_dual_max,
        "fact_style_tolerance": args.fact_style_tolerance,
        "fact_content_tolerance": args.fact_content_tolerance,
        "fact_dependence_tolerance": args.fact_dependence_tolerance,
    }
    for name, value in fact_overrides.items():
        if value is not None:
            if value < 0.0:
                raise ValueError(f"--{name.replace('_', '-')} must be non-negative")
            setattr(cfg, name, value)
    if args.fact_null_draws is not None:
        if args.fact_null_draws < 1:
            raise ValueError("--fact-null-draws must be positive")
        cfg.fact_null_draws = args.fact_null_draws
    fact_reference_overrides = {
        "fact_dual_reference_style": args.fact_dual_reference_style,
        "fact_dual_reference_content": args.fact_dual_reference_content,
        "fact_dual_reference_dependence": args.fact_dual_reference_dependence,
    }
    for name, value in fact_reference_overrides.items():
        if value is not None:
            if value <= 0.0:
                raise ValueError(f"--{name.replace('_', '-')} must be positive")
            setattr(cfg, name, value)
    if args.fact_dual_step_max is not None:
        if args.fact_dual_step_max <= 0.0:
            raise ValueError("--fact-dual-step-max must be positive")
        cfg.fact_dual_step_max = args.fact_dual_step_max
    if args.fact_label_hsic_weight is not None:
        if args.fact_label_hsic_weight < 0.0:
            raise ValueError("--fact-label-hsic-weight must be non-negative")
        cfg.fact_label_hsic_weight = args.fact_label_hsic_weight
    if args.fact_label_adversary_weight is not None:
        if args.fact_label_adversary_weight < 0.0:
            raise ValueError("--fact-label-adversary-weight must be non-negative")
        cfg.fact_label_adversary_weight = args.fact_label_adversary_weight
    if args.fact_label_adversary_lr is not None:
        if args.fact_label_adversary_lr <= 0.0:
            raise ValueError("--fact-label-adversary-lr must be positive")
        cfg.fact_label_adversary_lr = args.fact_label_adversary_lr
    if args.fact_label_adversary_steps is not None:
        if args.fact_label_adversary_steps < 1:
            raise ValueError("--fact-label-adversary-steps must be positive")
        cfg.fact_label_adversary_steps = args.fact_label_adversary_steps
    if args.fact_label_adversary_weight_decay is not None:
        if args.fact_label_adversary_weight_decay < 0.0:
            raise ValueError(
                "--fact-label-adversary-weight-decay must be non-negative"
            )
        cfg.fact_label_adversary_weight_decay = (
            args.fact_label_adversary_weight_decay
        )
    if args.fact_mean_hsic_weight is not None:
        if args.fact_mean_hsic_weight < 0.0:
            raise ValueError("--fact-mean-hsic-weight must be non-negative")
        cfg.fact_mean_hsic_weight = args.fact_mean_hsic_weight
    if not cfg.fact_enabled and (
        cfg.fact_label_hsic_weight > 0.0
        or cfg.fact_label_adversary_weight > 0.0
        or cfg.fact_mean_hsic_weight > 0.0
    ):
        raise ValueError("label HSIC/adversary penalties require --fact")
    if cfg.fact_dual_max < cfg.fact_dual_init:
        raise ValueError("--fact-dual-max must be at least --fact-dual-init")
    if args.batch_size is not None:
        if args.batch_size < 2:
            raise ValueError("--batch-size must be at least 2")
        cfg.batch_size = args.batch_size
    if args.max_train_batches is not None:
        if args.max_train_batches < 1:
            raise ValueError("--max-train-batches must be positive")
        cfg.max_train_batches = args.max_train_batches
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
    print(f"  joint_contract_final: {cfg.joint_contract_final}")
    print(f"  FACT       : {'enabled' if cfg.fact_enabled else 'disabled'}")
    if cfg.fact_enabled:
        print(
            "  FACT config: "
            f"start={cfg.fact_start_epoch} dual_lr={cfg.fact_dual_lr} "
            f"dual_init={cfg.fact_dual_init} dual_max={cfg.fact_dual_max} "
            f"style_only_dependence={cfg.fact_style_only_dependence} "
            f"null_draws={cfg.fact_null_draws}"
        )
        print(
            "  FACT dual references (style/content/dependence): "
            f"{cfg.fact_dual_reference_style}/{cfg.fact_dual_reference_content}/"
            f"{cfg.fact_dual_reference_dependence}; step_max={cfg.fact_dual_step_max}"
        )
        print(f"  FACT style-label HSIC weight: {cfg.fact_label_hsic_weight}")
        print(
            "  FACT nonlinear label adversary (weight/lr/steps/wd): "
            f"{cfg.fact_label_adversary_weight}/"
            f"{cfg.fact_label_adversary_lr}/"
            f"{cfg.fact_label_adversary_steps}/"
            f"{cfg.fact_label_adversary_weight_decay}"
        )
        print(f"  FACT audit-aligned mean HSIC weight: {cfg.fact_mean_hsic_weight}")
        print(
            "  FACT tolerances (style/content/dependence): "
            f"{cfg.fact_style_tolerance}/{cfg.fact_content_tolerance}/"
            f"{cfg.fact_dependence_tolerance}"
        )
    print(f"  batch_size : {cfg.batch_size}")
    if cfg.max_train_batches is not None:
        print(f"  max batches: {cfg.max_train_batches} (debug/smoke only)")
    print(f"  Output     : {save_dir}")
    print(f"  semantic_dim: {cfg.semantic_dim}  style_dim: {cfg.style_dim}")
    print(
        "  phase boundaries: "
        f"A={cfg.phase_a_end} B={cfg.phase_b_end} C={cfg.phase_c_end} D={cfg.phase_d_end}"
    )
    print(f"  phase weight freeze: {cfg.phase_weight_freeze_epoch}")
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
        "joint_contract_final": cfg.joint_contract_final,
        "fact_enabled": cfg.fact_enabled,
        "fact_start_epoch": cfg.fact_start_epoch,
        "fact_style_only_dependence": cfg.fact_style_only_dependence,
        "fact_dual_lr": cfg.fact_dual_lr,
        "fact_dual_init": cfg.fact_dual_init,
        "fact_dual_max": cfg.fact_dual_max,
        "fact_null_draws": cfg.fact_null_draws,
        "fact_dual_reference_style": cfg.fact_dual_reference_style,
        "fact_dual_reference_content": cfg.fact_dual_reference_content,
        "fact_dual_reference_dependence": cfg.fact_dual_reference_dependence,
        "fact_dual_step_max": cfg.fact_dual_step_max,
        "fact_label_hsic_weight": cfg.fact_label_hsic_weight,
        "fact_label_adversary_weight": cfg.fact_label_adversary_weight,
        "fact_label_adversary_lr": cfg.fact_label_adversary_lr,
        "fact_label_adversary_steps": cfg.fact_label_adversary_steps,
        "fact_label_adversary_weight_decay": (
            cfg.fact_label_adversary_weight_decay
        ),
        "fact_mean_hsic_weight": cfg.fact_mean_hsic_weight,
        "fact_style_tolerance": cfg.fact_style_tolerance,
        "fact_content_tolerance": cfg.fact_content_tolerance,
        "fact_dependence_tolerance": cfg.fact_dependence_tolerance,
        "batch_size": cfg.batch_size,
        "max_train_batches": cfg.max_train_batches,
        "style_sigma_floor": cfg.style_sigma_floor,
        "phase_a_end": cfg.phase_a_end,
        "phase_b_end": cfg.phase_b_end,
        "phase_c_end": cfg.phase_c_end,
        "phase_d_end": cfg.phase_d_end,
        "phase_weight_freeze_epoch": cfg.phase_weight_freeze_epoch,
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

"""
Ablation study runner for F-CS-WAE.

Trains all 9 ablation variants (or a specified subset) and exports results.

Variants
--------
    full              no_z_s           no_class_mmd
    no_style_mmd      no_classifier    gaussian_class_prior
    vmf_class_prior   single_center    learnable_centers

Usage
-----
    # All 9 variants
    python run_ablation_f_cs_wae.py --dataset cifar10 --device cuda:0

    # Specific variants
    python run_ablation_f_cs_wae.py --dataset cifar10 --device cuda:0 \\
        --variants full no_z_s no_class_mmd

    # Summarize from previously saved results
    python run_ablation_f_cs_wae.py --summarize-only --results-dir runs_f/cifar10/f_ablation_20260624_120000

Output
------
    <results-dir>/f_ablation_results.csv
    <results-dir>/f_ablation_results.json
"""

from __future__ import annotations

import argparse
import json
import os
import warnings
from datetime import datetime

import pandas as pd
import torch

warnings.filterwarnings("ignore")

from src.config_f_cs_wae import f_cs_wae_config as cfg
from src.models.f_cs_wae_ablation import (
    ABLATION_VARIANTS,
    FCSWAEAblation,
    create_f_cs_wae_ablation,
)
from src.datasets.loaders import get_loaders, get_dataset_info, SUPPORTED_DATASETS
from src.utils.dataset_config import apply_dataset_config
from src.trainers.trainer_f_cs_wae import FCSWAETrainer
from src.metrics.evaluation import ModelEvaluator
from src.utils.seed import set_seed
from src.utils.device import set_device
from src.utils.run_io import save_metrics
import lpips

ALL_VARIANTS = list(ABLATION_VARIANTS.keys())
METRIC_COLS  = ["ACC", "NMI", "ARI", "FID", "LPIPS", "SSIM", "PSNR"]


# ---------------------------------------------------------------------------
# Trainer adapter for ablation models
# ---------------------------------------------------------------------------

class FCSWAEAblationTrainer(FCSWAETrainer):
    """
    Subclass of FCSWAETrainer that handles FCSWAEAblation models.

    The key differences:
    - forward returns (x_hat, z_c, z_s, mu_c, mu_s, logvar_s) for full variants
      but z_s / mu_s / logvar_s may be None for no_z_s variant.
    - Loss weights follow the same 4-phase schedule as FCSWAETrainer.
    - EMA update is forwarded to model.update_ema_centers().
    """

    def train_epoch(self, epoch: int) -> dict[str, float]:
        from src.utils.loss_f_cs_wae import calculate_f_cs_wae_loss

        self.model.train()
        alpha, beta, gamma, delta, eta = self.get_phase_weights(epoch)

        # Override weights per variant config
        vc = self.model.variant_config
        if not vc["use_class_mmd"]:
            alpha = 0.0
        if not vc["use_style_mmd"]:
            gamma = 0.0
        if not vc["use_classifier"]:
            eta = 0.0
        # None of the 9 ABLATION_VARIANTS touch per-class style MMD (delta) —
        # it is orthogonal to this axis, so it passes through unmodified,
        # matching the "full" trainer's behavior.

        acc = {k: 0.0 for k in
               ("total", "rec", "class_mmd", "agg_mmd", "style_mmd", "style_cls_mmd", "cls", "var")}

        mu_c_accum: list[torch.Tensor] = []
        y_accum:    list[torch.Tensor] = []

        from tqdm import tqdm
        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch + 1} [{self.model.variant_name}] "
                 f"α={alpha:.2f} β={beta:.2f} γ={gamma:.2f} δ={delta:.2f} η={eta:.2f}",
        )

        for data, labels in pbar:
            data, labels = data.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()

            x_hat, z_c, z_s, mu_c, mu_s, logvar_s = self.model(data)
            mu_c_accum.append(mu_c.detach())
            y_accum.append(labels.detach())

            # For variants without style latent, pass zeros as placeholders
            if z_s is None:
                z_s_ph     = torch.zeros(data.shape[0], 1, device=self.device)
                mu_s_ph    = z_s_ph
                logvar_s_ph = z_s_ph
                gamma_eff  = 0.0
                delta_eff  = 0.0
                lv_eff     = 0.0
            else:
                z_s_ph     = z_s
                mu_s_ph    = mu_s
                logvar_s_ph = logvar_s
                gamma_eff  = gamma
                delta_eff  = delta
                lv_eff     = cfg.lambda_var

            # Classifier placeholder if not used
            if not vc["use_classifier"] and not hasattr(self.model, "classifier"):
                eta_eff = 0.0
            else:
                eta_eff = eta

            total, L_rec, L_class, L_agg, L_style, L_style_cls, L_cls, L_var = (
                calculate_f_cs_wae_loss(
                    data, labels, x_hat, z_c, z_s_ph, mu_c, mu_s_ph, logvar_s_ph,
                    self.model, self.loss_fn_vgg,
                    alpha, beta, gamma_eff, delta_eff, eta_eff, lv_eff,
                )
            )

            total.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)
            self.optimizer.step()

            acc["total"]         += total.item()
            acc["rec"]           += L_rec.item()
            acc["class_mmd"]     += L_class.item()
            acc["agg_mmd"]       += L_agg.item()
            acc["style_mmd"]     += L_style.item()
            acc["style_cls_mmd"] += L_style_cls.item()
            acc["cls"]           += L_cls.item()
            acc["var"]           += L_var.item()

            pbar.set_postfix({"Loss": f"{total.item():.3f}", "Rec": f"{L_rec.item():.3f}"})

        self.scheduler.step()
        self.model.update_ema_centers(mu_c_accum, y_accum)

        n = len(self.train_loader)
        avg = {k: v / n for k, v in acc.items()}
        print(
            f"====> Epoch {epoch + 1} [{self.model.variant_name}] "
            f"Total={avg['total']:.4f}  Rec={avg['rec']:.4f}  "
            f"ClassMMD={avg['class_mmd']:.4f}  Cls={avg['cls']:.4f}"
        )
        return avg


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="F-CS-WAE ablation study")
    p.add_argument("--dataset",     type=str, default="cifar10",
                   choices=list(SUPPORTED_DATASETS))
    p.add_argument("--seed",        type=int, default=0)
    p.add_argument("--device",      type=str, default=None)
    p.add_argument("--epochs",      type=int, default=None)
    p.add_argument("--variants",    type=str, nargs="+", default=None,
                   choices=ALL_VARIANTS,
                   help=f"Variants to run (default: all). Choices: {ALL_VARIANTS}")
    p.add_argument("--results-dir", type=str, default=None,
                   help="Output directory for this ablation run")
    p.add_argument("--delta-final", type=float, default=None,
                   help="Per-class style MMD weight override (default: cfg.delta_final=1.0). "
                        "Set 0 to disable the remedy, matching the delta=0 baseline in "
                        "train_f_cs_wae.py.")
    p.add_argument("--summarize-only", action="store_true",
                   help="Load saved metrics and print table only")
    p.add_argument("--skip-aggregate", action="store_true",
                   help="Train variants but skip final CSV aggregation")
    return p.parse_args()


def load_existing_metrics(results_dir: str) -> dict[str, dict]:
    all_results: dict[str, dict] = {}
    for entry in os.listdir(results_dir):
        variant_dir   = os.path.join(results_dir, entry)
        metrics_path  = os.path.join(variant_dir, "metrics.json")
        if os.path.isdir(variant_dir) and os.path.exists(metrics_path):
            with open(metrics_path) as f:
                all_results[entry] = json.load(f)
    return all_results


def print_and_save_table(all_results: dict[str, dict], results_dir: str) -> None:
    rows = []
    for variant_key, metrics in sorted(all_results.items()):
        vc = ABLATION_VARIANTS.get(variant_key, {})
        row = {
            "Variant":         vc.get("name", variant_key),
            "Key":             variant_key,
            "Description":     vc.get("description", ""),
        }
        for col in METRIC_COLS:
            row[col] = round(metrics.get(col, float("nan")), 4)
        rows.append(row)

    df = pd.DataFrame(rows).set_index("Key")
    print("\n" + "=" * 100)
    print(" F-CS-WAE ABLATION RESULTS ".center(100, "="))
    print("=" * 100)
    print(df[["Variant"] + METRIC_COLS].to_markdown(floatfmt=".4f"))
    print("=" * 100)

    csv_path = os.path.join(results_dir, "f_ablation_results.csv")
    df.to_csv(csv_path)
    print(f"\nSaved: {csv_path}")

    json_path = os.path.join(results_dir, "f_ablation_results.json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"Saved: {json_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args   = parse_args()
    set_seed(args.seed)
    device = set_device(args.device)

    dataset_info = apply_dataset_config(args.dataset, get_dataset_info, backbone="resnet18")
    n_classes    = dataset_info["n_classes"]
    in_channels  = dataset_info.get("in_channels", 3)
    image_size   = dataset_info.get("image_size",  32)

    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = args.results_dir or os.path.join(
        f"runs_f/{args.dataset}", f"f_ablation_{timestamp}"
    )
    os.makedirs(results_dir, exist_ok=True)

    # Summarize only
    if args.summarize_only:
        all_results = load_existing_metrics(results_dir)
        if not all_results:
            print(f"No metrics found under {results_dir}")
            return
        print_and_save_table(all_results, results_dir)
        return

    variants = args.variants or ALL_VARIANTS
    epochs   = args.epochs   or cfg.total_epochs
    if args.delta_final is not None:
        cfg.delta_final = args.delta_final

    print("=" * 60)
    print("F-CS-WAE Ablation Study")
    print("=" * 60)
    print(f"  Dataset  : {args.dataset}  Seed : {args.seed}  Device : {device}")
    print(f"  Variants : {variants}")
    print(f"  Epochs   : {epochs}")
    print(f"  delta_final: {cfg.delta_final}")
    print(f"  Output   : {results_dir}")

    train_loader, test_loader = get_loaders(
        dataset=args.dataset,
        seed=args.seed,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
    )
    evaluator = ModelEvaluator(device=device, dataset=args.dataset)

    all_results: dict[str, dict] = {}

    for variant_key in variants:
        print(f"\n{'=' * 60}")
        print(f"  Variant: {variant_key}  —  {ABLATION_VARIANTS[variant_key]['name']}")
        print(f"{'=' * 60}")

        variant_dir = os.path.join(results_dir, variant_key)
        os.makedirs(variant_dir, exist_ok=True)

        model = create_f_cs_wae_ablation(
            variant_key,
            semantic_dim=cfg.semantic_dim,
            style_dim=cfg.style_dim,
            n_classes=n_classes,
            in_channels=in_channels,
            image_size=image_size,
        ).to(device)

        trainer = FCSWAEAblationTrainer(model, train_loader, device=device)
        trainer.train(epochs=epochs)

        # Save weights
        torch.save(model.state_dict(), os.path.join(variant_dir, "model.pth"))

        # Evaluate — clustering uses encode_to_distribution (returns mu_c)
        model_label = f"F-CS-WAE ({variant_key})"
        metrics = evaluator.comprehensive_evaluation(
            model, model_label, test_loader, variant_dir
        )
        save_metrics(variant_dir, metrics)
        all_results[variant_key] = metrics

        print(f"  ACC={metrics.get('ACC', float('nan')):.4f}  "
              f"FID={metrics.get('FID', float('nan')):.2f}")

    if not args.skip_aggregate:
        print_and_save_table(all_results, results_dir)

    print("\nAblation study complete.")


if __name__ == "__main__":
    main()

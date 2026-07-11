#!/usr/bin/env python3
"""Compare F-CS-WAE class-conditional generation strategies.

The current F-CS-WAE sampling path draws z_c from the class prior and z_s from
N(0, I).  This script tests post-hoc alternatives for z_s that respect the
factorized representation learned by a checkpoint:

  - global_gaussian:       z_s ~ N(0, I)
  - class_mean:            z_s = mean_k
  - class_diag_t*:         z_s ~ N(mean_k, t^2 diag(var_k))
  - class_empirical:       z_s sampled from encoded training examples of class k
  - semantic_center_*:     use class center z_c instead of Spherical Cauchy z_c

Outputs:
  - strategy_grid.png
  - metrics.csv / metrics.json
  - selected_prior_grid_<strategy>.png for each strategy

The metrics are lightweight diagnostics, not paper-final FID.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba-cache")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config_f_cs_wae import f_cs_wae_config as cfg
from src.datasets.loaders import get_dataset_info
from src.models.f_cs_wae import FCSWAE
from src.utils.dataset_config import apply_dataset_config
from src.utils.device import set_device
from src.utils.utils import mobius_reparam, sample_uniform_sphere


@dataclass
class StyleStats:
    labels: torch.Tensor
    mu_s: torch.Tensor
    mean: torch.Tensor
    std: torch.Tensor


def build_dataset(dataset: str, root: Path, train: bool):
    transform = transforms.ToTensor()
    if dataset == "mnist":
        return datasets.MNIST(root / "data", train=train, download=False, transform=transform)
    if dataset == "fashion_mnist":
        return datasets.FashionMNIST(root / "data", train=train, download=False, transform=transform)
    if dataset == "cifar10":
        return datasets.CIFAR10(root / "data", train=train, download=False, transform=transform)
    raise ValueError("This diagnostic currently supports mnist, fashion_mnist, and cifar10.")


def load_model(run_dir: Path, dataset: str, device: torch.device) -> FCSWAE:
    info = apply_dataset_config(dataset, get_dataset_info, backbone="resnet18")
    model = FCSWAE(
        semantic_dim=cfg.semantic_dim,
        style_dim=cfg.style_dim,
        n_classes=info["n_classes"],
        in_channels=info.get("in_channels", 3),
        image_size=info.get("image_size", 32),
        n_centers=cfg.n_centers,
        rho_prior=cfg.rho_prior,
        ema_momentum=cfg.ema_momentum,
    )
    state = torch.load(run_dir / "f_cs_wae_model.pth", map_location=device, weights_only=False)
    model.load_state_dict(state)
    return model.to(device).eval()


@torch.no_grad()
def collect_style_stats(
    model: FCSWAE,
    dataset: str,
    device: torch.device,
    max_samples: int,
    batch_size: int,
) -> StyleStats:
    train_set = build_dataset(dataset, ROOT, train=True)
    loader = DataLoader(train_set, batch_size=batch_size, shuffle=False, num_workers=0)
    labels, styles = [], []
    total = 0
    for images, y in loader:
        images = images.to(device)
        _, _, mu_s, _ = model.encode(images)
        labels.append(y.cpu())
        styles.append(mu_s.cpu())
        total += images.shape[0]
        if total >= max_samples:
            break

    labels_t = torch.cat(labels)[:max_samples]
    mu_s_t = torch.cat(styles)[:max_samples]
    means, stds = [], []
    for k in range(model.n_classes):
        class_styles = mu_s_t[labels_t == k]
        means.append(class_styles.mean(dim=0))
        stds.append(class_styles.std(dim=0).clamp_min(1e-4))
    return StyleStats(
        labels=labels_t,
        mu_s=mu_s_t,
        mean=torch.stack(means),
        std=torch.stack(stds),
    )


def sample_zc(
    model: FCSWAE,
    class_idx: int,
    n: int,
    device: torch.device,
    mode: str,
) -> torch.Tensor:
    center = F.normalize(model.ema_centers[class_idx, 0], p=2, dim=0)
    if mode == "center":
        return center.unsqueeze(0).expand(n, -1)
    if mode == "prior":
        eps = sample_uniform_sphere(n, model.semantic_dim, device=device)
        rho = torch.full((n,), model.rho_p, device=device)
        return mobius_reparam(eps, center.unsqueeze(0).expand(n, -1), rho)
    raise ValueError(f"unknown z_c mode: {mode}")


def sample_zs(
    strategy: str,
    class_idx: int,
    n: int,
    stats: StyleStats,
    device: torch.device,
) -> torch.Tensor:
    if strategy == "global_gaussian":
        return torch.randn(n, stats.mu_s.shape[1], device=device)
    if strategy == "class_mean":
        return stats.mean[class_idx].to(device).unsqueeze(0).expand(n, -1)
    if strategy.startswith("class_diag_t"):
        temp = float(strategy.replace("class_diag_t", ""))
        mean = stats.mean[class_idx].to(device)
        std = stats.std[class_idx].to(device) * temp
        return mean.unsqueeze(0) + torch.randn(n, mean.numel(), device=device) * std.unsqueeze(0)
    if strategy == "class_empirical":
        pool = stats.mu_s[stats.labels == class_idx]
        idx = torch.randint(0, pool.shape[0], (n,))
        return pool[idx].to(device)
    raise ValueError(f"unknown z_s strategy: {strategy}")


@torch.no_grad()
def generate_batch(
    model: FCSWAE,
    stats: StyleStats,
    strategy: str,
    zc_mode: str,
    n_per_class: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    images, labels = [], []
    for k in range(model.n_classes):
        z_c = sample_zc(model, k, n_per_class, device, zc_mode)
        z_s = sample_zs(strategy, k, n_per_class, stats, device)
        x = model.decoder(torch.cat([z_c, z_s], dim=1)).clamp(0, 1)
        images.append(x.cpu())
        labels.append(torch.full((n_per_class,), k, dtype=torch.long))
    return torch.cat(images), torch.cat(labels)


@torch.no_grad()
def evaluate_self_consistency(
    model: FCSWAE,
    images: torch.Tensor,
    target: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> dict[str, float]:
    preds, max_probs = [], []
    for start in range(0, images.shape[0], batch_size):
        x = images[start : start + batch_size].to(device)
        mu_c, _ = model.encode_to_distribution(x)
        logits = model.classifier(mu_c)
        probs = logits.softmax(dim=1)
        preds.append(probs.argmax(dim=1).cpu())
        max_probs.append(probs.max(dim=1).values.cpu())
    pred = torch.cat(preds)
    conf = torch.cat(max_probs)
    acc = (pred == target).float().mean().item()
    per_class = []
    for k in range(model.n_classes):
        mask = target == k
        per_class.append((pred[mask] == k).float().mean().item())
    diversity = images.flatten(1).std(dim=0).mean().item()
    foreground = images.mean().item()
    return {
        "self_acc": acc,
        "self_conf": conf.mean().item(),
        "min_class_acc": min(per_class),
        "pixel_diversity": diversity,
        "mean_intensity": foreground,
    }


def image_for_plot(x: torch.Tensor) -> np.ndarray:
    x = x.detach().cpu().clamp(0, 1)
    if x.shape[0] == 1:
        return x.squeeze(0).numpy()
    return x.permute(1, 2, 0).numpy()


def save_class_grid(images: torch.Tensor, out_path: Path, title: str, n_classes: int, cols: int) -> None:
    fig, axes = plt.subplots(n_classes, cols, figsize=(cols * 1.0, n_classes * 1.0))
    if n_classes == 1:
        axes = np.expand_dims(axes, 0)
    for k in range(n_classes):
        for j in range(cols):
            ax = axes[k, j]
            arr = image_for_plot(images[k * cols + j])
            if arr.ndim == 2:
                ax.imshow(arr, cmap="gray", vmin=0, vmax=1)
            else:
                ax.imshow(arr)
            ax.axis("off")
            if j == 0:
                ax.set_ylabel(str(k), fontsize=8)
    fig.suptitle(title, y=0.995)
    plt.subplots_adjust(wspace=0.03, hspace=0.03)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def save_strategy_panel(
    strategy_images: dict[str, torch.Tensor],
    out_path: Path,
    n_classes: int,
    cols: int,
) -> None:
    rows = len(strategy_images) * n_classes
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.0, rows * 0.95))
    row = 0
    for name, images in strategy_images.items():
        for k in range(n_classes):
            for j in range(cols):
                ax = axes[row, j]
                arr = image_for_plot(images[k * cols + j])
                if arr.ndim == 2:
                    ax.imshow(arr, cmap="gray", vmin=0, vmax=1)
                else:
                    ax.imshow(arr)
                ax.axis("off")
                if j == 0:
                    ax.set_ylabel(f"{name}\n{k}", fontsize=5)
            row += 1
    plt.subplots_adjust(wspace=0.02, hspace=0.02)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default="mnist", choices=["mnist", "fashion_mnist", "cifar10"])
    p.add_argument("--run-dir", default=None)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--style-bank-max", type=int, default=12000)
    p.add_argument("--cols", type=int, default=8)
    p.add_argument("--eval-per-class", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--skip-self-eval", action="store_true")
    p.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="Optional strategy names to run, e.g. prior_class_diag_t0.50 prior_class_empirical",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = set_device(args.device)
    run_dir = Path(args.run_dir) if args.run_dir else ROOT / "runs_f" / args.dataset / "seed_0"
    out_dir = Path(args.output_dir) if args.output_dir else ROOT / "paper-f-cs-wae" / "experiments" / f"{args.dataset}_sampling"
    out_dir.mkdir(parents=True, exist_ok=True)

    model = load_model(run_dir, args.dataset, device)
    stats = collect_style_stats(model, args.dataset, device, args.style_bank_max, args.batch_size)

    strategies = [
        ("prior_global_gaussian", "global_gaussian", "prior"),
        ("prior_class_mean", "class_mean", "prior"),
        ("prior_class_diag_t0.25", "class_diag_t0.25", "prior"),
        ("prior_class_diag_t0.50", "class_diag_t0.50", "prior"),
        ("prior_class_diag_t1.00", "class_diag_t1.00", "prior"),
        ("prior_class_empirical", "class_empirical", "prior"),
        ("center_class_diag_t0.50", "class_diag_t0.50", "center"),
        ("center_class_empirical", "class_empirical", "center"),
    ]
    if args.only:
        selected = set(args.only)
        strategies = [s for s in strategies if s[0] in selected]
        missing = selected - {s[0] for s in strategies}
        if missing:
            raise ValueError(f"Unknown strategy names in --only: {sorted(missing)}")

    metrics = []
    panel_images: dict[str, torch.Tensor] = {}
    for name, zs_strategy, zc_mode in strategies:
        print(f"Generating {name} ...", flush=True)
        grid_images, _ = generate_batch(model, stats, zs_strategy, zc_mode, args.cols, device)
        save_class_grid(
            grid_images,
            out_dir / f"selected_prior_grid_{name}.png",
            title=name,
            n_classes=model.n_classes,
            cols=args.cols,
        )
        panel_images[name] = grid_images

        if args.skip_self_eval:
            row = {
                "strategy": name,
                "z_s": zs_strategy,
                "z_c": zc_mode,
                "self_acc": float("nan"),
                "self_conf": float("nan"),
                "min_class_acc": float("nan"),
                "pixel_diversity": grid_images.flatten(1).std(dim=0).mean().item(),
                "mean_intensity": grid_images.mean().item(),
            }
        else:
            eval_images, targets = generate_batch(
                model, stats, zs_strategy, zc_mode, args.eval_per_class, device
            )
            row = {
                "strategy": name,
                "z_s": zs_strategy,
                "z_c": zc_mode,
                **evaluate_self_consistency(model, eval_images, targets, device, args.batch_size),
            }
        metrics.append(row)

    save_strategy_panel(panel_images, out_dir / "strategy_grid.png", model.n_classes, args.cols)

    with (out_dir / "metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics[0].keys()))
        writer.writeheader()
        writer.writerows(metrics)
    with (out_dir / "metrics.json").open("w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Wrote sampling diagnostics to {out_dir}")
    for row in metrics:
        print(
            f"{row['strategy']:<28} self_acc={row['self_acc']:.3f} "
            f"conf={row['self_conf']:.3f} min_cls={row['min_class_acc']:.3f} "
            f"div={row['pixel_diversity']:.4f}"
        )


if __name__ == "__main__":
    main()

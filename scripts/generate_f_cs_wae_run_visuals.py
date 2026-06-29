#!/usr/bin/env python3
"""Generate run-level visualizations for trained F-CS-WAE checkpoints.

Outputs are written into the run directory, mirroring the CS-WAE sample files:
loss_history.png, reconstructions.png, latent_space_umap.png,
random_samples_from_priors.png, slerp_interpolation.png, plus F-CS-WAE-specific
class_conditional_prior_grid.png and style_semantic_diagnostic.png.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba-cache")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config_f_cs_wae import f_cs_wae_config as cfg
from src.datasets.loaders import SUPPORTED_DATASETS, get_dataset_info, get_loaders
from src.models.f_cs_wae import FCSWAE
from src.utils.dataset_config import apply_dataset_config
from src.utils.device import set_device


FASHION_MNIST_CLASSES = [
    "T-shirt",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]


def class_names(dataset: str, n_classes: int) -> list[str]:
    if dataset == "fashion_mnist" and n_classes == 10:
        return FASHION_MNIST_CLASSES
    return [str(i) for i in range(n_classes)]


def read_json(path: Path):
    with path.open() as f:
        return json.load(f)


def image_for_plot(x: torch.Tensor) -> np.ndarray:
    x = x.detach().cpu().clamp(0, 1)
    if x.shape[0] == 1:
        return x.squeeze(0).numpy()
    return x.permute(1, 2, 0).numpy()


def show_image(ax, x: torch.Tensor, title: str | None = None) -> None:
    arr = image_for_plot(x)
    if arr.ndim == 2:
        ax.imshow(arr, cmap="gray", vmin=0, vmax=1)
    else:
        ax.imshow(arr)
    if title:
        ax.set_title(title, fontsize=8)
    ax.axis("off")


def slerp(p0: torch.Tensor, p1: torch.Tensor, t: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    p0 = F.normalize(p0, p=2, dim=-1)
    p1 = F.normalize(p1, p=2, dim=-1)
    dot = torch.clamp((p0 * p1).sum(dim=-1, keepdim=True), -0.9995, 0.9995)
    omega = torch.acos(dot)
    sin_omega = torch.sin(omega).clamp_min(eps)
    return (
        torch.sin((1.0 - t) * omega) / sin_omega * p0
        + torch.sin(t * omega) / sin_omega * p1
    )


def load_model(run_dir: Path, dataset: str, device: torch.device, n_centers: int | None = None) -> FCSWAE:
    info = apply_dataset_config(dataset, get_dataset_info, backbone="resnet18")
    model = FCSWAE(
        semantic_dim=cfg.semantic_dim,
        style_dim=cfg.style_dim,
        n_classes=info["n_classes"],
        in_channels=info.get("in_channels", 3),
        image_size=info.get("image_size", 32),
        n_centers=n_centers or cfg.n_centers,
        rho_prior=cfg.rho_prior,
        ema_momentum=cfg.ema_momentum,
    )
    state = torch.load(run_dir / "f_cs_wae_model.pth", map_location=device, weights_only=False)
    model.load_state_dict(state)
    return model.to(device).eval()


def plot_loss_history(run_dir: Path) -> None:
    history_path = run_dir / "training_history.json"
    if not history_path.exists():
        print(f"skip loss_history: missing {history_path}")
        return
    history = read_json(history_path)
    if not history:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    epochs = np.arange(1, len(history) + 1)
    for key in ("total", "rec", "class_mmd", "agg_mmd", "style_mmd", "cls"):
        if key in history[0]:
            axes[0].plot(epochs, [h.get(key, np.nan) for h in history], label=key)
    axes[0].set_title("F-CS-WAE Training Terms")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=8)

    if "var" in history[0]:
        axes[1].plot(epochs, [h.get("var", np.nan) for h in history], color="#C82423")
        axes[1].set_title("Style Variance Diagnostic")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Var term")
        axes[1].grid(alpha=0.3)
    else:
        axes[1].axis("off")

    fig.tight_layout()
    fig.savefig(run_dir / "loss_history.png", dpi=250)
    plt.close(fig)


@torch.no_grad()
def plot_reconstructions(model: FCSWAE, loader, run_dir: Path, device: torch.device) -> None:
    images, labels = next(iter(loader))
    images = images.to(device)
    x_hat = model(images)[0]
    n = min(10, images.shape[0])
    fig, axes = plt.subplots(2, n, figsize=(1.4 * n, 3.0))
    for i in range(n):
        show_image(axes[0, i], images[i], "Original" if i == 0 else None)
        show_image(axes[1, i], x_hat[i], "Recon" if i == 0 else None)
    fig.suptitle("F-CS-WAE Reconstructions", y=1.02)
    fig.tight_layout()
    fig.savefig(run_dir / "reconstructions.png", dpi=250)
    plt.close(fig)


@torch.no_grad()
def collect_latents(model: FCSWAE, loader, device: torch.device, max_samples: int):
    latents, labels = [], []
    total = 0
    for images, y in loader:
        images = images.to(device)
        mu_c, _ = model.encode_to_distribution(images)
        latents.append(mu_c.detach().cpu().numpy())
        labels.append(y.numpy())
        total += images.shape[0]
        if total >= max_samples:
            break
    return np.concatenate(latents, axis=0)[:max_samples], np.concatenate(labels, axis=0)[:max_samples]


def reduce_2d(x: np.ndarray) -> tuple[np.ndarray, str]:
    try:
        import umap

        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
        return reducer.fit_transform(x), "UMAP"
    except Exception:
        from sklearn.manifold import TSNE

        return TSNE(n_components=2, init="pca", learning_rate="auto", perplexity=30, random_state=42).fit_transform(x), "t-SNE"


@torch.no_grad()
def plot_latent_umap(model: FCSWAE, loader, run_dir: Path, device: torch.device, max_samples: int) -> None:
    z, y = collect_latents(model, loader, device, max_samples)
    centers = F.normalize(model.ema_centers, p=2, dim=-1).detach().cpu().numpy()
    centers_flat = centers.reshape(-1, centers.shape[-1])
    emb_all, method = reduce_2d(np.vstack([z, centers_flat]))
    emb = emb_all[: len(z)]
    cemb = emb_all[len(z) :]

    n_classes = model.n_classes
    fig, ax = plt.subplots(figsize=(9, 7))
    scatter = ax.scatter(emb[:, 0], emb[:, 1], c=y, cmap="tab10", s=5, alpha=0.65, linewidths=0)
    center_classes = np.repeat(np.arange(n_classes), model.n_centers)
    ax.scatter(
        cemb[:, 0],
        cemb[:, 1],
        c=center_classes,
        cmap="tab10",
        marker="X",
        s=180,
        edgecolor="black",
        linewidth=0.8,
        label="class centers",
    )
    ax.set_title(f"F-CS-WAE semantic latent $z_c$ ({method})")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(*scatter.legend_elements(num=n_classes), title="class", loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(run_dir / "latent_space_umap.png", dpi=250)
    plt.close(fig)


@torch.no_grad()
def decode(model: FCSWAE, z_c: torch.Tensor, z_s: torch.Tensor) -> torch.Tensor:
    return model.decoder(torch.cat([z_c, z_s], dim=1)).clamp(0, 1)


@torch.no_grad()
def plot_random_samples(model: FCSWAE, run_dir: Path, device: torch.device, names: list[str], cols: int = 8) -> None:
    fig, axes = plt.subplots(model.n_classes, cols, figsize=(cols * 1.25, model.n_classes * 1.25))
    if model.n_classes == 1:
        axes = np.expand_dims(axes, 0)
    for k in range(model.n_classes):
        z_c, z_s = model.sample_from_class_prior(k, cols, device)
        samples = decode(model, z_c, z_s)
        for j in range(cols):
            show_image(axes[k, j], samples[j])
            if j == 0:
                axes[k, j].set_ylabel(names[k], fontsize=8)
    fig.suptitle("Random Samples from F-CS-WAE Class Priors", y=0.995)
    plt.subplots_adjust(wspace=0.03, hspace=0.03)
    fig.savefig(run_dir / "random_samples_from_priors.png", dpi=250)
    fig.savefig(run_dir / "class_conditional_prior_grid.png", dpi=250)
    plt.close(fig)


@torch.no_grad()
def plot_slerp(model: FCSWAE, run_dir: Path, device: torch.device, names: list[str], num_steps: int = 10) -> None:
    default_pairs = [(1, 7), (3, 5), (2, 8), (4, 9)]
    pairs = [(a, b) for a, b in default_pairs if a < model.n_classes and b < model.n_classes]
    centers = F.normalize(model.ema_centers[:, 0, :], p=2, dim=1)
    t_values = torch.linspace(0, 1, num_steps, device=device).unsqueeze(1)
    fixed_zs = torch.randn(len(pairs), model.style_dim, device=device)

    fig, axes = plt.subplots(len(pairs), num_steps, figsize=(num_steps * 1.25, len(pairs) * 1.45))
    if len(pairs) == 1:
        axes = np.expand_dims(axes, 0)
    for row, (a, b) in enumerate(pairs):
        z_c = slerp(centers[a].unsqueeze(0), centers[b].unsqueeze(0), t_values)
        z_s = fixed_zs[row].unsqueeze(0).expand(num_steps, -1)
        images = decode(model, z_c, z_s)
        for col in range(num_steps):
            title = None
            if col == 0:
                title = names[a]
            elif col == num_steps - 1:
                title = names[b]
            show_image(axes[row, col], images[col], title)
    fig.suptitle("Slerp Interpolation between F-CS-WAE Class Centers", y=1.02)
    fig.tight_layout()
    fig.savefig(run_dir / "slerp_interpolation.png", dpi=250)
    plt.close(fig)


@torch.no_grad()
def plot_style_semantic(model: FCSWAE, run_dir: Path, device: torch.device, names: list[str]) -> None:
    n_rows = min(5, model.n_classes)
    cols = 8
    centers = F.normalize(model.ema_centers[:, 0, :], p=2, dim=1)
    fig, axes = plt.subplots(n_rows + 3, cols, figsize=(cols * 1.25, (n_rows + 3) * 1.25))

    for row, k in enumerate(range(n_rows)):
        z_c = centers[k].unsqueeze(0).expand(cols, -1)
        z_s = torch.randn(cols, model.style_dim, device=device)
        images = decode(model, z_c, z_s)
        for col in range(cols):
            show_image(axes[row, col], images[col])
            if col == 0:
                axes[row, col].set_ylabel(f"fixed {names[k]}", fontsize=8)

    for row in range(3):
        z_c = centers[:cols]
        z_s = torch.randn(1, model.style_dim, device=device).expand(cols, -1)
        images = decode(model, z_c, z_s)
        out_row = n_rows + row
        for col in range(cols):
            show_image(axes[out_row, col], images[col])
            if row == 0:
                axes[out_row, col].set_title(names[col], fontsize=7)
            if col == 0:
                axes[out_row, col].set_ylabel(f"fixed style {row+1}", fontsize=8)

    fig.suptitle("F-CS-WAE Style/Semantic Diagnostic", y=0.995)
    plt.subplots_adjust(wspace=0.03, hspace=0.04)
    fig.savefig(run_dir / "style_semantic_diagnostic.png", dpi=250)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate F-CS-WAE run visualizations")
    parser.add_argument("--dataset", choices=list(SUPPORTED_DATASETS), required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--run-dir", type=str, default=None)
    parser.add_argument("--max-samples", type=int, default=3000)
    parser.add_argument("--n-centers", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.device and args.device.startswith("cuda") and not torch.cuda.is_available():
        print(f"CUDA requested ({args.device}) but unavailable; falling back to CPU for visualization.")
        device = set_device("cpu")
    else:
        device = set_device(args.device)
    run_dir = Path(args.run_dir) if args.run_dir else ROOT / "runs_f" / args.dataset / f"seed_{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    info = get_dataset_info(args.dataset)
    names = class_names(args.dataset, info["n_classes"])
    model = load_model(run_dir, args.dataset, device, args.n_centers)
    _, test_loader = get_loaders(
        dataset=args.dataset,
        seed=args.seed,
        batch_size=cfg.batch_size,
        num_workers=0,
    )

    print(f"Generating F-CS-WAE visuals for {args.dataset} seed {args.seed} -> {run_dir}")
    plot_loss_history(run_dir)
    plot_reconstructions(model, test_loader, run_dir, device)
    plot_latent_umap(model, test_loader, run_dir, device, args.max_samples)
    plot_random_samples(model, run_dir, device, names)
    plot_slerp(model, run_dir, device, names)
    plot_style_semantic(model, run_dir, device, names)
    print("Done.")


if __name__ == "__main__":
    main()

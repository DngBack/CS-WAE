#!/usr/bin/env python3
"""Generate additional F-CS-WAE CIFAR-10 visual figures for paper claims."""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "paper_outputs" / "f_cs_wae_cifar10" / "visual_figures"

import sys

sys.path.insert(0, str(ROOT))

from src.config import config
from src.config_f_cs_wae import f_cs_wae_config as fcfg
from src.datasets.loaders import get_loaders
from src.models import SphericalWAE_Supervised
from src.models.baselines_extended import ResNetAE
from src.models.f_cs_wae import FCSWAE
from src.utils.utils import mobius_reparam, sample_uniform_sphere


CIFAR10_CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


def read_json(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def choose_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda:1" if torch.cuda.device_count() > 1 else "cuda:0")
    return torch.device("cpu")


def save_fig(fig: plt.Figure, name: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT_DIR / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def img_np(x: torch.Tensor) -> np.ndarray:
    x = x.detach().cpu().clamp(0, 1)
    return x.permute(1, 2, 0).numpy()


def load_f_model(device: torch.device, ckpt: Path | None = None) -> FCSWAE:
    ckpt = ckpt or ROOT / "runs_f" / "cifar10" / "seed_0" / "f_cs_wae_model.pth"
    model = FCSWAE(
        semantic_dim=fcfg.semantic_dim,
        style_dim=fcfg.style_dim,
        n_classes=10,
        in_channels=3,
        image_size=32,
        n_centers=1,
        rho_prior=fcfg.rho_prior,
        ema_momentum=fcfg.ema_momentum,
    )
    state = torch.load(ckpt, map_location=device, weights_only=False)
    model.load_state_dict(state)
    return model.to(device).eval()


def load_cs_model(device: torch.device) -> SphericalWAE_Supervised:
    config.latent_dim = 32
    config.backbone = "resnet18"
    config.in_channels = 3
    config.image_size = 32
    model = SphericalWAE_Supervised(
        latent_dim=32,
        n_classes=10,
        in_channels=3,
        image_size=32,
        backbone="resnet18",
    )
    state = torch.load(
        ROOT / "runs" / "cifar10_resnet18" / "seed_0" / "cs_wae_model.pth",
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(state)
    return model.to(device).eval()


def load_ae_model(device: torch.device) -> ResNetAE:
    model = ResNetAE(latent_dim=fcfg.semantic_dim + fcfg.style_dim, in_channels=3, image_size=32)
    state = torch.load(
        ROOT / "runs_f" / "cifar10" / "baselines_extended" / "seed_0" / "ResNetAE" / "model.pth",
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(state)
    return model.to(device).eval()


def get_test_loader(batch_size: int = 256):
    _, test_loader = get_loaders(
        dataset="cifar10",
        seed=0,
        batch_size=batch_size,
        num_workers=0,
    )
    return test_loader


@torch.no_grad()
def collect_latents(model, model_kind: str, loader, device: torch.device, max_samples: int = 2000):
    latents, labels = [], []
    for images, batch_labels in loader:
        images = images.to(device)
        if model_kind == "ae":
            z = model.encode(images)
        elif model_kind == "cs":
            z, _ = model.encode_to_distribution(images)
        elif model_kind == "f":
            z, _ = model.encode_to_distribution(images)
        else:
            raise ValueError(model_kind)
        latents.append(z.detach().cpu().numpy())
        labels.append(batch_labels.numpy())
        if sum(len(x) for x in labels) >= max_samples:
            break
    return np.concatenate(latents, axis=0)[:max_samples], np.concatenate(labels, axis=0)[:max_samples]


def reduce_2d(x: np.ndarray, random_state: int = 0) -> np.ndarray:
    try:
        import umap

        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=30,
            min_dist=0.08,
            metric="euclidean",
            random_state=random_state,
        )
        return reducer.fit_transform(x)
    except Exception:
        from sklearn.manifold import TSNE

        return TSNE(n_components=2, init="pca", learning_rate="auto", perplexity=30, random_state=random_state).fit_transform(x)


def plot_latent_structure(ae, cs, f_model, loader, device: torch.device) -> None:
    panels = [
        ("AE latent", "ae", ae, None),
        ("single-latent CS-WAE", "cs", cs, F.normalize(cs.prior_mus, p=2, dim=1).detach().cpu().numpy()),
        (
            "F-CS-WAE $z_c$",
            "f",
            f_model,
            F.normalize(f_model.ema_centers[:, 0, :], p=2, dim=1).detach().cpu().numpy(),
        ),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    cmap = plt.get_cmap("tab10")

    for ax, (title, kind, model, centers) in zip(axes, panels):
        z, y = collect_latents(model, kind, loader, device)
        if centers is not None:
            z_all = np.vstack([z, centers])
            emb_all = reduce_2d(z_all)
            emb, cemb = emb_all[:-len(centers)], emb_all[-len(centers):]
        else:
            emb = reduce_2d(z)
            cemb = None

        ax.scatter(emb[:, 0], emb[:, 1], c=y, s=4, cmap=cmap, alpha=0.55, linewidths=0)
        if cemb is not None:
            ax.scatter(cemb[:, 0], cemb[:, 1], c=np.arange(10), s=120, cmap=cmap, marker="X", edgecolor="black", linewidth=0.8)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])

    handles = [
        plt.Line2D([0], [0], marker="o", color="w", label=name, markerfacecolor=cmap(i), markersize=6)
        for i, name in enumerate(CIFAR10_CLASSES)
    ]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.08))
    fig.suptitle("CIFAR-10 Latent Structure (UMAP/t-SNE fallback; X markers denote class centers)", y=1.02)
    save_fig(fig, "fig04_cifar10_latent_structure_umap")


@torch.no_grad()
def f_decode(model: FCSWAE, z_c: torch.Tensor, z_s: torch.Tensor) -> torch.Tensor:
    return model.decoder(torch.cat([z_c, z_s], dim=1)).clamp(0, 1)


@torch.no_grad()
def plot_class_conditional_grid(model: FCSWAE, device: torch.device, cols: int = 8) -> None:
    torch.manual_seed(7)
    rows = 10
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.15, rows * 1.15))
    for k in range(rows):
        zc, zs = model.sample_from_class_prior(k, cols, device)
        images = f_decode(model, zc, zs)
        for j in range(cols):
            axes[k, j].imshow(img_np(images[j]))
            axes[k, j].axis("off")
            if j == 0:
                axes[k, j].set_ylabel(CIFAR10_CLASSES[k], fontsize=8)
    fig.suptitle("F-CS-WAE Class-Conditional Prior Samples: rows = class, columns = style samples", y=0.995)
    plt.subplots_adjust(wspace=0.03, hspace=0.03)
    save_fig(fig, "fig05_f_cs_wae_class_conditional_prior_grid")


def slerp(a: torch.Tensor, b: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    a = F.normalize(a, p=2, dim=-1)
    b = F.normalize(b, p=2, dim=-1)
    dot = torch.clamp((a * b).sum(dim=-1, keepdim=True), -0.9995, 0.9995)
    omega = torch.acos(dot)
    so = torch.sin(omega)
    return (
        torch.sin((1.0 - t) * omega) / so * a
        + torch.sin(t * omega) / so * b
    )


@torch.no_grad()
def plot_style_semantic_traversal(model: FCSWAE, device: torch.device) -> None:
    torch.manual_seed(11)
    classes = [0, 1, 3, 5, 8]
    cols = 8
    centers = F.normalize(model.ema_centers[:, 0, :], p=2, dim=1)

    fig, axes = plt.subplots(len(classes) + 4, cols, figsize=(cols * 1.15, 9 * 1.15))

    for row, k in enumerate(classes):
        zc = centers[k].unsqueeze(0).expand(cols, -1)
        zs = torch.randn(cols, model.style_dim, device=device)
        images = f_decode(model, zc, zs)
        for col in range(cols):
            axes[row, col].imshow(img_np(images[col]))
            axes[row, col].axis("off")
            if col == 0:
                axes[row, col].set_ylabel(f"fixed {CIFAR10_CLASSES[k]}", fontsize=8)

    fixed_styles = torch.randn(4, model.style_dim, device=device)
    semantic_classes = list(range(cols))
    for r in range(4):
        zc = centers[semantic_classes]
        zs = fixed_styles[r].unsqueeze(0).expand(cols, -1)
        images = f_decode(model, zc, zs)
        out_row = len(classes) + r
        for col in range(cols):
            axes[out_row, col].imshow(img_np(images[col]))
            axes[out_row, col].axis("off")
            if r == 0:
                axes[out_row, col].set_title(CIFAR10_CLASSES[semantic_classes[col]], fontsize=7)
            if col == 0:
                axes[out_row, col].set_ylabel(f"fixed style {r+1}", fontsize=8)

    fig.suptitle("Style-Semantic Diagnostic: fixed $z_c$ varies style; fixed $z_s$ varies class center", y=0.995)
    plt.subplots_adjust(wspace=0.03, hspace=0.03)
    save_fig(fig, "fig06_style_semantic_disentanglement_grid")


@torch.no_grad()
def plot_class_center_slerp(model: FCSWAE, device: torch.device) -> None:
    torch.manual_seed(13)
    pairs = [(0, 8), (1, 9), (3, 5), (4, 7)]
    steps = 9
    t_values = torch.linspace(0, 1, steps, device=device).unsqueeze(1)
    centers = F.normalize(model.ema_centers[:, 0, :], p=2, dim=1)
    fixed_zs = torch.randn(len(pairs), model.style_dim, device=device)

    fig, axes = plt.subplots(len(pairs), steps, figsize=(steps * 1.05, len(pairs) * 1.2))
    for row, (a, b) in enumerate(pairs):
        zc = slerp(centers[a].unsqueeze(0), centers[b].unsqueeze(0), t_values)
        zs = fixed_zs[row].unsqueeze(0).expand(steps, -1)
        images = f_decode(model, zc, zs)
        for col in range(steps):
            axes[row, col].imshow(img_np(images[col]))
            axes[row, col].axis("off")
            if row == 0:
                axes[row, col].set_title(f"{float(t_values[col]):.2f}", fontsize=7)
            if col == 0:
                axes[row, col].set_ylabel(f"{CIFAR10_CLASSES[a]}\n→ {CIFAR10_CLASSES[b]}", fontsize=8)
    fig.suptitle("Semantic Slerp Between Learned Class Centers (fixed style code per row)", y=1.02)
    plt.subplots_adjust(wspace=0.03, hspace=0.08)
    save_fig(fig, "fig06b_class_center_slerp")


@torch.no_grad()
def sample_cs_prior_images(model: SphericalWAE_Supervised, device: torch.device, n: int) -> torch.Tensor:
    random_classes = torch.arange(10, device=device).repeat_interleave(math.ceil(n / 10))[:n]
    centers = F.normalize(model.prior_mus, p=2, dim=1)[random_classes]
    eps = sample_uniform_sphere(n, config.latent_dim, device=device)
    rho = torch.full((n,), model.rho_p, device=device)
    z = mobius_reparam(eps, centers, rho)
    return model.decoder(z).clamp(0, 1)


@torch.no_grad()
def plot_factorization_samples(cs: SphericalWAE_Supervised, f_model: FCSWAE, device: torch.device) -> None:
    torch.manual_seed(17)
    cols = 8
    cs_images = sample_cs_prior_images(cs, device, cols)
    f_zc, f_zs = [], []
    for k in range(cols):
        zc, zs = f_model.sample_from_class_prior(k, 1, device)
        f_zc.append(zc)
        f_zs.append(zs)
    f_images = f_decode(f_model, torch.cat(f_zc, dim=0), torch.cat(f_zs, dim=0))

    fig, axes = plt.subplots(2, cols, figsize=(cols * 1.15, 2.4))
    for col in range(cols):
        axes[0, col].imshow(img_np(cs_images[col]))
        axes[0, col].axis("off")
        axes[1, col].imshow(img_np(f_images[col]))
        axes[1, col].axis("off")
        axes[1, col].set_title(CIFAR10_CLASSES[col], fontsize=7)
    axes[0, 0].set_ylabel("single-latent\nCS-WAE", fontsize=8)
    axes[1, 0].set_ylabel("Full\nF-CS-WAE", fontsize=8)
    fig.suptitle("Single-Latent vs Factorized Prior Samples (available completed checkpoints)", y=1.08)
    plt.subplots_adjust(wspace=0.03, hspace=0.05)
    save_fig(fig, "fig03_available_factorization_samples")


def load_metric(path: Path) -> dict:
    return read_json(path) if path.exists() else {}


def plot_tradeoff_scatter() -> None:
    rows = []

    def add(method: str, path: Path, generative: bool = True) -> None:
        if not path.exists():
            return
        m = load_metric(path)
        if "ACC" in m and "FID" in m:
            rows.append({"method": method, "ACC": float(m["ACC"]), "FID": float(m["FID"]), "generative": generative})

    add("VAE", ROOT / "runs" / "cifar10" / "baselines" / "seed_0" / "VAE" / "metrics.json")
    add("WAE-MMD", ROOT / "runs" / "cifar10" / "baselines" / "seed_0" / "WAE-MMD" / "metrics.json")
    add("VaDE", ROOT / "runs" / "cifar10" / "baselines" / "seed_0" / "VaDE" / "metrics.json")
    add("single-latent\nCS-WAE", ROOT / "runs" / "cifar10_resnet18" / "seed_0" / "metrics.json")
    add("CS-WAE\nEuclidean", ROOT / "runs" / "cifar10" / "ablation_20260622_201844" / "euclidean" / "metrics.json")
    add("CS-WAE\nvMF", ROOT / "runs" / "cifar10" / "ablation_20260622_201844" / "vmf_prior" / "metrics.json")
    add("ResNetAE", ROOT / "runs_f" / "cifar10" / "baselines_extended" / "seed_0" / "ResNetAE" / "metrics.json")
    add("F-CS-WAE", ROOT / "runs_f" / "cifar10" / "seed_0" / "metrics.json")
    add("F-CS-WAE\nfull ablation", ROOT / "runs_f" / "cifar10" / "f_ablation_20260625_183045" / "full" / "metrics.json")

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "fig07_tradeoff_points.csv", index=False)

    fig, ax = plt.subplots(figsize=(7.3, 5.2))
    for _, row in df.iterrows():
        is_f = row["method"].startswith("F-CS-WAE")
        ax.scatter(row["FID"], row["ACC"] * 100, s=130 if is_f else 80, color="#C82423" if is_f else "#2878B5", edgecolor="black", linewidth=0.7, zorder=3)
        ax.annotate(row["method"], (row["FID"], row["ACC"] * 100), textcoords="offset points", xytext=(6, 5), fontsize=8)
    ax.set_xlabel("FID (lower is better)")
    ax.set_ylabel("ACC (%) (higher is better)")
    ax.set_title("Clustering-Generation Trade-off on CIFAR-10")
    ax.grid(alpha=0.25)
    save_fig(fig, "fig07_clustering_generation_tradeoff")


def write_visual_summary() -> None:
    lines = [
        "# Visual Figures for F-CS-WAE CIFAR-10",
        "",
        "These figures are generated from completed local checkpoints only.",
        "",
        "## Generated Figures",
        "",
        "- `fig03_available_factorization_samples.*`: single-latent CS-WAE vs full F-CS-WAE prior samples.",
        "- `fig04_cifar10_latent_structure_umap.*`: AE, single-latent CS-WAE, and F-CS-WAE semantic latent structure with class centers.",
        "- `fig05_f_cs_wae_class_conditional_prior_grid.*`: class-conditional prior grid, rows are CIFAR-10 classes.",
        "- `fig06_style_semantic_disentanglement_grid.*`: fixed semantic/vary style and fixed style/vary semantic center diagnostics.",
        "- `fig06b_class_center_slerp.*`: spherical interpolation between learned class centers.",
        "- `fig07_clustering_generation_tradeoff.*`: ACC-FID trade-off scatter.",
        "",
        "## Missing Exact Requested Panel",
        "",
        "`F-CS-WAE without style MMD` is not plotted because there is no completed checkpoint for `no_style_mmd` in `runs_f/cifar10/f_ablation_20260625_183045`.",
        "Train that variant and re-run this script to make the exact three-column Figure 3.",
    ]
    (OUT_DIR / "VISUAL_SUMMARY.md").write_text("\n".join(lines))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    np.random.seed(0)
    device = choose_device()

    loader = get_test_loader()
    ae = load_ae_model(device)
    cs = load_cs_model(device)
    f_model = load_f_model(device)

    plot_factorization_samples(cs, f_model, device)
    plot_latent_structure(ae, cs, f_model, loader, device)
    plot_class_conditional_grid(f_model, device)
    plot_style_semantic_traversal(f_model, device)
    plot_class_center_slerp(f_model, device)
    plot_tradeoff_scatter()
    write_visual_summary()

    print(f"Generated visual figures under {OUT_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

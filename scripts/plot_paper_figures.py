#!/usr/bin/env python3
"""
Generate paper-ready tables and figures from completed experiment runs.

Outputs under paper_outputs/ by default:
  tables/   - LaTeX + CSV exports
  figures/  - PNG + PDF figures
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from matplotlib import image as mpimg
from matplotlib.gridspec import GridSpec
from PIL import Image
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import config
from src.datasets.loaders import get_dataset_info, get_loaders
from src.models import SphericalWAE_Supervised, VAE, VaDE, WAE_MMD
from src.utils.device import set_device
from src.utils.seed import set_seed
from src.utils.utils import mobius_reparam, sample_uniform_sphere

DATASETS = ("mnist", "fashion_mnist", "cifar10")
BASELINE_METHODS = ("CS-WAE", "VaDE", "VAE", "WAE-MMD")
ABLATION_VARIANTS = (
    ("baseline", "CS-WAE (Baseline)"),
    ("no_sup_mmd", "CS-WAE w/o Supervised MMD"),
    ("euclidean", "CS-WAE (Euclidean)"),
    ("vmf_prior", "CS-WAE (von Mises-Fisher)"),
    ("minimal", "Minimal CS-WAE"),
)

RUN_PATHS = {
    "mnist": {
        "aggregated": "runs/mnist/aggregated_metrics.json",
        "baselines": "runs/mnist/baselines/seed_0",
        "ablation": "runs/mnist/ablation_20260621_061506",
    },
    "fashion_mnist": {
        "aggregated": "runs/fashion_mnist/aggregated_metrics.json",
        "baselines": "runs/fashion_mnist/baselines/seed_0",
        "ablation": "runs/fashion_mnist/ablation_20260621_192630",
    },
    "cifar10": {
        "aggregated": "runs/cifar10/aggregated_metrics.json",
        "baselines": "runs/cifar10/baselines/seed_0",
        "ablation": "runs/cifar10/ablation_20260622_201844",
    },
}

plt.rcParams.update(
    {
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    }
)


def _save_fig(fig: plt.Figure, out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{name}.png")
    fig.savefig(out_dir / f"{name}.pdf")
    plt.close(fig)
    print(f"  saved {name}.png / .pdf")


def load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def pct(x: float, digits: int = 1) -> str:
    return f"{100 * x:.{digits}f}"


def export_latex_tables(table_dir: Path) -> None:
    table_dir.mkdir(parents=True, exist_ok=True)
    rows_main = []
    for ds in DATASETS:
        agg = load_json(ROOT / RUN_PATHS[ds]["aggregated"])
        s = agg["summary"]
        label = "MNIST" if ds == "mnist" else "Fashion-MNIST"
        rows_main.append(
            {
                "Dataset": label,
                "ACC": f"{pct(s['ACC']['mean'])} $\\pm$ {pct(s['ACC']['std'])}",
                "NMI": f"{pct(s['NMI']['mean'])} $\\pm$ {pct(s['NMI']['std'])}",
                "ARI": f"{pct(s['ARI']['mean'])} $\\pm$ {pct(s['ARI']['std'])}",
                "FID": f"{s['FID']['mean']:.1f} $\\pm$ {s['FID']['std']:.1f}",
            }
        )
    df_main = pd.DataFrame(rows_main)
    df_main.to_csv(table_dir / "table01_main_results.csv", index=False)

    tex_main = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Main CS-WAE results (mean $\pm$ std over 3 seeds, 50 epochs).}",
        r"\label{tab:main-results}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Dataset & ACC$\uparrow$ & NMI$\uparrow$ & ARI$\uparrow$ & FID$\downarrow$ \\",
        r"\midrule",
    ]
    for row in rows_main:
        tex_main.append(
            f"{row['Dataset']} & {row['ACC']} & {row['NMI']} & {row['ARI']} & {row['FID']} \\\\"
        )
    tex_main += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (table_dir / "table01_main_results.tex").write_text("\n".join(tex_main))

    tex_base = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Baseline comparison (seed 0, 50 epochs).}",
        r"\label{tab:baselines}",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"Method & Dataset & ACC$\uparrow$ & NMI$\uparrow$ & ARI$\uparrow$ & FID$\downarrow$ \\",
        r"\midrule",
    ]
    rows_base = []
    for ds in DATASETS:
        label = "MNIST" if ds == "mnist" else "Fashion-MNIST"
        df = pd.read_csv(ROOT / RUN_PATHS[ds]["baselines"] / "comparison_results.csv", index_col=0)
        for method in BASELINE_METHODS:
            if method not in df.index:
                continue
            r = df.loc[method]
            rows_base.append(
                {
                    "Method": method,
                    "Dataset": label,
                    "ACC": pct(r["ACC"]),
                    "NMI": pct(r["NMI"]),
                    "ARI": pct(r["ARI"]),
                    "FID": f"{r['FID']:.1f}",
                }
            )
            if method == "CS-WAE":
                tex_base.append(
                    f"\\textbf{{CS-WAE}} & {label} & \\textbf{{{pct(r['ACC'])}}} & "
                    f"\\textbf{{{pct(r['NMI'])}}} & \\textbf{{{pct(r['ARI'])}}} & "
                    f"\\textbf{{{r['FID']:.1f}}} \\\\"
                )
            else:
                tex_base.append(
                    f"{method} & {label} & {pct(r['ACC'])} & {pct(r['NMI'])} & "
                    f"{pct(r['ARI'])} & {r['FID']:.1f} \\\\"
                )
    pd.DataFrame(rows_base).to_csv(table_dir / "table02_baselines.csv", index=False)
    tex_base += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (table_dir / "table02_baselines.tex").write_text("\n".join(tex_base))

    tex_abl = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Ablation study (seed 0, 50 epochs).}",
        r"\label{tab:ablation}",
        r"\begin{tabular}{llccc}",
        r"\toprule",
        r"Dataset & Variant & ACC$\uparrow$ & NMI$\uparrow$ & FID$\downarrow$ \\",
        r"\midrule",
    ]
    rows_abl = []
    for ds in DATASETS:
        label = "MNIST" if ds == "mnist" else "Fashion-MNIST"
        df = pd.read_csv(ROOT / RUN_PATHS[ds]["ablation"] / "ablation_results.csv", index_col=0)
        for _, vname in ABLATION_VARIANTS:
            if vname not in df.index:
                continue
            r = df.loc[vname]
            rows_abl.append(
                {
                    "Dataset": label,
                    "Variant": vname.replace("CS-WAE ", ""),
                    "ACC": pct(r["ACC"]),
                    "NMI": pct(r["NMI"]),
                    "FID": f"{r['FID']:.1f}",
                }
            )
            short = vname.replace("CS-WAE (Baseline)", "Full").replace("CS-WAE ", "")
            tex_abl.append(
                f"{label} & {short} & {pct(r['ACC'])} & {pct(r['NMI'])} & {r['FID']:.1f} \\\\"
            )
    pd.DataFrame(rows_abl).to_csv(table_dir / "table03_ablation.csv", index=False)
    tex_abl += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (table_dir / "table03_ablation.tex").write_text("\n".join(tex_abl))
    print(f"Exported LaTeX tables to {table_dir}")


def plot_baseline_bars(fig_dir: Path) -> None:
    acc_mnist, acc_fashion, fid_mnist, fid_fashion = [], [], [], []
    for ds, acc_list, fid_list in (
        ("mnist", acc_mnist, fid_mnist),
        ("fashion_mnist", acc_fashion, fid_fashion),
    ):
        df = pd.read_csv(ROOT / RUN_PATHS[ds]["baselines"] / "comparison_results.csv", index_col=0)
        for m in BASELINE_METHODS:
            acc_list.append(df.loc[m, "ACC"] * 100)
            fid_list.append(df.loc[m, "FID"])

    x = np.arange(len(BASELINE_METHODS))
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, mnist_vals, fashion_vals, ylabel, title in (
        (axes[0], acc_mnist, acc_fashion, "ACC (%)", "Clustering Accuracy"),
        (axes[1], fid_mnist, fid_fashion, "FID", "Fréchet Inception Distance"),
    ):
        ax.bar(x - width / 2, mnist_vals, width, label="MNIST", color="#4C72B0")
        ax.bar(x + width / 2, fashion_vals, width, label="Fashion-MNIST", color="#DD8452")
        ax.set_xticks(x)
        ax.set_xticklabels(BASELINE_METHODS, rotation=15, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Baseline Comparison (seed 0, 50 epochs)", fontsize=12, y=1.02)
    _save_fig(fig, fig_dir, "fig02_baseline_comparison")


def plot_ablation_bars(fig_dir: Path) -> None:
    labels = ["Full", "w/o Sup\nMMD", "Euclidean", "vMF", "Minimal"]
    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 4))
    for offset, ds, color in (
        (-width / 2, "mnist", "#4C72B0"),
        (width / 2, "fashion_mnist", "#DD8452"),
    ):
        df = pd.read_csv(ROOT / RUN_PATHS[ds]["ablation"] / "ablation_results.csv", index_col=0)
        accs = [df.loc[vname, "ACC"] * 100 for _, vname in ABLATION_VARIANTS]
        name = "MNIST" if ds == "mnist" else "Fashion-MNIST"
        ax.bar(x + offset, accs, width, label=name, color=color)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("ACC (%)")
    ax.set_title("Ablation Study — Clustering Accuracy (seed 0)")
    ax.axhline(10, color="gray", linestyle="--", linewidth=0.8, label="Chance (10%)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    _save_fig(fig, fig_dir, "fig03_ablation_acc")


def load_baseline_model(name: str, n_classes: int, ckpt: Path, device: torch.device):
    if name == "CS-WAE":
        model = SphericalWAE_Supervised(config.latent_dim, n_classes)
    elif name == "VAE":
        model = VAE(config.latent_dim)
    elif name == "VaDE":
        model = VaDE(config.latent_dim, n_classes)
    elif name == "WAE-MMD":
        model = WAE_MMD(config.latent_dim)
    else:
        raise ValueError(name)
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=False))
    model.to(device).eval()
    return model


def reconstruct_batch(model, images: torch.Tensor, model_name: str) -> torch.Tensor:
    with torch.no_grad():
        if model_name == "CS-WAE":
            return model(images)[0]
        if model_name == "VAE":
            return model(images)[0]
        if model_name == "VaDE":
            return model(images)[0]
        if model_name == "WAE-MMD":
            return model(images)[0]
    raise ValueError(model_name)


def extract_latents(model, loader, model_name: str, device: torch.device, max_samples: int = 2000):
    latents, labels = [], []
    with torch.no_grad():
        for images, batch_labels in loader:
            if len(labels) * images.size(0) >= max_samples:
                break
            images = images.to(device)
            if model_name == "CS-WAE":
                mu, _ = model.encode_to_distribution(images)
                z = mu
            elif model_name in ("VAE", "VaDE"):
                z, _ = model.encoder(images)
            elif model_name == "WAE-MMD":
                z = model.encoder(images)
            else:
                raise ValueError(model_name)
            latents.append(z.cpu().numpy())
            labels.append(batch_labels.numpy())
    latents_np = np.concatenate(latents, axis=0)[:max_samples]
    labels_np = np.concatenate(labels, axis=0)[:max_samples]
    return latents_np, labels_np


def cluster_assignments(latents: np.ndarray, labels: np.ndarray, n_classes: int, init_centers=None):
    if init_centers is not None:
        kmeans = KMeans(n_clusters=n_classes, init=init_centers, n_init=1, random_state=42)
    else:
        kmeans = KMeans(n_clusters=n_classes, random_state=42, n_init="auto")
    preds = kmeans.fit_predict(latents)
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for p, t in zip(preds, labels):
        cm[p, t] += 1
    row_ind, col_ind = linear_sum_assignment(-cm)
    mapping = {row: col for row, col in zip(row_ind, col_ind)}
    mapped = np.array([mapping.get(p, p) for p in preds])
    correct = mapped == labels
    acc = correct.mean()
    return preds, correct, acc


def generate_class_sample(model, model_name: str, class_idx: int, device: torch.device) -> np.ndarray:
    with torch.no_grad():
        if model_name == "CS-WAE":
            mu = F.normalize(model.prior_mus[class_idx : class_idx + 1], p=2, dim=1)
            eps = sample_uniform_sphere(1, config.latent_dim, device=device)
            z = mobius_reparam(eps, mu, torch.tensor([model.rho_p], device=device))
            img = model.decoder(z)
        elif model_name == "VaDE":
            mu = model.mu_c[class_idx : class_idx + 1]
            log_var = model.log_var_c[class_idx : class_idx + 1]
            z = model.reparameterize(mu, log_var)
            img = model.decoder(z)
        else:
            z = torch.randn(1, config.latent_dim, device=device)
            img = model.decoder(z)
    return img.cpu().squeeze().numpy()


def plot_reconstruction_grid(dataset: str, baseline_dir: Path, device: torch.device, fig_dir: Path, n_cols: int = 10):
    n_classes = get_dataset_info(dataset)["n_classes"]
    _, test_loader = get_loaders(dataset=dataset, seed=0, batch_size=128, num_workers=0)
    images, _ = next(iter(test_loader))
    images = images[:n_cols].to(device)

    fig, axes = plt.subplots(len(BASELINE_METHODS) * 2, n_cols, figsize=(n_cols * 1.2, len(BASELINE_METHODS) * 2.2))
    if n_cols == 1:
        axes = np.array([[axes]])

    for row_idx, method in enumerate(BASELINE_METHODS):
        ckpt = baseline_dir / method / f"{method}.pth"
        model = load_baseline_model(method, n_classes, ckpt, device)
        recons = reconstruct_batch(model, images, method).cpu()

        orig_row = row_idx * 2
        recon_row = row_idx * 2 + 1
        for col in range(n_cols):
            axes[orig_row, col].imshow(images[col].cpu().squeeze(), cmap="gray")
            axes[orig_row, col].axis("off")
            axes[recon_row, col].imshow(recons[col].squeeze(), cmap="gray")
            axes[recon_row, col].axis("off")
            if col == 0:
                axes[orig_row, col].set_ylabel(f"{method}\nOriginal", fontsize=9)
                axes[recon_row, col].set_ylabel("Recon", fontsize=9)

    title = "MNIST" if dataset == "mnist" else "Fashion-MNIST"
    fig.suptitle(f"Reconstruction Comparison — {title} (seed 0)", fontsize=12, y=1.01)
    plt.tight_layout()
    _save_fig(fig, fig_dir, f"fig05_reconstruction_grid_{dataset}")


def plot_generation_grid(dataset: str, baseline_dir: Path, device: torch.device, fig_dir: Path):
    n_classes = get_dataset_info(dataset)["n_classes"]
    fig, axes = plt.subplots(n_classes, len(BASELINE_METHODS), figsize=(len(BASELINE_METHODS) * 2, n_classes * 2))
    if n_classes == 1:
        axes = np.array([axes])

    for col, method in enumerate(BASELINE_METHODS):
        ckpt = baseline_dir / method / f"{method}.pth"
        model = load_baseline_model(method, n_classes, ckpt, device)
        for row in range(n_classes):
            img = generate_class_sample(model, method, row, device)
            axes[row, col].imshow(img, cmap="gray")
            axes[row, col].axis("off")
            if row == 0:
                axes[row, col].set_title(method, fontsize=10)
            if col == 0:
                axes[row, col].set_ylabel(f"Class {row}", fontsize=9)

    title = "MNIST" if dataset == "mnist" else "Fashion-MNIST"
    fig.suptitle(f"Class-conditional Generation — {title}", fontsize=12, y=1.01)
    plt.tight_layout()
    _save_fig(fig, fig_dir, f"fig06_generation_grid_{dataset}")


def plot_tsne_clustering(dataset: str, baseline_dir: Path, device: torch.device, fig_dir: Path, max_samples: int = 2000):
    n_classes = get_dataset_info(dataset)["n_classes"]
    _, test_loader = get_loaders(dataset=dataset, seed=0, batch_size=256, num_workers=0)

    methods_for_tsne = ("CS-WAE", "VaDE", "VAE")
    fig, axes = plt.subplots(len(methods_for_tsne) + 1, 1, figsize=(6, 3 * (len(methods_for_tsne) + 1)))

    cs_ckpt = baseline_dir / "CS-WAE" / "CS-WAE.pth"
    cs_model = load_baseline_model("CS-WAE", n_classes, cs_ckpt, device)
    latents, labels = extract_latents(cs_model, test_loader, "CS-WAE", device, max_samples)
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
    emb_gt = tsne.fit_transform(latents)

    axes[0].scatter(emb_gt[:, 0], emb_gt[:, 1], c=labels, cmap="tab10", s=6, alpha=0.7)
    axes[0].set_title("Ground-truth labels (CS-WAE latent t-SNE)")
    axes[0].set_xticks([])
    axes[0].set_yticks([])

    for ax_idx, method in enumerate(methods_for_tsne, start=1):
        ckpt = baseline_dir / method / f"{method}.pth"
        model = load_baseline_model(method, n_classes, ckpt, device)
        lat, lab = extract_latents(model, test_loader, method, device, max_samples)
        emb = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000).fit_transform(lat)
        init = model.mu_c.detach().cpu().numpy() if method == "VaDE" else None
        _, correct, acc = cluster_assignments(lat, lab, n_classes, init_centers=init)
        colors = np.where(correct, "#2ca02c", "#d62728")
        axes[ax_idx].scatter(emb[:, 0], emb[:, 1], c=colors, s=6, alpha=0.7)
        axes[ax_idx].set_title(f"{method} — ACC={acc * 100:.1f}% (green=correct, red=wrong)")
        axes[ax_idx].set_xticks([])
        axes[ax_idx].set_yticks([])

    title = "MNIST" if dataset == "mnist" else "Fashion-MNIST"
    fig.suptitle(f"Latent Space Clustering — {title}", fontsize=12, y=1.01)
    plt.tight_layout()
    _save_fig(fig, fig_dir, f"fig04_tsne_clustering_{dataset}")


def plot_ablation_recon_composite(dataset: str, ablation_dir: Path, fig_dir: Path):
    variant_labels = ["Full", "w/o Sup MMD", "Euclidean", "vMF", "Minimal"]
    panels = []
    for (vdir, _), label in zip(ABLATION_VARIANTS, variant_labels):
        path = ablation_dir / vdir / "reconstructions.png"
        if not path.exists():
            print(f"  skip missing {path}")
            continue
        img = Image.open(path).convert("RGB")
        w, h = img.size
        # Bottom half = reconstructions row in standard layout (2 rows x 10 cols)
        recon = img.crop((0, h // 2, w, h))
        panels.append((label, np.asarray(recon)))

    if not panels:
        return

    fig = plt.figure(figsize=(12, 2.5 * len(panels)))
    gs = GridSpec(len(panels), 1, figure=fig, hspace=0.35)
    for i, (label, arr) in enumerate(panels):
        ax = fig.add_subplot(gs[i, 0])
        ax.imshow(arr)
        ax.set_ylabel(label, fontsize=10, rotation=0, labelpad=60, va="center")
        ax.set_xticks([])
        ax.set_yticks([])

    title = "MNIST" if dataset == "mnist" else "Fashion-MNIST"
    fig.suptitle(f"Ablation Reconstructions — {title}", fontsize=12, y=1.02)
    _save_fig(fig, fig_dir, f"fig11_ablation_recon_{dataset}")


def plot_fid_sample_panel(dataset: str, baseline_dir: Path, fig_dir: Path, n_samples: int = 8):
    fig, axes = plt.subplots(len(BASELINE_METHODS), n_samples, figsize=(n_samples * 1.2, len(BASELINE_METHODS) * 1.3))
    rng = np.random.default_rng(42)

    for row, method in enumerate(BASELINE_METHODS):
        gen_dir = baseline_dir / method / f"fid_images_{method}" / "generated"
        if not gen_dir.exists():
            print(f"  skip FID panel — missing {gen_dir}")
            return
        files = sorted(gen_dir.glob("gen_*.png"))
        picks = rng.choice(len(files), size=min(n_samples, len(files)), replace=False)
        for col, idx in enumerate(picks):
            img = mpimg.imread(files[idx])
            axes[row, col].imshow(img.squeeze(), cmap="gray")
            axes[row, col].axis("off")
            if col == 0:
                axes[row, col].set_ylabel(method, fontsize=9)
            if row == 0:
                axes[row, col].set_title(f"#{idx}", fontsize=8)

    title = "MNIST" if dataset == "mnist" else "Fashion-MNIST"
    fig.suptitle(f"Random Generated Samples (FID pool) — {title}", fontsize=12, y=1.02)
    plt.tight_layout()
    _save_fig(fig, fig_dir, f"fig15_fid_samples_{dataset}")


def parse_args():
    p = argparse.ArgumentParser(description="Generate paper tables and figures")
    p.add_argument("--output-dir", type=str, default="paper_outputs")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--skip-model-plots", action="store_true", help="Only CSV/LaTeX and bar charts")
    p.add_argument("--max-tsne-samples", type=int, default=2000)
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    out = ROOT / args.output_dir
    table_dir = out / "tables"
    fig_dir = out / "figures"

    print("=== Paper figure generation ===")
    print(f"Output: {out}")

    print("\n[1/4] Exporting LaTeX tables...")
    export_latex_tables(table_dir)

    print("\n[2/4] Plotting bar charts...")
    plot_baseline_bars(fig_dir)
    plot_ablation_bars(fig_dir)

    print("\n[3/4] Ablation reconstruction composites...")
    for ds in DATASETS:
        plot_ablation_recon_composite(ds, ROOT / RUN_PATHS[ds]["ablation"], fig_dir)

    if args.skip_model_plots:
        print("\nSkipped model-based plots (--skip-model-plots).")
        print("Done.")
        return

    device = set_device(args.device)
    print(f"\n[4/4] Model-based figures on {device}...")

    for ds in DATASETS:
        baseline_dir = ROOT / RUN_PATHS[ds]["baselines"]
        print(f"\n  Dataset: {ds}")
        print("  - reconstruction grid")
        plot_reconstruction_grid(ds, baseline_dir, device, fig_dir)
        print("  - generation grid")
        plot_generation_grid(ds, baseline_dir, device, fig_dir)
        print("  - t-SNE clustering")
        plot_tsne_clustering(ds, baseline_dir, device, fig_dir, args.max_tsne_samples)
        print("  - FID sample panel")
        plot_fid_sample_panel(ds, baseline_dir, fig_dir)

    print("\n=== All paper outputs written to", out, "===")


if __name__ == "__main__":
    main()

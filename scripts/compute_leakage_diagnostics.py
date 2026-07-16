"""
compute_leakage_diagnostics.py

Compute conditional style leakage diagnostics for any factorized generative model.

Diagnostics (Section 3.4 of paper):
  1. Global MMD:     MMD²(q(z_s), N(0,I))           — should be ≈0 if global style reg works
  2. Δ_inter:        mean inter-class style mean sep   — should be ≈0 if z_s ⊥ y
  3. LP(z_s → y):    linear probe accuracy on z_s      — should be ≈ 1/K (10%) if z_s ⊥ y
  4. HSIC(z_s, y):   kernel independence test          — should be ≈0 if z_s ⊥ y
  5. Gen self-ACC:   classifier accuracy on generated  — should be ≈1.0 for class-cond gen

Usage
-----
python scripts/compute_leakage_diagnostics.py \\
    --model-type fcswae \\
    --checkpoint runs_f/mnist/seed_0/f_cs_wae_model.pth \\
    --dataset mnist \\
    --device cpu \\
    --n-samples 2048 \\
    --gen-per-class 50

Model types supported: fcswae, vae, waemmd, vade, betavae, factorvae, betatcvae
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.datasets.loaders import get_dataloader
from src.utils.seed import set_seed


# ---------------------------------------------------------------------------
# Kernel helpers
# ---------------------------------------------------------------------------

def _rbf(x: torch.Tensor, y: torch.Tensor, sigma: float) -> torch.Tensor:
    dist_sq = torch.cdist(x, y, p=2).pow(2)
    return torch.exp(-dist_sq / (2.0 * sigma ** 2 + 1e-8))


def mmd2_rbf(q: torch.Tensor, p: torch.Tensor) -> float:
    """Multi-scale RBF MMD² between two Euclidean sample sets."""
    if q.shape[0] < 2 or p.shape[0] < 2:
        return 0.0
    sigmas = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]
    val = 0.0
    for s in sigmas:
        kqq = _rbf(q, q, s).mean().item()
        kpp = _rbf(p, p, s).mean().item()
        kqp = _rbf(q, p, s).mean().item()
        val += kqq + kpp - 2.0 * kqp
    return val / len(sigmas)


def hsic(z: torch.Tensor, y: torch.Tensor, sigma_z: float = 1.0) -> float:
    """Empirical HSIC between continuous z_s and discrete y (one-hot kernel).

    Uses RBF kernel on z and delta kernel on y.
    Reference: Gretton et al. 2005.
    """
    n = z.shape[0]
    if n < 2:
        return 0.0

    # Kernel on z: RBF
    Kz = _rbf(z, z, sigma_z)

    # Kernel on y: delta (= outer equality product)
    y_vec = y.unsqueeze(1).float()  # (n, 1)
    Ky = (y_vec == y_vec.T).float()

    # Centre both kernels
    H = torch.eye(n, device=z.device) - 1.0 / n
    KzH = Kz @ H
    KyH = Ky @ H
    hsic_val = (KzH * KyH.T).sum() / ((n - 1) ** 2)
    return hsic_val.item()


# ---------------------------------------------------------------------------
# Leakage metrics
# ---------------------------------------------------------------------------

def compute_global_mmd(z_s: torch.Tensor) -> float:
    """MMD²(q(z_s), N(0,I))."""
    z_p = torch.randn_like(z_s)
    return mmd2_rbf(z_s, z_p)


def compute_delta_inter(z_s: torch.Tensor, labels: torch.Tensor, n_classes: int) -> float:
    """Mean pairwise distance between class-conditional style means.

    Δ_inter = (1 / C(K,2)) * Σ_{j<k} ||μ_s^(j) - μ_s^(k)||₂
    """
    class_means = []
    for k in range(n_classes):
        mask = labels == k
        if mask.sum() > 0:
            class_means.append(z_s[mask].mean(dim=0))

    if len(class_means) < 2:
        return 0.0

    total = 0.0
    count = 0
    for i in range(len(class_means)):
        for j in range(i + 1, len(class_means)):
            total += (class_means[i] - class_means[j]).norm().item()
            count += 1
    return total / count if count > 0 else 0.0


# ---------------------------------------------------------------------------
# JointMMD — proxy for Theorem term (4), I_q(z_c; z_s | y) (main_v6.tex Sec. 3.5)
# ---------------------------------------------------------------------------

_SIGMAS_SPHERICAL = [0.1, 0.3, 0.5, 1.0, 2.0]  # chordal dist_sq in [0,4] for unit vectors
_SIGMAS_EUCLIDEAN = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]  # matches mmd2_rbf's ladder


def _kc_multiscale(zc_a: torch.Tensor, zc_b: torch.Tensor) -> torch.Tensor:
    """Multi-scale spherical RBF kernel matrix on unit-norm z_c vectors."""
    from src.utils.utils import rbf_kernel as _rbf_spherical
    k = torch.zeros(zc_a.shape[0], zc_b.shape[0], device=zc_a.device)
    for s in _SIGMAS_SPHERICAL:
        k = k + _rbf_spherical(zc_a, zc_b, s)
    return k / len(_SIGMAS_SPHERICAL)


def _ks_multiscale(zs_a: torch.Tensor, zs_b: torch.Tensor) -> torch.Tensor:
    """Multi-scale Euclidean RBF kernel matrix on z_s vectors (same ladder as mmd2_rbf)."""
    k = torch.zeros(zs_a.shape[0], zs_b.shape[0], device=zs_a.device)
    for s in _SIGMAS_EUCLIDEAN:
        k = k + _rbf(zs_a, zs_b, s)
    return k / len(_SIGMAS_EUCLIDEAN)


def _joint_product_mmd2(zc1, zs1, zc2, zs2) -> float:
    """Biased MMD² between two sets of (z_c, z_s) pairs under a product kernel
    (spherical on z_c, Euclidean on z_s). Both kernel components use a
    multi-scale bandwidth ladder rather than a single fixed sigma, to avoid
    the saturation failure mode found in this script's own hsic() (single
    sigma_z=1.0, empirically flat regardless of true dependence — see
    main_v6.tex's HSIC discussion)."""
    def K(zc_a, zs_a, zc_b, zs_b):
        return _kc_multiscale(zc_a, zc_b) * _ks_multiscale(zs_a, zs_b)
    k11 = K(zc1, zs1, zc1, zs1).mean()
    k22 = K(zc2, zs2, zc2, zs2).mean()
    k12 = K(zc1, zs1, zc2, zs2).mean()
    return (k11 + k22 - 2.0 * k12).item()


def compute_joint_mmd(mu_c: torch.Tensor, mu_s: torch.Tensor, labels: torch.Tensor,
                       n_classes: int, seed: int = 0) -> float:
    """JointMMD (main_v6.tex Eq. in Sec. 3.5, Theorem term (4) proxy):

        (1/K) sum_k MMD²( {(z_c^i,z_s^i)}_{y_i=k}, {(z_c^i,z_s^{pi(i)})}_{y_i=k} )

    pi permutes style codes within class k, breaking exactly the within-class
    z_c-z_s coupling while leaving both per-class marginals (terms 1-3)
    unchanged. mu_c must already be unit-norm (posterior-mean semantic codes);
    mu_s is the posterior-mean style code. Requires >=4 samples per class to
    form a stable permuted comparison; classes with fewer are skipped.
    """
    zc_all = F.normalize(mu_c, p=2, dim=1)
    generator = torch.Generator().manual_seed(seed)
    vals = []
    for k in range(n_classes):
        mask = labels == k
        n_k = int(mask.sum().item())
        if n_k < 4:
            continue
        zc_k = zc_all[mask]
        zs_k = mu_s[mask]
        perm = torch.randperm(n_k, generator=generator)
        vals.append(_joint_product_mmd2(zc_k, zs_k, zc_k, zs_k[perm]))
    return float(sum(vals) / len(vals)) if vals else 0.0


def compute_linear_probe(
    z_s: torch.Tensor,
    labels: torch.Tensor,
    z_s_val: torch.Tensor | None = None,
    labels_val: torch.Tensor | None = None,
    n_epochs: int = 100,
    lr: float = 1e-2,
    return_model: bool = False,
):
    """Train a linear probe z_s → y and return accuracy.

    Trains on (z_s, labels), evaluates on (z_s_val, labels_val) if provided,
    else uses the same training set for evaluation (quick estimate).

    If return_model is True, returns (accuracy, trained probe) instead of
    just accuracy — callers that need to evaluate the probe on synthetic
    z_s draws afterward (e.g. a wrong-style-rate diagnostic) need the model.
    """
    n_classes = int(labels.max().item()) + 1
    d = z_s.shape[1]
    clf = nn.Linear(d, n_classes).to(z_s.device)
    opt = torch.optim.Adam(clf.parameters(), lr=lr, weight_decay=1e-4)

    for _ in range(n_epochs):
        logits = clf(z_s.detach())
        loss = F.cross_entropy(logits, labels)
        opt.zero_grad()
        loss.backward()
        opt.step()

    clf.eval()
    with torch.no_grad():
        z_eval = z_s_val if z_s_val is not None else z_s
        l_eval = labels_val if labels_val is not None else labels
        preds = clf(z_eval).argmax(dim=1)
        acc = (preds == l_eval).float().mean().item()

    return (acc, clf) if return_model else acc


def compute_gen_self_accuracy(
    model,
    aux_classifier: nn.Module,
    n_classes: int,
    gen_per_class: int,
    device: torch.device,
    style_mode: str = "global_gaussian",
    style_stats: dict | None = None,
) -> float:
    """Generate images class-conditionally, classify with aux_classifier, return self-ACC.

    style_mode:
        "global_gaussian"  : z_s ~ N(0,I)
        "class_diag_t025"  : z_s ~ N(mu_s^k, 0.25² * sigma_s^k)
        "class_mean"       : z_s = mu_s^k (deterministic)
    style_stats: dict with keys "means" (K, d_s) and "stds" (K, d_s) tensors
    """
    model.eval()
    aux_classifier.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for k in range(n_classes):
            z_c, z_s = model.sample_from_class_prior(k, gen_per_class, device)

            if style_mode == "global_gaussian":
                z_s = torch.randn(gen_per_class, z_s.shape[1], device=device)
            elif style_mode == "class_mean" and style_stats is not None:
                z_s = style_stats["means"][k].unsqueeze(0).expand(gen_per_class, -1)
            elif style_mode == "class_diag_t025" and style_stats is not None:
                mu_k = style_stats["means"][k]
                std_k = style_stats["stds"][k]
                z_s = mu_k + 0.25 * std_k * torch.randn(gen_per_class, z_s.shape[1], device=device)

            x_gen = model.decoder(torch.cat([z_c, z_s], dim=1)).clamp(0, 1)

            mu_c_gen, _ = model.encode_to_distribution(x_gen)
            logits = aux_classifier(mu_c_gen)
            preds = logits.argmax(dim=1)
            correct += (preds == k).sum().item()
            total += gen_per_class

    return correct / total if total > 0 else 0.0


# ---------------------------------------------------------------------------
# Extract z_s from a trained model on a dataset
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_style_latents(model, loader, device, model_type: str = "fcswae"):
    """Return (z_s, labels) tensors from a trained model."""
    model.eval()
    z_s_list, label_list = [], []

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        if model_type == "fcswae":
            mu_c, rho_c, mu_s, logvar_s = model.encoder(x)
            std_s = torch.exp(0.5 * logvar_s.clamp(-10, 10))
            eps = torch.randn_like(std_s)
            z_s = mu_s + std_s * eps
        elif model_type in ("vae", "waemmd"):
            # Standard VAE/WAE encoder: returns (mu, logvar) for full latent
            # Treat second half of latent as "style"
            mu, logvar = model.encode(x)
            std = torch.exp(0.5 * logvar.clamp(-10, 10))
            z_s = mu + std * torch.randn_like(std)
        elif model_type == "betavae":
            mu, logvar = model.encode(x)
            std = torch.exp(0.5 * logvar.clamp(-10, 10))
            z_s = mu + std * torch.randn_like(std)
        elif model_type in ("factorvae", "betatcvae"):
            mu, logvar = model.encode(x)
            std = torch.exp(0.5 * logvar.clamp(-10, 10))
            z_s = mu + std * torch.randn_like(std)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        z_s_list.append(z_s.cpu())
        label_list.append(y.cpu())

    return torch.cat(z_s_list, dim=0), torch.cat(label_list, dim=0)


@torch.no_grad()
def extract_full_encodings(model, loader, device):
    """Return (mu_c, mu_s, labels) — deterministic (posterior-mean) encodings.

    Unlike extract_style_latents (which reparameterizes z_s with sampling
    noise), this returns the raw posterior means for both the semantic and
    style heads. Used by diagnostics that intervene directly on latent
    points (latent swap, conditional-MMD) where isolating "which point in
    latent space" from reparameterization noise matters. fcswae-only.
    """
    model.eval()
    mu_c_list, mu_s_list, label_list = [], [], []

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        mu_c, _rho_c, mu_s, _logvar_s = model.encode(x)
        mu_c_list.append(mu_c.cpu())
        mu_s_list.append(mu_s.cpu())
        label_list.append(y.cpu())

    return (
        torch.cat(mu_c_list, dim=0),
        torch.cat(mu_s_list, dim=0),
        torch.cat(label_list, dim=0),
    )


# ---------------------------------------------------------------------------
# Full diagnostic suite
# ---------------------------------------------------------------------------

def run_diagnostics(
    model,
    loader,
    device: torch.device,
    n_classes: int = 10,
    model_type: str = "fcswae",
    gen_per_class: int = 50,
    aux_classifier: nn.Module | None = None,
    verbose: bool = True,
) -> dict:
    """Run all leakage diagnostics and return results dict."""

    print("Extracting style latents...")
    z_s, labels = extract_style_latents(model, loader, device, model_type)
    z_s = z_s.to(device)
    labels = labels.to(device)

    results = {}

    # 1. Global MMD
    print("Computing global MMD...")
    results["global_mmd"] = compute_global_mmd(z_s)

    # 2. Δ_inter
    print("Computing inter-class style separation...")
    results["delta_inter"] = compute_delta_inter(z_s, labels, n_classes)

    # 3. Linear probe LP(z_s → y)
    print("Computing linear probe accuracy...")
    results["lp_accuracy"] = compute_linear_probe(z_s, labels)

    # 4. HSIC(z_s, y)
    print("Computing HSIC...")
    # Use subset for HSIC (O(n²) kernel)
    n_hsic = min(1024, z_s.shape[0])
    idx = torch.randperm(z_s.shape[0])[:n_hsic]
    results["hsic"] = hsic(z_s[idx], labels[idx])

    # 4b. JointMMD — Theorem term (4) proxy, I_q(z_c;z_s|y). fcswae-only:
    # needs both z_c and z_s from the *same* forward pass (extract_full_encodings),
    # not just z_s (extract_style_latents above already discarded z_c).
    results["joint_mmd"] = None
    if model_type == "fcswae" and hasattr(model, "encode"):
        print("Computing JointMMD (term 4 proxy)...")
        mu_c_full, mu_s_full, labels_full = extract_full_encodings(model, loader, device)
        n_jmmd = min(2048, mu_c_full.shape[0])
        idx_j = torch.randperm(mu_c_full.shape[0])[:n_jmmd]
        results["joint_mmd"] = compute_joint_mmd(
            mu_c_full[idx_j].to(device), mu_s_full[idx_j].to(device),
            labels_full[idx_j].to(device), n_classes,
        )

    # 5. Gen self-ACC (only for models with sample_from_class_prior)
    results["gen_self_acc_global_gaussian"] = None
    results["gen_self_acc_class_diag_t025"] = None

    if aux_classifier is None and hasattr(model, "classifier"):
        # FCSWAE's own jointly-trained classifier head is the natural default
        # aux classifier — without this, gen_self_acc_* silently stays null
        # unless a separate --aux-classifier-checkpoint is passed (none exists
        # in this repo today).
        aux_classifier = model.classifier

    if aux_classifier is not None and hasattr(model, "sample_from_class_prior"):
        print("Computing gen self-accuracy (global Gaussian style)...")
        results["gen_self_acc_global_gaussian"] = compute_gen_self_accuracy(
            model, aux_classifier, n_classes, gen_per_class, device,
            style_mode="global_gaussian",
        )

        # Compute class-conditional style stats for conditional sampling
        print("Computing class style stats for conditional sampling...")
        class_means, class_stds = [], []
        for k in range(n_classes):
            mask = labels == k
            if mask.sum() > 0:
                zk = z_s[mask]
                class_means.append(zk.mean(dim=0))
                class_stds.append(zk.std(dim=0).clamp(min=1e-6))
            else:
                d = z_s.shape[1]
                class_means.append(torch.zeros(d, device=device))
                class_stds.append(torch.ones(d, device=device))
        style_stats = {
            "means": torch.stack(class_means),
            "stds": torch.stack(class_stds),
        }

        print("Computing gen self-accuracy (class diag t=0.25)...")
        results["gen_self_acc_class_diag_t025"] = compute_gen_self_accuracy(
            model, aux_classifier, n_classes, gen_per_class, device,
            style_mode="class_diag_t025",
            style_stats=style_stats,
        )

    if verbose:
        print("\n" + "=" * 60)
        print("LEAKAGE DIAGNOSTIC RESULTS")
        print("=" * 60)
        print(f"  Global MMD²(q(z_s), N(0,I))  : {results['global_mmd']:.6f}  (should ≈ 0)")
        print(f"  Δ_inter (inter-class sep)     : {results['delta_inter']:.4f}  (should ≈ 0)")
        print(f"  LP(z_s → y) accuracy          : {results['lp_accuracy']:.4f}  (should ≈ {1/n_classes:.2f})")
        print(f"  HSIC(z_s, y)                  : {results['hsic']:.6f}  (should ≈ 0)")
        if results["joint_mmd"] is not None:
            print(f"  JointMMD (term 4 proxy)       : {results['joint_mmd']:.6f}  (should ≈ 0)")
        if results["gen_self_acc_global_gaussian"] is not None:
            print(f"  Gen self-ACC (global N(0,I)) : {results['gen_self_acc_global_gaussian']:.4f}  (should ≈ 1.0)")
        if results["gen_self_acc_class_diag_t025"] is not None:
            print(f"  Gen self-ACC (class t=0.25)  : {results['gen_self_acc_class_diag_t025']:.4f}  (should ≈ 1.0)")
        print("=" * 60)

    return results


# ---------------------------------------------------------------------------
# Shared loading helpers (reused by other diagnostic scripts — see
# scripts/latent_swap_diagnostics.py, conditional_mmd_diagnostics.py,
# wrong_style_rate_diagnostics.py, run_delta_sweep.py)
# ---------------------------------------------------------------------------

def load_fcswae_checkpoint(checkpoint_path, dataset: str, device, variant: str | None = None):
    """Load a trained FCSWAE (or FCSWAEAblation) model from a checkpoint.

    variant: an ABLATION_VARIANTS key (e.g. "no_classifier", "gaussian_class_prior")
    if the checkpoint came from run_ablation_f_cs_wae.py -- its encoder/decoder
    structure (e.g. euclidean vs. spherical z_c, presence of a classifier head)
    differs from plain FCSWAE, so the state_dict keys would otherwise mismatch.
    """
    from src.config_f_cs_wae import f_cs_wae_config as cfg
    from src.utils.dataset_config import apply_dataset_config

    apply_dataset_config(cfg, dataset, backbone="resnet18")
    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt.get("model_state_dict", ckpt)
    # Infer semantic_dim/style_dim directly from the checkpoint rather than
    # trusting cfg's defaults: checkpoints trained with --dc/--ds overrides
    # (e.g. capacity-ablation runs) have non-default dims that a fresh cfg
    # instance in this process has no way to know about.
    semantic_dim = state["encoder.fc_mu_c.weight"].shape[0]
    style_dim = state["encoder.fc_mu_s.weight"].shape[0]
    if variant is not None:
        from src.models.f_cs_wae_ablation import create_f_cs_wae_ablation
        model = create_f_cs_wae_ablation(
            variant,
            semantic_dim=semantic_dim,
            style_dim=style_dim,
            n_classes=cfg.n_classes,
            in_channels=cfg.in_channels,
            image_size=cfg.image_size,
        ).to(device)
    else:
        from src.models.f_cs_wae import FCSWAE
        model = FCSWAE(
            semantic_dim=semantic_dim,
            style_dim=style_dim,
            n_classes=cfg.n_classes,
            in_channels=cfg.in_channels,
            image_size=cfg.image_size,
        ).to(device)
    model.load_state_dict(state)
    model.eval()
    return model


def get_eval_subset_loader(
    dataset: str,
    n_samples: int,
    batch_size: int = 256,
    seed: int = 0,
    split: str = "test",
    num_workers: int = 0,
) -> DataLoader:
    """Build a seeded, reproducible subset DataLoader for diagnostics.

    split: "train" or "test" — which underlying loader's dataset to subset.
    Seeded so repeated invocations (e.g. across delta-sweep checkpoints)
    evaluate on the same subset instead of adding unnecessary extra noise
    on top of single-seed training runs.
    """
    train_loader, test_loader = get_dataloader(dataset, batch_size=batch_size, num_workers=num_workers)
    base_loader = train_loader if split == "train" else test_loader
    full_dataset = base_loader.dataset
    n = min(n_samples, len(full_dataset))
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(full_dataset), generator=generator)[:n].tolist()
    subset = Subset(full_dataset, indices)
    return DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=num_workers)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Compute conditional style leakage diagnostics")
    p.add_argument("--model-type", default="fcswae",
                   choices=["fcswae", "vae", "waemmd", "betavae", "factorvae", "betatcvae"],
                   help="Model type to load")
    p.add_argument("--checkpoint", required=True, help="Path to model checkpoint (.pth)")
    p.add_argument("--variant", default=None,
                   help="F-CS-WAE ablation variant key (e.g. no_classifier, "
                        "gaussian_class_prior) if the checkpoint was produced by "
                        "run_ablation_f_cs_wae.py rather than train_f_cs_wae.py. "
                        "Default: plain FCSWAE.")
    p.add_argument("--dataset", default="mnist",
                   choices=["mnist", "fashion_mnist", "cifar10"])
    p.add_argument("--device", default="cpu")
    p.add_argument("--n-samples", type=int, default=2048,
                   help="Number of test samples to use")
    p.add_argument("--gen-per-class", type=int, default=50,
                   help="Generated images per class for self-ACC")
    p.add_argument("--aux-classifier-checkpoint", default=None,
                   help="Path to auxiliary classifier checkpoint for gen self-ACC "
                        "(defaults to the model's own jointly-trained classifier head)")
    p.add_argument("--seed", type=int, default=0,
                   help="Seed for reproducible eval-subset sampling")
    p.add_argument("--out", default=None, help="Save results JSON to this path")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    set_seed(args.seed)

    print(f"Loading {args.model_type} from {args.checkpoint}...")
    if args.model_type == "fcswae":
        model = load_fcswae_checkpoint(args.checkpoint, args.dataset, device, variant=args.variant)
    else:
        raise NotImplementedError(
            f"Loading for model_type={args.model_type} not implemented yet. "
            "Add model-specific loading here."
        )

    # Optionally load a separate auxiliary classifier; run_diagnostics()
    # falls back to model.classifier when this stays None.
    aux_clf = None
    if args.aux_classifier_checkpoint:
        print(f"Loading aux classifier from {args.aux_classifier_checkpoint}...")
        clf_ckpt = torch.load(args.aux_classifier_checkpoint, map_location=device)
        aux_clf = nn.Linear(model.semantic_dim, 10).to(device)
        aux_clf.load_state_dict(clf_ckpt)

    print(f"Loading {args.dataset} test subset (n={args.n_samples}, seed={args.seed})...")
    loader = get_eval_subset_loader(
        args.dataset, args.n_samples, batch_size=256, seed=args.seed, split="test",
    )

    results = run_diagnostics(
        model, loader, device,
        n_classes=10,
        model_type=args.model_type,
        gen_per_class=args.gen_per_class,
        aux_classifier=aux_clf,
        verbose=True,
    )

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()

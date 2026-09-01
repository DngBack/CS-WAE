"""
compute_leakage_diagnostics.py

Compute conditional style leakage diagnostics for any factorized generative model.

Canonical Stage-0 diagnostics:
  1. Sampling-contract view: posterior-sampled z_s, MMD²_U, Delta_inter,
     calibrated multi-scale HSIC, and a secondary probe suite.
  2. Representation view: deterministic mu_s, train/validation/test logistic,
     MLP, RBF-SVM and k-NN probes, Delta_inter, and calibrated HSIC.
  3. Within-class view: JointMMD²_U on deterministic (mu_c, mu_s).
  4. Generation: primary pixel-space Gen-ACC from an independent real-image
     classifier; the model's internal re-encoding score is robustness-only.

Usage
-----
python scripts/compute_leakage_diagnostics.py \\
    --model-type fcswae \\
    --checkpoint runs_f/mnist/seed_0/f_cs_wae_model.pth \\
    --dataset mnist \\
    --device cpu \\
    --n-samples 2048 \\
    --gen-per-class 50

The CLI currently loads F-CS-WAE checkpoints.  Compatibility helpers for the
historical cross-model script remain importable, but new baseline adapters
must emit the same named-view result schema before their results are pooled.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.datasets.loaders import get_dataloader
from src.metrics.audit_protocol import (
    AUDIT_PROTOCOL_VERSION,
    DEFAULT_EVAL_SAMPLES,
    DEFAULT_HSIC_PERMUTATIONS,
    DEFAULT_HSIC_SAMPLES,
    classwise_conditional_hsic_permutation_test,
    conditional_mmd_to_standard_normal,
    delta_inter as protocol_delta_inter,
    fit_logistic_probe,
    global_mmd_to_standard_normal,
    hsic_permutation_test,
    joint_mmd_unbiased,
    mmd2_unbiased,
    multiscale_hsic_statistic,
    rbf_kernel,
    run_probe_suite,
    stratified_probe_split,
)
from src.metrics.factorized_adapter import (
    build_factorized_audit_adapter,
    summarize_style_posterior,
)
from src.models.f_cs_wae import sample_style_posterior
from src.models.external_classifiers import load_external_classifier_checkpoint
from src.utils.provenance import build_manifest, save_result_with_manifest
from src.utils.seed import set_seed


# ---------------------------------------------------------------------------
# Kernel helpers
# ---------------------------------------------------------------------------

def _rbf(x: torch.Tensor, y: torch.Tensor, sigma: float) -> torch.Tensor:
    """Compatibility alias for the canonical Stage-0 RBF kernel."""

    return rbf_kernel(x, y, sigma)


def mmd2_rbf(q: torch.Tensor, p: torch.Tensor) -> float:
    """Compatibility alias for evaluation MMD²_U (unbiased)."""

    return mmd2_unbiased(q, p)


def hsic(z: torch.Tensor, y: torch.Tensor, sigma_z: float = 1.0) -> float:
    """Compatibility scalar for multi-scale HSIC (use calibration in new runs)."""

    del sigma_z
    return multiscale_hsic_statistic(z, y)[0]


# ---------------------------------------------------------------------------
# Leakage metrics
# ---------------------------------------------------------------------------

def compute_global_mmd(z_s: torch.Tensor, seed: int = 0) -> float:
    """MMD²(q(z_s), N(0,I))."""
    return global_mmd_to_standard_normal(z_s, seed=seed)


def compute_delta_inter(z_s: torch.Tensor, labels: torch.Tensor, n_classes: int) -> float:
    """Mean pairwise distance between class-conditional style means.

    Δ_inter = (1 / C(K,2)) * Σ_{j<k} ||μ_s^(j) - μ_s^(k)||₂
    """
    return protocol_delta_inter(z_s, labels, n_classes)


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
    return joint_mmd_unbiased(mu_c, mu_s, labels, n_classes, seed=seed)


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

    Trains on (z_s, labels), evaluates on a held-out set.  If no held-out set
    is supplied, a deterministic stratified 60/20/20 split is created and
    the test partition is reported.  Standardization is fitted on train only.

    If return_model is True, returns (accuracy, trained probe) instead of
    just accuracy — callers that need to evaluate the probe on synthetic
    z_s draws afterward (e.g. a wrong-style-rate diagnostic) need the model.
    """
    del lr
    original_device = z_s.device
    if z_s_val is not None and labels_val is not None:
        accuracy, probe = fit_logistic_probe(
            z_s, labels, z_s_val, labels_val, seed=0, epochs=n_epochs
        )
        probe = probe.to(original_device)
        return (accuracy, probe) if return_model else accuracy

    split = stratified_probe_split(labels, seed=0)
    _validation_accuracy, probe = fit_logistic_probe(
        z_s[split.train], labels[split.train],
        z_s[split.validation], labels[split.validation],
        seed=0, epochs=n_epochs,
    )
    probe = probe.to(original_device)
    with torch.no_grad():
        predictions = probe(z_s[split.test]).argmax(1)
        accuracy = float((predictions == labels[split.test]).float().mean().item())
    return (accuracy, probe) if return_model else accuracy


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
            if hasattr(model, "sample_style"):
                z_s, _std_s = model.sample_style(mu_s, logvar_s)
            else:
                z_s, _std_s = sample_style_posterior(mu_s, logvar_s)
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


@torch.no_grad()
def extract_latent_views(model, loader, device, model_type: str = "fcswae", seed: int = 0):
    """Extract named posterior-sample and posterior-mean latent views once.

    The seeded CPU generator makes posterior sampling reproducible across
    repeated runs and independent of the accelerator's RNG state.
    """

    model.eval()
    generator = torch.Generator(device="cpu").manual_seed(seed)
    z_s_samples, mu_s_values, mu_c_values, z_c_samples = [], [], [], []
    logvar_values, effective_std_values, labels_values = [], [], []
    adapter = build_factorized_audit_adapter(model) if model_type == "fcswae" else None
    for images, labels in loader:
        images = images.to(device)
        if adapter is not None:
            named = adapter.encode_views(images, generator=generator)
            mu_c = named.content_mean
            z_c = named.content_sample
            mu_s = named.style_mean
            z_s = named.style_sample
            logvar_s = named.style_logvar
            effective_std = named.style_effective_std
        else:
            encoded = model.encode(images)
            mu, logvar_s = encoded
            split_at = mu.shape[1] // 2
            mu_c, mu_s = mu[:, :split_at], mu[:, split_at:]
            logvar_s = logvar_s[:, split_at:]
            z_s, effective_std = sample_style_posterior(
                mu_s, logvar_s, generator=generator
            )
            z_c = mu_c
        z_s_samples.append(z_s.cpu())
        mu_s_values.append(mu_s.cpu())
        mu_c_values.append(mu_c.cpu())
        z_c_samples.append(z_c.cpu())
        if logvar_s is not None:
            logvar_values.append(logvar_s.cpu())
        if effective_std is not None:
            effective_std_values.append(effective_std.cpu())
        labels_values.append(labels.cpu())
    return {
        "z_s_sample": torch.cat(z_s_samples),
        "mu_s": torch.cat(mu_s_values),
        "mu_c": torch.cat(mu_c_values),
        "z_c_sample": torch.cat(z_c_samples),
        "logvar_s": torch.cat(logvar_values) if logvar_values else None,
        "style_effective_std": (
            torch.cat(effective_std_values) if effective_std_values else None
        ),
        "labels": torch.cat(labels_values),
        "adapter_metadata": None if adapter is None else adapter.metadata(),
    }


@torch.no_grad()
def compute_external_generation_metrics(
    model,
    classifier: nn.Module,
    n_classes: int,
    gen_per_class: int,
    device: torch.device,
    style_mode: str,
    style_stats: dict | None = None,
) -> dict:
    """Pixel-space generated-class metrics using an independent classifier."""

    model.eval()
    classifier.eval()
    confusion = torch.zeros((n_classes, n_classes), dtype=torch.long)
    for requested_class in range(n_classes):
        z_c, z_s = model.sample_from_class_prior(requested_class, gen_per_class, device)
        if style_mode == "global_gaussian":
            z_s = torch.randn_like(z_s)
        elif style_mode == "class_diag_t025" and style_stats is not None:
            mean = style_stats["means"][requested_class]
            std = style_stats["stds"][requested_class]
            z_s = mean + 0.25 * std * torch.randn_like(z_s)
        elif style_mode == "class_mean" and style_stats is not None:
            z_s = style_stats["means"][requested_class].expand_as(z_s)
        generated = model.decoder(torch.cat([z_c, z_s], dim=1)).clamp(0, 1)
        predictions = classifier(generated).argmax(1).cpu()
        confusion[requested_class] = torch.bincount(predictions, minlength=n_classes)
    per_class = confusion.diagonal().float() / confusion.sum(1).clamp_min(1)
    return {
        "macro_gen_accuracy": float(per_class.mean().item()),
        "micro_gen_accuracy": float(confusion.diagonal().sum().item() / confusion.sum().item()),
        "per_class_gen_accuracy": per_class.tolist(),
        "confusion_counts": confusion.tolist(),
        "n_per_class": int(gen_per_class),
        "style_mode": style_mode,
        "evaluator": "independent pixel-space classifier",
    }


def compute_class_style_stats(
    mu_s: torch.Tensor, labels: torch.Tensor, n_classes: int, device: torch.device
) -> dict:
    """Fit per-class diagonal style summaries from an explicitly named split."""

    class_means, class_stds, class_counts = [], [], []
    for class_index in range(n_classes):
        class_values = mu_s[labels == class_index]
        class_counts.append(int(class_values.shape[0]))
        if class_values.shape[0]:
            class_means.append(class_values.mean(0))
            class_stds.append(class_values.std(0).clamp_min(1e-6))
        else:
            class_means.append(torch.zeros(mu_s.shape[1], device=mu_s.device))
            class_stds.append(torch.ones(mu_s.shape[1], device=mu_s.device))
    return {
        "means": torch.stack(class_means).to(device),
        "stds": torch.stack(class_stds).to(device),
        "counts": class_counts,
    }


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
    external_classifier: nn.Module | None = None,
    generation_style_stats: dict | None = None,
    generation_style_stats_source: str | None = None,
    seed: int = 0,
    probe_epochs: int = 300,
    hsic_permutations: int = DEFAULT_HSIC_PERMUTATIONS,
    verbose: bool = True,
) -> dict:
    """Run the canonical Stage-0 diagnostic suite.

    Sampling-contract metrics use ``z_s_sample``.  Representation probes,
    deterministic interventions, and JointMMD use posterior means.  The
    output names both views and retains flat aliases for old plotting code.
    """

    print("Extracting named latent views (z_s_sample, mu_s, mu_c)...")
    views = extract_latent_views(model, loader, device, model_type=model_type, seed=seed)
    z_s_sample = views["z_s_sample"].to(device)
    mu_s = views["mu_s"].to(device)
    mu_c = views["mu_c"].to(device)
    logvar_s = None if views["logvar_s"] is None else views["logvar_s"].to(device)
    labels = views["labels"].to(device)

    print("Computing U-statistic global MMD on posterior samples...")
    # Keep the Gaussian reference independent from the seeded posterior draw.
    # Reusing ``seed`` here reproduces the same CPU-normal stream used by
    # ``extract_latent_views`` and silently couples the two MMD samples.
    global_mmd_reference_seed = seed + 10_000
    global_mmd = compute_global_mmd(z_s_sample, seed=global_mmd_reference_seed)
    conditional_mmd_reference_seed = seed + 11_000
    conditional_mmd = conditional_mmd_to_standard_normal(
        z_s_sample,
        labels,
        n_classes,
        seed=conditional_mmd_reference_seed,
    )
    conditional_mmd["conditional_to_global_ratio"] = (
        None if global_mmd == 0.0 else conditional_mmd["mean_mmd2_u"] / global_mmd
    )
    delta_sample = compute_delta_inter(z_s_sample, labels, n_classes)
    delta_mean = compute_delta_inter(mu_s, labels, n_classes)

    print("Running train/validation/test probe suites...")
    probes_mean = run_probe_suite(mu_s, labels, seed=seed, epochs=probe_epochs)
    probes_sample = run_probe_suite(z_s_sample, labels, seed=seed, epochs=probe_epochs)

    generator = torch.Generator(device="cpu").manual_seed(seed)
    n_hsic = min(DEFAULT_HSIC_SAMPLES, labels.shape[0])
    hsic_indices = torch.randperm(labels.shape[0], generator=generator)[:n_hsic].to(device)
    print(f"Computing multi-scale HSIC calibration (n={n_hsic}, B={hsic_permutations})...")
    hsic_mean = hsic_permutation_test(
        mu_s[hsic_indices], labels[hsic_indices], seed=seed,
        n_permutations=hsic_permutations,
    )
    hsic_sample = hsic_permutation_test(
        z_s_sample[hsic_indices], labels[hsic_indices], seed=seed,
        n_permutations=hsic_permutations,
    )

    joint_mmd = None
    conditional_hsic = None
    posterior_summary = None
    if model_type == "fcswae" and hasattr(model, "encode"):
        print("Computing mean-view JointMMD²_U...")
        joint_mmd = compute_joint_mmd(mu_c, mu_s, labels, n_classes, seed=seed)
        print("Computing classwise conditional HSIC calibration...")
        conditional_hsic = classwise_conditional_hsic_permutation_test(
            mu_c, mu_s, labels, n_classes, seed=seed,
            n_permutations=hsic_permutations,
        )
        if logvar_s is not None:
            posterior_summary = summarize_style_posterior(
                logvar_s,
                float(getattr(model, "style_sigma_floor", 0.0)),
            )

    style_stats = generation_style_stats
    if hasattr(model, "sample_from_class_prior"):
        if style_stats is None:
            style_stats = compute_class_style_stats(mu_s, labels, n_classes, device)
            generation_style_stats_source = "evaluation subset (legacy fallback)"

    external_generation = None
    if external_classifier is not None and style_stats is not None:
        print("Computing external pixel-space generation metrics...")
        external_generation = {
            "global_gaussian": compute_external_generation_metrics(
                model, external_classifier, n_classes, gen_per_class, device, "global_gaussian"
            ),
            "class_diag_t025": compute_external_generation_metrics(
                model, external_classifier, n_classes, gen_per_class, device,
                "class_diag_t025", style_stats,
            ),
        }

    internal_generation = None
    if aux_classifier is None and hasattr(model, "classifier"):
        aux_classifier = model.classifier
    if aux_classifier is not None and style_stats is not None:
        internal_generation = {
            "warning": "robustness-only internal re-encoding evaluator; not a primary Gen-ACC",
            "global_gaussian": compute_gen_self_accuracy(
                model, aux_classifier, n_classes, gen_per_class, device, "global_gaussian"
            ),
            "class_diag_t025": compute_gen_self_accuracy(
                model, aux_classifier, n_classes, gen_per_class, device,
                "class_diag_t025", style_stats,
            ),
        }

    results = {
        "protocol": {
            "version": AUDIT_PROTOCOL_VERSION,
            "n_evaluation_samples": int(labels.shape[0]),
            "latent_views": {
                "z_s_sample": "primary for q(z_s), q(z_s|y), and sampling-contract metrics",
                "mu_s": "primary for representation probes and deterministic interventions",
                "mu_c_mu_s": "posterior-mean pair used by JointMMD",
            },
            "mmd_estimator": "unbiased U-statistic; fixed multi-scale RBF",
            "global_mmd_reference_seed": global_mmd_reference_seed,
            "conditional_mmd_reference_seed": conditional_mmd_reference_seed,
            "probe_split": "stratified 60/20/20 train/validation/test",
            "factorized_adapter": views["adapter_metadata"],
        },
        "sampling_contract": {
            "global_mmd2_u_z_s_sample": global_mmd,
            "conditional_mmd2_u_z_s_sample": conditional_mmd,
            "delta_inter_z_s_sample": delta_sample,
            "hsic_z_s_sample": hsic_sample,
            "probe_suite_z_s_sample": probes_sample,
            "style_posterior": posterior_summary,
        },
        "representation": {
            "delta_inter_mu_s": delta_mean,
            "hsic_mu_s": hsic_mean,
            "probe_suite_mu_s": probes_mean,
        },
        "within_class_dependence": {
            "joint_mmd2_u_mu_c_mu_s": joint_mmd,
            "classwise_conditional_hsic_mu_c_mu_s": conditional_hsic,
        },
        "generation": {
            "external": external_generation,
            "internal_robustness_only": internal_generation,
            "style_stats_source": generation_style_stats_source,
            "style_stats_per_class_counts": None if style_stats is None else style_stats.get("counts"),
        },
        # Compatibility fields for legacy aggregators.  New tables should use
        # the explicitly named nested fields above.
        "global_mmd": global_mmd,
        "delta_inter": delta_sample,
        "lp_accuracy": probes_mean["models"]["logistic"]["test"]["accuracy"],
        "hsic": hsic_mean["statistic"],
        "joint_mmd": joint_mmd,
        "gen_self_acc_global_gaussian": None if internal_generation is None else internal_generation["global_gaussian"],
        "gen_self_acc_class_diag_t025": None if internal_generation is None else internal_generation["class_diag_t025"],
    }

    if verbose:
        print("\n" + "=" * 60)
        print("LEAKAGE DIAGNOSTIC RESULTS")
        print("=" * 60)
        print(f"  Global MMD²_U(z_s_sample)     : {global_mmd:.6f}")
        print(f"  Δ_inter(z_s_sample)           : {delta_sample:.4f}")
        print(f"  Δ_inter(mu_s)                 : {delta_mean:.4f}")
        print(f"  Logistic probe(mu_s) test ACC : {results['lp_accuracy']:.4f}")
        print(f"  HSIC(mu_s,y), permutation p   : {hsic_mean['statistic']:.6f}, p={hsic_mean['p_value']:.4f}")
        if results["joint_mmd"] is not None:
            print(f"  JointMMD²_U(mu_c,mu_s|y)      : {results['joint_mmd']:.6f}")
        if conditional_hsic is not None:
            print(
                "  Conditional HSIC, permutation p: "
                f"{conditional_hsic['statistic']:.6f}, p={conditional_hsic['p_value']:.4f}"
            )
        if external_generation is not None:
            score = external_generation["global_gaussian"]["macro_gen_accuracy"]
            print(f"  External macro Gen-ACC        : {score:.4f}")
        print("=" * 60)

    return results


# ---------------------------------------------------------------------------
# Shared loading helpers (reused by other diagnostic scripts — see
# scripts/latent_swap_diagnostics.py, conditional_mmd_diagnostics.py,
# wrong_style_rate_diagnostics.py, run_delta_sweep.py)
# ---------------------------------------------------------------------------

def load_fcswae_checkpoint(
    checkpoint_path,
    dataset: str,
    device,
    variant: str | None = None,
    style_sigma_floor: float | None = None,
):
    """Load a trained FCSWAE (or FCSWAEAblation) model from a checkpoint.

    variant: an ABLATION_VARIANTS key (e.g. "no_classifier", "gaussian_class_prior")
    if the checkpoint came from run_ablation_f_cs_wae.py -- its encoder/decoder
    structure (e.g. euclidean vs. spherical z_c, presence of a classifier head)
    differs from plain FCSWAE, so the state_dict keys would otherwise mismatch.
    """
    from src.config_f_cs_wae import f_cs_wae_config as cfg
    from src.utils.dataset_config import apply_dataset_config

    apply_dataset_config(cfg, dataset, backbone="resnet18")
    # Audit inputs are trusted local training artifacts.  PyTorch 2.6 changed
    # ``torch.load`` to default to ``weights_only=True``; crash-safe training
    # checkpoints also contain NumPy/DataLoader RNG state and therefore need
    # the full checkpoint loader.
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    if style_sigma_floor is None:
        run_config_path = Path(checkpoint_path).parent / "run_config.json"
        if run_config_path.exists():
            with run_config_path.open() as handle:
                run_metadata = json.load(handle)
            nested_config = run_metadata.get("config", {})
            style_sigma_floor = run_metadata.get(
                "style_sigma_floor",
                nested_config.get("style_sigma_floor", 0.0),
            )
        else:
            style_sigma_floor = 0.0
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
            style_sigma_floor=float(style_sigma_floor),
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
    p.add_argument("--n-samples", type=int, default=DEFAULT_EVAL_SAMPLES,
                   help=f"Number of test samples to use (Stage-0 default: {DEFAULT_EVAL_SAMPLES})")
    p.add_argument("--gen-per-class", type=int, default=50,
                   help="Generated images per class for self-ACC")
    p.add_argument("--generation-style-bank-samples", type=int, default=12000,
                   help="Training examples used to fit post-hoc class-conditional style summaries")
    p.add_argument("--probe-epochs", type=int, default=300,
                   help="Maximum epochs for fixed Torch probes")
    p.add_argument("--hsic-permutations", type=int, default=DEFAULT_HSIC_PERMUTATIONS,
                   help="Permutation count for calibrated multi-scale HSIC")
    p.add_argument("--external-classifier-checkpoint", default=None,
                   help="Stage-0 independent pixel classifier checkpoint. Required for primary Gen-ACC.")
    p.add_argument("--aux-classifier-checkpoint", default=None,
                   help="Legacy latent classifier checkpoint for robustness-only internal self-ACC")
    p.add_argument("--seed", type=int, default=0,
                   help="Seed for reproducible eval-subset sampling")
    p.add_argument("--out", default=None,
                   help="Result JSON path (default: runs_diag/stage0/<dataset>/<checkpoint>_seed<N>.json)")
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

    # Optional robustness-only latent classifier.  Primary generation metrics
    # require the independent pixel classifier loaded below.
    aux_clf = None
    if args.aux_classifier_checkpoint:
        print(f"Loading aux classifier from {args.aux_classifier_checkpoint}...")
        clf_ckpt = torch.load(args.aux_classifier_checkpoint, map_location=device)
        aux_clf = nn.Linear(model.semantic_dim, 10).to(device)
        aux_clf.load_state_dict(clf_ckpt)

    external_clf = None
    external_metadata = None
    if args.external_classifier_checkpoint:
        print(f"Loading external pixel classifier from {args.external_classifier_checkpoint}...")
        external_clf, external_metadata = load_external_classifier_checkpoint(
            args.external_classifier_checkpoint, args.dataset, device
        )
        print(f"  real-test accuracy: {external_metadata.get('real_test_accuracy')}")

    print(f"Loading {args.dataset} test subset (n={args.n_samples}, seed={args.seed})...")
    loader = get_eval_subset_loader(
        args.dataset, args.n_samples, batch_size=256, seed=args.seed, split="test",
    )

    generation_style_stats = None
    generation_style_stats_source = None
    if external_clf is not None:
        print(
            "Fitting post-hoc style summaries on the training split "
            f"(n={args.generation_style_bank_samples}, seed={args.seed})..."
        )
        style_loader = get_eval_subset_loader(
            args.dataset, args.generation_style_bank_samples,
            batch_size=256, seed=args.seed, split="train",
        )
        _style_mu_c, style_mu_s, style_labels = extract_full_encodings(
            model, style_loader, device
        )
        generation_style_stats = compute_class_style_stats(
            style_mu_s.to(device), style_labels.to(device), model.n_classes, device
        )
        generation_style_stats_source = (
            f"training split subset, n={len(style_loader.dataset)}, seed={args.seed}"
        )

    results = run_diagnostics(
        model, loader, device,
        n_classes=10,
        model_type=args.model_type,
        gen_per_class=args.gen_per_class,
        aux_classifier=aux_clf,
        external_classifier=external_clf,
        generation_style_stats=generation_style_stats,
        generation_style_stats_source=generation_style_stats_source,
        seed=args.seed,
        probe_epochs=args.probe_epochs,
        hsic_permutations=args.hsic_permutations,
        verbose=True,
    )
    if external_metadata is not None:
        results["generation"]["external_classifier_metadata"] = external_metadata

    checkpoint_path = Path(args.checkpoint)
    run_config_path = checkpoint_path.parent / "run_config.json"
    if run_config_path.exists():
        with run_config_path.open() as handle:
            model_config = json.load(handle)
    else:
        model_config = {
            "semantic_dim": getattr(model, "semantic_dim", None),
            "style_dim": getattr(model, "style_dim", None),
            "n_classes": getattr(model, "n_classes", None),
            "variant": args.variant,
            "run_config_missing": True,
        }
    subset_indices = getattr(loader.dataset, "indices", [])
    subset_digest = None
    if subset_indices:
        packed = torch.tensor(subset_indices, dtype=torch.int64).numpy().tobytes()
        import hashlib
        subset_digest = hashlib.sha256(packed).hexdigest()
    evaluation_config = {
        "n_samples": args.n_samples,
        "actual_n_samples": len(loader.dataset),
        "eval_subset_seed": args.seed,
        "eval_subset_indices_sha256": subset_digest,
        "probe_epochs": args.probe_epochs,
        "probe_models": ["logistic", "mlp", "rbf_svm", "knn"],
        "probe_split": [0.60, 0.20, 0.20],
        "hsic_max_samples": DEFAULT_HSIC_SAMPLES,
        "hsic_permutations": args.hsic_permutations,
        "gen_per_class": args.gen_per_class,
        "generation_style_bank_samples": args.generation_style_bank_samples,
        "generation_style_bank_split": "train" if external_clf is not None else None,
        "latent_primary_sampling_contract": "z_s_sample",
        "latent_primary_probe_swap": "mu_s",
        "global_mmd_reference_seed": args.seed + 10_000,
        "conditional_mmd_reference_seed": args.seed + 11_000,
    }
    manifest = build_manifest(
        repository_root=ROOT,
        checkpoint_path=args.checkpoint,
        dataset=args.dataset,
        seed=args.seed,
        evaluation_config=evaluation_config,
        model_config=model_config,
        external_classifier_path=args.external_classifier_checkpoint,
    )
    out_path = Path(args.out) if args.out else (
        ROOT / "runs_diag" / "stage0" / args.dataset /
        f"{checkpoint_path.stem}_seed{args.seed}.json"
    )
    digest = save_result_with_manifest(out_path, results, manifest)
    print(f"\nResults saved to {out_path}")
    print(f"Result SHA-256: {digest}")


if __name__ == "__main__":
    main()

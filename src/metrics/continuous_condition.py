"""Audits for continuously conditioned factorized samplers.

The primary test uses paired conditional randomization.  For each observed
condition ``y_i`` it receives one posterior draw and one independent sampler
draw.  Under conditional equality those two draws are exchangeable, so the
null swaps their bank assignments independently within each pair.  This keeps
the continuous conditions fixed and avoids discretizing them for inference.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F

from .audit_protocol import EUCLIDEAN_SIGMAS, multiscale_rbf_kernel


CONTINUOUS_AUDIT_VERSION = "continuous-condition-audit-1.0.0"
CONDITION_SIGMAS = (0.25, 0.5, 1.0, 2.0)


def _center(kernel: torch.Tensor) -> torch.Tensor:
    return kernel - kernel.mean(0, keepdim=True) - kernel.mean(1, keepdim=True) + kernel.mean()


def _standardize(values: torch.Tensor) -> torch.Tensor:
    if values.ndim == 1:
        values = values[:, None]
    return (values - values.mean(0, keepdim=True)) / values.std(
        0, unbiased=False, keepdim=True
    ).clamp_min(1e-6)


def _spherical_kernel(values: torch.Tensor) -> torch.Tensor:
    values = F.normalize(values, p=2, dim=1)
    chordal_sq = (2.0 - 2.0 * (values @ values.T)).clamp_min(0.0)
    kernel = torch.zeros_like(chordal_sq)
    for sigma in (0.1, 0.3, 0.5, 1.0, 2.0):
        kernel.add_(torch.exp(-chordal_sq / (2.0 * sigma**2 + 1e-12)))
    return kernel / 5.0


def _condition_kernel(
    conditions: torch.Tensor,
    *,
    kind: str,
    sigmas: Sequence[float] = CONDITION_SIGMAS,
) -> torch.Tensor:
    if kind == "continuous":
        values = _standardize(conditions.float())
        return multiscale_rbf_kernel(values, values, sigmas)
    if kind == "categorical":
        values = conditions.flatten()
        return (values[:, None] == values[None, :]).float()
    raise ValueError("condition kind must be 'continuous' or 'categorical'")


def _mmd_from_signs(kernel: torch.Tensor, signs: torch.Tensor, n: int) -> torch.Tensor:
    kernel = kernel.clone()
    kernel.fill_diagonal_(0.0)
    total = kernel.sum()
    quadratic = (signs * (kernel @ signs)).sum(0)
    within_ordered = (total + quadratic) / 2.0
    cross_ordered = (total - quadratic) / 2.0
    return within_ordered / float(n * (n - 1)) - cross_ordered / float(n * n)


def _paired_permutation_summary(
    kernel: torch.Tensor,
    *,
    n: int,
    seed: int,
    n_permutations: int,
) -> dict:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    assignments = [torch.ones(n, dtype=kernel.dtype)]
    for _ in range(n_permutations):
        swap = torch.randint(0, 2, (n,), generator=generator)
        assignments.append(torch.where(swap == 0, 1.0, -1.0).to(kernel.dtype))
    first_bank = torch.stack(assignments, dim=1).to(kernel.device)
    signs = torch.cat([first_bank, -first_bank], dim=0)
    values = _mmd_from_signs(kernel, signs, n)
    observed = float(values[0].item())
    null = values[1:].detach().cpu().double().numpy()
    exceedances = int(np.count_nonzero(null >= observed))
    p_value = float((exceedances + 1) / (n_permutations + 1))
    return {
        "statistic": observed,
        "p_value": p_value,
        "reject_0.05": bool(p_value <= 0.05),
        "null_mean": float(null.mean()),
        "null_std": float(null.std(ddof=1)) if null.size > 1 else None,
        "null_quantiles": {
            "q50": float(np.quantile(null, 0.50)),
            "q95": float(np.quantile(null, 0.95)),
            "q99": float(np.quantile(null, 0.99)),
        },
        "n_pairs": int(n),
        "n_permutations": int(n_permutations),
    }


def paired_conditional_mmd_permutation_test(
    posterior: torch.Tensor,
    sampler: torch.Tensor,
    conditions: torch.Tensor,
    *,
    seed: int = 0,
    n_permutations: int = 199,
    condition_kind: str = "continuous",
    geometry: str = "euclidean",
    condition_sigmas: Sequence[float] = CONDITION_SIGMAS,
    feature_sigmas: Sequence[float] = EUCLIDEAN_SIGMAS,
) -> dict:
    """Test integrated conditional equality with within-condition pair swaps.

    ``posterior[i]`` and ``sampler[i]`` must be independent draws associated
    with the same ``conditions[i]``.  Product kernels on condition and latent
    coordinates retain locality in continuous ``y`` while paired swaps give a
    finite-sample randomization null under conditional exchangeability.
    """

    if posterior.ndim != 2 or sampler.ndim != 2 or posterior.shape != sampler.shape:
        raise ValueError("posterior and sampler must have the same rank-2 shape")
    n = posterior.shape[0]
    if n < 4 or conditions.shape[0] != n:
        raise ValueError("paired conditional MMD requires at least four aligned pairs")
    if n_permutations < 1:
        raise ValueError("n_permutations must be positive")

    pooled = torch.cat([posterior, sampler], dim=0)
    pooled_conditions = torch.cat([conditions, conditions], dim=0).to(pooled.device)
    condition_kernel = _condition_kernel(
        pooled_conditions,
        kind=condition_kind,
        sigmas=condition_sigmas,
    ).to(device=pooled.device, dtype=pooled.dtype)
    if geometry == "spherical":
        feature_kernel = _spherical_kernel(pooled)
    elif geometry == "euclidean":
        standardized = _standardize(pooled)
        feature_kernel = multiscale_rbf_kernel(
            standardized, standardized, feature_sigmas
        )
    else:
        raise ValueError("geometry must be 'spherical' or 'euclidean'")
    kernel = condition_kernel * feature_kernel

    result = _paired_permutation_summary(
        kernel, n=n, seed=seed, n_permutations=n_permutations
    )
    result.update({
        "condition_kind": condition_kind,
        "geometry": geometry,
        "condition_sigmas": [float(value) for value in condition_sigmas],
        "feature_sigmas": [float(value) for value in feature_sigmas],
        "estimator": (
            "condition-latent product-kernel MMD^2_U with paired within-condition swaps"
        ),
        "assumption": (
            "Each posterior/sampler pair is independent conditional on its shared condition."
        ),
    })
    return result


def paired_factorized_joint_mmd_permutation_test(
    posterior_content: torch.Tensor,
    posterior_style: torch.Tensor,
    sampler_content: torch.Tensor,
    sampler_style: torch.Tensor,
    conditions: torch.Tensor,
    *,
    seed: int = 0,
    n_permutations: int = 199,
    condition_kind: str = "continuous",
    condition_sigmas: Sequence[float] = CONDITION_SIGMAS,
    style_sigmas: Sequence[float] = EUCLIDEAN_SIGMAS,
) -> dict:
    """Direct mixed-geometry test of ``Q(z_c,z_s|y) = P(z_c,z_s|y)``."""

    if posterior_content.shape != sampler_content.shape:
        raise ValueError("posterior/sampler content banks must match")
    if posterior_style.shape != sampler_style.shape:
        raise ValueError("posterior/sampler style banks must match")
    n = posterior_content.shape[0]
    if posterior_style.shape[0] != n or conditions.shape[0] != n or n < 4:
        raise ValueError("joint conditional MMD requires at least four aligned pairs")
    if n_permutations < 1:
        raise ValueError("n_permutations must be positive")
    pooled_conditions = torch.cat([conditions, conditions], dim=0).to(
        posterior_content.device
    )
    pooled_content = torch.cat([posterior_content, sampler_content], dim=0)
    pooled_style = _standardize(torch.cat([posterior_style, sampler_style], dim=0))
    condition_kernel = _condition_kernel(
        pooled_conditions, kind=condition_kind, sigmas=condition_sigmas
    ).to(device=pooled_content.device, dtype=pooled_content.dtype)
    content_kernel = _spherical_kernel(pooled_content)
    style_kernel = multiscale_rbf_kernel(pooled_style, pooled_style, style_sigmas)
    result = _paired_permutation_summary(
        condition_kernel * content_kernel * style_kernel,
        n=n,
        seed=seed,
        n_permutations=n_permutations,
    )
    result.update(
        {
            "condition_kind": condition_kind,
            "geometry": "product(spherical-content, Euclidean-style)",
            "condition_sigmas": [float(value) for value in condition_sigmas],
            "style_sigmas": [float(value) for value in style_sigmas],
            "estimator": (
                "condition/content/style product-kernel MMD^2_U with paired "
                "within-condition swaps"
            ),
            "assumption": (
                "Each posterior/sampler pair is independent conditional on its shared condition."
            ),
        }
    )
    return result


def continuous_hsic_permutation_test(
    features: torch.Tensor,
    conditions: torch.Tensor,
    *,
    seed: int = 0,
    n_permutations: int = 199,
    condition_sigmas: Sequence[float] = CONDITION_SIGMAS,
    feature_sigmas: Sequence[float] = EUCLIDEAN_SIGMAS,
    permutation_batch_size: int = 32,
) -> dict:
    """Permutation-calibrated HSIC between a representation and real-valued y."""

    if features.ndim != 2 or features.shape[0] != conditions.shape[0]:
        raise ValueError("features and conditions must have matching leading dimensions")
    if features.shape[0] < 4 or n_permutations < 1:
        raise ValueError("continuous HSIC requires four samples and positive permutations")
    if permutation_batch_size < 1:
        raise ValueError("permutation_batch_size must be positive")
    features = _standardize(features)
    conditions = conditions.to(features.device)
    k = _center(multiscale_rbf_kernel(features, features, feature_sigmas))
    l = _center(
        _condition_kernel(conditions, kind="continuous", sigmas=condition_sigmas)
    ).to(device=features.device, dtype=features.dtype)
    denominator = float((features.shape[0] - 1) ** 2)
    observed = float(((k * l).sum() / denominator).item())
    generator = torch.Generator(device="cpu").manual_seed(seed)
    permutations = [
        torch.randperm(features.shape[0], generator=generator)
        for _ in range(n_permutations)
    ]
    null = []
    # Batched advanced indexing is algebraically identical to materializing
    # one P L P^T matrix at a time, but substantially reduces Python overhead.
    for start in range(0, n_permutations, permutation_batch_size):
        indices = torch.stack(
            permutations[start : start + permutation_batch_size], dim=0
        ).to(l.device)
        permuted = l[indices[:, :, None], indices[:, None, :]]
        statistics = (permuted * k.unsqueeze(0)).sum(dim=(1, 2)) / denominator
        null.extend(statistics.detach().cpu().double().tolist())
    null_array = np.asarray(null, dtype=np.float64)
    exceedances = int(np.count_nonzero(null_array >= observed))
    return {
        "statistic": observed,
        "p_value": float((exceedances + 1) / (n_permutations + 1)),
        "reject_0.05": bool((exceedances + 1) / (n_permutations + 1) <= 0.05),
        "null_mean": float(null_array.mean()),
        "null_std": float(null_array.std(ddof=1)) if null_array.size > 1 else None,
        "n_samples": int(features.shape[0]),
        "n_permutations": int(n_permutations),
        "permutation_batch_size": int(permutation_batch_size),
        "condition_sigmas": [float(value) for value in condition_sigmas],
        "feature_sigmas": [float(value) for value in feature_sigmas],
        "estimator": "biased centered RBF/RBF HSIC with global condition permutation",
    }


__all__ = [
    "CONDITION_SIGMAS",
    "CONTINUOUS_AUDIT_VERSION",
    "continuous_hsic_permutation_test",
    "paired_factorized_joint_mmd_permutation_test",
    "paired_conditional_mmd_permutation_test",
]

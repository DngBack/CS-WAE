"""
Loss functions for F-CS-WAE.

Total objective:
    L = L_rec
      + alpha(t) * L_class
      + beta(t)  * L_agg
      + gamma(t) * L_style
      + eta(t)   * L_cls
      + lambda_var * L_var          (subtracted — higher var is better)

where:
    L_rec   = l1_weight * L1(x, x_hat) + lpips_weight * LPIPS(x, x_hat)
    L_class = (1/K) sum_k MMD(z_c[y==k],  SpherCauchy(m_k, rho_p))   [on sphere]
    L_agg   = MMD(z_c_all,  z_p_mix)  with z_p_mix ~ mixture of class priors
    L_style = MMD(z_s, N(0,I))  [Euclidean RBF kernel]
    L_cls   = CE(classifier(mu_c), y)
    L_var   = -Var(z_s).mean()  (so minimising total loss → maximising var)
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .utils import mmd_loss, mobius_reparam, sample_uniform_sphere, to_rgb_for_lpips
from ..config_f_cs_wae import f_cs_wae_config as cfg


SPHERICAL_JOINT_SIGMAS = (0.1, 0.3, 0.5, 1.0, 2.0)
EUCLIDEAN_JOINT_SIGMAS = (0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0)


# ---------------------------------------------------------------------------
# Euclidean MMD (for style latent vs N(0,I))
# ---------------------------------------------------------------------------

def _rbf_euclidean(x: torch.Tensor, y: torch.Tensor, sigma: float) -> torch.Tensor:
    """RBF kernel using squared Euclidean distances."""
    dist_sq = torch.cdist(x, y, p=2).pow(2)
    return torch.exp(-dist_sq / (2.0 * sigma ** 2 + cfg.epsilon))


def mmd_euclidean(q: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    """MMD² with multi-scale RBF bandwidths between two Euclidean sample sets.

    Uses fixed bandwidths spanning several orders of magnitude so the kernel
    remains sensitive even when q has large variance (median-heuristic would
    saturate in that regime and return ~0 regardless of distribution mismatch).
    Bandwidth values are chosen to cover N(0,I) typical distances for common
    style_dim values (64–256).
    """
    if q.shape[0] < 2 or p.shape[0] < 2:
        return torch.tensor(0.0, device=q.device)
    _SIGMAS = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]
    mmd_val = torch.tensor(0.0, device=q.device)
    for s in _SIGMAS:
        k_qq = _rbf_euclidean(q, q, s).mean()
        k_pp = _rbf_euclidean(p, p, s).mean()
        k_qp = _rbf_euclidean(q, p, s).mean()
        mmd_val = mmd_val + (k_qq + k_pp - 2.0 * k_qp)
    return mmd_val / len(_SIGMAS)


def _multiscale_rbf(
    x: torch.Tensor,
    y: torch.Tensor,
    sigmas: tuple[float, ...],
) -> torch.Tensor:
    """Differentiable multiscale Euclidean RBF kernel matrix."""

    dist_sq = torch.cdist(x, y, p=2).square()
    kernel = torch.zeros_like(dist_sq)
    for sigma in sigmas:
        kernel = kernel + torch.exp(
            -dist_sq / (2.0 * float(sigma) ** 2 + cfg.epsilon)
        )
    return kernel / float(len(sigmas))


def joint_product_mmd(
    q_content: torch.Tensor,
    q_style: torch.Tensor,
    p_content: torch.Tensor,
    p_style: torch.Tensor,
) -> torch.Tensor:
    """Biased MMD² on spherical-content x Euclidean-style product space.

    Content and style retain separate bandwidth ladders, matching the held-out
    complete-contract audit rather than concatenating incompatible geometries.
    The biased V-statistic is intentional for stable minibatch optimization.
    """

    tensors = (q_content, q_style, p_content, p_style)
    if any(tensor.ndim != 2 for tensor in tensors):
        raise ValueError("joint product MMD inputs must be rank-2 tensors")
    if q_content.shape[0] != q_style.shape[0]:
        raise ValueError("posterior content/style banks must have equal size")
    if p_content.shape[0] != p_style.shape[0]:
        raise ValueError("prior content/style banks must have equal size")
    if q_content.shape[1] != p_content.shape[1]:
        raise ValueError("posterior/prior content dimensions must match")
    if q_style.shape[1] != p_style.shape[1]:
        raise ValueError("posterior/prior style dimensions must match")
    if q_content.shape[0] < 2 or p_content.shape[0] < 2:
        return q_content.new_zeros(())

    q_content = F.normalize(q_content, p=2, dim=1)
    p_content = F.normalize(p_content, p=2, dim=1)
    k_qq = _multiscale_rbf(
        q_content, q_content, SPHERICAL_JOINT_SIGMAS
    ) * _multiscale_rbf(q_style, q_style, EUCLIDEAN_JOINT_SIGMAS)
    k_pp = _multiscale_rbf(
        p_content, p_content, SPHERICAL_JOINT_SIGMAS
    ) * _multiscale_rbf(p_style, p_style, EUCLIDEAN_JOINT_SIGMAS)
    k_qp = _multiscale_rbf(
        q_content, p_content, SPHERICAL_JOINT_SIGMAS
    ) * _multiscale_rbf(q_style, p_style, EUCLIDEAN_JOINT_SIGMAS)
    return k_qq.mean() + k_pp.mean() - 2.0 * k_qp.mean()


def _sample_class_prior_for_loss(model, class_idx: int, n: int, device: torch.device) -> torch.Tensor:
    if hasattr(model, "_sample_prior_z_c"):
        return model._sample_prior_z_c(class_idx, n, device)

    normalized_centers = F.normalize(model.ema_centers, p=2, dim=-1)
    if model.n_centers == 1:
        center = normalized_centers[class_idx, 0].unsqueeze(0).expand(n, -1)
    else:
        r_idx = torch.randint(0, model.n_centers, (n,), device=device)
        center = normalized_centers[class_idx][r_idx]
    rho = torch.full((n,), model.rho_p, device=device)
    eps = sample_uniform_sphere(n, model.semantic_dim, device=device)
    return mobius_reparam(eps, center, rho)


def _sample_prior_mixture_for_loss(model, n: int, device: torch.device) -> torch.Tensor:
    if hasattr(model, "_sample_prior_z_c"):
        rand_k = torch.randint(0, model.n_classes, (n,), device=device)
        return torch.cat(
            [model._sample_prior_z_c(k.item(), 1, device) for k in rand_k],
            dim=0,
        )

    normalized_centers = F.normalize(model.ema_centers, p=2, dim=-1)
    rand_k = torch.randint(0, model.n_classes, (n,), device=device)
    if model.n_centers == 1:
        center_mix = normalized_centers[rand_k, 0]
    else:
        rand_r = torch.randint(0, model.n_centers, (n,), device=device)
        center_mix = normalized_centers[rand_k, rand_r]
    rho_mix = torch.full((n,), model.rho_p, device=device)
    eps_mix = sample_uniform_sphere(n, model.semantic_dim, device=device)
    return mobius_reparam(eps_mix, center_mix, rho_mix)


def joint_contract_mmd_loss(
    z_c: torch.Tensor,
    z_s: torch.Tensor,
    y: torch.Tensor,
    model,
) -> torch.Tensor:
    """Compare q(z_c,z_s|y) with independent p(z_c|y)p(z_s) draws.

    Classes represented by fewer than two minibatch examples are omitted.
    Fresh content and style prior banks are drawn independently for every
    represented class, while gradients flow only through posterior codes.
    """

    if z_c.ndim != 2 or z_s.ndim != 2 or y.ndim != 1:
        raise ValueError("expected rank-2 latents and rank-1 labels")
    if not (z_c.shape[0] == z_s.shape[0] == y.shape[0]):
        raise ValueError("latent and label batch sizes must match")

    loss = z_c.new_zeros(())
    represented_classes = 0
    for class_idx in range(model.n_classes):
        mask = y == class_idx
        n_class = int(mask.sum().item())
        if n_class < 2:
            continue
        prior_content = _sample_class_prior_for_loss(
            model, class_idx, n_class, z_c.device
        ).detach()
        prior_style = torch.randn(
            n_class,
            z_s.shape[1],
            device=z_s.device,
            dtype=z_s.dtype,
        )
        loss = loss + joint_product_mmd(
            z_c[mask], z_s[mask], prior_content, prior_style
        )
        represented_classes += 1

    if represented_classes == 0:
        return loss
    return loss / float(represented_classes)


# ---------------------------------------------------------------------------
# FACT: clause-aligned, null-calibrated contract constraints
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FactConstraintValues:
    """Differentiable FACT violations plus raw and matched-null diagnostics."""

    style: torch.Tensor
    content: torch.Tensor
    dependence: torch.Tensor
    label_dependence: torch.Tensor
    mean_dependence: torch.Tensor
    style_raw: torch.Tensor
    content_raw: torch.Tensor
    dependence_raw: torch.Tensor
    label_dependence_raw: torch.Tensor
    mean_dependence_raw: torch.Tensor
    style_null: torch.Tensor
    content_null: torch.Tensor
    dependence_null: torch.Tensor
    label_dependence_null: torch.Tensor
    mean_dependence_null: torch.Tensor
    represented_classes: int
    mean_represented_classes: int


def mmd_spherical_multiscale(q: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    """Biased fixed-bandwidth MMD on the unit sphere, aligned with the audit."""

    if q.ndim != 2 or p.ndim != 2:
        raise ValueError("spherical MMD inputs must be rank-2 tensors")
    if q.shape[1] != p.shape[1]:
        raise ValueError("spherical MMD inputs must have equal feature dimensions")
    if q.shape[0] < 2 or p.shape[0] < 2:
        return q.new_zeros(())
    q = F.normalize(q, p=2, dim=1)
    p = F.normalize(p, p=2, dim=1)
    k_qq = _multiscale_rbf(q, q, SPHERICAL_JOINT_SIGMAS)
    k_pp = _multiscale_rbf(p, p, SPHERICAL_JOINT_SIGMAS)
    k_qp = _multiscale_rbf(q, p, SPHERICAL_JOINT_SIGMAS)
    return k_qq.mean() + k_pp.mean() - 2.0 * k_qp.mean()


def _center_kernel(kernel: torch.Tensor) -> torch.Tensor:
    return (
        kernel
        - kernel.mean(dim=0, keepdim=True)
        - kernel.mean(dim=1, keepdim=True)
        + kernel.mean()
    )


def conditional_product_hsic(
    content: torch.Tensor,
    style: torch.Tensor,
) -> torch.Tensor:
    """Biased HSIC for one condition using geometry-specific kernels."""

    if content.ndim != 2 or style.ndim != 2:
        raise ValueError("HSIC inputs must be rank-2 tensors")
    if content.shape[0] != style.shape[0]:
        raise ValueError("HSIC inputs must have equal sample counts")
    if content.shape[0] < 2:
        return content.new_zeros(())
    content = F.normalize(content, p=2, dim=1)
    content_kernel = _center_kernel(
        _multiscale_rbf(content, content, SPHERICAL_JOINT_SIGMAS)
    )
    style_kernel = _center_kernel(
        _multiscale_rbf(style, style, EUCLIDEAN_JOINT_SIGMAS)
    )
    value = (content_kernel * style_kernel).mean()
    return value.clamp_min(0.0)


def _style_label_hsic_from_kernel(
    centered_style_kernel: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    label_kernel = (labels[:, None] == labels[None, :]).to(
        centered_style_kernel.dtype
    )
    label_kernel = _center_kernel(label_kernel)
    return (centered_style_kernel * label_kernel).mean().clamp_min(0.0)


def style_label_hsic(style: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Biased RBF/delta-kernel HSIC for sampled style and a condition."""

    if style.ndim != 2 or labels.ndim != 1:
        raise ValueError("style must be rank 2 and labels rank 1")
    if style.shape[0] != labels.shape[0]:
        raise ValueError("style and labels must have equal sample counts")
    if style.shape[0] < 2:
        return style.new_zeros(())
    style_kernel = _center_kernel(
        _multiscale_rbf(style, style, EUCLIDEAN_JOINT_SIGMAS)
    )
    return _style_label_hsic_from_kernel(style_kernel, labels)


def _mean_constraint_terms(
    terms: list[tuple[torch.Tensor, torch.Tensor]],
    reference: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Average positive excess, raw statistic, and detached null floor."""

    if not terms:
        zero = reference.new_zeros(())
        return zero, zero, zero
    raw = torch.stack([observed for observed, _null in terms]).mean()
    null = torch.stack([null for _observed, null in terms]).mean().detach()
    excess = torch.stack(
        [F.relu(observed - null.detach()) for observed, null in terms]
    ).mean()
    return excess, raw, null


def audit_aligned_mean_hsic(
    mean_content: torch.Tensor,
    mean_style: torch.Tensor,
) -> torch.Tensor:
    """Differentiable per-class posterior-mean HSIC used by the audit.

    This mirrors ``classwise_conditional_hsic_permutation_test``: spherical
    normalization for content, per-feature standardization for style, the
    fixed audit bandwidth ladders, centered kernels, and ``(n-1)^2`` scaling.
    """

    if mean_content.ndim != 2 or mean_style.ndim != 2:
        raise ValueError("mean HSIC inputs must be rank-2 tensors")
    if mean_content.shape[0] != mean_style.shape[0]:
        raise ValueError("mean HSIC inputs must have equal sample counts")
    n = mean_content.shape[0]
    if n < 4:
        return mean_style.new_zeros(())
    content = F.normalize(mean_content, p=2, dim=1)
    style_std = mean_style.std(dim=0, keepdim=True).clamp_min(1e-6)
    style = (mean_style - mean_style.mean(dim=0, keepdim=True)) / style_std
    content_kernel = _center_kernel(
        _multiscale_rbf(content, content, SPHERICAL_JOINT_SIGMAS)
    )
    style_kernel = _center_kernel(
        _multiscale_rbf(style, style, EUCLIDEAN_JOINT_SIGMAS)
    )
    value = (content_kernel * style_kernel).sum() / float((n - 1) ** 2)
    return value.clamp_min(0.0)


def mean_conditional_hsic_constraint(
    mean_content: torch.Tensor,
    mean_style: torch.Tensor,
    labels: torch.Tensor,
    n_classes: int,
    *,
    null_draws: int = 1,
    style_only: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Null-calibrated classwise posterior-mean HSIC constraint."""

    if null_draws < 1:
        raise ValueError("null_draws must be positive")
    if mean_content.ndim != 2 or mean_style.ndim != 2 or labels.ndim != 1:
        raise ValueError("expected rank-2 means and rank-1 labels")
    if not (mean_content.shape[0] == mean_style.shape[0] == labels.shape[0]):
        raise ValueError("posterior means and labels must have equal sizes")
    terms: list[tuple[torch.Tensor, torch.Tensor]] = []
    for class_idx in range(n_classes):
        mask = labels == class_idx
        n_class = int(mask.sum().item())
        if n_class < 4:
            continue
        content = mean_content[mask]
        if style_only:
            content = content.detach()
        style = mean_style[mask]
        observed = audit_aligned_mean_hsic(content, style)
        null_values = []
        for _ in range(null_draws):
            permutation = torch.randperm(n_class, device=mean_style.device)
            null_values.append(
                audit_aligned_mean_hsic(
                    content.detach(), style.detach()[permutation]
                )
            )
        terms.append((observed, torch.stack(null_values).mean()))
    excess, raw, null = _mean_constraint_terms(terms, mean_style)
    return excess, raw, null, len(terms)


def fact_contract_constraints(
    z_c: torch.Tensor,
    z_s: torch.Tensor,
    y: torch.Tensor,
    model,
    *,
    style_only_dependence: bool = True,
    null_draws: int = 1,
    label_dependence: bool = False,
    mean_content: torch.Tensor | None = None,
    mean_style: torch.Tensor | None = None,
    mean_dependence: bool = False,
) -> FactConstraintValues:
    """Compute the three FACT constraints on represented minibatch classes.

    Each observed statistic is compared with a same-size matched-null statistic:
    prior-vs-prior for content/style and a within-class style permutation for
    dependence.  The null is detached and only positive excess is optimized.
    With ``style_only_dependence=True``, conditional HSIC cannot move content.
    """

    if null_draws < 1:
        raise ValueError("null_draws must be positive")
    if z_c.ndim != 2 or z_s.ndim != 2 or y.ndim != 1:
        raise ValueError("expected rank-2 latents and rank-1 labels")
    if not (z_c.shape[0] == z_s.shape[0] == y.shape[0]):
        raise ValueError("latent and label batch sizes must match")
    if mean_dependence and (mean_content is None or mean_style is None):
        raise ValueError("mean dependence requires mean_content and mean_style")

    style_terms: list[tuple[torch.Tensor, torch.Tensor]] = []
    content_terms: list[tuple[torch.Tensor, torch.Tensor]] = []
    dependence_terms: list[tuple[torch.Tensor, torch.Tensor]] = []

    for class_idx in range(model.n_classes):
        mask = y == class_idx
        n_class = int(mask.sum().item())
        if n_class < 2:
            continue

        posterior_content = z_c[mask]
        posterior_style = z_s[mask]

        content_prior = _sample_class_prior_for_loss(
            model, class_idx, n_class, z_c.device
        ).detach()
        content_null_values = []
        for _ in range(null_draws):
            content_null = _sample_class_prior_for_loss(
                model, class_idx, n_class, z_c.device
            ).detach()
            content_null_values.append(
                mmd_spherical_multiscale(content_null, content_prior)
            )
        content_terms.append(
            (
                mmd_spherical_multiscale(posterior_content, content_prior),
                torch.stack(content_null_values).mean(),
            )
        )

        style_prior = torch.randn_like(posterior_style)
        style_null_values = [
            mmd_euclidean(torch.randn_like(posterior_style), style_prior)
            for _ in range(null_draws)
        ]
        style_terms.append(
            (
                mmd_euclidean(posterior_style, style_prior),
                torch.stack(style_null_values).mean(),
            )
        )

        hsic_content = (
            posterior_content.detach()
            if style_only_dependence
            else posterior_content
        )
        observed_hsic = conditional_product_hsic(hsic_content, posterior_style)
        null_hsic_values = []
        for _ in range(null_draws):
            permutation = torch.randperm(n_class, device=z_s.device)
            null_hsic_values.append(
                conditional_product_hsic(
                    hsic_content.detach(), posterior_style.detach()[permutation]
                )
            )
        null_hsic = torch.stack(null_hsic_values).mean()
        dependence_terms.append((observed_hsic, null_hsic))

    style, style_raw, style_null = _mean_constraint_terms(style_terms, z_s)
    content, content_raw, content_null = _mean_constraint_terms(content_terms, z_c)
    dependence, dependence_raw, dependence_null = _mean_constraint_terms(
        dependence_terms, z_s
    )
    if label_dependence:
        label_style_kernel = _center_kernel(
            _multiscale_rbf(z_s, z_s, EUCLIDEAN_JOINT_SIGMAS)
        )
        label_dependence_raw = _style_label_hsic_from_kernel(
            label_style_kernel, y
        )
        label_null_values = [
            _style_label_hsic_from_kernel(
                label_style_kernel.detach(),
                y[torch.randperm(y.shape[0], device=y.device)],
            )
            for _ in range(null_draws)
        ]
        label_dependence_null = torch.stack(label_null_values).mean().detach()
        label_dependence_value = F.relu(
            label_dependence_raw - label_dependence_null
        )
    else:
        label_dependence_value = z_s.new_zeros(())
        label_dependence_raw = z_s.new_zeros(())
        label_dependence_null = z_s.new_zeros(())
    if mean_dependence:
        (
            mean_dependence_value,
            mean_dependence_raw,
            mean_dependence_null,
            mean_represented_classes,
        ) = mean_conditional_hsic_constraint(
            mean_content,
            mean_style,
            y,
            model.n_classes,
            null_draws=null_draws,
            style_only=style_only_dependence,
        )
    else:
        mean_dependence_value = z_s.new_zeros(())
        mean_dependence_raw = z_s.new_zeros(())
        mean_dependence_null = z_s.new_zeros(())
        mean_represented_classes = 0
    return FactConstraintValues(
        style=style,
        content=content,
        dependence=dependence,
        label_dependence=label_dependence_value,
        mean_dependence=mean_dependence_value,
        style_raw=style_raw,
        content_raw=content_raw,
        dependence_raw=dependence_raw,
        label_dependence_raw=label_dependence_raw,
        mean_dependence_raw=mean_dependence_raw,
        style_null=style_null,
        content_null=content_null,
        dependence_null=dependence_null,
        label_dependence_null=label_dependence_null,
        mean_dependence_null=mean_dependence_null,
        represented_classes=len(style_terms),
        mean_represented_classes=mean_represented_classes,
    )


# ---------------------------------------------------------------------------
# Main loss function
# ---------------------------------------------------------------------------

def mmd_per_class_style(
    z_s: torch.Tensor,
    y: torch.Tensor,
    n_classes: int,
) -> torch.Tensor:
    """Per-class style MMD: enforces q(z_s | y=k) ≈ N(0,I) for each class k.

    This directly targets conditional style leakage that global aggregate MMD
    cannot prevent (Proposition 1 in paper).
    """
    L = torch.tensor(0.0, device=z_s.device)
    count = 0
    for k in range(n_classes):
        mask = y == k
        if mask.sum() < 2:
            continue
        z_s_k = z_s[mask]
        z_p_k = torch.randn_like(z_s_k)
        L = L + mmd_euclidean(z_s_k, z_p_k)
        count += 1
    return L / count if count > 0 else L


def calculate_f_cs_wae_loss(
    x: torch.Tensor,
    y: torch.Tensor,
    x_hat: torch.Tensor,
    z_c: torch.Tensor,
    z_s: torch.Tensor,
    mu_c: torch.Tensor,
    mu_s: torch.Tensor,
    logvar_s: torch.Tensor,
    model,
    loss_fn_vgg,
    alpha: float,
    beta: float,
    gamma: float,
    delta: float,
    eta: float,
    lambda_var: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
           torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute all F-CS-WAE loss terms.

    Parameters
    ----------
    x, y        : input images and class labels
    x_hat       : reconstructed images
    z_c, z_s    : sampled semantic and style latents
    mu_c        : L2-normalised semantic direction (used for L_cls, L_class)
    mu_s, logvar_s : style distribution parameters
    model       : FCSWAE instance (needs .ema_centers, .n_classes, .rho_p, .classifier)
    loss_fn_vgg : lpips.LPIPS instance
    alpha, beta, gamma, delta, eta, lambda_var : current loss weights
        delta = per-class style MMD weight (0 disables per-class fix)

    Returns
    -------
    (total, L_rec, L_class, L_agg, L_style, L_style_cls, L_cls, L_var)
    """
    device = x.device

    # ------------------------------------------------------------------ #
    # 1. Reconstruction: L1 + LPIPS
    # ------------------------------------------------------------------ #
    l1 = F.l1_loss(x_hat, x)

    x_lpips = to_rgb_for_lpips(x * 2.0 - 1.0)
    xh_lpips = to_rgb_for_lpips(x_hat * 2.0 - 1.0)
    lpips_val = loss_fn_vgg(xh_lpips, x_lpips).mean()

    L_rec = cfg.l1_weight * l1 + cfg.lpips_weight * lpips_val

    # ------------------------------------------------------------------ #
    # 2. Supervised class MMD  (spherical Cauchy prior per class)
    # ------------------------------------------------------------------ #
    L_class = torch.tensor(0.0, device=device)
    if alpha > 0.0:
        count = 0
        for k in range(model.n_classes):
            mask = y == k
            if mask.sum() < 2:
                continue
            n_k = mask.sum().item()
            z_p_k = _sample_class_prior_for_loss(model, k, n_k, device)
            L_class = L_class + mmd_loss(z_c[mask], z_p_k)
            count += 1
        if count > 0:
            L_class = L_class / count

    # ------------------------------------------------------------------ #
    # 3. Aggregated semantic MMD  (global prior mixture matching)
    # ------------------------------------------------------------------ #
    L_agg = torch.tensor(0.0, device=device)
    if beta > 0.0:
        B = x.shape[0]
        z_p_mix = _sample_prior_mixture_for_loss(model, B, device)
        L_agg = mmd_loss(z_c, z_p_mix)

    # ------------------------------------------------------------------ #
    # 4. Global style prior MMD  (z_s vs N(0, I)) — necessary but not sufficient
    # ------------------------------------------------------------------ #
    L_style = torch.tensor(0.0, device=device)
    if gamma > 0.0:
        z_s_prior = torch.randn_like(z_s)
        L_style = mmd_euclidean(z_s, z_s_prior)

    # ------------------------------------------------------------------ #
    # 5. Per-class style MMD  (q(z_s|y=k) vs N(0,I) for each k) — proposed fix
    # ------------------------------------------------------------------ #
    L_style_cls = torch.tensor(0.0, device=device)
    if delta > 0.0:
        L_style_cls = mmd_per_class_style(z_s, y, model.n_classes)

    # ------------------------------------------------------------------ #
    # 6. Auxiliary classification loss  (prevents semantic collapse)
    # ------------------------------------------------------------------ #
    L_cls = torch.tensor(0.0, device=device)
    if eta > 0.0:
        logits = model.classifier(mu_c)
        L_cls = F.cross_entropy(logits, y)

    # ------------------------------------------------------------------ #
    # 7. Diversity regularization  (prevent style latent collapse)
    # ------------------------------------------------------------------ #
    # L_var is the *negative* variance; we add lambda_var * L_var to total,
    # so minimising total maximises var(z_s).
    L_var = -z_s.var(dim=0).mean()

    # ------------------------------------------------------------------ #
    # Total
    # ------------------------------------------------------------------ #
    total = (
        L_rec
        + alpha * L_class
        + beta  * L_agg
        + gamma * L_style
        + delta * L_style_cls
        + eta   * L_cls
        + lambda_var * L_var
    )

    return total, L_rec, L_class, L_agg, L_style, L_style_cls, L_cls, L_var

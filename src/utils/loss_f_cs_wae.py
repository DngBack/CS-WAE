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

import torch
import torch.nn.functional as F

from .utils import mmd_loss, mobius_reparam, sample_uniform_sphere, to_rgb_for_lpips
from ..config_f_cs_wae import f_cs_wae_config as cfg


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


# ---------------------------------------------------------------------------
# Main loss function
# ---------------------------------------------------------------------------

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
    eta: float,
    lambda_var: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
           torch.Tensor, torch.Tensor, torch.Tensor]:
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
    alpha, beta, gamma, eta, lambda_var : current loss weights

    Returns
    -------
    (total, L_rec, L_class, L_agg, L_style, L_cls, L_var)
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
        normalized_centers = F.normalize(model.ema_centers, p=2, dim=-1)  # (K, R, d_c)
        count = 0
        for k in range(model.n_classes):
            mask = y == k
            if mask.sum() < 2:
                continue
            n_k = mask.sum().item()
            # Pick center (single-center: R=1; multi-center: random pick)
            if model.n_centers == 1:
                center_k = normalized_centers[k, 0].unsqueeze(0).expand(n_k, -1)
            else:
                r_idx = torch.randint(0, model.n_centers, (n_k,), device=device)
                center_k = normalized_centers[k][r_idx]
            rho_k = torch.full((n_k,), model.rho_p, device=device)
            eps_k = sample_uniform_sphere(n_k, model.semantic_dim, device=device)
            z_p_k = mobius_reparam(eps_k, center_k, rho_k)
            L_class = L_class + mmd_loss(z_c[mask], z_p_k)
            count += 1
        if count > 0:
            L_class = L_class / count

    # ------------------------------------------------------------------ #
    # 3. Aggregated semantic MMD  (global prior mixture matching)
    # ------------------------------------------------------------------ #
    L_agg = torch.tensor(0.0, device=device)
    if beta > 0.0:
        normalized_centers = F.normalize(model.ema_centers, p=2, dim=-1)
        B = x.shape[0]
        rand_k = torch.randint(0, model.n_classes, (B,), device=device)
        if model.n_centers == 1:
            center_mix = normalized_centers[rand_k, 0]           # (B, d_c)
        else:
            rand_r = torch.randint(0, model.n_centers, (B,), device=device)
            center_mix = normalized_centers[rand_k, rand_r]      # (B, d_c)
        rho_mix = torch.full((B,), model.rho_p, device=device)
        eps_mix = sample_uniform_sphere(B, model.semantic_dim, device=device)
        z_p_mix = mobius_reparam(eps_mix, center_mix, rho_mix)
        L_agg = mmd_loss(z_c, z_p_mix)

    # ------------------------------------------------------------------ #
    # 4. Style prior MMD  (z_s vs N(0, I))
    # ------------------------------------------------------------------ #
    L_style = torch.tensor(0.0, device=device)
    if gamma > 0.0:
        z_s_prior = torch.randn_like(z_s)
        L_style = mmd_euclidean(z_s, z_s_prior)

    # ------------------------------------------------------------------ #
    # 5. Auxiliary classification loss  (prevents semantic collapse)
    # ------------------------------------------------------------------ #
    L_cls = torch.tensor(0.0, device=device)
    if eta > 0.0:
        logits = model.classifier(mu_c)
        L_cls = F.cross_entropy(logits, y)

    # ------------------------------------------------------------------ #
    # 6. Diversity regularization  (prevent style latent collapse)
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
        + eta   * L_cls
        + lambda_var * L_var
    )

    return total, L_rec, L_class, L_agg, L_style, L_cls, L_var

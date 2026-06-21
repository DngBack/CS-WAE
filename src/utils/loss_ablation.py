"""
Loss functions for CS-WAE ablation studies
"""

import torch
import torch.nn.functional as F
import lpips
from .utils import mmd_loss, sample_uniform_sphere, mobius_reparam
from ..config_ablation import ablation_config


def calculate_ablation_loss(
    x,
    y,
    x_hat,
    z_q,
    mu_q,
    param_q,
    model,
    loss_fn_vgg,
    sup_mmd_weight,
    unsup_mmd_weight,
    epoch=0,
):
    """
    Calculate CS-WAE loss for ablation variants

    Args:
        x (torch.Tensor): Original input images
        y (torch.Tensor): Class labels
        x_hat (torch.Tensor): Reconstructed images
        z_q (torch.Tensor): Sampled latent vectors
        mu_q (torch.Tensor): Latent means
        param_q (torch.Tensor): Second latent parameter (rho_q or logvar_q)
        model: The ablation model
        loss_fn_vgg: LPIPS loss function
        sup_mmd_weight (float): Weight for supervised MMD loss
        unsup_mmd_weight (float): Weight for unsupervised MMD loss
        epoch (int): Current epoch for annealing

    Returns:
        tuple: (total_loss, recon_loss, supervised_mmd_loss, unsupervised_mmd_loss, kld_loss)
    """
    # --- Reconstruction Loss Components ---
    bce_loss = F.binary_cross_entropy(x_hat, x, reduction="mean")

    # Convert image scale from [0, 1] to [-1, 1] for LPIPS
    x_rescaled = (x * 2) - 1
    x_hat_rescaled = (x_hat * 2) - 1
    # LPIPS requires 3-channel images, repeat grayscale 3 times
    x_rescaled_rgb = x_rescaled.repeat(1, 3, 1, 1)
    x_hat_rescaled_rgb = x_hat_rescaled.repeat(1, 3, 1, 1)

    lpips_loss = loss_fn_vgg(x_hat_rescaled_rgb, x_rescaled_rgb).mean()

    # Combine reconstruction losses
    recon_loss = (
        ablation_config.bce_weight * bce_loss
        + ablation_config.lpips_weight * lpips_loss
    )

    # --- Initialize loss components ---
    supervised_mmd_loss = torch.tensor(0.0, device=x.device)
    unsupervised_mmd_loss = torch.tensor(0.0, device=x.device)
    kld_loss = torch.tensor(0.0, device=x.device)

    # --- Regularization Loss Components (variant-dependent) ---

    if model.use_spherical_space:
        # === SPHERICAL SPACE VARIANTS ===

        if model.use_supervised_mmd:
            # Supervised MMD loss: match each class to its corresponding prior
            normalized_prior_mus = F.normalize(model.prior_mus, p=2, dim=1)

            for c in range(ablation_config.n_classes):
                class_mask = y == c
                if class_mask.sum() > 1:
                    # Sample from class-specific prior
                    if model.prior_type == "spherical_cauchy":
                        z_p_class = mobius_reparam(
                            sample_uniform_sphere(
                                class_mask.sum(),
                                ablation_config.latent_dim,
                                device=x.device,
                            ),
                            normalized_prior_mus[c].expand(class_mask.sum(), -1),
                            torch.full(
                                (class_mask.sum(),), model.rho_p, device=x.device
                            ),
                        )
                    elif model.prior_type == "von_mises_fisher":
                        z_p_class = model._sample_von_mises_fisher(
                            c, class_mask.sum().item(), x.device
                        )

                    supervised_mmd_loss += mmd_loss(z_q[class_mask], z_p_class)

            supervised_mmd_loss /= ablation_config.n_classes

        # Unsupervised MMD loss (always used for spherical variants)
        random_classes = torch.randint(
            0, ablation_config.n_classes, (x.size(0),), device=x.device
        )

        if model.prior_type == "spherical_cauchy":
            normalized_prior_mus = F.normalize(model.prior_mus, p=2, dim=1)
            z_p_unsupervised = mobius_reparam(
                sample_uniform_sphere(
                    x.size(0), ablation_config.latent_dim, device=x.device
                ),
                normalized_prior_mus[random_classes],
                torch.full((x.size(0),), model.rho_p, device=x.device),
            )
        elif model.prior_type == "von_mises_fisher":
            # Sample from random class priors
            z_p_list = []
            for i, c in enumerate(random_classes):
                z_p_single = model._sample_von_mises_fisher(c.item(), 1, x.device)
                z_p_list.append(z_p_single)
            z_p_unsupervised = torch.cat(z_p_list, dim=0)

        unsupervised_mmd_loss = mmd_loss(z_q, z_p_unsupervised)

    else:
        # === EUCLIDEAN SPACE VARIANTS ===

        if model.use_supervised_mmd:
            # Supervised MMD in Euclidean space
            for c in range(ablation_config.n_classes):
                class_mask = y == c
                if class_mask.sum() > 1:
                    # Sample from class-specific Gaussian prior
                    mu_p = model.prior_mus[c].expand(class_mask.sum(), -1)
                    logvar_p = model.prior_logvars[c].expand(class_mask.sum(), -1)
                    std_p = torch.exp(0.5 * logvar_p)
                    eps = torch.randn_like(std_p)
                    z_p_class = mu_p + eps * std_p

                    supervised_mmd_loss += mmd_loss(z_q[class_mask], z_p_class)

            supervised_mmd_loss /= ablation_config.n_classes

        # Standard VAE KL divergence as regularization in Euclidean space
        logvar_q = param_q
        # KL divergence to standard Gaussian
        kld_loss = (
            -0.5 * torch.sum(1 + logvar_q - mu_q.pow(2) - logvar_q.exp()) / x.size(0)
        )

        # Unsupervised MMD to standard Gaussian
        z_p_standard = torch.randn_like(z_q)
        unsupervised_mmd_loss = mmd_loss(z_q, z_p_standard)

    # --- Total Loss Calculation ---
    total_loss = recon_loss
    if model.use_supervised_mmd:
        total_loss = total_loss + sup_mmd_weight * supervised_mmd_loss
    total_loss = total_loss + unsup_mmd_weight * unsupervised_mmd_loss
    if not model.use_spherical_space:
        total_loss = total_loss + kld_loss

    return total_loss, recon_loss, supervised_mmd_loss, unsupervised_mmd_loss, kld_loss


def add_noise_to_images(images, noise_type="gaussian", noise_level=0.1):
    """
    Add noise to images for robustness testing

    Args:
        images (torch.Tensor): Input images [B, C, H, W]
        noise_type (str): Type of noise ('gaussian', 'salt_pepper', 'uniform')
        noise_level (float): Intensity of noise

    Returns:
        torch.Tensor: Noisy images
    """
    if noise_type == "gaussian":
        noise = torch.randn_like(images) * noise_level
        noisy_images = images + noise
    elif noise_type == "salt_pepper":
        mask = torch.rand_like(images) < noise_level
        salt_pepper = torch.randint_like(images, 0, 2, dtype=images.dtype)
        noisy_images = torch.where(mask, salt_pepper, images)
    elif noise_type == "uniform":
        noise = (torch.rand_like(images) - 0.5) * 2 * noise_level
        noisy_images = images + noise
    else:
        raise ValueError(f"Unknown noise type: {noise_type}")

    # Clamp to valid range [0, 1]
    return torch.clamp(noisy_images, 0, 1)

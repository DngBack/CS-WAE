import torch
import torch.nn.functional as F
import lpips
from .utils import mmd_loss, sample_uniform_sphere, mobius_reparam
from ..config import config

"""
Loss functions for CS-WAE, including combined reconstruction and regularization losses.
"""


def calculate_cs_wae_loss(x, y, x_hat, z_q, model, loss_fn_vgg, sup_mmd_weight, unsup_mmd_weight):
    """
    Calculate the complete CS-WAE loss with LPIPS and MMD components.
    
    Args:
        x (torch.Tensor): Original input images
        y (torch.Tensor): Class labels
        x_hat (torch.Tensor): Reconstructed images
        z_q (torch.Tensor): Encoded latent vectors
        model: The CS-WAE model
        loss_fn_vgg: LPIPS loss function
        sup_mmd_weight (float): Weight for supervised MMD loss
        unsup_mmd_weight (float): Weight for unsupervised MMD loss
        
    Returns:
        tuple: (total_loss, recon_loss, supervised_mmd_loss, unsupervised_mmd_loss)
    """
    # --- Reconstruction Loss Components ---
    bce_loss = F.binary_cross_entropy(x_hat, x, reduction='mean')

    # Convert image scale from [0, 1] to [-1, 1] for LPIPS
    x_rescaled = (x * 2) - 1
    x_hat_rescaled = (x_hat * 2) - 1
    # LPIPS requires 3-channel images, repeat grayscale 3 times
    x_rescaled_rgb = x_rescaled.repeat(1, 3, 1, 1)
    x_hat_rescaled_rgb = x_hat_rescaled.repeat(1, 3, 1, 1)

    lpips_loss = loss_fn_vgg(x_hat_rescaled_rgb, x_rescaled_rgb).mean()

    # Combine reconstruction losses
    recon_loss = config.bce_weight * bce_loss + config.lpips_weight * lpips_loss

    # --- MMD Loss Components ---
    device = x.device
    supervised_mmd_loss = 0.0
    normalized_prior_mus = F.normalize(model.prior_mus, p=2, dim=1)
    
    for c in range(config.n_classes):
        class_mask = (y == c)
        if class_mask.sum() > 1:
            n = class_mask.sum().item()
            supervised_mmd_loss += mmd_loss(
                z_q[class_mask],
                mobius_reparam(
                    sample_uniform_sphere(n, config.latent_dim, device=device),
                    normalized_prior_mus[c].expand(n, -1),
                    torch.full((n,), model.rho_p, device=device),
                )
            )
    supervised_mmd_loss /= config.n_classes

    # Unsupervised MMD loss
    random_classes = torch.randint(0, config.n_classes, (x.size(0),), device=device)
    z_p_unsupervised = mobius_reparam(
        sample_uniform_sphere(x.size(0), config.latent_dim, device=device),
        normalized_prior_mus[random_classes],
        torch.full((x.size(0),), model.rho_p, device=device),
    )
    unsupervised_mmd_loss = mmd_loss(z_q, z_p_unsupervised)

    # --- Total Loss ---
    total_loss = recon_loss + \
                 (sup_mmd_weight * supervised_mmd_loss) + \
                 (unsup_mmd_weight * unsupervised_mmd_loss)

    return total_loss, recon_loss, supervised_mmd_loss, unsupervised_mmd_loss
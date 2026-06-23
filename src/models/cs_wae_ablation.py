"""
CS-WAE ablation variants for systematic evaluation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ..config_ablation import ablation_config
from ..utils.utils import sample_uniform_sphere, mobius_reparam


from ..config import config
from .image_cnn import DecoderCNN, encoder_spatial_size
from .backbone import build_cs_wae_encoder, encoder_spatial_for_backbone


class EncoderCNN_Ablation(nn.Module):
    """Ablation encoder — spherical or Euclidean heads, multi-channel support."""

    def __init__(self, latent_dim, output_type="spherical", in_channels=None, image_size=None):
        super().__init__()
        self.output_type = output_type
        in_channels = in_channels if in_channels is not None else config.in_channels
        image_size = image_size if image_size is not None else config.image_size
        spatial = encoder_spatial_size(image_size, "cs_wae")
        flat = 128 * spatial * spatial

        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
        )
        self.fc_block = nn.Sequential(
            nn.Flatten(), nn.Linear(flat, 256), nn.ReLU(True)
        )
        self.fc_mu = nn.Linear(256, latent_dim)
        if output_type == "spherical":
            self.fc_s = nn.Linear(256, 1)
        else:
            self.fc_logvar = nn.Linear(256, latent_dim)

    def forward(self, x):
        x = self.conv_block(x)
        x = self.fc_block(x)
        if self.output_type == "spherical":
            return self.fc_mu(x), self.fc_s(x)
        return self.fc_mu(x), self.fc_logvar(x)


class DecoderCNN_Ablation(nn.Module):
    """Ablation decoder with configurable channels."""

    def __init__(self, latent_dim, in_channels=None, image_size=None):
        super().__init__()
        in_channels = in_channels if in_channels is not None else config.in_channels
        image_size = image_size if image_size is not None else config.image_size
        spatial = encoder_spatial_for_backbone(config.backbone, image_size, "cs_wae")
        self.spatial = spatial
        self.decoder = DecoderCNN(latent_dim, in_channels, spatial, image_size)

    def forward(self, z):
        return self.decoder(z)


class CSWAEAblation(nn.Module):
    """CS-WAE with ablation capabilities"""

    def __init__(self, latent_dim, n_classes, variant="baseline"):
        super(CSWAEAblation, self).__init__()

        # Get variant configuration
        if variant not in ablation_config.ablation_variants:
            raise ValueError(f"Unknown variant: {variant}")

        self.variant_config = ablation_config.ablation_variants[variant]
        self.variant_name = variant

        # Configure encoder based on space type
        encoder_type = (
            "spherical" if self.variant_config["use_spherical_space"] else "euclidean"
        )
        self.encoder = build_cs_wae_encoder(
            config.backbone,
            latent_dim,
            config.in_channels,
            config.image_size,
            output_type=encoder_type,
        )
        self.decoder = DecoderCNN_Ablation(latent_dim)

        # Initialize priors based on variant
        if self.variant_config["use_spherical_space"]:
            # Spherical space: learnable directions on sphere
            self.prior_mus = nn.Parameter(torch.randn(n_classes, latent_dim))
            self.rho_p = ablation_config.rho_prior
        else:
            # Euclidean space: standard Gaussian priors or learnable means
            self.prior_mus = nn.Parameter(torch.randn(n_classes, latent_dim))
            self.prior_logvars = nn.Parameter(torch.zeros(n_classes, latent_dim))

        # Store configuration
        self.latent_dim = latent_dim
        self.n_classes = n_classes
        self.use_supervised_mmd = self.variant_config["use_supervised_mmd"]
        self.use_spherical_space = self.variant_config["use_spherical_space"]
        self.prior_type = self.variant_config["prior_type"]

    def encode_to_distribution(self, x):
        """Encode input to distribution parameters"""
        if self.use_spherical_space:
            # Spherical encoding
            mu_q_unnormalized, s_q = self.encoder(x)
            mu_q = F.normalize(mu_q_unnormalized, p=2, dim=1)
            rho_q = torch.sigmoid(s_q).squeeze(-1) * (1 - ablation_config.epsilon)
            return mu_q, rho_q
        else:
            # Euclidean encoding (VAE-style)
            mu_q, logvar_q = self.encoder(x)
            return mu_q, logvar_q

    def sample_latent(self, mu_q, param_q):
        """Sample from latent distribution"""
        if self.use_spherical_space:
            # Spherical sampling with Möbius reparameterization
            rho_q = param_q
            eps = sample_uniform_sphere(
                mu_q.shape[0], self.latent_dim, device=mu_q.device
            )
            z_q = mobius_reparam(eps, mu_q, rho_q)
        else:
            # Euclidean sampling (VAE reparameterization)
            logvar_q = param_q
            std = torch.exp(0.5 * logvar_q)
            eps = torch.randn_like(std)
            z_q = mu_q + eps * std
        return z_q

    def sample_from_prior(self, class_idx, num_samples, device):
        """Sample from class-specific prior"""
        if self.use_spherical_space:
            # Spherical prior sampling
            if self.prior_type == "spherical_cauchy":
                # Use Möbius reparameterization with learned centers
                normalized_prior_mus = F.normalize(self.prior_mus, p=2, dim=1)
                eps = sample_uniform_sphere(num_samples, self.latent_dim, device=device)
                z_p = mobius_reparam(
                    eps,
                    normalized_prior_mus[class_idx].expand(num_samples, -1),
                    torch.full((num_samples,), self.rho_p, device=device),
                )
            elif self.prior_type == "von_mises_fisher":
                # von Mises-Fisher sampling (approximate with rejection sampling)
                z_p = self._sample_von_mises_fisher(class_idx, num_samples, device)
            else:
                raise ValueError(f"Unknown spherical prior type: {self.prior_type}")
        else:
            # Euclidean prior sampling
            if self.prior_type == "gaussian":
                # Sample from learned Gaussian priors
                mu_p = self.prior_mus[class_idx].expand(num_samples, -1)
                logvar_p = self.prior_logvars[class_idx].expand(num_samples, -1)
                std_p = torch.exp(0.5 * logvar_p)
                eps = torch.randn_like(std_p)
                z_p = mu_p + eps * std_p
            else:
                raise ValueError(f"Unknown Euclidean prior type: {self.prior_type}")

        return z_p

    def _sample_von_mises_fisher(self, class_idx, num_samples, device):
        """Sample from von Mises-Fisher distribution (simplified implementation)"""
        # Approximate vMF with concentrated spherical Gaussian
        normalized_prior_mus = F.normalize(self.prior_mus, p=2, dim=1)
        mu = normalized_prior_mus[class_idx]

        # Use higher concentration for vMF (less heavy-tailed than Cauchy)
        kappa = 10.0  # Concentration parameter

        # Sample using rejection sampling approximation
        samples = []
        while len(samples) < num_samples:
            # Sample from spherical Gaussian and normalize
            z = torch.randn(min(num_samples * 2, 1000), self.latent_dim, device=device)
            z = F.normalize(z, p=2, dim=1)

            # Accept samples based on vMF likelihood
            dots = torch.mv(z, mu)
            probs = torch.exp(kappa * (dots - 1))
            accept = torch.rand(len(z), device=device) < probs

            accepted_samples = z[accept]
            samples.append(accepted_samples)

            if sum(len(s) for s in samples) >= num_samples:
                break

        all_samples = torch.cat(samples, dim=0)
        return all_samples[:num_samples]

    def forward(self, x):
        """Forward pass"""
        mu_q, param_q = self.encode_to_distribution(x)
        z_q = self.sample_latent(mu_q, param_q)
        x_hat = self.decoder(z_q)
        return x_hat, z_q, mu_q, param_q


def create_ablation_model(variant="baseline"):
    """Factory function to create ablation models.

    The baseline variant uses the same model class as main training
    (SphericalWAE_Supervised) for identical results.
    """
    if variant == "baseline":
        from .cs_wae import SphericalWAE_Supervised

        return SphericalWAE_Supervised(
            latent_dim=ablation_config.latent_dim,
            n_classes=ablation_config.n_classes,
            in_channels=config.in_channels,
            image_size=config.image_size,
        )

    return CSWAEAblation(
        latent_dim=ablation_config.latent_dim,
        n_classes=ablation_config.n_classes,
        variant=variant,
    )

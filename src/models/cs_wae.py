"""
CS-WAE (Spherical Wasserstein Autoencoder) model implementation
"""
import torch
import torch.nn.functional as F
from ..config import config
from ..utils.utils import sample_uniform_sphere, mobius_reparam
from .image_cnn import DecoderCNN, EncoderCNN, encoder_spatial_size
from .backbone import build_cs_wae_encoder, encoder_spatial_for_backbone


class SphericalWAE_Supervised(torch.nn.Module):
    """Spherical Wasserstein Autoencoder with Supervised Learning"""

    def __init__(
        self,
        latent_dim,
        n_classes,
        in_channels: int = 1,
        image_size: int = 28,
        backbone: str | None = None,
    ):
        super().__init__()
        backbone = backbone or config.backbone
        spatial = encoder_spatial_for_backbone(backbone, image_size, "cs_wae")
        self.encoder = build_cs_wae_encoder(
            backbone, latent_dim, in_channels, image_size, output_type="spherical"
        )
        self.decoder = DecoderCNN(latent_dim, in_channels, spatial, image_size)
        self.prior_mus = torch.nn.Parameter(torch.randn(n_classes, latent_dim))
        self.rho_p = config.rho_prior
        self.n_classes = n_classes
        self.in_channels = in_channels
        self.image_size = image_size
        self.backbone = backbone

    def encode_to_distribution(self, x):
        mu_q_unnormalized, s_q = self.encoder(x)
        mu_q = F.normalize(mu_q_unnormalized, p=2, dim=1)
        rho_q = torch.sigmoid(s_q).squeeze(-1) * (1 - config.epsilon)
        return mu_q, rho_q

    def forward(self, x):
        mu_q, rho_q = self.encode_to_distribution(x)
        eps = sample_uniform_sphere(x.shape[0], config.latent_dim, device=x.device)
        z_q = mobius_reparam(eps, mu_q, rho_q)
        x_hat = self.decoder(z_q)
        return x_hat, z_q


# Backward-compatible re-exports
EncoderCNN = EncoderCNN
DecoderCNN = DecoderCNN

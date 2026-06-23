"""
Baseline model implementations for comparison with CS-WAE
Includes VAE, WAE-MMD, S-VAE, and VaDE
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .image_cnn import BaseDecoderCNN, baseline_decoder_spatial
from .backbone import build_baseline_encoder
from ..config import config


class VAE(nn.Module):
    """Vanilla Variational Autoencoder"""

    def __init__(
        self,
        latent_dim,
        in_channels: int = 1,
        image_size: int = 28,
        backbone: str | None = None,
    ):
        super().__init__()
        backbone = backbone or config.backbone
        dec_spatial = baseline_decoder_spatial(image_size)
        self.encoder = build_baseline_encoder(
            backbone, latent_dim, in_channels, image_size, vae_mode=True
        )
        self.decoder = BaseDecoderCNN(latent_dim, in_channels, dec_spatial, image_size)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        return self.decoder(z), mu, log_var

    def loss_function(self, recon_x, x, mu, log_var):
        BCE = F.binary_cross_entropy(recon_x, x, reduction="sum")
        KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        return BCE + KLD


def rbf_kernel(x, y, sigma=1.0):
    dist_sq = torch.cdist(x, y, p=2).pow(2)
    return torch.exp(-dist_sq / (2 * sigma**2))


def mmd_loss(q_samples, p_samples, sigma=1.0):
    k_qq = rbf_kernel(q_samples, q_samples, sigma).mean()
    k_pp = rbf_kernel(p_samples, p_samples, sigma).mean()
    k_qp = rbf_kernel(q_samples, p_samples, sigma).mean()
    return k_qq + k_pp - 2 * k_qp


class WAE_MMD(nn.Module):
    """Wasserstein Autoencoder with MMD regularization"""

    def __init__(
        self,
        latent_dim,
        in_channels: int = 1,
        image_size: int = 28,
        backbone: str | None = None,
    ):
        super().__init__()
        backbone = backbone or config.backbone
        dec_spatial = baseline_decoder_spatial(image_size)
        self.encoder = build_baseline_encoder(backbone, latent_dim, in_channels, image_size)
        self.decoder = BaseDecoderCNN(latent_dim, in_channels, dec_spatial, image_size)
        self.latent_dim = latent_dim

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z

    def loss_function(self, recon_x, x, z, mmd_weight=10.0):
        recon_loss = F.binary_cross_entropy(recon_x, x, reduction="mean")
        true_samples = torch.randn(z.size(0), self.latent_dim, device=z.device)
        mmd = mmd_loss(z, true_samples)
        return recon_loss + mmd_weight * mmd


class S_VAE(nn.Module):
    """Spherical VAE"""

    def __init__(
        self,
        latent_dim,
        in_channels: int = 1,
        image_size: int = 28,
        backbone: str | None = None,
    ):
        super().__init__()
        backbone = backbone or config.backbone
        dec_spatial = baseline_decoder_spatial(image_size)
        self.encoder = build_baseline_encoder(
            backbone, latent_dim + 1, in_channels, image_size
        )
        self.decoder = BaseDecoderCNN(latent_dim, in_channels, dec_spatial, image_size)
        self.latent_dim = latent_dim

    def forward(self, x):
        try:
            from hyperspherical_vae.distributions import VonMisesFisher, HypersphericalUniform

            q_params = self.encoder(x)
            mu = F.normalize(q_params[:, :-1], p=2, dim=1)
            kappa = F.softplus(q_params[:, -1].unsqueeze(-1)) + 1e-6

            q_z = VonMisesFisher(mu, kappa)
            p_z = HypersphericalUniform(self.latent_dim, device=x.device)

            z = q_z.sample()
            recon_x = self.decoder(z)
            return recon_x, q_z, p_z
        except ImportError:
            raise ImportError(
                "Please install hyperspherical_vae: "
                "pip install git+https://github.com/nicola-decao/s-vae-pytorch.git"
            )

    def loss_function(self, recon_x, x, q_z, p_z):
        recon_loss = F.binary_cross_entropy(recon_x, x, reduction="sum")
        kld = torch.distributions.kl.kl_divergence(q_z, p_z).sum()
        return recon_loss + kld


class VaDE(nn.Module):
    """Variational Deep Embedding"""

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
        dec_spatial = baseline_decoder_spatial(image_size)
        self.encoder = build_baseline_encoder(
            backbone, latent_dim, in_channels, image_size, vae_mode=True
        )
        self.decoder = BaseDecoderCNN(latent_dim, in_channels, dec_spatial, image_size)
        self.latent_dim = latent_dim
        self.n_classes = n_classes

        self.pi = nn.Parameter(torch.ones(n_classes) / n_classes, requires_grad=True)
        self.mu_c = nn.Parameter(torch.zeros(n_classes, latent_dim), requires_grad=True)
        self.log_var_c = nn.Parameter(torch.zeros(n_classes, latent_dim), requires_grad=True)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        return self.decoder(z), mu, log_var, z

    def loss_function(self, recon_x, x, mu, log_var, z):
        eps = 1e-10
        recon_loss = F.binary_cross_entropy(
            recon_x.clamp(eps, 1 - eps), x.clamp(0, 1), reduction="mean"
        )

        log_var = log_var.clamp(-6, 6)
        log_var_c = self.log_var_c.clamp(-6, 6)

        z_expanded = z.unsqueeze(1)
        mu_c_expanded = self.mu_c.unsqueeze(0)
        log_var_c_expanded = log_var_c.unsqueeze(0)
        var_c = log_var_c_expanded.exp()

        log_2pi = torch.log(
            torch.tensor(2 * 3.141592653589793, device=x.device, dtype=x.dtype)
        )
        log_p_z_given_c = -0.5 * (
            log_var_c_expanded.sum(dim=2)
            + ((z_expanded - mu_c_expanded).pow(2) / var_c).sum(dim=2)
            + self.latent_dim * log_2pi
        )
        log_prior_c = F.log_softmax(self.pi, dim=0)
        log_p_c_z = log_prior_c.unsqueeze(0) + log_p_z_given_c
        log_gamma = log_p_c_z - torch.logsumexp(log_p_c_z, dim=1, keepdim=True)
        gamma = log_gamma.exp()

        log_var_exp = log_var.unsqueeze(1)
        mu_exp = mu.unsqueeze(1)
        var = log_var_exp.exp()

        log_p_z_given_c_post = -0.5 * (
            log_var_c_expanded.sum(dim=2)
            + ((z_expanded - mu_c_expanded).pow(2) / var_c).sum(dim=2)
        )
        log_q_z_x = -0.5 * (
            log_var_exp.sum(dim=2)
            + ((z_expanded - mu_exp).pow(2) / var).sum(dim=2)
        )

        kl_loss = (gamma * (log_p_z_given_c_post - log_q_z_x)).sum() / z.size(0)
        return recon_loss + kl_loss

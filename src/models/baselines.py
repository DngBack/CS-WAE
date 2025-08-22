"""
Baseline model implementations for comparison with CS-WAE
Includes VAE, WAE-MMD, S-VAE, and VaDE
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class BaseEncoderCNN(nn.Module):
    """Base CNN Encoder for baseline models"""
    
    def __init__(self, latent_dim, vae_mode=False):
        super(BaseEncoderCNN, self).__init__()
        self.vae_mode = vae_mode
        self.conv_block = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1), nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(True),
            nn.Conv2d(64, 128, 4, 2, 1), nn.ReLU(True),
        )
        self.fc_block = nn.Sequential(
            nn.Flatten(), 
            nn.Linear(128 * 3 * 3, 256), 
            nn.ReLU(True)
        )
        self.fc_out1 = nn.Linear(256, latent_dim)  # mu for VAE, z for WAE
        if self.vae_mode:
            self.fc_out2 = nn.Linear(256, latent_dim)  # log_var for VAE

    def forward(self, x):
        x = self.conv_block(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc_block(x)
        out1 = self.fc_out1(x)
        if self.vae_mode:
            out2 = self.fc_out2(x)
            return out1, out2
        return out1


class BaseDecoderCNN(nn.Module):
    """Base CNN Decoder for baseline models"""
    
    def __init__(self, latent_dim):
        super(BaseDecoderCNN, self).__init__()
        self.fc_block = nn.Sequential(
            nn.Linear(latent_dim, 256), 
            nn.ReLU(True), 
            nn.Linear(256, 128 * 7 * 7), 
            nn.ReLU(True)
        )
        self.deconv_block = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(True),
            nn.ConvTranspose2d(32, 1, 4, 2, 1), nn.Sigmoid()
        )
        
    def forward(self, z):
        x = self.fc_block(z)
        x = x.view(-1, 128, 7, 7)
        x = self.deconv_block(x)
        # Crop to 28x28
        return x[:, :, :28, :28]


class VAE(nn.Module):
    """Vanilla Variational Autoencoder"""
    
    def __init__(self, latent_dim):
        super(VAE, self).__init__()
        self.encoder = BaseEncoderCNN(latent_dim, vae_mode=True)
        self.decoder = BaseDecoderCNN(latent_dim)

    def reparameterize(self, mu, log_var):
        """Reparameterization trick"""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        return self.decoder(z), mu, log_var

    def loss_function(self, recon_x, x, mu, log_var):
        """VAE loss function"""
        BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
        KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        return BCE + KLD


def rbf_kernel(x, y, sigma=1.0):
    """RBF kernel for MMD computation"""
    dist_sq = torch.cdist(x, y, p=2).pow(2)
    return torch.exp(-dist_sq / (2 * sigma**2))


def mmd_loss(q_samples, p_samples, sigma=1.0):
    """Maximum Mean Discrepancy loss"""
    k_qq = rbf_kernel(q_samples, q_samples, sigma).mean()
    k_pp = rbf_kernel(p_samples, p_samples, sigma).mean()
    k_qp = rbf_kernel(q_samples, p_samples, sigma).mean()
    return k_qq + k_pp - 2 * k_qp


class WAE_MMD(nn.Module):
    """Wasserstein Autoencoder with MMD regularization"""
    
    def __init__(self, latent_dim):
        super(WAE_MMD, self).__init__()
        self.encoder = BaseEncoderCNN(latent_dim)
        self.decoder = BaseDecoderCNN(latent_dim)
        self.latent_dim = latent_dim

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z

    def loss_function(self, recon_x, x, z, mmd_weight=10.0):
        """WAE-MMD loss function"""
        recon_loss = F.binary_cross_entropy(recon_x, x, reduction='mean')
        true_samples = torch.randn(z.size(0), self.latent_dim, device=z.device)
        mmd = mmd_loss(z, true_samples)
        return recon_loss + mmd_weight * mmd


class S_VAE(nn.Module):
    """Spherical VAE"""
    
    def __init__(self, latent_dim):
        super(S_VAE, self).__init__()
        self.encoder = BaseEncoderCNN(latent_dim + 1)
        self.decoder = BaseDecoderCNN(latent_dim)
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
            raise ImportError("Please install hyperspherical_vae: pip install git+https://github.com/nicola-decao/s-vae-pytorch.git")

    def loss_function(self, recon_x, x, q_z, p_z):
        """S-VAE loss function"""
        recon_loss = F.binary_cross_entropy(recon_x, x, reduction='sum')
        kld = torch.distributions.kl.kl_divergence(q_z, p_z).sum()
        return recon_loss + kld


class VaDE(nn.Module):
    """Variational Deep Embedding"""
    
    def __init__(self, latent_dim, n_classes):
        super(VaDE, self).__init__()
        self.encoder = BaseEncoderCNN(latent_dim, vae_mode=True)
        self.decoder = BaseDecoderCNN(latent_dim)
        self.latent_dim = latent_dim
        self.n_classes = n_classes

        # GMM parameters
        self.pi = nn.Parameter(torch.ones(n_classes) / n_classes, requires_grad=True)
        self.mu_c = nn.Parameter(torch.randn(n_classes, latent_dim), requires_grad=True)
        self.log_var_c = nn.Parameter(torch.randn(n_classes, latent_dim), requires_grad=True)

    def reparameterize(self, mu, log_var):
        """Reparameterization trick"""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        return self.decoder(z), mu, log_var, z

    def loss_function(self, recon_x, x, mu, log_var, z):
        """VaDE loss function"""
        # Reconstruction Loss
        recon_loss = F.binary_cross_entropy(recon_x, x, reduction='sum')

        # GMM Prior Loss (KL Divergence)
        z_expanded = z.unsqueeze(1)
        mu_c_expanded = self.mu_c.unsqueeze(0)
        log_var_c_expanded = self.log_var_c.unsqueeze(0)

        # Calculate gamma (responsibilities)
        log_p_z_given_c = -0.5 * (
            torch.sum(log_var_c_expanded, dim=2) +
            torch.sum((z_expanded - mu_c_expanded).pow(2) / torch.exp(log_var_c_expanded), dim=2) +
            self.latent_dim * np.log(2 * np.pi)
        )
        log_p_c_p_z_given_c = F.log_softmax(self.pi, dim=0) + log_p_z_given_c
        gamma = F.softmax(log_p_c_p_z_given_c, dim=1)

        # Calculate KL Divergence part
        kl_div = 0.5 * torch.sum(gamma * (
            torch.sum(log_var_c_expanded - log_var.unsqueeze(1), dim=2) +
            torch.sum((torch.exp(log_var.unsqueeze(1)) + (mu.unsqueeze(1) - mu_c_expanded).pow(2)) / torch.exp(log_var_c_expanded), dim=2) -
            self.latent_dim
        ))
        kl_div -= torch.sum(gamma * torch.log(self.pi.unsqueeze(0) / (gamma + 1e-10)))

        return recon_loss + kl_div

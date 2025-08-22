"""
CS-WAE (Spherical Wasserstein Autoencoder) model implementation
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..config import config
from ..utils.utils import sample_uniform_sphere, mobius_reparam


class EncoderCNN(nn.Module):
    """CNN Encoder for CS-WAE"""
    
    def __init__(self, latent_dim):
        super(EncoderCNN, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1), 
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
            nn.Flatten(), 
            nn.Linear(128 * 4 * 4, 256), 
            nn.ReLU(True)
        )
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_s = nn.Linear(256, 1)
        
    def forward(self, x):
        x = self.conv_block(x)
        x = self.fc_block(x)
        return self.fc_mu(x), self.fc_s(x)


class DecoderCNN(nn.Module):
    """CNN Decoder for CS-WAE"""
    
    def __init__(self, latent_dim):
        super(DecoderCNN, self).__init__()
        self.fc_block = nn.Sequential(
            nn.Linear(latent_dim, 256), 
            nn.ReLU(True), 
            nn.Linear(256, 128 * 4 * 4), 
            nn.ReLU(True)
        )
        self.deconv_block = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1), 
            nn.BatchNorm2d(64), 
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1), 
            nn.BatchNorm2d(32), 
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1), 
            nn.Sigmoid()
        )
        
    def forward(self, z):
        x = self.fc_block(z)
        x = x.view(-1, 128, 4, 4)
        return self.deconv_block(x)


class SphericalWAE_Supervised(nn.Module):
    """Spherical Wasserstein Autoencoder with Supervised Learning"""
    
    def __init__(self, latent_dim, n_classes):
        super(SphericalWAE_Supervised, self).__init__()
        self.encoder = EncoderCNN(latent_dim)
        self.decoder = DecoderCNN(latent_dim)
        self.prior_mus = nn.Parameter(torch.randn(n_classes, latent_dim))
        self.rho_p = config.rho_prior
        
    def encode_to_distribution(self, x):
        """Encode input to distribution parameters"""
        mu_q_unnormalized, s_q = self.encoder(x)
        mu_q = F.normalize(mu_q_unnormalized, p=2, dim=1)
        rho_q = torch.sigmoid(s_q).squeeze(-1) * (1 - config.epsilon)
        return mu_q, rho_q
        
    def forward(self, x):
        """Forward pass"""
        mu_q, rho_q = self.encode_to_distribution(x)
        eps = sample_uniform_sphere(x.shape[0], config.latent_dim, device=x.device)
        z_q = mobius_reparam(eps, mu_q, rho_q)
        x_hat = self.decoder(z_q)
        return x_hat, z_q

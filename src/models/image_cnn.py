"""CNN encoder/decoder helpers for grayscale or RGB inputs."""

from __future__ import annotations

import torch
import torch.nn as nn


def _conv_out(size: int, kernel: int, stride: int, padding: int) -> int:
    return (size + 2 * padding - kernel) // stride + 1


def baseline_decoder_spatial(image_size: int) -> int:
    """Legacy decoder init spatial (7 for 28² was used in original baselines)."""
    return 7 if image_size == 28 else 4


def encoder_spatial_size(image_size: int, style: str = "cs_wae") -> int:
    """Spatial H=W after the encoder conv stack."""
    if style == "cs_wae":
        size = image_size
        size = _conv_out(size, 4, 2, 1)
        size = _conv_out(size, 4, 2, 1)
        size = _conv_out(size, 3, 2, 1)
        return size
    if style == "baseline":
        size = image_size
        for _ in range(3):
            size = _conv_out(size, 4, 2, 1)
        return size
    raise ValueError(f"Unknown CNN style: {style}")


def verify_decoder_output(
    latent_dim: int,
    in_channels: int,
    image_size: int,
    style: str = "cs_wae",
) -> int:
    """Return decoder output side length; used in tests."""
    spatial = encoder_spatial_size(image_size, style)
    if style == "cs_wae":
        dec = DecoderCNN(latent_dim, in_channels, spatial, image_size)
    else:
        dec = BaseDecoderCNN(latent_dim, in_channels, spatial, image_size)
    z = torch.zeros(1, latent_dim)
    return dec(z).shape[-1]


class EncoderCNN(nn.Module):
    """CS-WAE encoder (spherical / Euclidean ablation-compatible)."""

    def __init__(self, latent_dim, in_channels: int = 1, image_size: int = 28):
        super().__init__()
        spatial = encoder_spatial_size(image_size, "cs_wae")
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
        flat = 128 * spatial * spatial
        self.fc_block = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat, 256),
            nn.ReLU(True),
        )
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_s = nn.Linear(256, 1)

    def forward(self, x):
        x = self.conv_block(x)
        x = self.fc_block(x)
        return self.fc_mu(x), self.fc_s(x)


class DecoderCNN(nn.Module):
    """CS-WAE decoder."""

    def __init__(self, latent_dim, in_channels: int = 1, spatial: int = 4, image_size: int = 28):
        super().__init__()
        self.spatial = spatial
        self.image_size = image_size
        self.fc_block = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(True),
            nn.Linear(256, 128 * spatial * spatial),
            nn.ReLU(True),
        )
        self.deconv_block = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, in_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, z):
        x = self.fc_block(z)
        x = x.view(-1, 128, self.spatial, self.spatial)
        x = self.deconv_block(x)
        if x.shape[-1] != self.image_size or x.shape[-2] != self.image_size:
            x = torch.nn.functional.interpolate(
                x,
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            )
        return x


class BaseEncoderCNN(nn.Module):
    """Baseline VAE / WAE encoder."""

    def __init__(self, latent_dim, in_channels: int = 1, image_size: int = 28, vae_mode=False):
        super().__init__()
        self.vae_mode = vae_mode
        spatial = encoder_spatial_size(image_size, "baseline")
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.ReLU(True),
        )
        flat = 128 * spatial * spatial
        self.fc_block = nn.Sequential(nn.Flatten(), nn.Linear(flat, 256), nn.ReLU(True))
        self.fc_out1 = nn.Linear(256, latent_dim)
        if self.vae_mode:
            self.fc_out2 = nn.Linear(256, latent_dim)

    def forward(self, x):
        x = self.conv_block(x)
        x = x.view(x.size(0), -1)
        x = self.fc_block(x)
        out1 = self.fc_out1(x)
        if self.vae_mode:
            return out1, self.fc_out2(x)
        return out1


class BaseDecoderCNN(nn.Module):
    """Baseline decoder with optional crop to target image size."""

    def __init__(self, latent_dim, in_channels: int = 1, spatial: int = 7, image_size: int = 28):
        super().__init__()
        self.image_size = image_size
        if spatial is None:
            spatial = baseline_decoder_spatial(image_size)
        self.spatial = spatial
        self.fc_block = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(True),
            nn.Linear(256, 128 * spatial * spatial),
            nn.ReLU(True),
        )
        self.deconv_block = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, in_channels, 4, 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, z):
        x = self.fc_block(z)
        x = x.view(-1, 128, self.spatial, self.spatial)
        x = self.deconv_block(x)
        if x.shape[-1] != self.image_size:
            x = x[:, :, : self.image_size, : self.image_size]
        return x

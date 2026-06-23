"""Encoder backbone selection (lightweight CNN vs ResNet-18 for CIFAR-scale inputs)."""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import resnet18

from .image_cnn import BaseEncoderCNN, EncoderCNN, encoder_spatial_size

SUPPORTED_BACKBONES = ("cnn", "resnet18")

_RESNET_IMAGE_SIZE = 32
_RESNET_SPATIAL = 4
_RESNET_CHANNELS = 512


class ResNet18Body(nn.Module):
    """ResNet-18 feature trunk adapted for 32×32 inputs (CIFAR-style)."""

    def __init__(self, in_channels: int = 3):
        super().__init__()
        m = resnet18(weights=None)
        m.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        m.maxpool = nn.Identity()
        self.body = nn.Sequential(
            m.conv1,
            m.bn1,
            m.relu,
            m.layer1,
            m.layer2,
            m.layer3,
            m.layer4,
        )
        self.out_channels = _RESNET_CHANNELS
        self.spatial = _RESNET_SPATIAL

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class ResNet18EncoderCSWAE(nn.Module):
    """CS-WAE spherical encoder with ResNet-18 trunk."""

    def __init__(self, latent_dim: int, in_channels: int = 3, image_size: int = 32):
        super().__init__()
        if image_size != _RESNET_IMAGE_SIZE:
            raise ValueError(
                f"ResNet18 backbone requires image_size={_RESNET_IMAGE_SIZE}, got {image_size}"
            )
        self.features = ResNet18Body(in_channels)
        flat = self.features.out_channels * self.features.spatial**2
        self.fc_block = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat, 256),
            nn.ReLU(True),
        )
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_s = nn.Linear(256, 1)

    def forward(self, x: torch.Tensor):
        x = self.features(x)
        x = self.fc_block(x)
        return self.fc_mu(x), self.fc_s(x)


class ResNet18EncoderAblation(nn.Module):
    """Ablation encoder (spherical or Euclidean) with ResNet-18 trunk."""

    def __init__(
        self,
        latent_dim: int,
        output_type: str = "spherical",
        in_channels: int = 3,
        image_size: int = 32,
    ):
        super().__init__()
        if image_size != _RESNET_IMAGE_SIZE:
            raise ValueError(
                f"ResNet18 backbone requires image_size={_RESNET_IMAGE_SIZE}, got {image_size}"
            )
        self.output_type = output_type
        self.features = ResNet18Body(in_channels)
        flat = self.features.out_channels * self.features.spatial**2
        self.fc_block = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat, 256),
            nn.ReLU(True),
        )
        self.fc_mu = nn.Linear(256, latent_dim)
        if output_type == "spherical":
            self.fc_s = nn.Linear(256, 1)
        else:
            self.fc_logvar = nn.Linear(256, latent_dim)

    def forward(self, x: torch.Tensor):
        x = self.features(x)
        x = self.fc_block(x)
        if self.output_type == "spherical":
            return self.fc_mu(x), self.fc_s(x)
        return self.fc_mu(x), self.fc_logvar(x)


class ResNet18EncoderBase(nn.Module):
    """Baseline VAE / WAE / VaDE encoder with ResNet-18 trunk."""

    def __init__(
        self,
        latent_dim: int,
        in_channels: int = 3,
        image_size: int = 32,
        vae_mode: bool = False,
    ):
        super().__init__()
        if image_size != _RESNET_IMAGE_SIZE:
            raise ValueError(
                f"ResNet18 backbone requires image_size={_RESNET_IMAGE_SIZE}, got {image_size}"
            )
        self.vae_mode = vae_mode
        self.features = ResNet18Body(in_channels)
        flat = self.features.out_channels * self.features.spatial**2
        self.fc_block = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat, 256),
            nn.ReLU(True),
        )
        self.fc_out1 = nn.Linear(256, latent_dim)
        if vae_mode:
            self.fc_out2 = nn.Linear(256, latent_dim)

    def forward(self, x: torch.Tensor):
        x = self.features(x)
        x = self.fc_block(x)
        out1 = self.fc_out1(x)
        if self.vae_mode:
            return out1, self.fc_out2(x)
        return out1


def normalize_backbone(backbone: str) -> str:
    backbone = (backbone or "cnn").lower()
    if backbone not in SUPPORTED_BACKBONES:
        raise ValueError(f"Unsupported backbone: {backbone}. Choose from {SUPPORTED_BACKBONES}")
    return backbone


def encoder_spatial_for_backbone(backbone: str, image_size: int, style: str = "cs_wae") -> int:
    backbone = normalize_backbone(backbone)
    if backbone == "resnet18":
        if image_size != _RESNET_IMAGE_SIZE:
            raise ValueError(f"ResNet18 only supports image_size={_RESNET_IMAGE_SIZE}")
        return _RESNET_SPATIAL
    return encoder_spatial_size(image_size, style)


def build_cs_wae_encoder(
    backbone: str,
    latent_dim: int,
    in_channels: int = 1,
    image_size: int = 28,
    output_type: str = "spherical",
) -> nn.Module:
    """Build CS-WAE / ablation encoder."""
    backbone = normalize_backbone(backbone)
    if backbone == "resnet18":
        if output_type == "spherical":
            return ResNet18EncoderCSWAE(latent_dim, in_channels, image_size)
        return ResNet18EncoderAblation(
            latent_dim, output_type=output_type, in_channels=in_channels, image_size=image_size
        )

    if output_type == "spherical":
        return EncoderCNN(latent_dim, in_channels, image_size)

    from .cs_wae_ablation import EncoderCNN_Ablation

    return EncoderCNN_Ablation(
        latent_dim, output_type=output_type, in_channels=in_channels, image_size=image_size
    )


def build_baseline_encoder(
    backbone: str,
    latent_dim: int,
    in_channels: int = 1,
    image_size: int = 28,
    vae_mode: bool = False,
) -> nn.Module:
    backbone = normalize_backbone(backbone)
    if backbone == "resnet18":
        return ResNet18EncoderBase(latent_dim, in_channels, image_size, vae_mode=vae_mode)
    return BaseEncoderCNN(latent_dim, in_channels, image_size, vae_mode=vae_mode)

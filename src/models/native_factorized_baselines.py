"""Compact native DRIT and DIVA models for the common Rotated-MNIST audit.

These implementations preserve the latent factorization and defining losses
of the original model families while using a shared lightweight 28x28
backbone.  They are not generic VAEs with a post-hoc latent split.

Primary sources:
  DRIT: Lee et al., ECCV 2018; official HsinYingLee/DRIT repository.
  DIVA: Ilse et al., MIDL 2020; official AMLab-Amsterdam/DIVA repository.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


def _sample_diagonal_gaussian(
    mean: torch.Tensor,
    logvar: torch.Tensor,
    *,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    std = torch.exp(0.5 * logvar.clamp(-10.0, 10.0))
    if generator is None:
        noise = torch.randn_like(std)
    else:
        generator_device = torch.device(getattr(generator, "device", "cpu"))
        noise = torch.randn(
            std.shape,
            generator=generator,
            dtype=std.dtype,
            device=generator_device,
        ).to(std.device)
    return mean + std * noise, std


def diagonal_gaussian_kl(
    q_mean: torch.Tensor,
    q_logvar: torch.Tensor,
    p_mean: torch.Tensor | None = None,
    p_logvar: torch.Tensor | None = None,
) -> torch.Tensor:
    """Mean batch KL for diagonal Gaussian distributions."""

    if p_mean is None:
        p_mean = torch.zeros_like(q_mean)
    if p_logvar is None:
        p_logvar = torch.zeros_like(q_logvar)
    q_logvar = q_logvar.clamp(-10.0, 10.0)
    p_logvar = p_logvar.clamp(-10.0, 10.0)
    variance_ratio = torch.exp(q_logvar - p_logvar)
    mean_term = (q_mean - p_mean).square() * torch.exp(-p_logvar)
    return 0.5 * (
        p_logvar - q_logvar + variance_ratio + mean_term - 1.0
    ).sum(dim=1).mean()


class _ConvStem(nn.Module):
    def __init__(self, channels: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1),
            nn.InstanceNorm2d(32, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, channels, 4, 2, 1),
            nn.InstanceNorm2d(channels, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.net(images)


class _VectorDecoder(nn.Module):
    def __init__(self, latent_dim: int):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 128 * 7 * 7),
            nn.ReLU(inplace=True),
        )
        self.net = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.InstanceNorm2d(64, affine=True),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.InstanceNorm2d(32, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 3, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.net(self.fc(latent).view(-1, 128, 7, 7))


class _ImageDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.InstanceNorm2d(64, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.InstanceNorm2d(128, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, 1),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.net(images).squeeze(1)


class DRITContentEncoder(nn.Module):
    """Domain-specific stems followed by a shared content bottleneck."""

    def __init__(self, content_dim: int):
        super().__init__()
        self.stems = nn.ModuleList([_ConvStem(), _ConvStem()])
        self.shared = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.InstanceNorm2d(128, affine=True),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, content_dim),
        )

    def forward(self, images: torch.Tensor, domain: int) -> torch.Tensor:
        return self.shared(self.stems[int(domain)](images))


class DRITAttributeEncoder(nn.Module):
    """Domain-specific Gaussian attribute encoders, as in DRIT concat mode."""

    def __init__(self, style_dim: int):
        super().__init__()
        self.stems = nn.ModuleList([_ConvStem(), _ConvStem()])
        self.heads = nn.ModuleList(
            [
                nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(64, 2 * style_dim)),
                nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(64, 2 * style_dim)),
            ]
        )
        self.style_dim = int(style_dim)

    def forward(
        self, images: torch.Tensor, domain: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        parameters = self.heads[int(domain)](self.stems[int(domain)](images))
        mean, logvar = parameters.chunk(2, dim=1)
        return mean, logvar.clamp(-10.0, 10.0)


class NativeDRIT(nn.Module):
    """Two-domain DRIT family model with native content/attribute blocks."""

    model_family = "DRIT"
    native_factorization = True

    def __init__(self, content_dim: int = 64, style_dim: int = 8):
        super().__init__()
        self.content_dim = int(content_dim)
        self.style_dim = int(style_dim)
        self.content_encoder = DRITContentEncoder(content_dim)
        self.attribute_encoder = DRITAttributeEncoder(style_dim)
        self.decoders = nn.ModuleList(
            [_VectorDecoder(content_dim + style_dim) for _ in range(2)]
        )
        self.image_discriminators = nn.ModuleList(
            [_ImageDiscriminator(), _ImageDiscriminator()]
        )
        self.content_discriminator = nn.Sequential(
            nn.Linear(content_dim, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 1),
        )

    def encode_content(self, images: torch.Tensor, domain: int) -> torch.Tensor:
        return self.content_encoder(images, domain)

    def encode_style(
        self,
        images: torch.Tensor,
        domain: int,
        *,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, logvar = self.attribute_encoder(images, domain)
        sample, std = _sample_diagonal_gaussian(mean, logvar, generator=generator)
        return mean, logvar, sample, std

    def decode(
        self, content: torch.Tensor, style: torch.Tensor, domain: int
    ) -> torch.Tensor:
        return self.decoders[int(domain)](torch.cat([content, style], dim=1))

    def reconstruct(
        self,
        images: torch.Tensor,
        domain: int,
        *,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        content = self.encode_content(images, domain)
        mean, logvar, style, std = self.encode_style(
            images, domain, generator=generator
        )
        reconstruction = self.decode(content, style, domain)
        return reconstruction, {
            "content": content,
            "style_mean": mean,
            "style_logvar": logvar,
            "style_sample": style,
            "style_std": std,
        }


class _DIVAEncoder(nn.Module):
    def __init__(self, latent_dim: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 5, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Flatten(),
        )
        self.mean = nn.Linear(1024, latent_dim)
        self.logvar = nn.Linear(1024, latent_dim)

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.features(images)
        return self.mean(features), self.logvar(features).clamp(-10.0, 10.0)


class _ConditionalGaussianPrior(nn.Module):
    def __init__(self, n_conditions: int, latent_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_conditions, latent_dim),
            nn.ReLU(inplace=True),
        )
        self.mean = nn.Linear(latent_dim, latent_dim)
        self.logvar = nn.Linear(latent_dim, latent_dim)
        self.n_conditions = int(n_conditions)

    def forward(self, conditions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if conditions.ndim == 1:
            conditions = F.one_hot(
                conditions.long(), num_classes=self.n_conditions
            ).float()
        hidden = self.net(conditions)
        return self.mean(hidden), self.logvar(hidden).clamp(-10.0, 10.0)


@dataclass
class DIVAViews:
    zd_mean: torch.Tensor
    zd_logvar: torch.Tensor
    zd_sample: torch.Tensor
    zx_mean: torch.Tensor
    zx_logvar: torch.Tensor
    zx_sample: torch.Tensor
    zy_mean: torch.Tensor
    zy_logvar: torch.Tensor
    zy_sample: torch.Tensor


class NativeDIVA(nn.Module):
    """DIVA with native domain, residual and class-specific latent spaces."""

    model_family = "DIVA"
    native_factorization = True

    def __init__(
        self,
        domain_dim: int = 32,
        style_dim: int = 32,
        semantic_dim: int = 32,
        n_domains: int = 2,
        n_classes: int = 10,
    ):
        super().__init__()
        self.domain_dim = int(domain_dim)
        self.style_dim = int(style_dim)
        self.semantic_dim = int(semantic_dim)
        self.n_domains = int(n_domains)
        self.n_classes = int(n_classes)
        self.qzd = _DIVAEncoder(domain_dim)
        self.qzx = _DIVAEncoder(style_dim)
        self.qzy = _DIVAEncoder(semantic_dim)
        self.pzd = _ConditionalGaussianPrior(n_domains, domain_dim)
        self.pzy = _ConditionalGaussianPrior(n_classes, semantic_dim)
        self.decoder = _VectorDecoder(domain_dim + style_dim + semantic_dim)
        self.domain_classifier = nn.Linear(domain_dim, n_domains)
        self.label_classifier = nn.Linear(semantic_dim, n_classes)

    def encode_views(
        self,
        images: torch.Tensor,
        *,
        generator: torch.Generator | None = None,
    ) -> DIVAViews:
        zd_mean, zd_logvar = self.qzd(images)
        zx_mean, zx_logvar = self.qzx(images)
        zy_mean, zy_logvar = self.qzy(images)
        zd_sample, _ = _sample_diagonal_gaussian(
            zd_mean, zd_logvar, generator=generator
        )
        zx_sample, _ = _sample_diagonal_gaussian(
            zx_mean, zx_logvar, generator=generator
        )
        zy_sample, _ = _sample_diagonal_gaussian(
            zy_mean, zy_logvar, generator=generator
        )
        return DIVAViews(
            zd_mean=zd_mean,
            zd_logvar=zd_logvar,
            zd_sample=zd_sample,
            zx_mean=zx_mean,
            zx_logvar=zx_logvar,
            zx_sample=zx_sample,
            zy_mean=zy_mean,
            zy_logvar=zy_logvar,
            zy_sample=zy_sample,
        )

    def decode(
        self,
        zd: torch.Tensor,
        zx: torch.Tensor,
        zy: torch.Tensor,
    ) -> torch.Tensor:
        return self.decoder(torch.cat([zd, zx, zy], dim=1))

    def forward(
        self,
        images: torch.Tensor,
        *,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, DIVAViews]:
        views = self.encode_views(images, generator=generator)
        reconstruction = self.decode(
            views.zd_sample, views.zx_sample, views.zy_sample
        )
        return reconstruction, views

    def conditional_domain_prior(
        self, domain: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.pzd(domain)

    def conditional_semantic_prior(
        self, labels: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.pzy(labels)


__all__ = [
    "DIVAViews",
    "NativeDIVA",
    "NativeDRIT",
    "diagonal_gaussian_kl",
]

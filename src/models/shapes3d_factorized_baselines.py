"""Native factorized Shapes3D baselines for the ground-truth factor audit."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import ResNet18Body
from .f_cs_wae import ResBlockDecoder, sample_style_posterior


def _diagonal_kl_mean(
    mean: torch.Tensor,
    logvar: torch.Tensor,
    prior_mean: torch.Tensor | None = None,
) -> torch.Tensor:
    """KL averaged over batch and latent coordinates for stable loss scaling."""

    logvar = logvar.clamp(-10.0, 10.0)
    centered = mean if prior_mean is None else mean - prior_mean
    return 0.5 * (centered.square() + logvar.exp() - 1.0 - logvar).mean()


class _GaussianHead(nn.Module):
    def __init__(self, hidden_dim: int, latent_dim: int):
        super().__init__()
        self.mean = nn.Linear(hidden_dim, latent_dim)
        self.logvar = nn.Linear(hidden_dim, latent_dim)
        nn.init.zeros_(self.logvar.weight)
        nn.init.zeros_(self.logvar.bias)

    def forward(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.mean(hidden), self.logvar(hidden).clamp(-10.0, 10.0)


class Shapes3DConditionalVAE(nn.Module):
    """cVAE with observed shape content and a Gaussian residual style code."""

    model_family = "Conditional VAE"
    native_factorization = True

    def __init__(
        self,
        *,
        content_dim: int = 64,
        style_dim: int = 64,
        n_classes: int = 4,
        in_channels: int = 3,
        image_size: int = 64,
    ) -> None:
        super().__init__()
        self.content_dim = int(content_dim)
        self.style_dim = int(style_dim)
        self.n_classes = int(n_classes)
        self.features = ResNet18Body(in_channels)
        flat = self.features.out_channels * self.features.spatial**2
        self.hidden = nn.Sequential(
            nn.Flatten(), nn.Linear(flat, 256), nn.SiLU()
        )
        self.style_head = _GaussianHead(256, style_dim)
        self.content_embedding = nn.Embedding(n_classes, content_dim)
        self.decoder = ResBlockDecoder(
            content_dim, style_dim, in_channels, image_size
        )

    def encode_style(self, images: torch.Tensor):
        return self.style_head(self.hidden(self.features(images)))

    def content_from_labels(self, labels: torch.Tensor) -> torch.Tensor:
        return self.content_embedding(labels.long())

    def encode_views(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        *,
        generator: torch.Generator | None = None,
    ):
        mean, logvar = self.encode_style(images)
        sample, std = sample_style_posterior(mean, logvar, generator=generator)
        content = self.content_from_labels(labels)
        return content, mean, logvar, sample, std

    def decode(self, content: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        return self.decoder(torch.cat([content, style], dim=1))

    def forward(self, images: torch.Tensor, labels: torch.Tensor):
        content, mean, logvar, style, _std = self.encode_views(images, labels)
        return self.decode(content, style), mean, logvar

    def loss_terms(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        *,
        kl_weight: float,
    ) -> dict[str, torch.Tensor]:
        reconstruction, mean, logvar = self(images, labels)
        rec = F.l1_loss(reconstruction, images)
        style_kl = _diagonal_kl_mean(mean, logvar)
        return {
            "total": rec + float(kl_weight) * style_kl,
            "reconstruction": rec,
            "style_kl": style_kl,
            "shape_ce": torch.zeros((), device=images.device),
            "content_kl": torch.zeros((), device=images.device),
        }


@dataclass(frozen=True)
class ContentStyleViews:
    content_mean: torch.Tensor
    content_logvar: torch.Tensor
    content_sample: torch.Tensor
    style_mean: torch.Tensor
    style_logvar: torch.Tensor
    style_sample: torch.Tensor
    style_std: torch.Tensor


class Shapes3DContentStyleVAE(nn.Module):
    """DIVA-type VAE with learned semantic and residual Gaussian subspaces."""

    model_family = "Supervised content-style VAE"
    native_factorization = True

    def __init__(
        self,
        *,
        content_dim: int = 64,
        style_dim: int = 64,
        n_classes: int = 4,
        in_channels: int = 3,
        image_size: int = 64,
    ) -> None:
        super().__init__()
        self.content_dim = int(content_dim)
        self.style_dim = int(style_dim)
        self.n_classes = int(n_classes)
        self.features = ResNet18Body(in_channels)
        flat = self.features.out_channels * self.features.spatial**2
        self.hidden = nn.Sequential(
            nn.Flatten(), nn.Linear(flat, 256), nn.SiLU()
        )
        self.content_head = _GaussianHead(256, content_dim)
        self.style_head = _GaussianHead(256, style_dim)
        self.content_prior_mean = nn.Embedding(n_classes, content_dim)
        self.shape_classifier = nn.Linear(content_dim, n_classes)
        self.decoder = ResBlockDecoder(
            content_dim, style_dim, in_channels, image_size
        )

    def encode_views(
        self,
        images: torch.Tensor,
        *,
        generator: torch.Generator | None = None,
    ) -> ContentStyleViews:
        hidden = self.hidden(self.features(images))
        content_mean, content_logvar = self.content_head(hidden)
        style_mean, style_logvar = self.style_head(hidden)
        content_sample, _ = sample_style_posterior(
            content_mean, content_logvar, generator=generator
        )
        style_sample, style_std = sample_style_posterior(
            style_mean, style_logvar, generator=generator
        )
        return ContentStyleViews(
            content_mean=content_mean,
            content_logvar=content_logvar,
            content_sample=content_sample,
            style_mean=style_mean,
            style_logvar=style_logvar,
            style_sample=style_sample,
            style_std=style_std,
        )

    def decode(self, content: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        return self.decoder(torch.cat([content, style], dim=1))

    def forward(self, images: torch.Tensor):
        views = self.encode_views(images)
        return self.decode(views.content_sample, views.style_sample), views

    def loss_terms(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        *,
        kl_weight: float,
        classifier_weight: float,
    ) -> dict[str, torch.Tensor]:
        reconstruction, views = self(images)
        rec = F.l1_loss(reconstruction, images)
        prior_mean = self.content_prior_mean(labels.long())
        content_kl = _diagonal_kl_mean(
            views.content_mean, views.content_logvar, prior_mean
        )
        style_kl = _diagonal_kl_mean(views.style_mean, views.style_logvar)
        shape_ce = F.cross_entropy(
            self.shape_classifier(views.content_mean), labels.long()
        )
        total = (
            rec
            + float(kl_weight) * (content_kl + style_kl)
            + float(classifier_weight) * shape_ce
        )
        return {
            "total": total,
            "reconstruction": rec,
            "style_kl": style_kl,
            "content_kl": content_kl,
            "shape_ce": shape_ce,
        }


__all__ = [
    "ContentStyleViews",
    "Shapes3DConditionalVAE",
    "Shapes3DContentStyleVAE",
]

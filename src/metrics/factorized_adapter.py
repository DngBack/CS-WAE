"""Model adapters for the canonical factorized-latent audit.

The audit must use latent blocks defined by the model itself.  This module
provides a small interface that prevents diagnostic code from inferring a
"style" block by arbitrarily splitting a generic latent vector.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from ..models.f_cs_wae import effective_style_variance, sample_style_posterior
from ..utils.utils import mobius_reparam


@dataclass(frozen=True)
class FactorizedLatentViews:
    """Named posterior views emitted by a native factorized model."""

    content_mean: torch.Tensor
    content_sample: torch.Tensor
    style_mean: torch.Tensor
    style_sample: torch.Tensor
    style_logvar: torch.Tensor | None = None
    style_effective_std: torch.Tensor | None = None


class FactorizedAuditAdapter(ABC):
    """Required interface for a model included in cross-model audit tables."""

    def __init__(self, model: torch.nn.Module):
        self.model = model

    @abstractmethod
    def encode_views(
        self,
        images: torch.Tensor,
        *,
        generator: torch.Generator,
        domain: torch.Tensor | None = None,
    ) -> FactorizedLatentViews:
        """Return native content/style means and stochastic samples."""

    @abstractmethod
    def sample_style_prior(
        self,
        n_samples: int,
        *,
        generator: torch.Generator,
        device: torch.device,
        dtype: torch.dtype,
        domain: int | None = None,
    ) -> torch.Tensor:
        """Draw from the style prior used by the model's sampler."""

    @abstractmethod
    def decode(
        self,
        content: torch.Tensor,
        style: torch.Tensor,
        *,
        domain: torch.Tensor | int | None = None,
    ) -> torch.Tensor:
        """Decode a content/style recombination."""

    @abstractmethod
    def metadata(self) -> dict[str, Any]:
        """Describe the native factorization and sampling contract."""


class FCSWAEAuditAdapter(FactorizedAuditAdapter):
    """Canonical adapter for :class:`src.models.f_cs_wae.FCSWAE`."""

    @torch.no_grad()
    def encode_views(
        self,
        images: torch.Tensor,
        *,
        generator: torch.Generator,
        domain: torch.Tensor | None = None,
    ) -> FactorizedLatentViews:
        del domain
        mu_c, rho_c, mu_s, logvar_s = self.model.encode(images)

        generator_device = torch.device(getattr(generator, "device", "cpu"))
        epsilon_c = torch.randn(
            mu_c.shape,
            generator=generator,
            dtype=mu_c.dtype,
            device=generator_device,
        ).to(mu_c.device)
        epsilon_c = F.normalize(epsilon_c, p=2, dim=1)
        z_c = mobius_reparam(epsilon_c, mu_c, rho_c)
        z_s, std_s = self.model.sample_style(mu_s, logvar_s, generator=generator)
        return FactorizedLatentViews(
            content_mean=mu_c,
            content_sample=z_c,
            style_mean=mu_s,
            style_sample=z_s,
            style_logvar=logvar_s,
            style_effective_std=std_s,
        )

    def sample_style_prior(
        self,
        n_samples: int,
        *,
        generator: torch.Generator,
        device: torch.device,
        dtype: torch.dtype,
        domain: int | None = None,
    ) -> torch.Tensor:
        del domain
        generator_device = torch.device(getattr(generator, "device", "cpu"))
        return torch.randn(
            (n_samples, self.model.style_dim),
            generator=generator,
            dtype=dtype,
            device=generator_device,
        ).to(device)

    def decode(
        self,
        content: torch.Tensor,
        style: torch.Tensor,
        *,
        domain: torch.Tensor | int | None = None,
    ) -> torch.Tensor:
        del domain
        return self.model.decoder(torch.cat([content, style], dim=1))

    def metadata(self) -> dict[str, Any]:
        return {
            "model_family": "F-CS-WAE",
            "native_factorization": True,
            "content_view": "hyperspherical semantic z_c",
            "style_view": "Euclidean Gaussian z_s",
            "style_prior": "N(0, I)",
            "style_sigma_floor": float(self.model.style_sigma_floor),
            "sampler": "z_c ~ p(z_c|y), z_s ~ N(0,I), independently",
        }


class FCSWAEAblationAuditAdapter(FactorizedAuditAdapter):
    """Native adapter for style-bearing F-CS-WAE ablation checkpoints."""

    @torch.no_grad()
    def encode_views(
        self,
        images: torch.Tensor,
        *,
        generator: torch.Generator,
        domain: torch.Tensor | None = None,
    ) -> FactorizedLatentViews:
        del domain
        if not self.model.use_style_latent:
            raise ValueError("The no_z_s ablation has no style block to audit")
        mu_c_raw, parameter_c, mu_s, logvar_s = self.model.encoder(images)
        if self.model.z_c_type == "spherical":
            generator_device = torch.device(getattr(generator, "device", "cpu"))
            epsilon_c = torch.randn(
                mu_c_raw.shape,
                generator=generator,
                dtype=mu_c_raw.dtype,
                device=generator_device,
            ).to(mu_c_raw.device)
            mu_c = F.normalize(mu_c_raw, p=2, dim=1)
            rho_c = torch.sigmoid(parameter_c).squeeze(-1) * (1 - 1e-8)
            z_c = mobius_reparam(F.normalize(epsilon_c, p=2, dim=1), mu_c, rho_c)
        else:
            mu_c = mu_c_raw
            z_c, _ = sample_style_posterior(
                mu_c, parameter_c, generator=generator
            )
        z_s, std_s = sample_style_posterior(
            mu_s,
            logvar_s,
            sigma_floor=float(getattr(self.model, "style_sigma_floor", 0.0)),
            generator=generator,
        )
        return FactorizedLatentViews(
            content_mean=mu_c,
            content_sample=z_c,
            style_mean=mu_s,
            style_sample=z_s,
            style_logvar=logvar_s,
            style_effective_std=std_s,
        )

    def sample_style_prior(
        self,
        n_samples: int,
        *,
        generator: torch.Generator,
        device: torch.device,
        dtype: torch.dtype,
        domain: int | None = None,
    ) -> torch.Tensor:
        del domain
        generator_device = torch.device(getattr(generator, "device", "cpu"))
        return torch.randn(
            (n_samples, self.model.style_dim),
            generator=generator,
            dtype=dtype,
            device=generator_device,
        ).to(device)

    def decode(
        self,
        content: torch.Tensor,
        style: torch.Tensor,
        *,
        domain: torch.Tensor | int | None = None,
    ) -> torch.Tensor:
        del domain
        return self.model.decoder(torch.cat([content, style], dim=1))

    def metadata(self) -> dict[str, Any]:
        return {
            "model_family": "F-CS-WAE ablation",
            "variant": self.model.variant_name,
            "native_factorization": bool(self.model.use_style_latent),
            "content_view": f"{self.model.z_c_type} z_c",
            "style_view": "Euclidean Gaussian z_s",
            "style_prior": "N(0, I)",
            "style_sigma_floor": float(getattr(self.model, "style_sigma_floor", 0.0)),
        }


def build_factorized_audit_adapter(model: torch.nn.Module) -> FactorizedAuditAdapter:
    """Return a registered native adapter or fail instead of guessing a split."""

    if hasattr(model, "variant_config") and hasattr(model, "use_style_latent"):
        return FCSWAEAblationAuditAdapter(model)
    required = ("encode", "sample_style", "style_sigma_floor", "style_dim", "decoder")
    if all(hasattr(model, name) for name in required):
        return FCSWAEAuditAdapter(model)
    raise TypeError(
        f"No native factorized audit adapter is registered for {type(model).__name__}. "
        "Do not infer style by splitting a generic latent vector."
    )


def summarize_style_posterior(
    logvar_s: torch.Tensor,
    sigma_floor: float,
) -> dict[str, float]:
    """Summarize raw collapse and effective stochasticity for manifests."""

    variance = effective_style_variance(logvar_s, sigma_floor)
    std = variance.sqrt()
    entropy_per_example = 0.5 * (
        1.0
        + torch.log(torch.tensor(2.0 * torch.pi, dtype=logvar_s.dtype, device=logvar_s.device))
        + torch.log(variance)
    ).sum(dim=1)
    return {
        "configured_sigma_floor": float(sigma_floor),
        "raw_logvar_at_or_below_minus10_fraction": float((logvar_s <= -10.0).float().mean().item()),
        "effective_std_mean": float(std.mean().item()),
        "effective_std_median": float(std.median().item()),
        "effective_noise_rms_norm_mean": float(
            variance.sum(dim=1).sqrt().mean().item()
        ),
        "entropy_nats_mean": float(entropy_per_example.mean().item()),
        "entropy_nats_per_dimension_mean": float(
            (entropy_per_example / logvar_s.shape[1]).mean().item()
        ),
    }

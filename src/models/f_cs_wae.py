"""
F-CS-WAE: Factorized Class-Structured Spherical Cauchy Wasserstein Auto-Encoder.

Latent space is split into:
  z_c in S^{d_c-1}  — semantic/class latent on hypersphere
  z_s in R^{d_s}    — style latent in Euclidean space

Encoder: shared ResNet-18 backbone → semantic head (mu_c, rho_c) + style head (mu_s, logvar_s)
Decoder: concat(z_c, z_s) → native ResBlock decoder (4×4 → 32--256 px)
Priors:  z_c ~ SphericalCauchy(m_k, rho_p) per class k (EMA centers, no gradient)
         z_s ~ N(0, I)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config_f_cs_wae import f_cs_wae_config as cfg
from ..utils.utils import mobius_reparam, sample_uniform_sphere
from .backbone import ResNet18Body


def effective_style_variance(
    logvar_s: torch.Tensor,
    sigma_floor: float = 0.0,
) -> torch.Tensor:
    """Return the effective diagonal style-posterior variance.

    ``sigma_floor`` is an explicit controlled-noise intervention.  It is
    added in variance space so the historical model is recovered exactly at
    zero and the requested floor remains meaningful when the raw log-variance
    head collapses below the numerical clamp.
    """

    if sigma_floor < 0.0:
        raise ValueError("style sigma floor must be non-negative")
    variance = torch.exp(logvar_s.clamp(-10, 10))
    if sigma_floor:
        variance = variance + float(sigma_floor) ** 2
    return variance


def style_posterior_entropy(
    logvar_s: torch.Tensor,
    sigma_floor: float = 0.0,
) -> torch.Tensor:
    """Differential entropy in nats for each diagonal Gaussian posterior."""

    variance = effective_style_variance(logvar_s, sigma_floor)
    log_two_pi_e = 1.0 + torch.log(torch.tensor(2.0 * torch.pi, device=logvar_s.device, dtype=logvar_s.dtype))
    return 0.5 * (log_two_pi_e + torch.log(variance)).sum(dim=1)


def sample_style_posterior(
    mu_s: torch.Tensor,
    logvar_s: torch.Tensor,
    sigma_floor: float = 0.0,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample style with one canonical reparameterization implementation.

    A supplied generator may live on CPU while the posterior lives on an
    accelerator.  Drawing on the generator's device and moving the noise is
    deliberate: it gives reproducible audit samples independent of CUDA RNG
    state while training can keep using the efficient ``generator=None`` path.
    Returns both the sample and effective standard deviation.
    """

    std_s = effective_style_variance(logvar_s, sigma_floor).sqrt()
    if generator is None:
        epsilon = torch.randn_like(std_s)
    else:
        generator_device = torch.device(getattr(generator, "device", "cpu"))
        epsilon = torch.randn(
            std_s.shape,
            generator=generator,
            dtype=std_s.dtype,
            device=generator_device,
        ).to(std_s.device)
    return mu_s + std_s * epsilon, std_s


# ---------------------------------------------------------------------------
# ResBlock building blocks
# ---------------------------------------------------------------------------

class ResBlock(nn.Module):
    """Residual block with GroupNorm + SiLU activations."""

    def __init__(self, channels: int, num_groups: int = 8):
        super().__init__()
        # Clamp num_groups so it always divides channels
        num_groups = min(num_groups, channels)
        while channels % num_groups != 0 and num_groups > 1:
            num_groups -= 1
        self.net = nn.Sequential(
            nn.GroupNorm(num_groups, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class ResBlockUp(nn.Module):
    """Upsampling residual block: 2× bilinear upsample + channel projection + ResBlock."""

    def __init__(self, in_channels: int, out_channels: int, num_groups: int = 8):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.res = ResBlock(out_channels, num_groups)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        x = self.proj(x)
        return self.res(x)


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------

class ResBlockDecoder(nn.Module):
    """
    ResBlock-based decoder.

    Input:  latent vector of size (semantic_dim + style_dim)
    Output: image of shape (in_channels, image_size, image_size)

    Architecture:
        Linear → reshape (512, 4, 4)
        ResBlockUp 512 → 256   (4 → 8)
        ResBlockUp 256 → 128   (8 → 16)
        ResBlockUp 128 → 64    (16 → 32)
        optional ResBlockUp 64 → 32 (32 → 64)
        optional ResBlockUp 32 → 16 (64 → 128)
        optional ResBlockUp 16 → 16 (128 → 256)
        GN + SiLU → Conv2d → in_channels → Sigmoid
    """

    def __init__(
        self,
        semantic_dim: int,
        style_dim: int,
        in_channels: int = 3,
        image_size: int = 32,
    ):
        super().__init__()
        if image_size not in (28, 32, 64, 128, 256):
            raise ValueError(
                "F-CS-WAE decoder supports image_size in {28,32,64,128,256}"
            )
        self.image_size = image_size
        latent_dim = semantic_dim + style_dim

        self.fc = nn.Linear(latent_dim, 512 * 4 * 4)
        up_blocks = [
            ResBlockUp(512, 256),
            ResBlockUp(256, 128),
            ResBlockUp(128, 64),
        ]
        output_channels = 64
        if image_size == 64:
            up_blocks.append(ResBlockUp(64, 32))
            output_channels = 32
        elif image_size == 128:
            up_blocks.extend([ResBlockUp(64, 32), ResBlockUp(32, 16)])
            output_channels = 16
        elif image_size == 256:
            up_blocks.extend(
                [
                    ResBlockUp(64, 32),
                    ResBlockUp(32, 16),
                    ResBlockUp(16, 16),
                ]
            )
            output_channels = 16
        self.ups = nn.Sequential(*up_blocks)
        self.out = nn.Sequential(
            nn.GroupNorm(8, output_channels),
            nn.SiLU(),
            nn.Conv2d(output_channels, in_channels, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.fc(z)
        x = x.view(-1, 512, 4, 4)
        x = self.ups(x)
        x = self.out(x)
        # Only the legacy 28x28 path crops/interpolates the native 32x32
        # decoder. Shapes3D 64x64 always uses the fourth native upsampling block.
        if x.shape[-1] != self.image_size:
            x = F.interpolate(x, size=(self.image_size, self.image_size),
                              mode="bilinear", align_corners=False)
        return x


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

class FCSWAEEncoder(nn.Module):
    """
    Shared ResNet-18 backbone with two output heads:
      Semantic head: (mu_c_raw, rho_c_raw) → used to compute spherical distribution
      Style head:    (mu_s, logvar_s)       → Gaussian distribution parameters
    """

    def __init__(
        self,
        semantic_dim: int,
        style_dim: int,
        in_channels: int = 3,
        image_size: int = 32,
    ):
        super().__init__()
        self.features = ResNet18Body(in_channels)
        flat = self.features.out_channels * self.features.spatial ** 2

        self.fc_block = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat, 256),
            nn.SiLU(),
        )

        # Semantic head
        self.fc_mu_c = nn.Linear(256, semantic_dim)
        self.fc_rho_c = nn.Linear(256, 1)

        # Style head
        self.fc_mu_s = nn.Linear(256, style_dim)
        self.fc_logvar_s = nn.Linear(256, style_dim)
        # Init logvar head to near-zero so std_s ≈ 1 at the start
        nn.init.zeros_(self.fc_logvar_s.weight)
        nn.init.zeros_(self.fc_logvar_s.bias)

    def forward(self, x: torch.Tensor):
        h = self.features(x)
        h = self.fc_block(h)
        mu_c_raw = self.fc_mu_c(h)
        rho_c_raw = self.fc_rho_c(h)
        mu_s = self.fc_mu_s(h)
        logvar_s = self.fc_logvar_s(h)
        return mu_c_raw, rho_c_raw, mu_s, logvar_s


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class FCSWAE(nn.Module):
    """
    F-CS-WAE: Factorized Class-Structured Spherical Cauchy WAE.

    Attributes
    ----------
    ema_centers : buffer (K, R, d_c), L2-normalized
        Class center directions on the hypersphere.  Updated by the trainer
        via EMA after each epoch — NOT learned by gradient.
    classifier  : nn.Linear(d_c, K)
        Auxiliary head used only for L_cls (anti-collapse term).
    """

    def __init__(
        self,
        semantic_dim: int,
        style_dim: int,
        n_classes: int,
        in_channels: int = 3,
        image_size: int = 32,
        n_centers: int = 1,
        rho_prior: float = 0.7,
        ema_momentum: float = 0.95,
        style_sigma_floor: float = 0.0,
    ):
        super().__init__()
        self.semantic_dim = semantic_dim
        self.style_dim = style_dim
        self.n_classes = n_classes
        self.n_centers = n_centers
        self.rho_p = rho_prior
        self.ema_momentum = ema_momentum
        if style_sigma_floor < 0.0:
            raise ValueError("style_sigma_floor must be non-negative")
        self.style_sigma_floor = float(style_sigma_floor)

        self.encoder = FCSWAEEncoder(semantic_dim, style_dim, in_channels, image_size)
        self.decoder = ResBlockDecoder(semantic_dim, style_dim, in_channels, image_size)
        self.classifier = nn.Linear(semantic_dim, n_classes)

        # EMA class centers — stored as a buffer (no gradient, no optimizer update)
        init_centers = F.normalize(
            torch.randn(n_classes, n_centers, semantic_dim), p=2, dim=-1
        )
        self.register_buffer("ema_centers", init_centers)

    # ------------------------------------------------------------------
    # Encoding helpers
    # ------------------------------------------------------------------

    def encode(self, x: torch.Tensor):
        """Return (mu_c, rho_c, mu_s, logvar_s) from raw encoder outputs."""
        mu_c_raw, rho_c_raw, mu_s, logvar_s = self.encoder(x)
        mu_c = F.normalize(mu_c_raw, p=2, dim=1)
        rho_c = torch.sigmoid(rho_c_raw).squeeze(-1) * (1 - cfg.epsilon)
        return mu_c, rho_c, mu_s, logvar_s

    def encode_to_distribution(self, x: torch.Tensor):
        """
        Compatibility shim for ModelEvaluator clustering.
        Returns (mu_c, rho_c) — only the semantic direction is used for K-means.
        """
        mu_c, rho_c, _, _ = self.encode(x)
        return mu_c, rho_c

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor):
        """
        Returns
        -------
        x_hat     : (B, C, H, W) reconstructed image
        z_c       : (B, d_c)  semantic latent sample on S^{d_c-1}
        z_s       : (B, d_s)  style latent sample in R^{d_s}
        mu_c      : (B, d_c)  semantic direction (L2-normalized encoder mean)
        mu_s      : (B, d_s)  style mean
        logvar_s  : (B, d_s)  style log-variance
        """
        mu_c, rho_c, mu_s, logvar_s = self.encode(x)

        # Sample semantic latent via Möbius reparameterization
        eps_s = sample_uniform_sphere(x.shape[0], self.semantic_dim, device=x.device)
        z_c = mobius_reparam(eps_s, mu_c, rho_c)

        # Sample style latent through the canonical helper.  sigma_floor=0
        # exactly preserves the historical checkpoint behavior.
        z_s, _std_s = self.sample_style(mu_s, logvar_s)

        x_hat = self.decoder(torch.cat([z_c, z_s], dim=1))
        return x_hat, z_c, z_s, mu_c, mu_s, logvar_s

    def sample_style(
        self,
        mu_s: torch.Tensor,
        logvar_s: torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample the effective style posterior used by training and audit."""

        return sample_style_posterior(
            mu_s,
            logvar_s,
            sigma_floor=self.style_sigma_floor,
            generator=generator,
        )

    def style_entropy(self, logvar_s: torch.Tensor) -> torch.Tensor:
        """Per-example effective style-posterior entropy in nats."""

        return style_posterior_entropy(logvar_s, self.style_sigma_floor)

    # ------------------------------------------------------------------
    # Prior sampling
    # ------------------------------------------------------------------

    def sample_from_class_prior(
        self, class_idx: int, n_samples: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Sample (z_c, z_s) from the class-k prior.

        For R=1: sample from SphericalCauchy(ema_centers[k, 0], rho_p).
        For R>1:  uniformly pick one center per sample, then sample from it.

        Returns
        -------
        z_c : (n_samples, d_c)  on S^{d_c-1}
        z_s : (n_samples, d_s)  ~ N(0, I)
        """
        centers = self.ema_centers[class_idx]  # (R, d_c)

        if self.n_centers == 1:
            center = centers[0].unsqueeze(0).expand(n_samples, -1)  # (n, d_c)
        else:
            r_idx = torch.randint(0, self.n_centers, (n_samples,), device=device)
            center = centers[r_idx]  # (n, d_c)

        center = F.normalize(center, p=2, dim=1)
        eps_s = sample_uniform_sphere(n_samples, self.semantic_dim, device=device)
        rho = torch.full((n_samples,), self.rho_p, device=device)
        z_c = mobius_reparam(eps_s, center, rho)
        z_s = torch.randn(n_samples, self.style_dim, device=device)
        return z_c, z_s

    def decode_from_class(
        self, class_idx: int, n_samples: int, device: torch.device
    ) -> torch.Tensor:
        """Convenience: sample prior and decode to images."""
        z_c, z_s = self.sample_from_class_prior(class_idx, n_samples, device)
        return self.decoder(torch.cat([z_c, z_s], dim=1))

    def sample_from_continuous_prior(
        self,
        conditions: torch.Tensor,
        knots: torch.Tensor,
        *,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample a declared continuous conditional prior by center interpolation.

        ``knots[k]`` is the real-valued condition represented by EMA class
        center ``k``.  Adjacent centers are linearly interpolated and projected
        back to the unit sphere (normalized linear interpolation).  This makes
        the continuous sampler explicit and reproducible while retaining the
        age-bin centers used by the current trainer.  Multiple centers per bin
        are intentionally rejected because their cross-bin correspondence is
        not identified.
        """

        if self.n_centers != 1:
            raise ValueError(
                "continuous prior interpolation requires exactly one center per knot"
            )
        values = conditions.flatten().to(self.ema_centers)
        knots = knots.flatten().to(self.ema_centers)
        if knots.numel() != self.n_classes:
            raise ValueError("one strictly ordered knot is required per class center")
        if knots.numel() < 2 or not bool(torch.all(knots[1:] > knots[:-1])):
            raise ValueError("continuous-prior knots must be strictly increasing")
        clipped = values.clamp(float(knots[0]), float(knots[-1]))
        upper = torch.searchsorted(knots, clipped, right=True).clamp(1, knots.numel() - 1)
        lower = upper - 1
        denominator = (knots[upper] - knots[lower]).clamp_min(1e-8)
        weight = ((clipped - knots[lower]) / denominator).unsqueeze(1)
        centers = self.ema_centers[:, 0]
        interpolated = F.normalize(
            (1.0 - weight) * centers[lower] + weight * centers[upper],
            p=2,
            dim=1,
        )
        generator_device = (
            torch.device(getattr(generator, "device", "cpu"))
            if generator is not None
            else interpolated.device
        )
        noise = torch.randn(
            interpolated.shape,
            device=generator_device,
            dtype=interpolated.dtype,
            generator=generator,
        ).to(interpolated.device)
        epsilon = F.normalize(noise, p=2, dim=1)
        rho = torch.full(
            (values.numel(),), self.rho_p, device=interpolated.device, dtype=interpolated.dtype
        )
        content = mobius_reparam(epsilon, interpolated, rho)
        style = torch.randn(
            (values.numel(), self.style_dim),
            device=generator_device,
            dtype=interpolated.dtype,
            generator=generator,
        ).to(interpolated.device)
        return content, style

    # ------------------------------------------------------------------
    # EMA center update (called by trainer, no gradient needed)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def update_ema_centers(
        self,
        mu_c_list: list[torch.Tensor],
        y_list: list[torch.Tensor],
    ) -> None:
        """
        Update EMA class centers using accumulated (mu_c, y) from the epoch.

        mu_c_list : list of (B_i, d_c) tensors (already L2-normalized)
        y_list    : list of (B_i,) label tensors
        """
        all_mu_c = torch.cat(mu_c_list, dim=0)   # (N, d_c)
        all_y = torch.cat(y_list, dim=0)          # (N,)
        tau = self.ema_momentum

        for k in range(self.n_classes):
            mask = all_y == k
            if mask.sum() == 0:
                continue
            batch_mean = all_mu_c[mask].mean(dim=0)  # (d_c,)

            if self.n_centers == 1:
                new_center = tau * self.ema_centers[k, 0] + (1 - tau) * batch_mean
                self.ema_centers[k, 0] = F.normalize(new_center, p=2, dim=0)
            else:
                # Assign each sample to the nearest current center, then update
                class_mu = all_mu_c[mask]  # (M, d_c)
                dots = class_mu @ self.ema_centers[k].T  # (M, R)
                assignments = dots.argmax(dim=1)          # (M,)
                for r in range(self.n_centers):
                    r_mask = assignments == r
                    if r_mask.sum() == 0:
                        continue
                    sub_mean = class_mu[r_mask].mean(dim=0)
                    new_c = tau * self.ema_centers[k, r] + (1 - tau) * sub_mean
                    self.ema_centers[k, r] = F.normalize(new_c, p=2, dim=0)

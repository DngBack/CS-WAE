"""
Ablation variants for F-CS-WAE.

9 variants covering the main design axes:

    full              — complete F-CS-WAE (reference)
    no_z_s            — single semantic latent z_c only (no style)
    no_class_mmd      — drop supervised L_class (alpha=0 always)
    no_style_mmd      — drop style prior L_style (gamma=0 always)
    no_classifier     — drop auxiliary CE L_cls (eta=0 always)
    gaussian_class_prior — Euclidean z_c with Gaussian class priors (no sphere)
    vmf_class_prior   — spherical z_c but vMF prior instead of Spherical Cauchy
    single_center     — explicitly R=1 (same as default; here for paper labelling)
    learnable_centers — prior_mus as nn.Parameter + gradient (no EMA)

Usage
-----
    model = create_f_cs_wae_ablation("no_z_s")
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config_f_cs_wae import f_cs_wae_config as cfg
from ..utils.utils import mobius_reparam, sample_uniform_sphere
from .backbone import ResNet18Body
from .f_cs_wae import ResBlock, ResBlockDecoder, ResBlockUp


# ---------------------------------------------------------------------------
# Variant config registry
# ---------------------------------------------------------------------------

ABLATION_VARIANTS: dict[str, dict] = {
    "full": {
        "name": "F-CS-WAE (Full)",
        "use_style_latent":   True,
        "use_class_mmd":      True,
        "use_style_mmd":      True,
        "use_classifier":     True,
        "z_c_type":           "spherical",   # "spherical" | "euclidean"
        "prior_type":         "spherical_cauchy",  # "spherical_cauchy"|"vmf"|"gaussian"
        "ema_centers":        True,
        "description": "Complete F-CS-WAE model — all components enabled.",
    },
    "no_z_s": {
        "name": "F-CS-WAE (no z_s)",
        "use_style_latent":   False,
        "use_class_mmd":      True,
        "use_style_mmd":      False,
        "use_classifier":     True,
        "z_c_type":           "spherical",
        "prior_type":         "spherical_cauchy",
        "ema_centers":        True,
        "description": "Single z_c latent only — no style factorization.",
    },
    "no_class_mmd": {
        "name": "F-CS-WAE (no L_class)",
        "use_style_latent":   True,
        "use_class_mmd":      False,
        "use_style_mmd":      True,
        "use_classifier":     True,
        "z_c_type":           "spherical",
        "prior_type":         "spherical_cauchy",
        "ema_centers":        True,
        "description": "Remove supervised class MMD — tests class prior matching importance.",
    },
    "no_style_mmd": {
        "name": "F-CS-WAE (no L_style)",
        "use_style_latent":   True,
        "use_class_mmd":      True,
        "use_style_mmd":      False,
        "use_classifier":     True,
        "z_c_type":           "spherical",
        "prior_type":         "spherical_cauchy",
        "ema_centers":        True,
        "description": "Remove style prior MMD — style latent not regularized to N(0,I).",
    },
    "no_classifier": {
        "name": "F-CS-WAE (no L_cls)",
        "use_style_latent":   True,
        "use_class_mmd":      True,
        "use_style_mmd":      True,
        "use_classifier":     False,
        "z_c_type":           "spherical",
        "prior_type":         "spherical_cauchy",
        "ema_centers":        True,
        "description": "Remove auxiliary CE — tests anti-collapse loss importance.",
    },
    "gaussian_class_prior": {
        "name": "F-CS-WAE (Euclidean z_c)",
        "use_style_latent":   True,
        "use_class_mmd":      True,
        "use_style_mmd":      True,
        "use_classifier":     True,
        "z_c_type":           "euclidean",
        "prior_type":         "gaussian",
        "ema_centers":        False,
        "description": "Replace spherical z_c with Euclidean Gaussian — no hypersphere.",
    },
    "vmf_class_prior": {
        "name": "F-CS-WAE (vMF prior)",
        "use_style_latent":   True,
        "use_class_mmd":      True,
        "use_style_mmd":      True,
        "use_classifier":     True,
        "z_c_type":           "spherical",
        "prior_type":         "vmf",
        "ema_centers":        True,
        "description": "Spherical Cauchy prior replaced with von Mises-Fisher.",
    },
    "single_center": {
        "name": "F-CS-WAE (R=1 explicit)",
        "use_style_latent":   True,
        "use_class_mmd":      True,
        "use_style_mmd":      True,
        "use_classifier":     True,
        "z_c_type":           "spherical",
        "prior_type":         "spherical_cauchy",
        "ema_centers":        True,
        "description": "Explicit single center per class (R=1) — paper clarity ablation.",
    },
    "learnable_centers": {
        "name": "F-CS-WAE (learnable centers)",
        "use_style_latent":   True,
        "use_class_mmd":      True,
        "use_style_mmd":      True,
        "use_classifier":     True,
        "z_c_type":           "spherical",
        "prior_type":         "spherical_cauchy",
        "ema_centers":        False,  # use nn.Parameter instead
        "description": "Class centers as nn.Parameter with gradient (no EMA).",
    },
}


# ---------------------------------------------------------------------------
# Ablation encoder
# ---------------------------------------------------------------------------

class FCSWAEAblationEncoder(nn.Module):
    """
    Shared ResNet-18 backbone with configurable latent heads.

    z_c_type == "spherical": outputs (mu_c_raw, rho_c_raw, mu_s, logvar_s)
    z_c_type == "euclidean": outputs (mu_c, logvar_c, mu_s, logvar_s)
                             (logvar_c used for VAE-style reparameterization)
    use_style_latent == False: mu_s and logvar_s heads are omitted (return None)
    """

    def __init__(
        self,
        semantic_dim: int,
        style_dim: int,
        in_channels: int,
        z_c_type: str = "spherical",
        use_style_latent: bool = True,
    ):
        super().__init__()
        self.z_c_type = z_c_type
        self.use_style_latent = use_style_latent
        self.semantic_dim = semantic_dim
        self.style_dim = style_dim

        body = ResNet18Body(in_channels)
        flat = body.out_channels * body.spatial ** 2
        self.features = body
        self.fc_block = nn.Sequential(
            nn.Flatten(), nn.Linear(flat, 256), nn.SiLU()
        )

        # Semantic head
        self.fc_mu_c = nn.Linear(256, semantic_dim)
        if z_c_type == "spherical":
            self.fc_rho_c = nn.Linear(256, 1)
        else:
            self.fc_logvar_c = nn.Linear(256, semantic_dim)

        # Style head (optional)
        if use_style_latent:
            self.fc_mu_s     = nn.Linear(256, style_dim)
            self.fc_logvar_s = nn.Linear(256, style_dim)
            nn.init.zeros_(self.fc_logvar_s.weight)
            nn.init.zeros_(self.fc_logvar_s.bias)

    def forward(self, x: torch.Tensor):
        h = self.fc_block(self.features(x))
        mu_c_raw = self.fc_mu_c(h)

        if self.z_c_type == "spherical":
            rho_c_raw = self.fc_rho_c(h)
        else:
            rho_c_raw = self.fc_logvar_c(h)  # used as logvar_c

        if self.use_style_latent:
            mu_s     = self.fc_mu_s(h)
            logvar_s = self.fc_logvar_s(h)
        else:
            mu_s = logvar_s = None

        return mu_c_raw, rho_c_raw, mu_s, logvar_s


# ---------------------------------------------------------------------------
# Ablation model
# ---------------------------------------------------------------------------

class FCSWAEAblation(nn.Module):
    """
    F-CS-WAE with configurable ablation variant.

    All variants expose the same forward signature for the trainer and evaluator:
        forward(x) → (x_hat, z_c, z_s, mu_c, mu_s, logvar_s)
    where z_s / mu_s / logvar_s are None when use_style_latent == False.
    """

    def __init__(
        self,
        variant: str,
        semantic_dim: int | None = None,
        style_dim: int | None = None,
        n_classes: int | None = None,
        in_channels: int | None = None,
        image_size: int | None = None,
    ):
        super().__init__()
        if variant not in ABLATION_VARIANTS:
            raise ValueError(
                f"Unknown variant '{variant}'. Choose from: {list(ABLATION_VARIANTS)}"
            )

        self.variant_config = ABLATION_VARIANTS[variant]
        self.variant_name   = variant

        self.semantic_dim       = semantic_dim or cfg.semantic_dim
        self.style_dim          = style_dim    or cfg.style_dim
        self.n_classes          = n_classes    or cfg.n_classes
        self.n_centers          = 1
        in_channels             = in_channels  or cfg.in_channels
        image_size              = image_size   or cfg.image_size

        vc = self.variant_config
        self.use_style_latent   = vc["use_style_latent"]
        self.use_class_mmd      = vc["use_class_mmd"]
        self.use_style_mmd      = vc["use_style_mmd"]
        self.use_classifier     = vc["use_classifier"]
        self.z_c_type           = vc["z_c_type"]
        self.prior_type         = vc["prior_type"]
        self.use_ema_centers    = vc["ema_centers"]

        self.rho_p = cfg.rho_prior

        # Encoder
        self.encoder = FCSWAEAblationEncoder(
            self.semantic_dim, self.style_dim, in_channels,
            z_c_type=self.z_c_type,
            use_style_latent=self.use_style_latent,
        )

        # Decoder — if no style latent, only takes z_c
        if self.use_style_latent:
            self.decoder = ResBlockDecoder(
                self.semantic_dim, self.style_dim, in_channels, image_size
            )
        else:
            # decoder input dim = semantic_dim; use sem=semantic_dim//3, sty=rest
            sem = self.semantic_dim // 3
            sty = self.semantic_dim - sem
            self.decoder = ResBlockDecoder(sem, sty, in_channels, image_size)

        # Class centers
        if self.use_ema_centers:
            init_centers = F.normalize(
                torch.randn(self.n_classes, 1, self.semantic_dim), p=2, dim=-1
            )
            self.register_buffer("ema_centers", init_centers)
        else:
            # Learnable centers (gradient)
            self.prior_mus = nn.Parameter(
                F.normalize(torch.randn(self.n_classes, self.semantic_dim), p=2, dim=-1)
            )
            if self.prior_type == "gaussian":
                self.prior_logvars = nn.Parameter(
                    torch.zeros(self.n_classes, self.semantic_dim)
                )

        # Auxiliary classifier (optional)
        if self.use_classifier:
            self.classifier = nn.Linear(self.semantic_dim, self.n_classes)

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    def encode_to_distribution(self, x: torch.Tensor):
        """Returns (mu_c, rho_or_logvar_c) — compat with ModelEvaluator."""
        mu_c_raw, param_c, _, _ = self.encoder(x)
        if self.z_c_type == "spherical":
            mu_c  = F.normalize(mu_c_raw, p=2, dim=1)
            rho_c = torch.sigmoid(param_c).squeeze(-1) * (1 - cfg.epsilon)
            return mu_c, rho_c
        else:
            return mu_c_raw, param_c

    # ------------------------------------------------------------------
    # Sampling helpers
    # ------------------------------------------------------------------

    def _sample_z_c(self, mu_c_raw, param_c):
        if self.z_c_type == "spherical":
            mu_c  = F.normalize(mu_c_raw, p=2, dim=1)
            rho_c = torch.sigmoid(param_c).squeeze(-1) * (1 - cfg.epsilon)
            eps   = sample_uniform_sphere(mu_c.shape[0], self.semantic_dim, device=mu_c.device)
            return mobius_reparam(eps, mu_c, rho_c), mu_c, rho_c
        else:
            # Euclidean: VAE reparameterization
            mu_c   = mu_c_raw
            logvar = param_c
            std    = torch.exp(0.5 * logvar)
            return mu_c + std * torch.randn_like(std), mu_c, logvar

    def _get_class_centers(self, device: torch.device) -> torch.Tensor:
        """Return (K, d_c) normalized centers."""
        if self.use_ema_centers:
            return F.normalize(self.ema_centers[:, 0, :], p=2, dim=-1)
        else:
            if self.z_c_type == "spherical":
                return F.normalize(self.prior_mus, p=2, dim=-1)
            return self.prior_mus

    def _sample_prior_z_c(self, k: int, n: int, device: torch.device) -> torch.Tensor:
        centers = self._get_class_centers(device)
        if self.prior_type == "spherical_cauchy":
            center = centers[k].unsqueeze(0).expand(n, -1)
            eps = sample_uniform_sphere(n, self.semantic_dim, device=device)
            rho = torch.full((n,), self.rho_p, device=device)
            return mobius_reparam(eps, center, rho)
        elif self.prior_type == "vmf":
            return self._sample_vmf(centers[k], n, device)
        else:  # gaussian
            mu_k = self.prior_mus[k]
            std  = torch.exp(0.5 * self.prior_logvars[k])
            return mu_k + std * torch.randn(n, self.semantic_dim, device=device)

    def _sample_vmf(self, mu: torch.Tensor, n: int, device: torch.device) -> torch.Tensor:
        """Rejection-sample approximation of von Mises-Fisher."""
        kappa = 10.0
        collected, samples = 0, []
        while collected < n:
            z    = F.normalize(torch.randn(n * 4, self.semantic_dim, device=device), p=2, dim=1)
            dots = (z * mu.unsqueeze(0)).sum(dim=1)
            probs = torch.exp(kappa * (dots - 1.0))
            accept = torch.rand(len(z), device=device) < probs
            batch  = z[accept]
            samples.append(batch)
            collected += batch.shape[0]
        return torch.cat(samples, dim=0)[:n]

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor):
        mu_c_raw, param_c, mu_s_raw, logvar_s = self.encoder(x)
        z_c, mu_c, _ = self._sample_z_c(mu_c_raw, param_c)

        if self.use_style_latent and mu_s_raw is not None:
            std_s = torch.exp(0.5 * logvar_s.clamp(-10, 10))
            z_s   = mu_s_raw + std_s * torch.randn_like(std_s)
            dec_input = torch.cat([z_c, z_s], dim=1)
        else:
            z_s = mu_s = logvar_s = None
            dec_input = z_c

        x_hat = self.decoder(dec_input)
        return x_hat, z_c, z_s, mu_c, mu_s_raw, logvar_s

    # ------------------------------------------------------------------
    # EMA update (only when use_ema_centers == True)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def update_ema_centers(
        self,
        mu_c_list: list[torch.Tensor],
        y_list: list[torch.Tensor],
    ) -> None:
        if not self.use_ema_centers:
            return
        all_mu_c = torch.cat(mu_c_list, dim=0)
        all_y    = torch.cat(y_list,    dim=0)
        tau = cfg.ema_momentum

        for k in range(self.n_classes):
            mask = all_y == k
            if mask.sum() == 0:
                continue
            batch_mean = all_mu_c[mask].mean(dim=0)
            new_c = tau * self.ema_centers[k, 0] + (1 - tau) * batch_mean
            self.ema_centers[k, 0] = F.normalize(new_c, p=2, dim=0)

    # ------------------------------------------------------------------
    # Compat: prior sampling for evaluator
    # ------------------------------------------------------------------

    def sample_from_class_prior(
        self, k: int, n: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        z_c = self._sample_prior_z_c(k, n, device)
        if self.use_style_latent:
            z_s = torch.randn(n, self.style_dim, device=device)
        else:
            z_s = None
        return z_c, z_s


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_f_cs_wae_ablation(
    variant: str,
    semantic_dim: int | None = None,
    style_dim: int | None = None,
    n_classes: int | None = None,
    in_channels: int | None = None,
    image_size: int | None = None,
) -> FCSWAEAblation:
    """
    Factory function.  variant must be one of ABLATION_VARIANTS keys.

    The 'full' variant creates the complete F-CS-WAE — identical in structure
    to FCSWAE but implemented via FCSWAEAblation for unified ablation tracking.
    """
    return FCSWAEAblation(
        variant=variant,
        semantic_dim=semantic_dim,
        style_dim=style_dim,
        n_classes=n_classes,
        in_channels=in_channels,
        image_size=image_size,
    )

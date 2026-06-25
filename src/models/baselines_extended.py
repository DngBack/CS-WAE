"""
Extended baselines for F-CS-WAE comparison study.

Three groups, all using ResNet-18 backbone for fair comparison:

Group 1 — Unsupervised generative clustering
    ResNetAE           — plain autoencoder + K-means (no label)

Group 2 — Label-guided representation learning
    AEWithCE           — AE + cross-entropy classifier head
    AEWithSupCon       — AE + supervised contrastive loss
    AEWithCenterLoss   — AE + center loss (class compactness)
    AEWithTriplet      — AE + triplet margin loss

Group 3 — Conditional generative models
    ConditionalVAE      — cVAE: label embedding concat to decoder
    ConditionalWAE_MMD  — cWAE-MMD: class-conditioned MMD matching
    GaussianClassPriorWAE — WAE with learnable Gaussian class centers
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import ResNet18Body
from .f_cs_wae import ResBlockDecoder
from ..config_f_cs_wae import f_cs_wae_config as cfg


# ---------------------------------------------------------------------------
# Shared encoder helper
# ---------------------------------------------------------------------------

def _build_resnet_encoder(latent_dim: int, in_channels: int = 3) -> tuple[nn.Module, nn.Sequential]:
    """Return (ResNet18Body, fc_head) that maps features → R^{latent_dim}."""
    body = ResNet18Body(in_channels)
    flat = body.out_channels * body.spatial ** 2
    fc = nn.Sequential(
        nn.Flatten(),
        nn.Linear(flat, 256),
        nn.SiLU(),
        nn.Linear(256, latent_dim),
    )
    return body, fc


# ---------------------------------------------------------------------------
# Euclidean MMD helper (shared with loss_f_cs_wae but reproduced to avoid
# circular imports when this module is used standalone)
# ---------------------------------------------------------------------------

def _rbf_euclidean(x: torch.Tensor, y: torch.Tensor, sigma: float) -> torch.Tensor:
    dist_sq = torch.cdist(x, y, p=2).pow(2)
    return torch.exp(-dist_sq / (2.0 * sigma ** 2 + 1e-8))


def _mmd_euclidean(q: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    if q.shape[0] < 2 or p.shape[0] < 2:
        return torch.tensor(0.0, device=q.device)
    with torch.no_grad():
        dists = torch.pdist(torch.cat([q, p], dim=0))
        sigma = max(dists.median().item(), 1e-8)
    k_qq = _rbf_euclidean(q, q, sigma).mean()
    k_pp = _rbf_euclidean(p, p, sigma).mean()
    k_qp = _rbf_euclidean(q, p, sigma).mean()
    return k_qq + k_pp - 2.0 * k_qp


# ===========================================================================
# GROUP 1: Unsupervised
# ===========================================================================

class ResNetAE(nn.Module):
    """
    Plain ResNet-18 autoencoder (no label, no prior regularization).
    Latent dim = semantic_dim + style_dim = 192 by default to match
    F-CS-WAE's total latent capacity.

    Evaluation: extract latent z, then run K-means externally.
    """

    def __init__(
        self,
        latent_dim: int | None = None,
        in_channels: int = 3,
        image_size: int = 32,
    ):
        super().__init__()
        latent_dim = latent_dim or (cfg.semantic_dim + cfg.style_dim)
        self.latent_dim = latent_dim
        self.features, self.fc = _build_resnet_encoder(latent_dim, in_channels)
        # Use a unified decoder dim split into "semantic" and "style" parts
        # so we can reuse ResBlockDecoder (which expects two dims)
        sem = latent_dim // 3
        sty = latent_dim - sem
        self.decoder = ResBlockDecoder(sem, sty, in_channels, image_size)
        self._sem = sem
        self._sty = sty

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.features(x))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        x_hat = self.decoder(z)
        return x_hat, z

    def loss_function(self, x_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return F.l1_loss(x_hat, x)


# ===========================================================================
# GROUP 2: Label-guided
# ===========================================================================

class AEWithCE(nn.Module):
    """
    AE + cross-entropy classification head.
    Loss = L1(recon) + lambda_ce * CE(z, y)
    """

    def __init__(
        self,
        latent_dim: int | None = None,
        n_classes: int | None = None,
        in_channels: int = 3,
        image_size: int = 32,
        lambda_ce: float = 1.0,
    ):
        super().__init__()
        latent_dim = latent_dim or (cfg.semantic_dim + cfg.style_dim)
        n_classes  = n_classes  or cfg.n_classes
        self.lambda_ce = lambda_ce

        self.features, self.fc = _build_resnet_encoder(latent_dim, in_channels)
        sem = latent_dim // 3
        sty = latent_dim - sem
        self.decoder = ResBlockDecoder(sem, sty, in_channels, image_size)
        self.classifier = nn.Linear(latent_dim, n_classes)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.features(x))

    def forward(self, x: torch.Tensor):
        z = self.encode(x)
        x_hat = self.decoder(z)
        logits = self.classifier(z)
        return x_hat, z, logits

    def loss_function(self, x_hat, x, logits, y):
        return F.l1_loss(x_hat, x) + self.lambda_ce * F.cross_entropy(logits, y)


class AEWithSupCon(nn.Module):
    """
    AE + Supervised Contrastive Loss (SupCon).
    Uses L2-normalized latent projections; loss balances recon + SupCon.
    """

    def __init__(
        self,
        latent_dim: int | None = None,
        proj_dim: int = 128,
        in_channels: int = 3,
        image_size: int = 32,
        temperature: float = 0.07,
        lambda_sc: float = 1.0,
    ):
        super().__init__()
        latent_dim = latent_dim or (cfg.semantic_dim + cfg.style_dim)
        self.temperature = temperature
        self.lambda_sc = lambda_sc

        self.features, self.fc = _build_resnet_encoder(latent_dim, in_channels)
        sem = latent_dim // 3
        sty = latent_dim - sem
        self.decoder = ResBlockDecoder(sem, sty, in_channels, image_size)
        self.projector = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, proj_dim),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.features(x))

    def forward(self, x: torch.Tensor):
        z = self.encode(x)
        x_hat = self.decoder(z)
        proj = F.normalize(self.projector(z), p=2, dim=1)
        return x_hat, z, proj

    def supcon_loss(self, proj: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Supervised contrastive loss (SimCLR-style, single view)."""
        device = proj.device
        B = proj.shape[0]
        sim = torch.matmul(proj, proj.T) / self.temperature  # (B, B)
        # Mask: same class, excluding self
        y_eq = (y.unsqueeze(0) == y.unsqueeze(1)).float()  # (B, B)
        mask_self = torch.eye(B, device=device)
        positives = y_eq * (1 - mask_self)
        # log-softmax denominator: all pairs excluding self
        sim_exp = torch.exp(sim) * (1 - mask_self)
        log_prob = sim - torch.log(sim_exp.sum(dim=1, keepdim=True) + 1e-8)
        # Average over positives
        n_pos = positives.sum(dim=1).clamp(min=1)
        loss = -(positives * log_prob).sum(dim=1) / n_pos
        return loss.mean()

    def loss_function(self, x_hat, x, proj, y):
        return F.l1_loss(x_hat, x) + self.lambda_sc * self.supcon_loss(proj, y)


class AEWithCenterLoss(nn.Module):
    """
    AE + Center Loss: pulls latent of each class toward a learnable center.
    Loss = L1(recon) + lambda_cl * (1/B) sum ||z_i - center_{y_i}||^2
    """

    def __init__(
        self,
        latent_dim: int | None = None,
        n_classes: int | None = None,
        in_channels: int = 3,
        image_size: int = 32,
        lambda_cl: float = 0.5,
    ):
        super().__init__()
        latent_dim = latent_dim or (cfg.semantic_dim + cfg.style_dim)
        n_classes  = n_classes  or cfg.n_classes
        self.lambda_cl = lambda_cl

        self.features, self.fc = _build_resnet_encoder(latent_dim, in_channels)
        sem = latent_dim // 3
        sty = latent_dim - sem
        self.decoder = ResBlockDecoder(sem, sty, in_channels, image_size)
        self.centers = nn.Parameter(torch.zeros(n_classes, latent_dim))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.features(x))

    def forward(self, x: torch.Tensor):
        z = self.encode(x)
        x_hat = self.decoder(z)
        return x_hat, z

    def loss_function(self, x_hat, x, z, y):
        recon = F.l1_loss(x_hat, x)
        center_y = self.centers[y]  # (B, d)
        center_l = ((z - center_y) ** 2).sum(dim=1).mean()
        return recon + self.lambda_cl * center_l


class AEWithTriplet(nn.Module):
    """
    AE + Triplet Margin Loss (online hard triplet mining within batch).
    Loss = L1(recon) + lambda_tri * TripletMarginLoss
    """

    def __init__(
        self,
        latent_dim: int | None = None,
        in_channels: int = 3,
        image_size: int = 32,
        margin: float = 1.0,
        lambda_tri: float = 1.0,
    ):
        super().__init__()
        latent_dim = latent_dim or (cfg.semantic_dim + cfg.style_dim)
        self.lambda_tri = lambda_tri
        self.triplet_loss = nn.TripletMarginLoss(margin=margin, p=2)

        self.features, self.fc = _build_resnet_encoder(latent_dim, in_channels)
        sem = latent_dim // 3
        sty = latent_dim - sem
        self.decoder = ResBlockDecoder(sem, sty, in_channels, image_size)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.features(x))

    def forward(self, x: torch.Tensor):
        z = self.encode(x)
        x_hat = self.decoder(z)
        return x_hat, z

    def _mine_hard_triplets(
        self, z: torch.Tensor, y: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Mine hardest positive and hardest negative per anchor within the batch."""
        dists = torch.cdist(z, z, p=2)  # (B, B)
        B = z.shape[0]
        anchors, positives, negatives = [], [], []

        for i in range(B):
            pos_mask = (y == y[i]) & (torch.arange(B, device=z.device) != i)
            neg_mask = y != y[i]
            if pos_mask.sum() == 0 or neg_mask.sum() == 0:
                continue
            # Hardest positive: farthest same-class sample
            pos_idx = dists[i].where(pos_mask, torch.tensor(-1e9, device=z.device)).argmax()
            # Hardest negative: closest different-class sample
            neg_idx = dists[i].where(neg_mask, torch.tensor(1e9, device=z.device)).argmin()
            anchors.append(z[i])
            positives.append(z[pos_idx])
            negatives.append(z[neg_idx])

        if not anchors:
            return z[:1], z[:1], z[:1]
        return (torch.stack(anchors),
                torch.stack(positives),
                torch.stack(negatives))

    def loss_function(self, x_hat, x, z, y):
        recon = F.l1_loss(x_hat, x)
        a, p, n = self._mine_hard_triplets(z, y)
        tri = self.triplet_loss(a, p, n)
        return recon + self.lambda_tri * tri


# ===========================================================================
# GROUP 3: Conditional generative
# ===========================================================================

class ConditionalVAE(nn.Module):
    """
    Conditional VAE: label embedding is concatenated to the decoder input.

    Encoder: x → (mu, logvar)  [Euclidean Gaussian]
    Decoder: concat(z, label_embed(y)) → x_hat
    Loss:    L1(recon) + beta * KL
    """

    def __init__(
        self,
        latent_dim: int | None = None,
        n_classes: int | None = None,
        label_emb_dim: int = 16,
        in_channels: int = 3,
        image_size: int = 32,
        beta: float = 1.0,
    ):
        super().__init__()
        latent_dim = latent_dim or (cfg.semantic_dim + cfg.style_dim)
        n_classes  = n_classes  or cfg.n_classes
        self.latent_dim = latent_dim
        self.beta = beta

        # Encoder: x → (mu, logvar)
        body = ResNet18Body(in_channels)
        flat = body.out_channels * body.spatial ** 2
        self.enc_body = body
        self.enc_fc   = nn.Sequential(nn.Flatten(), nn.Linear(flat, 256), nn.SiLU())
        self.fc_mu    = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)

        # Label embedding (for decoder)
        self.label_emb = nn.Embedding(n_classes, label_emb_dim)

        # Decoder: concat(z, label_emb) → image
        sem = latent_dim // 3
        sty = latent_dim + label_emb_dim - sem
        self.decoder = ResBlockDecoder(sem, sty, in_channels, image_size)

    def encode(self, x: torch.Tensor):
        h = self.enc_fc(self.enc_body(x))
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        emb = self.label_emb(y)  # (B, label_emb_dim)
        z_cond = torch.cat([z, emb], dim=1)
        x_hat = self.decoder(z_cond)
        return x_hat, mu, logvar

    def loss_function(self, x_hat, x, mu, logvar):
        recon = F.l1_loss(x_hat, x)
        kl    = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return recon + self.beta * kl

    def generate(self, y: torch.Tensor, device: torch.device) -> torch.Tensor:
        """Sample z ~ N(0,I) and decode conditioned on y."""
        B = y.shape[0]
        z = torch.randn(B, self.latent_dim, device=device)
        emb = self.label_emb(y.to(device))
        z_cond = torch.cat([z, emb], dim=1)
        return self.decoder(z_cond)


class ConditionalWAE_MMD(nn.Module):
    """
    Conditional WAE-MMD: encoder is deterministic; prior is class-conditioned.

    For each class k, prior is N(label_embed(k), I).
    Loss = L1(recon) + mmd_weight * (1/K) sum_k MMD(z[y==k], N(embed_k, I))
    """

    def __init__(
        self,
        latent_dim: int | None = None,
        n_classes: int | None = None,
        label_emb_dim: int = 32,
        in_channels: int = 3,
        image_size: int = 32,
        mmd_weight: float = 10.0,
    ):
        super().__init__()
        latent_dim = latent_dim or (cfg.semantic_dim + cfg.style_dim)
        n_classes  = n_classes  or cfg.n_classes
        self.n_classes = n_classes
        self.mmd_weight = mmd_weight

        # Deterministic encoder
        body = ResNet18Body(in_channels)
        flat = body.out_channels * body.spatial ** 2
        self.enc_body = body
        self.enc_fc   = nn.Sequential(
            nn.Flatten(), nn.Linear(flat, 256), nn.SiLU(), nn.Linear(256, latent_dim)
        )

        # Learnable class embeddings in latent space (used as prior centers)
        self.class_embed = nn.Embedding(n_classes, latent_dim)

        sem = latent_dim // 3
        sty = latent_dim - sem
        self.decoder = ResBlockDecoder(sem, sty, in_channels, image_size)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.enc_fc(self.enc_body(x))

    def forward(self, x: torch.Tensor):
        z = self.encode(x)
        x_hat = self.decoder(z)
        return x_hat, z

    def loss_function(self, x_hat, x, z, y):
        recon = F.l1_loss(x_hat, x)
        # Per-class MMD vs N(embed_k, I)
        class_mmd = torch.tensor(0.0, device=x.device)
        count = 0
        for k in range(self.n_classes):
            mask = y == k
            if mask.sum() < 2:
                continue
            mu_k = self.class_embed.weight[k]  # (d,)
            n_k  = mask.sum().item()
            z_p  = mu_k.unsqueeze(0) + torch.randn(n_k, mu_k.shape[0], device=x.device)
            class_mmd = class_mmd + _mmd_euclidean(z[mask], z_p)
            count += 1
        if count > 0:
            class_mmd = class_mmd / count
        return recon + self.mmd_weight * class_mmd

    def generate(self, y: torch.Tensor, device: torch.device) -> torch.Tensor:
        """Sample z ~ N(embed_k, I) and decode."""
        mu_k = self.class_embed(y.to(device))            # (B, d)
        z = mu_k + torch.randn_like(mu_k)
        return self.decoder(z)


class GaussianClassPriorWAE(nn.Module):
    """
    WAE with learnable Gaussian class prior centers (Euclidean).

    Similar to a Euclidean version of CS-WAE's class-conditional matching,
    without spherical geometry.
    Loss = L1(recon) + mmd_weight * (1/K) sum_k MMD(z[y==k], N(mu_k, I))
         + agg_weight * MMD(z_all, mixture prior)
    """

    def __init__(
        self,
        latent_dim: int | None = None,
        n_classes: int | None = None,
        in_channels: int = 3,
        image_size: int = 32,
        mmd_weight: float = 10.0,
        agg_weight: float = 1.0,
    ):
        super().__init__()
        latent_dim = latent_dim or (cfg.semantic_dim + cfg.style_dim)
        n_classes  = n_classes  or cfg.n_classes
        self.n_classes = n_classes
        self.mmd_weight = mmd_weight
        self.agg_weight = agg_weight
        self.latent_dim = latent_dim

        # Deterministic encoder
        body = ResNet18Body(in_channels)
        flat = body.out_channels * body.spatial ** 2
        self.enc_body = body
        self.enc_fc   = nn.Sequential(
            nn.Flatten(), nn.Linear(flat, 256), nn.SiLU(), nn.Linear(256, latent_dim)
        )

        # Learnable class prior centers
        self.prior_mus = nn.Parameter(torch.randn(n_classes, latent_dim) * 0.1)

        sem = latent_dim // 3
        sty = latent_dim - sem
        self.decoder = ResBlockDecoder(sem, sty, in_channels, image_size)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.enc_fc(self.enc_body(x))

    def forward(self, x: torch.Tensor):
        z = self.encode(x)
        x_hat = self.decoder(z)
        return x_hat, z

    def loss_function(self, x_hat, x, z, y):
        recon = F.l1_loss(x_hat, x)
        device = x.device

        # Supervised class MMD
        class_mmd = torch.tensor(0.0, device=device)
        count = 0
        for k in range(self.n_classes):
            mask = y == k
            if mask.sum() < 2:
                continue
            n_k = mask.sum().item()
            mu_k = self.prior_mus[k]
            z_p = mu_k.unsqueeze(0) + torch.randn(n_k, self.latent_dim, device=device)
            class_mmd = class_mmd + _mmd_euclidean(z[mask], z_p)
            count += 1
        if count > 0:
            class_mmd = class_mmd / count

        # Aggregated mixture MMD
        B = x.shape[0]
        rand_k = torch.randint(0, self.n_classes, (B,), device=device)
        mu_mix = self.prior_mus[rand_k]  # (B, d)
        z_p_mix = mu_mix + torch.randn_like(mu_mix)
        agg_mmd = _mmd_euclidean(z, z_p_mix)

        return recon + self.mmd_weight * class_mmd + self.agg_weight * agg_mmd

    def generate(self, y: torch.Tensor, device: torch.device) -> torch.Tensor:
        mu_k = self.prior_mus[y.to(device)]  # (B, d)
        z = mu_k + torch.randn_like(mu_k)
        return self.decoder(z)


# ---------------------------------------------------------------------------
# Model registry (for compare_baselines_extended.py)
# ---------------------------------------------------------------------------

GROUP1_MODELS = {"ResNetAE": ResNetAE}

GROUP2_MODELS = {
    "AEWithCE":          AEWithCE,
    "AEWithSupCon":      AEWithSupCon,
    "AEWithCenterLoss":  AEWithCenterLoss,
    "AEWithTriplet":     AEWithTriplet,
}

GROUP3_MODELS = {
    "ConditionalVAE":         ConditionalVAE,
    "ConditionalWAE_MMD":     ConditionalWAE_MMD,
    "GaussianClassPriorWAE":  GaussianClassPriorWAE,
}

ALL_EXTENDED_MODELS = {**GROUP1_MODELS, **GROUP2_MODELS, **GROUP3_MODELS}

# Metadata for comparison table
MODEL_META = {
    "ResNetAE":              {"uses_labels": False, "generative_prior": False},
    "AEWithCE":              {"uses_labels": True,  "generative_prior": False},
    "AEWithSupCon":          {"uses_labels": True,  "generative_prior": False},
    "AEWithCenterLoss":      {"uses_labels": True,  "generative_prior": False},
    "AEWithTriplet":         {"uses_labels": True,  "generative_prior": False},
    "ConditionalVAE":        {"uses_labels": True,  "generative_prior": True},
    "ConditionalWAE_MMD":    {"uses_labels": True,  "generative_prior": True},
    "GaussianClassPriorWAE": {"uses_labels": True,  "generative_prior": True},
    "F-CS-WAE":              {"uses_labels": True,  "generative_prior": True},
}

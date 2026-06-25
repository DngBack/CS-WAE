"""
Trainers for extended baselines (Groups 1, 2, 3).

A single ExtendedBaselineTrainer dispatches to the correct loss call based on
the model class name, keeping the training loop uniform.
"""

from __future__ import annotations

import torch
import torch.optim as optim
from tqdm import tqdm

from ..config_f_cs_wae import f_cs_wae_config as cfg
from ..models.baselines_extended import (
    ResNetAE,
    AEWithCE,
    AEWithSupCon,
    AEWithCenterLoss,
    AEWithTriplet,
    ConditionalVAE,
    ConditionalWAE_MMD,
    GaussianClassPriorWAE,
)

# Models whose forward/loss need the label tensor y
_LABEL_REQUIRED_FORWARD = (AEWithCE, AEWithSupCon, AEWithCenterLoss, AEWithTriplet,
                            ConditionalVAE, ConditionalWAE_MMD, GaussianClassPriorWAE)

# Models whose forward call takes (x, y) instead of just (x)
_LABEL_IN_FORWARD = (ConditionalVAE,)


class ExtendedBaselineTrainer:
    """
    Generic trainer for all extended baseline models.

    Parameters
    ----------
    model       : one of the extended baseline model instances
    model_name  : string key (used for dispatch and logging)
    train_loader: DataLoader yielding (images, labels)
    device      : torch.device
    epochs      : override from config if provided
    lr          : learning rate (default from cfg)
    """

    def __init__(
        self,
        model: torch.nn.Module,
        model_name: str,
        train_loader,
        device: torch.device | None = None,
        epochs: int | None = None,
        lr: float | None = None,
    ):
        self.model = model
        self.model_name = model_name
        self.train_loader = train_loader
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.epochs = epochs or cfg.total_epochs
        self.model.to(self.device)

        self.optimizer = optim.Adam(model.parameters(), lr=lr or cfg.lr)
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=cfg.lr_scheduler_step,
            gamma=cfg.lr_scheduler_gamma,
        )

    # ------------------------------------------------------------------
    # Loss dispatch
    # ------------------------------------------------------------------

    def _compute_loss(
        self, model, data: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """Forward + loss for any extended baseline model."""
        if isinstance(model, ConditionalVAE):
            x_hat, mu, logvar = model(data, labels)
            return model.loss_function(x_hat, data, mu, logvar)

        elif isinstance(model, AEWithCE):
            x_hat, z, logits = model(data)
            return model.loss_function(x_hat, data, logits, labels)

        elif isinstance(model, AEWithSupCon):
            x_hat, z, proj = model(data)
            return model.loss_function(x_hat, data, proj, labels)

        elif isinstance(model, AEWithCenterLoss):
            x_hat, z = model(data)
            return model.loss_function(x_hat, data, z, labels)

        elif isinstance(model, AEWithTriplet):
            x_hat, z = model(data)
            return model.loss_function(x_hat, data, z, labels)

        elif isinstance(model, ConditionalWAE_MMD):
            x_hat, z = model(data)
            return model.loss_function(x_hat, data, z, labels)

        elif isinstance(model, GaussianClassPriorWAE):
            x_hat, z = model(data)
            return model.loss_function(x_hat, data, z, labels)

        else:
            # ResNetAE (unsupervised)
            x_hat, z = model(data)
            return model.loss_function(x_hat, data)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0
        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch + 1}/{self.epochs} [{self.model_name}]",
        )

        for data, labels in pbar:
            data   = data.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            loss = self._compute_loss(self.model, data, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)
            self.optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({"Loss": f"{loss.item():.4f}"})

        self.scheduler.step()
        avg = total_loss / len(self.train_loader)
        print(f"====> Epoch {epoch + 1} [{self.model_name}] Avg Loss: {avg:.4f}")
        return avg

    def train(self) -> list[float]:
        print(f"Training {self.model_name} on {self.device} — {self.epochs} epochs")
        history = []
        for epoch in range(self.epochs):
            avg = self.train_epoch(epoch)
            history.append(avg)
        print(f"Done training {self.model_name}.")
        return history

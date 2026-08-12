"""
Trainer for F-CS-WAE with 4-phase training schedule and EMA class center updates.

Phase schedule:
    A [0,   50):  L_rec only
    B [50, 100):  + L_style + L_cls  (gamma and eta ramp up)
    C [100, 200): + weak L_class + L_agg  (alpha and beta ramp 0 → final/2)
    D [200, 300): full objective  (alpha and beta ramp final/2 → final)

EMA center update strategy:
    During the training forward pass, mu_c tensors are accumulated (detached)
    in lists per epoch.  After all batches complete, update_ema_centers() is
    called once.  This avoids a second data pass.
"""

from __future__ import annotations

import os
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm
import lpips

from ..config_f_cs_wae import f_cs_wae_config as cfg
from ..utils.loss_f_cs_wae import calculate_f_cs_wae_loss


TRAINING_CHECKPOINT_SCHEMA_VERSION = "fcswae-training-1.0.0"


def _capture_rng_state(train_loader) -> dict[str, Any]:
    """Capture every RNG stream that affects the next training epoch."""

    loader_generator = getattr(train_loader, "generator", None)
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "data_loader_generator": (
            loader_generator.get_state() if loader_generator is not None else None
        ),
    }


def _restore_rng_state(state: dict[str, Any], train_loader) -> None:
    """Restore RNG state after model/optimizer construction has consumed RNG."""

    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    cuda_state = state.get("torch_cuda")
    if cuda_state is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(cuda_state)
    loader_state = state.get("data_loader_generator")
    loader_generator = getattr(train_loader, "generator", None)
    if loader_state is not None and loader_generator is not None:
        loader_generator.set_state(loader_state)


def atomic_torch_save(payload: dict, path: str | Path) -> None:
    """Write a checkpoint without exposing a partially-written target file."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, target)


class FCSWAETrainer:
    """Trainer for FCSWAE model."""

    def __init__(self, model, train_loader, device: torch.device | None = None):
        self.model = model
        self.train_loader = train_loader
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=cfg.lr_scheduler_step,
            gamma=cfg.lr_scheduler_gamma,
        )
        self.loss_fn_vgg = lpips.LPIPS(net="vgg").to(self.device)

    # ------------------------------------------------------------------
    # Crash-safe checkpointing
    # ------------------------------------------------------------------

    def save_training_checkpoint(
        self,
        checkpoint_path: str | Path,
        *,
        completed_epochs: int,
        history: list[dict[str, float]],
        run_config: dict[str, Any],
    ) -> None:
        """Atomically save all state needed to continue at an epoch boundary."""

        payload = {
            "schema_version": TRAINING_CHECKPOINT_SCHEMA_VERSION,
            "completed_epochs": int(completed_epochs),
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "history": history,
            "run_config": run_config,
            "rng_state": _capture_rng_state(self.train_loader),
        }
        atomic_torch_save(payload, checkpoint_path)

    def load_training_checkpoint(
        self,
        checkpoint_path: str | Path,
        *,
        expected_run_config: dict[str, Any] | None = None,
    ) -> tuple[int, list[dict[str, float]]]:
        """Restore a trusted local checkpoint and return the next epoch/history."""

        checkpoint = torch.load(
            checkpoint_path,
            # RNG snapshots must remain CPU ByteTensors. Model and optimizer
            # loaders copy their own state to the parameter device.
            map_location="cpu",
            weights_only=False,
        )
        if checkpoint.get("schema_version") != TRAINING_CHECKPOINT_SCHEMA_VERSION:
            raise ValueError(
                "Unsupported training checkpoint schema: "
                f"{checkpoint.get('schema_version')!r}"
            )
        if expected_run_config is not None:
            stored = checkpoint.get("run_config", {})
            keys = (
                "dataset",
                "seed",
                "epochs",
                "n_centers",
                "semantic_dim",
                "style_dim",
                "delta_final",
                "style_sigma_floor",
                "phase_a_end",
                "phase_b_end",
            )
            mismatches = {
                key: (stored.get(key), expected_run_config.get(key))
                for key in keys
                if stored.get(key) != expected_run_config.get(key)
            }
            if mismatches:
                raise ValueError(f"Resume configuration mismatch: {mismatches}")

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        _restore_rng_state(checkpoint["rng_state"], self.train_loader)
        completed_epochs = int(checkpoint["completed_epochs"])
        history = list(checkpoint["history"])
        if completed_epochs != len(history):
            raise ValueError(
                "Checkpoint history length does not match completed_epochs: "
                f"{len(history)} != {completed_epochs}"
            )
        return completed_epochs, history

    # ------------------------------------------------------------------
    # Phase weight schedule
    # ------------------------------------------------------------------

    def get_phase_weights(self, epoch: int) -> tuple[float, float, float, float, float]:
        """
        Return (alpha, beta, gamma, delta, eta) for the given epoch.

        alpha = supervised class MMD weight
        beta  = aggregated semantic MMD weight
        gamma = global style MMD weight
        delta = per-class style MMD weight (new: enforces z_s ⊥ y)
        eta   = auxiliary classifier weight

        Epoch is 0-indexed.
        """
        e = epoch  # 0-indexed

        A_end = cfg.phase_a_end      # 50
        B_end = cfg.phase_b_end      # 100
        C_end = cfg.phase_c_end      # 200
        D_end = cfg.phase_d_end      # 300

        alpha_f = cfg.alpha_final    # 2.0
        beta_f  = cfg.beta_final     # 5.0
        gamma_f = cfg.gamma_final    # 1.0
        delta_f = cfg.delta_final    # 1.0
        eta_i   = cfg.eta_init       # 0.1
        eta_f   = cfg.eta_final      # 0.3

        if e < A_end:
            # Phase A: reconstruction only
            return 0.0, 0.0, 0.0, 0.0, 0.0

        elif e < B_end:
            # Phase B: style (global + per-class) + cls ramp up
            t = (e - A_end) / (B_end - A_end)   # 0 → 1
            gamma = gamma_f * t
            delta = delta_f * t
            eta   = eta_i + (0.2 - eta_i) * t   # 0.1 → 0.2
            return 0.0, 0.0, gamma, delta, eta

        elif e < C_end:
            # Phase C: class + agg MMD ramp 0 → half-final
            t = (e - B_end) / (C_end - B_end)   # 0 → 1
            alpha = (alpha_f / 2.0) * t
            beta  = (beta_f  / 2.0) * t
            eta   = 0.2 + (eta_f - 0.2) * t     # 0.2 → 0.3
            return alpha, beta, gamma_f, delta_f, eta

        else:
            # Phase D: class + agg MMD ramp half-final → final
            t = min(1.0, (e - C_end) / (D_end - C_end))  # 0 → 1
            alpha = alpha_f / 2.0 + (alpha_f / 2.0) * t
            beta  = beta_f  / 2.0 + (beta_f  / 2.0) * t
            return alpha, beta, gamma_f, delta_f, eta_f

    # ------------------------------------------------------------------
    # Single epoch
    # ------------------------------------------------------------------

    def train_epoch(self, epoch: int) -> dict[str, float]:
        """Train one epoch. Returns dict of average losses."""
        self.model.train()
        alpha, beta, gamma, delta, eta = self.get_phase_weights(epoch)

        acc = {k: 0.0 for k in
               ("total", "rec", "class_mmd", "agg_mmd", "style_mmd",
                "style_cls_mmd", "cls", "var")}

        # Buffers for EMA center update (accumulated, detached)
        mu_c_accum: list[torch.Tensor] = []
        y_accum: list[torch.Tensor] = []

        pbar = tqdm(
            self.train_loader,
            desc=(
                f"Epoch {epoch + 1} "
                f"[α={alpha:.2f} β={beta:.2f} γ={gamma:.2f} δ={delta:.2f} η={eta:.2f}]"
            ),
        )

        for data, labels in pbar:
            data, labels = data.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()

            x_hat, z_c, z_s, mu_c, mu_s, logvar_s = self.model(data)

            # Accumulate for EMA (detach so no graph is kept)
            mu_c_accum.append(mu_c.detach())
            y_accum.append(labels.detach())

            # lambda_var is only active once style latent is being trained (phase B+)
            lv = cfg.lambda_var if gamma > 0.0 else 0.0
            total, L_rec, L_class, L_agg, L_style, L_style_cls, L_cls, L_var = (
                calculate_f_cs_wae_loss(
                    data, labels, x_hat, z_c, z_s, mu_c, mu_s, logvar_s,
                    self.model, self.loss_fn_vgg,
                    alpha, beta, gamma, delta, eta, lv,
                )
            )

            total.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)
            self.optimizer.step()

            acc["total"]         += total.item()
            acc["rec"]           += L_rec.item()
            acc["class_mmd"]     += L_class.item()
            acc["agg_mmd"]       += L_agg.item()
            acc["style_mmd"]     += L_style.item()
            acc["style_cls_mmd"] += L_style_cls.item()
            acc["cls"]           += L_cls.item()
            acc["var"]           += L_var.item()

            pbar.set_postfix({
                "Loss":    f"{total.item():.3f}",
                "Rec":     f"{L_rec.item():.3f}",
                "Cls":     f"{L_cls.item():.4f}",
                "StyCls":  f"{L_style_cls.item():.4f}",
            })

        self.scheduler.step()

        # EMA center update (no-grad, done once per epoch)
        self.model.update_ema_centers(mu_c_accum, y_accum)

        n = len(self.train_loader)
        avg = {k: v / n for k, v in acc.items()}

        print(
            f"====> Epoch {epoch + 1}  "
            f"Total={avg['total']:.4f}  Rec={avg['rec']:.4f}  "
            f"ClassMMD={avg['class_mmd']:.4f}  AggMMD={avg['agg_mmd']:.4f}  "
            f"StyleMMD={avg['style_mmd']:.4f}  StyleClsMMD={avg['style_cls_mmd']:.4f}  "
            f"Cls={avg['cls']:.4f}  Var={avg['var']:.4f}"
        )
        return avg

    # ------------------------------------------------------------------
    # Full training loop
    # ------------------------------------------------------------------

    def train(
        self,
        epochs: int | None = None,
        *,
        start_epoch: int = 0,
        history: list[dict[str, float]] | None = None,
        checkpoint_path: str | Path | None = None,
        checkpoint_every: int = 1,
        run_config: dict[str, Any] | None = None,
    ) -> list[dict[str, float]]:
        """Run through ``epochs``, optionally checkpointing at epoch boundaries."""

        epochs = epochs or cfg.total_epochs
        history = [] if history is None else list(history)
        if checkpoint_every < 1:
            raise ValueError("checkpoint_every must be positive")
        if start_epoch < 0 or start_epoch > epochs:
            raise ValueError(f"start_epoch must be in [0, {epochs}], got {start_epoch}")
        if len(history) != start_epoch:
            raise ValueError(
                f"history length ({len(history)}) must equal start_epoch ({start_epoch})"
            )

        print(
            f"F-CS-WAE training on {self.device} — target {epochs} epochs, "
            f"starting at epoch {start_epoch + 1 if start_epoch < epochs else epochs}"
        )
        for epoch in range(start_epoch, epochs):
            avg = self.train_epoch(epoch)
            history.append(avg)
            completed_epochs = epoch + 1
            should_checkpoint = (
                checkpoint_path is not None
                and (
                    completed_epochs % checkpoint_every == 0
                    or completed_epochs == epochs
                )
            )
            if should_checkpoint:
                self.save_training_checkpoint(
                    checkpoint_path,
                    completed_epochs=completed_epochs,
                    history=history,
                    run_config=run_config or {},
                )
                print(
                    f"Checkpoint saved after epoch {completed_epochs} "
                    f"→ {checkpoint_path}"
                )

        print("Training complete.")
        return history

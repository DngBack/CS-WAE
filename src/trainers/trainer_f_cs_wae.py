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
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import lpips

from ..config_f_cs_wae import f_cs_wae_config as cfg
from ..utils.loss_f_cs_wae import (
    calculate_f_cs_wae_loss,
    fact_contract_constraints,
    joint_contract_mmd_loss,
)


TRAINING_CHECKPOINT_SCHEMA_VERSION = "fcswae-training-1.0.0"


class StyleLabelAdversary(nn.Module):
    """Audit-aligned nonlinear classifier for sampled style labels."""

    def __init__(
        self,
        style_dim: int,
        n_classes: int,
        hidden_dims: tuple[int, ...] = (128, 64),
    ) -> None:
        super().__init__()
        dims = (style_dim, *hidden_dims, n_classes)
        layers: list[nn.Module] = []
        for index, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            layers.append(nn.Linear(in_dim, out_dim))
            if index < len(dims) - 2:
                layers.append(nn.ReLU())
        self.network = nn.Sequential(*layers)

    def forward(self, style: torch.Tensor) -> torch.Tensor:
        if style.ndim != 2:
            raise ValueError("style adversary expects a rank-2 tensor")
        mean = style.mean(dim=0, keepdim=True)
        variance = style.var(dim=0, unbiased=False, keepdim=True)
        standardized = (style - mean) * torch.rsqrt(variance + 1e-5)
        return self.network(standardized)


def uniform_label_confusion_loss(logits: torch.Tensor) -> torch.Tensor:
    """KL(U || q) confusion objective; zero iff predictions are uniform."""

    if logits.ndim != 2 or logits.shape[1] < 2:
        raise ValueError("logits must be rank 2 with at least two classes")
    uniform = torch.full_like(logits, 1.0 / logits.shape[1])
    return F.kl_div(
        F.log_softmax(logits, dim=1), uniform, reduction="batchmean"
    )


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
        self.fact_dual_weights = {
            "style": float(cfg.fact_dual_init),
            "content": float(cfg.fact_dual_init),
            "dependence": float(cfg.fact_dual_init),
        }
        self.label_adversary: StyleLabelAdversary | None = None
        self.label_adversary_optimizer: optim.Optimizer | None = None
        if cfg.fact_label_adversary_weight > 0.0:
            self.label_adversary = StyleLabelAdversary(
                model.style_dim, model.n_classes
            ).to(self.device)
            self.label_adversary_optimizer = optim.Adam(
                self.label_adversary.parameters(),
                lr=cfg.fact_label_adversary_lr,
                weight_decay=cfg.fact_label_adversary_weight_decay,
            )

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
            "fact_state": {
                "dual_weights": dict(
                    getattr(
                        self,
                        "fact_dual_weights",
                        {
                            "style": float(cfg.fact_dual_init),
                            "content": float(cfg.fact_dual_init),
                            "dependence": float(cfg.fact_dual_init),
                        },
                    )
                ),
                "label_adversary_state_dict": (
                    self.label_adversary.state_dict()
                    if getattr(self, "label_adversary", None) is not None
                    else None
                ),
                "label_adversary_optimizer_state_dict": (
                    self.label_adversary_optimizer.state_dict()
                    if getattr(self, "label_adversary_optimizer", None) is not None
                    else None
                ),
            },
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
                "batch_size",
                "delta_final",
                "joint_contract_final",
                "style_sigma_floor",
                "phase_a_end",
                "phase_b_end",
                "phase_c_end",
                "phase_d_end",
                "phase_weight_freeze_epoch",
                "max_train_batches",
            )
            mismatches = {}
            for key in keys:
                default = 0.0 if key == "joint_contract_final" else None
                stored_value = stored.get(key, default)
                expected_value = expected_run_config.get(key, default)
                if stored_value != expected_value:
                    mismatches[key] = (stored_value, expected_value)
            stored_fact = bool(stored.get("fact_enabled", False))
            expected_fact = bool(expected_run_config.get("fact_enabled", False))
            if stored_fact != expected_fact:
                mismatches["fact_enabled"] = (stored_fact, expected_fact)
            if stored_fact or expected_fact:
                fact_keys = (
                    "fact_start_epoch",
                    "fact_style_only_dependence",
                    "fact_dual_lr",
                    "fact_dual_init",
                    "fact_dual_max",
                    "fact_null_draws",
                    "fact_dual_reference_style",
                    "fact_dual_reference_content",
                    "fact_dual_reference_dependence",
                    "fact_dual_step_max",
                    "fact_label_hsic_weight",
                    "fact_label_adversary_weight",
                    "fact_label_adversary_lr",
                    "fact_label_adversary_steps",
                    "fact_label_adversary_weight_decay",
                    "fact_mean_hsic_weight",
                    "fact_style_tolerance",
                    "fact_content_tolerance",
                    "fact_dependence_tolerance",
                )
                fact_defaults = {
                    "fact_label_hsic_weight": 0.0,
                    "fact_label_adversary_weight": 0.0,
                    "fact_label_adversary_lr": 1e-3,
                    "fact_label_adversary_steps": 1,
                    "fact_label_adversary_weight_decay": 1e-4,
                    "fact_mean_hsic_weight": 0.0,
                }
                for key in fact_keys:
                    default = fact_defaults.get(key)
                    stored_value = stored.get(key, default)
                    expected_value = expected_run_config.get(key, default)
                    if stored_value != expected_value:
                        mismatches[key] = (stored_value, expected_value)
            if mismatches:
                raise ValueError(f"Resume configuration mismatch: {mismatches}")

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        fact_state = checkpoint.get("fact_state", {})
        stored_duals = fact_state.get("dual_weights")
        if stored_duals is not None:
            expected_names = {"style", "content", "dependence"}
            if set(stored_duals) != expected_names:
                raise ValueError(
                    "Invalid FACT dual state: expected weights for "
                    f"{sorted(expected_names)}, got {sorted(stored_duals)}"
                )
            self.fact_dual_weights = {
                key: float(stored_duals[key]) for key in expected_names
            }
        elif not hasattr(self, "fact_dual_weights"):
            self.fact_dual_weights = {
                "style": float(cfg.fact_dual_init),
                "content": float(cfg.fact_dual_init),
                "dependence": float(cfg.fact_dual_init),
            }
        adversary_state = fact_state.get("label_adversary_state_dict")
        adversary_optimizer_state = fact_state.get(
            "label_adversary_optimizer_state_dict"
        )
        if adversary_state is not None:
            if getattr(self, "label_adversary", None) is None:
                raise ValueError(
                    "Checkpoint contains a label adversary but this run disables it"
                )
            self.label_adversary.load_state_dict(adversary_state)
            if adversary_optimizer_state is None:
                raise ValueError("Checkpoint is missing label adversary optimizer state")
            self.label_adversary_optimizer.load_state_dict(
                adversary_optimizer_state
            )
        elif getattr(self, "label_adversary", None) is not None:
            raise ValueError("Checkpoint is missing enabled label adversary state")
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
        freeze_epoch = cfg.phase_weight_freeze_epoch
        e = epoch if freeze_epoch is None else min(epoch, freeze_epoch)

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

    def get_joint_contract_weight(self, epoch: int) -> float:
        """Ramp joint matching after class-conditional priors become active."""

        final = float(cfg.joint_contract_final)
        if final <= 0.0 or epoch < cfg.phase_b_end:
            return 0.0
        if epoch < cfg.phase_c_end:
            progress = (epoch - cfg.phase_b_end) / (
                cfg.phase_c_end - cfg.phase_b_end
            )
            return 0.5 * final * progress
        progress = min(
            1.0,
            (epoch - cfg.phase_c_end) / (cfg.phase_d_end - cfg.phase_c_end),
        )
        return 0.5 * final + 0.5 * final * progress

    def fact_is_active(self, epoch: int) -> bool:
        return bool(cfg.fact_enabled and epoch >= cfg.fact_start_epoch)

    def update_fact_dual_weights(self, averages: dict[str, float]) -> None:
        """One projected dual-ascent update from epoch-average violations."""

        specifications = {
            "style": (
                "fact_style",
                float(cfg.fact_style_tolerance),
                float(cfg.fact_dual_reference_style),
            ),
            "content": (
                "fact_content",
                float(cfg.fact_content_tolerance),
                float(cfg.fact_dual_reference_content),
            ),
            "dependence": (
                "fact_dependence", float(cfg.fact_dependence_tolerance),
                float(cfg.fact_dual_reference_dependence),
            ),
        }
        for name, (metric, tolerance, reference) in specifications.items():
            step = float(cfg.fact_dual_lr) * (
                float(averages[metric]) - tolerance
            ) / reference
            if cfg.fact_dual_step_max is not None:
                step_limit = float(cfg.fact_dual_step_max)
                step = min(step_limit, max(-step_limit, step))
            updated = self.fact_dual_weights[name] + step
            self.fact_dual_weights[name] = min(
                float(cfg.fact_dual_max), max(0.0, updated)
            )

    # ------------------------------------------------------------------
    # Single epoch
    # ------------------------------------------------------------------

    def train_epoch(self, epoch: int) -> dict[str, float]:
        """Train one epoch. Returns dict of average losses."""
        self.model.train()
        if self.label_adversary is not None:
            self.label_adversary.train()
        alpha, beta, gamma, delta, eta = self.get_phase_weights(epoch)
        joint_weight = self.get_joint_contract_weight(epoch)
        fact_active = self.fact_is_active(epoch)
        loss_alpha = 0.0 if fact_active else alpha
        loss_delta = 0.0 if fact_active else delta

        acc = {k: 0.0 for k in
               ("total", "rec", "class_mmd", "agg_mmd", "style_mmd",
                "style_cls_mmd", "joint_contract_mmd", "cls", "var",
                "fact_style", "fact_content", "fact_dependence",
                "fact_label_dependence",
                "fact_mean_dependence",
                "fact_style_raw", "fact_content_raw", "fact_dependence_raw",
                "fact_label_dependence_raw",
                "fact_mean_dependence_raw",
                "fact_style_null", "fact_content_null", "fact_dependence_null",
                "fact_label_dependence_null",
                "fact_mean_dependence_null",
                "fact_represented_classes", "fact_mean_represented_classes",
                "label_adversary_ce",
                "label_adversary_accuracy", "label_adversary_confusion")}

        # Buffers for EMA center update (accumulated, detached)
        mu_c_accum: list[torch.Tensor] = []
        y_accum: list[torch.Tensor] = []

        fact_description = ""
        if fact_active:
            fact_description = (
                " FACT["
                f"λs={self.fact_dual_weights['style']:.2f} "
                f"λc={self.fact_dual_weights['content']:.2f} "
                f"λd={self.fact_dual_weights['dependence']:.2f}]"
            )
        pbar = tqdm(
            self.train_loader,
            desc=(
                f"Epoch {epoch + 1} "
                f"[α={alpha:.2f} β={beta:.2f} γ={gamma:.2f} δ={delta:.2f} "
                f"λJ={joint_weight:.2f} η={eta:.2f}]"
                f"{fact_description}"
            ),
        )

        batches_seen = 0
        for batch in pbar:
            data, labels = batch[:2]
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
                    loss_alpha, beta, gamma, loss_delta, eta, lv,
                )
            )
            L_joint = data.new_zeros(())
            if joint_weight > 0.0:
                L_joint = joint_contract_mmd_loss(z_c, z_s, labels, self.model)
                total = total + joint_weight * L_joint

            fact_values = None
            if fact_active:
                fact_values = fact_contract_constraints(
                    z_c,
                    z_s,
                    labels,
                    self.model,
                    style_only_dependence=cfg.fact_style_only_dependence,
                    null_draws=cfg.fact_null_draws,
                    label_dependence=cfg.fact_label_hsic_weight > 0.0,
                    mean_content=mu_c,
                    mean_style=mu_s,
                    mean_dependence=cfg.fact_mean_hsic_weight > 0.0,
                )
                total = total + (
                    self.fact_dual_weights["style"] * fact_values.style
                    + self.fact_dual_weights["content"] * fact_values.content
                    + self.fact_dual_weights["dependence"] * fact_values.dependence
                    + cfg.fact_label_hsic_weight * fact_values.label_dependence
                    + cfg.fact_mean_hsic_weight * fact_values.mean_dependence
                )

            adversary_active = (
                fact_active
                and self.label_adversary is not None
                and self.label_adversary_optimizer is not None
            )
            L_adversary_ce = data.new_zeros(())
            L_adversary_confusion = data.new_zeros(())
            adversary_accuracy = data.new_zeros(())
            if adversary_active:
                for _ in range(cfg.fact_label_adversary_steps):
                    self.label_adversary_optimizer.zero_grad(set_to_none=True)
                    adversary_logits = self.label_adversary(z_s.detach())
                    L_adversary_ce = F.cross_entropy(adversary_logits, labels)
                    L_adversary_ce.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.label_adversary.parameters(), cfg.grad_clip
                    )
                    self.label_adversary_optimizer.step()
                with torch.no_grad():
                    adversary_accuracy = (
                        self.label_adversary(z_s.detach()).argmax(dim=1) == labels
                    ).float().mean()
                for parameter in self.label_adversary.parameters():
                    parameter.requires_grad_(False)
                adversary_logits_for_encoder = self.label_adversary(z_s)
                L_adversary_confusion = uniform_label_confusion_loss(
                    adversary_logits_for_encoder
                )
                total = total + (
                    cfg.fact_label_adversary_weight * L_adversary_confusion
                )

            try:
                total.backward()
            finally:
                if adversary_active:
                    for parameter in self.label_adversary.parameters():
                        parameter.requires_grad_(True)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)
            self.optimizer.step()

            acc["total"]         += total.item()
            acc["rec"]           += L_rec.item()
            acc["class_mmd"]     += L_class.item()
            acc["agg_mmd"]       += L_agg.item()
            acc["style_mmd"]     += L_style.item()
            acc["style_cls_mmd"] += L_style_cls.item()
            acc["joint_contract_mmd"] += L_joint.item()
            acc["cls"]           += L_cls.item()
            acc["var"]           += L_var.item()
            if fact_values is not None:
                acc["fact_style"] += fact_values.style.item()
                acc["fact_content"] += fact_values.content.item()
                acc["fact_dependence"] += fact_values.dependence.item()
                acc["fact_label_dependence"] += fact_values.label_dependence.item()
                acc["fact_mean_dependence"] += fact_values.mean_dependence.item()
                acc["fact_style_raw"] += fact_values.style_raw.item()
                acc["fact_content_raw"] += fact_values.content_raw.item()
                acc["fact_dependence_raw"] += fact_values.dependence_raw.item()
                acc["fact_label_dependence_raw"] += (
                    fact_values.label_dependence_raw.item()
                )
                acc["fact_mean_dependence_raw"] += (
                    fact_values.mean_dependence_raw.item()
                )
                acc["fact_style_null"] += fact_values.style_null.item()
                acc["fact_content_null"] += fact_values.content_null.item()
                acc["fact_dependence_null"] += fact_values.dependence_null.item()
                acc["fact_label_dependence_null"] += (
                    fact_values.label_dependence_null.item()
                )
                acc["fact_mean_dependence_null"] += (
                    fact_values.mean_dependence_null.item()
                )
                acc["fact_represented_classes"] += float(
                    fact_values.represented_classes
                )
                acc["fact_mean_represented_classes"] += float(
                    fact_values.mean_represented_classes
                )
            acc["label_adversary_ce"] += L_adversary_ce.item()
            acc["label_adversary_accuracy"] += adversary_accuracy.item()
            acc["label_adversary_confusion"] += L_adversary_confusion.item()

            pbar.set_postfix({
                "Loss":    f"{total.item():.3f}",
                "Rec":     f"{L_rec.item():.3f}",
                "Cls":     f"{L_cls.item():.4f}",
                "StyCls":  f"{L_style_cls.item():.4f}",
                "Joint":   f"{L_joint.item():.4f}",
                "FACT": (
                    "off"
                    if fact_values is None
                    else (
                        f"{fact_values.style.item():.3f}/"
                        f"{fact_values.content.item():.3f}/"
                        f"{fact_values.dependence.item():.3f}/"
                        f"{fact_values.label_dependence.item():.3f}/"
                        f"{fact_values.mean_dependence.item():.3f}"
                    )
                ),
                "Adv": (
                    "off"
                    if not adversary_active
                    else (
                        f"{adversary_accuracy.item():.2f}/"
                        f"{L_adversary_confusion.item():.3f}"
                    )
                ),
            })
            batches_seen += 1
            if (
                cfg.max_train_batches is not None
                and batches_seen >= cfg.max_train_batches
            ):
                break

        self.scheduler.step()

        # EMA center update (no-grad, done once per epoch)
        self.model.update_ema_centers(mu_c_accum, y_accum)

        if batches_seen == 0:
            raise RuntimeError("training loader produced no batches")
        n = batches_seen
        avg = {k: v / n for k, v in acc.items()}
        if fact_active:
            self.update_fact_dual_weights(avg)
        avg.update(
            {
                "fact_lambda_style": float(self.fact_dual_weights["style"]),
                "fact_lambda_content": float(self.fact_dual_weights["content"]),
                "fact_lambda_dependence": float(
                    self.fact_dual_weights["dependence"]
                ),
            }
        )

        print(
            f"====> Epoch {epoch + 1}  "
            f"Total={avg['total']:.4f}  Rec={avg['rec']:.4f}  "
            f"ClassMMD={avg['class_mmd']:.4f}  AggMMD={avg['agg_mmd']:.4f}  "
            f"StyleMMD={avg['style_mmd']:.4f}  StyleClsMMD={avg['style_cls_mmd']:.4f}  "
            f"JointMMD={avg['joint_contract_mmd']:.4f}  "
            f"Cls={avg['cls']:.4f}  Var={avg['var']:.4f}  "
            f"FACT(s/c/d/y/m)={avg['fact_style']:.4f}/"
            f"{avg['fact_content']:.4f}/{avg['fact_dependence']:.4f}/"
            f"{avg['fact_label_dependence']:.4f}/"
            f"{avg['fact_mean_dependence']:.4f}  "
            f"Dual(s/c/d)={avg['fact_lambda_style']:.3f}/"
            f"{avg['fact_lambda_content']:.3f}/"
            f"{avg['fact_lambda_dependence']:.3f}  "
            f"Adv(acc/ce/conf)={avg['label_adversary_accuracy']:.3f}/"
            f"{avg['label_adversary_ce']:.3f}/"
            f"{avg['label_adversary_confusion']:.4f}"
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

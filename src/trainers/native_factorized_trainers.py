"""Crash-safe trainers for native DRIT and DIVA Rotated-MNIST pilots."""

from __future__ import annotations

import os
from pathlib import Path
import random
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F

from ..models.native_factorized_baselines import (
    NativeDIVA,
    NativeDRIT,
    diagonal_gaussian_kl,
)


CROSS_MODEL_CHECKPOINT_SCHEMA = "native-cross-model-training-1.0.0"


def _atomic_torch_save(payload: dict, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, target)


def _rng_state(loaders: Iterable) -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "loaders": [
            getattr(loader, "generator", None).get_state()
            if getattr(loader, "generator", None) is not None
            else None
            for loader in loaders
        ],
    }


def _restore_rng(state: dict[str, Any], loaders: Iterable) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if state.get("torch_cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])
    for loader, loader_state in zip(loaders, state.get("loaders", [])):
        generator = getattr(loader, "generator", None)
        if generator is not None and loader_state is not None:
            generator.set_state(loader_state)


def _configuration_mismatch(stored: dict, expected: dict) -> dict:
    return {
        key: (stored.get(key), value)
        for key, value in expected.items()
        if stored.get(key) != value
    }


class DIVATrainer:
    """Supervised DIVA objective with native ``z_d``, ``z_x`` and ``z_y``."""

    def __init__(
        self,
        model: NativeDIVA,
        train_loader,
        device: torch.device,
        *,
        lr: float = 1e-3,
        warmup_epochs: int = 20,
        auxiliary_domain_weight: float = 10.0,
        auxiliary_label_weight: float = 20.0,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.device = device
        self.warmup_epochs = int(warmup_epochs)
        self.auxiliary_domain_weight = float(auxiliary_domain_weight)
        self.auxiliary_label_weight = float(auxiliary_label_weight)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=100
        )

    def train_epoch(self, epoch: int) -> dict[str, float]:
        self.model.train()
        beta = min(1.0, float(epoch + 1) / max(1, self.warmup_epochs))
        totals = {
            "total": 0.0,
            "reconstruction": 0.0,
            "kl_domain": 0.0,
            "kl_style": 0.0,
            "kl_semantic": 0.0,
            "domain_ce": 0.0,
            "label_ce": 0.0,
        }
        n_batches = 0
        for images, labels, domains in self.train_loader:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            domains = domains.to(self.device, non_blocking=True)
            reconstruction, views = self.model(images)
            domain_prior_mean, domain_prior_logvar = (
                self.model.conditional_domain_prior(domains)
            )
            semantic_prior_mean, semantic_prior_logvar = (
                self.model.conditional_semantic_prior(labels)
            )
            reconstruction_loss = F.binary_cross_entropy(
                reconstruction, images, reduction="sum"
            ) / images.shape[0]
            kl_domain = diagonal_gaussian_kl(
                views.zd_mean,
                views.zd_logvar,
                domain_prior_mean,
                domain_prior_logvar,
            )
            kl_style = diagonal_gaussian_kl(
                views.zx_mean, views.zx_logvar
            )
            kl_semantic = diagonal_gaussian_kl(
                views.zy_mean,
                views.zy_logvar,
                semantic_prior_mean,
                semantic_prior_logvar,
            )
            domain_ce = F.cross_entropy(
                self.model.domain_classifier(views.zd_sample), domains
            )
            label_ce = F.cross_entropy(
                self.model.label_classifier(views.zy_sample), labels
            )
            total = (
                reconstruction_loss
                + beta * (kl_domain + kl_style + kl_semantic)
                + self.auxiliary_domain_weight * domain_ce
                + self.auxiliary_label_weight * label_ce
            )
            self.optimizer.zero_grad()
            total.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
            self.optimizer.step()
            values = {
                "total": total,
                "reconstruction": reconstruction_loss,
                "kl_domain": kl_domain,
                "kl_style": kl_style,
                "kl_semantic": kl_semantic,
                "domain_ce": domain_ce,
                "label_ce": label_ce,
            }
            for name, value in values.items():
                totals[name] += float(value.detach().item())
            n_batches += 1
        self.scheduler.step()
        result = {name: value / max(1, n_batches) for name, value in totals.items()}
        result["beta"] = beta
        return result

    def _checkpoint_payload(
        self, completed_epochs: int, history: list[dict], run_config: dict
    ) -> dict:
        return {
            "schema_version": CROSS_MODEL_CHECKPOINT_SCHEMA,
            "family": "diva",
            "completed_epochs": int(completed_epochs),
            "history": history,
            "run_config": run_config,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "rng_state": _rng_state([self.train_loader]),
        }

    def load_checkpoint(
        self, path: str | Path, expected_config: dict
    ) -> tuple[int, list[dict]]:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("schema_version") != CROSS_MODEL_CHECKPOINT_SCHEMA:
            raise ValueError("Unsupported cross-model checkpoint schema")
        mismatch = _configuration_mismatch(payload.get("run_config", {}), expected_config)
        if mismatch:
            raise ValueError(f"Resume configuration mismatch: {mismatch}")
        self.model.load_state_dict(payload["model_state_dict"])
        self.optimizer.load_state_dict(payload["optimizer_state_dict"])
        self.scheduler.load_state_dict(payload["scheduler_state_dict"])
        _restore_rng(payload["rng_state"], [self.train_loader])
        return int(payload["completed_epochs"]), list(payload["history"])

    def train(
        self,
        *,
        epochs: int,
        checkpoint_path: str | Path,
        checkpoint_every: int,
        run_config: dict,
        start_epoch: int = 0,
        history: list[dict] | None = None,
    ) -> list[dict]:
        history = [] if history is None else list(history)
        for epoch in range(start_epoch, epochs):
            metrics = self.train_epoch(epoch)
            history.append(metrics)
            print(
                f"DIVA epoch {epoch + 1}/{epochs} "
                f"loss={metrics['total']:.4f} rec={metrics['reconstruction']:.4f} "
                f"label_ce={metrics['label_ce']:.4f}",
                flush=True,
            )
            completed = epoch + 1
            if completed % checkpoint_every == 0 or completed == epochs:
                _atomic_torch_save(
                    self._checkpoint_payload(completed, history, run_config),
                    checkpoint_path,
                )
        return history


def _set_requires_grad(module: torch.nn.Module, enabled: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(enabled)


class DRITTrainer:
    """DRIT training with adversarial content alignment and cross-cycle loss."""

    def __init__(
        self,
        model: NativeDRIT,
        domain_loaders: tuple,
        device: torch.device,
        *,
        lr: float = 2e-4,
    ):
        self.model = model.to(device)
        self.domain_loaders = domain_loaders
        self.device = device
        discriminator_parameters = list(self.model.image_discriminators.parameters())
        generator_parameters = [
            parameter
            for name, parameter in self.model.named_parameters()
            if not name.startswith("image_discriminators")
            and not name.startswith("content_discriminator")
        ]
        self.generator_optimizer = torch.optim.Adam(
            generator_parameters, lr=lr, betas=(0.5, 0.999)
        )
        self.image_discriminator_optimizer = torch.optim.Adam(
            discriminator_parameters, lr=lr, betas=(0.5, 0.999)
        )
        self.content_discriminator_optimizer = torch.optim.Adam(
            self.model.content_discriminator.parameters(),
            lr=lr / 2.5,
            betas=(0.5, 0.999),
        )
        self.schedulers = [
            torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
            for optimizer in (
                self.generator_optimizer,
                self.image_discriminator_optimizer,
                self.content_discriminator_optimizer,
            )
        ]

    @staticmethod
    def _lsgan(logits: torch.Tensor, target: float) -> torch.Tensor:
        return (logits - float(target)).square().mean()

    def _encode_pair(self, images_a: torch.Tensor, images_b: torch.Tensor):
        content_a = self.model.encode_content(images_a, 0)
        content_b = self.model.encode_content(images_b, 1)
        mean_a, logvar_a, style_a, _ = self.model.encode_style(images_a, 0)
        mean_b, logvar_b, style_b, _ = self.model.encode_style(images_b, 1)
        return content_a, content_b, mean_a, logvar_a, style_a, mean_b, logvar_b, style_b

    def train_epoch(self, epoch: int) -> dict[str, float]:
        del epoch
        self.model.train()
        names = (
            "generator_total",
            "self_reconstruction",
            "cross_cycle",
            "style_kl",
            "latent_regression",
            "content_reconstruction",
            "image_adversarial",
            "content_confusion",
            "image_discriminator",
            "content_discriminator",
        )
        totals = {name: 0.0 for name in names}
        n_batches = 0
        for batch_a, batch_b in zip(*self.domain_loaders):
            images_a = batch_a[0].to(self.device, non_blocking=True)
            images_b = batch_b[0].to(self.device, non_blocking=True)
            batch_size = min(images_a.shape[0], images_b.shape[0])
            images_a, images_b = images_a[:batch_size], images_b[:batch_size]

            with torch.no_grad():
                content_a, content_b, _, _, style_a, _, _, style_b = self._encode_pair(
                    images_a, images_b
                )
                fake_a = self.model.decode(content_b, style_a, 0)
                fake_b = self.model.decode(content_a, style_b, 1)

            _set_requires_grad(self.model.image_discriminators, True)
            self.image_discriminator_optimizer.zero_grad()
            image_discriminator_loss = 0.5 * (
                self._lsgan(self.model.image_discriminators[0](images_a), 1.0)
                + self._lsgan(self.model.image_discriminators[0](fake_a.detach()), 0.0)
                + self._lsgan(self.model.image_discriminators[1](images_b), 1.0)
                + self._lsgan(self.model.image_discriminators[1](fake_b.detach()), 0.0)
            )
            image_discriminator_loss.backward()
            self.image_discriminator_optimizer.step()

            _set_requires_grad(self.model.content_discriminator, True)
            self.content_discriminator_optimizer.zero_grad()
            with torch.no_grad():
                content_a = self.model.encode_content(images_a, 0)
                content_b = self.model.encode_content(images_b, 1)
            content_logits_a = self.model.content_discriminator(content_a.detach()).squeeze(1)
            content_logits_b = self.model.content_discriminator(content_b.detach()).squeeze(1)
            content_discriminator_loss = 0.5 * (
                F.binary_cross_entropy_with_logits(
                    content_logits_a, torch.zeros_like(content_logits_a)
                )
                + F.binary_cross_entropy_with_logits(
                    content_logits_b, torch.ones_like(content_logits_b)
                )
            )
            content_discriminator_loss.backward()
            self.content_discriminator_optimizer.step()

            _set_requires_grad(self.model.image_discriminators, False)
            _set_requires_grad(self.model.content_discriminator, False)
            self.generator_optimizer.zero_grad()
            (
                content_a,
                content_b,
                mean_a,
                logvar_a,
                style_a,
                mean_b,
                logvar_b,
                style_b,
            ) = self._encode_pair(images_a, images_b)
            self_a = self.model.decode(content_a, style_a, 0)
            self_b = self.model.decode(content_b, style_b, 1)
            fake_a = self.model.decode(content_b, style_a, 0)
            fake_b = self.model.decode(content_a, style_b, 1)
            fake_a_content = self.model.encode_content(fake_a, 0)
            fake_b_content = self.model.encode_content(fake_b, 1)
            cycle_a = self.model.decode(fake_b_content, style_a, 0)
            cycle_b = self.model.decode(fake_a_content, style_b, 1)

            random_style_a = torch.randn_like(style_a)
            random_style_b = torch.randn_like(style_b)
            random_fake_a = self.model.decode(content_b, random_style_a, 0)
            random_fake_b = self.model.decode(content_a, random_style_b, 1)
            recovered_style_a = self.model.attribute_encoder(random_fake_a, 0)[0]
            recovered_style_b = self.model.attribute_encoder(random_fake_b, 1)[0]

            self_reconstruction = F.l1_loss(self_a, images_a) + F.l1_loss(
                self_b, images_b
            )
            cross_cycle = F.l1_loss(cycle_a, images_a) + F.l1_loss(
                cycle_b, images_b
            )
            style_kl = diagonal_gaussian_kl(mean_a, logvar_a) + diagonal_gaussian_kl(
                mean_b, logvar_b
            )
            latent_regression = F.l1_loss(
                recovered_style_a, random_style_a
            ) + F.l1_loss(recovered_style_b, random_style_b)
            content_reconstruction = F.l1_loss(
                fake_a_content, content_b
            ) + F.l1_loss(fake_b_content, content_a)
            image_adversarial = (
                self._lsgan(self.model.image_discriminators[0](fake_a), 1.0)
                + self._lsgan(self.model.image_discriminators[1](fake_b), 1.0)
                + self._lsgan(self.model.image_discriminators[0](random_fake_a), 1.0)
                + self._lsgan(self.model.image_discriminators[1](random_fake_b), 1.0)
            )
            confusion_a = self.model.content_discriminator(content_a).squeeze(1)
            confusion_b = self.model.content_discriminator(content_b).squeeze(1)
            content_confusion = F.binary_cross_entropy_with_logits(
                confusion_a, torch.full_like(confusion_a, 0.5)
            ) + F.binary_cross_entropy_with_logits(
                confusion_b, torch.full_like(confusion_b, 0.5)
            )
            generator_total = (
                10.0 * self_reconstruction
                + 10.0 * cross_cycle
                + 0.01 * style_kl
                + 10.0 * latent_regression
                + content_reconstruction
                + image_adversarial
                + content_confusion
                + 0.01 * (content_a.square().mean() + content_b.square().mean())
            )
            generator_total.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
            self.generator_optimizer.step()
            _set_requires_grad(self.model.image_discriminators, True)
            _set_requires_grad(self.model.content_discriminator, True)

            values = {
                "generator_total": generator_total,
                "self_reconstruction": self_reconstruction,
                "cross_cycle": cross_cycle,
                "style_kl": style_kl,
                "latent_regression": latent_regression,
                "content_reconstruction": content_reconstruction,
                "image_adversarial": image_adversarial,
                "content_confusion": content_confusion,
                "image_discriminator": image_discriminator_loss,
                "content_discriminator": content_discriminator_loss,
            }
            for name, value in values.items():
                totals[name] += float(value.detach().item())
            n_batches += 1
        for scheduler in self.schedulers:
            scheduler.step()
        return {name: value / max(1, n_batches) for name, value in totals.items()}

    def _checkpoint_payload(
        self, completed_epochs: int, history: list[dict], run_config: dict
    ) -> dict:
        return {
            "schema_version": CROSS_MODEL_CHECKPOINT_SCHEMA,
            "family": "drit",
            "completed_epochs": int(completed_epochs),
            "history": history,
            "run_config": run_config,
            "model_state_dict": self.model.state_dict(),
            "generator_optimizer": self.generator_optimizer.state_dict(),
            "image_discriminator_optimizer": self.image_discriminator_optimizer.state_dict(),
            "content_discriminator_optimizer": self.content_discriminator_optimizer.state_dict(),
            "schedulers": [scheduler.state_dict() for scheduler in self.schedulers],
            "rng_state": _rng_state(self.domain_loaders),
        }

    def load_checkpoint(
        self, path: str | Path, expected_config: dict
    ) -> tuple[int, list[dict]]:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("schema_version") != CROSS_MODEL_CHECKPOINT_SCHEMA:
            raise ValueError("Unsupported cross-model checkpoint schema")
        mismatch = _configuration_mismatch(payload.get("run_config", {}), expected_config)
        if mismatch:
            raise ValueError(f"Resume configuration mismatch: {mismatch}")
        self.model.load_state_dict(payload["model_state_dict"])
        self.generator_optimizer.load_state_dict(payload["generator_optimizer"])
        self.image_discriminator_optimizer.load_state_dict(
            payload["image_discriminator_optimizer"]
        )
        self.content_discriminator_optimizer.load_state_dict(
            payload["content_discriminator_optimizer"]
        )
        for scheduler, state in zip(self.schedulers, payload["schedulers"]):
            scheduler.load_state_dict(state)
        _restore_rng(payload["rng_state"], self.domain_loaders)
        return int(payload["completed_epochs"]), list(payload["history"])

    def train(
        self,
        *,
        epochs: int,
        checkpoint_path: str | Path,
        checkpoint_every: int,
        run_config: dict,
        start_epoch: int = 0,
        history: list[dict] | None = None,
    ) -> list[dict]:
        history = [] if history is None else list(history)
        for epoch in range(start_epoch, epochs):
            metrics = self.train_epoch(epoch)
            history.append(metrics)
            print(
                f"DRIT epoch {epoch + 1}/{epochs} "
                f"G={metrics['generator_total']:.4f} "
                f"self={metrics['self_reconstruction']:.4f} "
                f"cycle={metrics['cross_cycle']:.4f}",
                flush=True,
            )
            completed = epoch + 1
            if completed % checkpoint_every == 0 or completed == epochs:
                _atomic_torch_save(
                    self._checkpoint_payload(completed, history, run_config),
                    checkpoint_path,
                )
        return history


__all__ = ["DIVATrainer", "DRITTrainer"]

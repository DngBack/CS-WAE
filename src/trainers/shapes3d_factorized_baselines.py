"""Crash-safe trainers for the Shapes3D native factorized baselines."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from ..models.shapes3d_factorized_baselines import (
    Shapes3DConditionalVAE,
    Shapes3DContentStyleVAE,
)
from .trainer_f_cs_wae import atomic_torch_save


BASELINE_TRAINING_SCHEMA = "shapes3d-factorized-baseline-training-1.0.0"


def _capture_rng(loader) -> dict[str, Any]:
    generator = getattr(loader, "generator", None)
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "loader": generator.get_state() if generator is not None else None,
    }


def _restore_rng(state: dict[str, Any], loader) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if state.get("torch_cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])
    generator = getattr(loader, "generator", None)
    if state.get("loader") is not None and generator is not None:
        generator.set_state(state["loader"])


class Shapes3DFactorizedBaselineTrainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_loader,
        *,
        device: torch.device,
        learning_rate: float = 1e-3,
        kl_final: float = 0.01,
        classifier_weight: float = 0.1,
        warmup_epochs: int = 20,
    ) -> None:
        if not isinstance(model, (Shapes3DConditionalVAE, Shapes3DContentStyleVAE)):
            raise TypeError(f"Unsupported Shapes3D baseline: {type(model).__name__}")
        self.model = model.to(device)
        self.train_loader = train_loader
        self.device = device
        self.kl_final = float(kl_final)
        self.classifier_weight = float(classifier_weight)
        self.warmup_epochs = int(warmup_epochs)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=50, gamma=0.5
        )

    def _weights(self, epoch: int) -> tuple[float, float]:
        # Start with a small non-zero KL coefficient so the first epoch never
        # leaves Gaussian heads completely unconstrained.
        progress = min(1.0, max(0.0, (epoch + 1) / max(1, self.warmup_epochs)))
        return self.kl_final * progress, self.classifier_weight

    def train_epoch(self, epoch: int) -> dict[str, float]:
        self.model.train()
        kl_weight, classifier_weight = self._weights(epoch)
        totals = {
            "total": 0.0,
            "reconstruction": 0.0,
            "style_kl": 0.0,
            "content_kl": 0.0,
            "shape_ce": 0.0,
        }
        progress = tqdm(
            self.train_loader,
            desc=f"{self.model.model_family} epoch {epoch + 1}",
        )
        for images, labels, _factors in progress:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            self.optimizer.zero_grad()
            if isinstance(self.model, Shapes3DConditionalVAE):
                terms = self.model.loss_terms(
                    images, labels, kl_weight=kl_weight
                )
            else:
                terms = self.model.loss_terms(
                    images,
                    labels,
                    kl_weight=kl_weight,
                    classifier_weight=classifier_weight,
                )
            terms["total"].backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            for name in totals:
                totals[name] += float(terms[name].detach().item())
            progress.set_postfix(
                loss=f"{terms['total'].item():.4f}",
                rec=f"{terms['reconstruction'].item():.4f}",
            )
        self.scheduler.step()
        count = len(self.train_loader)
        result = {name: value / count for name, value in totals.items()}
        result["kl_weight"] = kl_weight
        result["classifier_weight"] = classifier_weight
        print(
            f"{self.model.model_family} epoch {epoch + 1}: "
            f"total={result['total']:.5f} rec={result['reconstruction']:.5f} "
            f"style_kl={result['style_kl']:.5f} "
            f"content_kl={result['content_kl']:.5f} "
            f"shape_ce={result['shape_ce']:.5f}",
            flush=True,
        )
        return result

    def save_checkpoint(
        self,
        path: str | Path,
        *,
        completed_epochs: int,
        history: list[dict[str, float]],
        run_config: dict[str, Any],
    ) -> None:
        atomic_torch_save(
            {
                "schema_version": BASELINE_TRAINING_SCHEMA,
                "completed_epochs": int(completed_epochs),
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "history": history,
                "run_config": run_config,
                "rng_state": _capture_rng(self.train_loader),
            },
            path,
        )

    def load_checkpoint(
        self, path: str | Path, *, expected_run_config: dict[str, Any]
    ) -> tuple[int, list[dict[str, float]]]:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("schema_version") != BASELINE_TRAINING_SCHEMA:
            raise ValueError("Unsupported Shapes3D baseline checkpoint schema")
        if payload.get("run_config") != expected_run_config:
            raise ValueError("Shapes3D baseline resume configuration mismatch")
        self.model.load_state_dict(payload["model_state_dict"])
        self.optimizer.load_state_dict(payload["optimizer_state_dict"])
        self.scheduler.load_state_dict(payload["scheduler_state_dict"])
        _restore_rng(payload["rng_state"], self.train_loader)
        completed = int(payload["completed_epochs"])
        history = list(payload["history"])
        if len(history) != completed:
            raise ValueError("Shapes3D baseline checkpoint history length mismatch")
        return completed, history

    def train(
        self,
        *,
        epochs: int,
        checkpoint_path: str | Path,
        checkpoint_every: int,
        run_config: dict[str, Any],
        auto_resume: bool,
    ) -> list[dict[str, float]]:
        start, history = 0, []
        checkpoint = Path(checkpoint_path)
        if auto_resume and checkpoint.exists():
            start, history = self.load_checkpoint(
                checkpoint, expected_run_config=run_config
            )
        print(
            f"{self.model.model_family} training on {self.device}: "
            f"target={epochs}, starting={start + 1 if start < epochs else epochs}",
            flush=True,
        )
        for epoch in range(start, epochs):
            history.append(self.train_epoch(epoch))
            completed = epoch + 1
            if completed % checkpoint_every == 0 or completed == epochs:
                self.save_checkpoint(
                    checkpoint,
                    completed_epochs=completed,
                    history=history,
                    run_config=run_config,
                )
                print(f"Checkpoint saved after epoch {completed} → {checkpoint}", flush=True)
        return history


__all__ = ["Shapes3DFactorizedBaselineTrainer"]

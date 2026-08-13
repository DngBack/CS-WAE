#!/usr/bin/env python3
"""Train and audit F-CS-WAE on Shapes3D ground-truth factors."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import Ridge
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision.utils import make_grid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config_f_cs_wae import f_cs_wae_config as fcfg
from src.datasets.shapes3d import (
    DEFAULT_SPLIT_COUNTS,
    FACTOR_CARDINALITIES,
    FACTOR_NAMES,
    SHAPE_FACTOR_INDEX,
    SHAPES3D_SHA256,
    build_shapes3d_loaders,
    save_shapes3d_manifest,
    verify_shapes3d_file,
)
from src.metrics.audit_protocol import (
    AUDIT_PROTOCOL_VERSION,
    conditional_mmd_to_standard_normal,
    hsic_permutation_test,
    mmd2_permutation_test,
    run_probe_suite,
    stratified_probe_split,
)
from src.metrics.factorized_adapter import (
    build_factorized_audit_adapter,
    summarize_style_posterior,
)
from src.metrics.statistical_reporting import holm_family
from src.models.f_cs_wae import FCSWAE
from src.models.shapes3d_factor_evaluator import Shapes3DFactorEvaluator
from src.trainers.trainer_f_cs_wae import FCSWAETrainer, atomic_torch_save
from src.utils.provenance import sha256_file


SHAPES3D_PROTOCOL_VERSION = "shapes3d-factor-audit-1.0.0"
SPLIT_SEED = 2027
AUDIT_PER_SHAPE = 1024
MMD_SAMPLES = 2048
HSIC_SAMPLES = 1024
FACTOR_EVALUATOR_THRESHOLDS = {
    "floor_hue": 0.90,
    "wall_hue": 0.90,
    "object_hue": 0.90,
    "scale": 0.90,
    "shape": 0.95,
    "orientation": 0.90,
}
MODEL_COMPETENCE_THRESHOLDS = {
    "reconstruction_l1_max": 0.12,
    "content_shape_logistic_accuracy_min": 0.80,
    "reconstruction_shape_accuracy_min": 0.80,
    "reconstruction_style_factor_accuracy_mean_min": 0.70,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("verify-data", "evaluator", "train", "audit", "seed0"),
        default="seed0",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--data", default="data/shapes3d/3dshapes.h5")
    parser.add_argument("--output-root", default="runs_shapes3d/factor_audit_v1")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--evaluator-epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--evaluator-batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--checkpoint-every", type=int, default=5)
    parser.add_argument("--probe-epochs", type=int, default=300)
    parser.add_argument("--mmd-permutations", type=int, default=500)
    parser.add_argument("--hsic-permutations", type=int, default=200)
    parser.add_argument("--auto-resume", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument(
        "--skip-source-hash",
        action="store_true",
        help="Development-only: validate HDF5 schema but skip full SHA-256.",
    )
    return parser.parse_args()


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _atomic_json(path: Path, payload: Any, *, checksum: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True))
    os.replace(temporary, path)
    if checksum:
        digest = sha256_file(path)
        sidecar = path.with_suffix(path.suffix + ".sha256")
        temporary_sidecar = sidecar.with_name(f".{sidecar.name}.tmp")
        temporary_sidecar.write_text(f"{digest}  {path.name}\n")
        os.replace(temporary_sidecar, sidecar)


def _git_metadata() -> dict:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=ROOT, text=True,
            stderr=subprocess.DEVNULL,
        ).strip())
        return {"commit": commit, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def _implementation_hashes() -> dict[str, str]:
    files = (
        "src/datasets/shapes3d.py",
        "src/metrics/audit_protocol.py",
        "src/metrics/factorized_adapter.py",
        "src/metrics/statistical_reporting.py",
        "src/models/backbone.py",
        "src/models/f_cs_wae.py",
        "src/models/shapes3d_factor_evaluator.py",
        "src/trainers/trainer_f_cs_wae.py",
        "scripts/run_shapes3d_factor_audit.py",
    )
    return {name: sha256_file(ROOT / name) for name in files}


def _root(args: argparse.Namespace) -> Path:
    return ROOT / args.output_root


def _run_dir(args: argparse.Namespace) -> Path:
    return _root(args) / "fcswae" / f"seed_{args.seed}"


def _evaluator_dir(args: argparse.Namespace) -> Path:
    return _root(args) / "evaluator" / "seed_0"


def _model_config() -> dict:
    return {
        "semantic_dim": 64,
        "style_dim": 64,
        "n_classes": 4,
        "in_channels": 3,
        "image_size": 64,
        "n_centers": 1,
        "rho_prior": 0.7,
        "ema_momentum": 0.95,
        "style_sigma_floor": 0.0,
        "conditional_style_mmd_weight": 0.0,
    }


def _run_config(args: argparse.Namespace) -> dict:
    return {
        "shapes3d_protocol_version": SHAPES3D_PROTOCOL_VERSION,
        "dataset": "shapes3d",
        "dataset_sha256": SHAPES3D_SHA256,
        "split_seed": SPLIT_SEED,
        "split_counts": dict(DEFAULT_SPLIT_COUNTS),
        "seed": int(args.seed),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "model": _model_config(),
        "n_centers": 1,
        "semantic_dim": 64,
        "style_dim": 64,
        "delta_final": 0.0,
        "style_sigma_floor": 0.0,
        "phase_a_end": 20,
        "phase_b_end": 40,
    }


def _acceptance_payload(args: argparse.Namespace, source: dict) -> dict:
    return {
        "shapes3d_protocol_version": SHAPES3D_PROTOCOL_VERSION,
        "stage0_audit_protocol_version": AUDIT_PROTOCOL_VERSION,
        "source": source,
        "split": {
            "seed": SPLIT_SEED,
            "counts": dict(DEFAULT_SPLIT_COUNTS),
            "type": "shape-stratified IID; no compositional holdout",
        },
        "designated_semantic_factor": "shape",
        "style_factors": [name for name in FACTOR_NAMES if name != "shape"],
        "model": _model_config(),
        "training": {
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "factor_evaluator_epochs": int(args.evaluator_epochs),
            "factor_evaluator_batch_size": int(args.evaluator_batch_size),
            "fixed_final_epoch": True,
            "no_audit_based_checkpoint_selection": True,
        },
        "evaluation": {
            "audit_per_shape": AUDIT_PER_SHAPE,
            "mmd_samples": MMD_SAMPLES,
            "hsic_samples": HSIC_SAMPLES,
            "mmd_permutations": int(args.mmd_permutations),
            "hsic_permutations": int(args.hsic_permutations),
            "multiple_testing": "Holm correction separately within each latent view",
            "primary_latent_view": "stochastic posterior sample",
        },
        "factor_evaluator_thresholds": FACTOR_EVALUATOR_THRESHOLDS,
        "model_competence_thresholds": MODEL_COMPETENCE_THRESHOLDS,
    }


def _prepare(args: argparse.Namespace):
    source = verify_shapes3d_file(
        ROOT / args.data, verify_sha256=not args.skip_source_hash
    )
    acceptance = _root(args) / "acceptance_criteria.json"
    payload = _acceptance_payload(args, source)
    if acceptance.exists():
        existing = json.loads(acceptance.read_text())
        if existing != payload:
            raise ValueError(f"Frozen Shapes3D criteria differ: {acceptance}")
    else:
        _atomic_json(acceptance, payload, checksum=True)
    return source


def _loaders(args: argparse.Namespace, *, batch_size: int | None = None):
    return build_shapes3d_loaders(
        path=ROOT / args.data,
        seed=args.seed,
        split_seed=SPLIT_SEED,
        batch_size=batch_size or args.batch_size,
        num_workers=args.num_workers,
        split_counts=DEFAULT_SPLIT_COUNTS,
        verify_sha256=not args.skip_source_hash,
    )


def _factor_losses(logits: dict[str, torch.Tensor], factors: torch.Tensor):
    values = {
        name: F.cross_entropy(logits[name], factors[:, column])
        for column, name in enumerate(FACTOR_NAMES)
    }
    return sum(values.values()) / len(values), values


@torch.no_grad()
def evaluate_factor_evaluator(model, loader, device: torch.device) -> dict:
    model.eval()
    correct = {name: 0 for name in FACTOR_NAMES}
    loss_sum = {name: 0.0 for name in FACTOR_NAMES}
    total = 0
    for images, _shape, factors in loader:
        images = images.to(device, non_blocking=True)
        factors = factors.to(device, non_blocking=True)
        logits = model(images)
        for column, name in enumerate(FACTOR_NAMES):
            targets = factors[:, column]
            loss_sum[name] += float(
                F.cross_entropy(logits[name], targets, reduction="sum").item()
            )
            correct[name] += int((logits[name].argmax(1) == targets).sum().item())
        total += images.shape[0]
    return {
        "n_samples": total,
        "accuracy": {name: correct[name] / total for name in FACTOR_NAMES},
        "cross_entropy_nats": {name: loss_sum[name] / total for name in FACTOR_NAMES},
        "macro_accuracy": float(np.mean([correct[name] / total for name in FACTOR_NAMES])),
    }


def train_evaluator(args: argparse.Namespace) -> Path:
    directory = _evaluator_dir(args)
    final_path = directory / "factor_evaluator.pth"
    result_path = directory / "result.json"
    if args.skip_existing and final_path.exists() and result_path.exists():
        result = json.loads(result_path.read_text())
        if not result["passes_competence_gate"]:
            raise RuntimeError("Stored Shapes3D factor evaluator failed its frozen gate")
        return final_path
    directory.mkdir(parents=True, exist_ok=True)
    _set_seed(0)
    original_seed = args.seed
    args.seed = 0
    loaders, manifest = _loaders(args, batch_size=args.evaluator_batch_size)
    args.seed = original_seed
    save_shapes3d_manifest(directory / "split_manifest.json", manifest)
    device = torch.device(args.device)
    model = Shapes3DFactorEvaluator().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.evaluator_epochs
    )
    checkpoint = directory / "training_checkpoint.pt"
    start_epoch, history = 0, []
    if args.auto_resume and checkpoint.exists():
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        expected = {
            "schema_version": "shapes3d-factor-evaluator-training-1.0.0",
            "target_epochs": args.evaluator_epochs,
        }
        for key, value in expected.items():
            if payload.get(key) != value:
                raise ValueError(f"Evaluator resume mismatch for {key}")
        model.load_state_dict(payload["state_dict"])
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        scheduler.load_state_dict(payload["scheduler_state_dict"])
        start_epoch = int(payload["completed_epochs"])
        history = list(payload["history"])
        torch.set_rng_state(payload["torch_rng_state"])
        if torch.cuda.is_available() and payload.get("cuda_rng_state") is not None:
            torch.cuda.set_rng_state_all(payload["cuda_rng_state"])
        if loaders["train"].generator is not None:
            loaders["train"].generator.set_state(payload["loader_rng_state"])

    for epoch in range(start_epoch, args.evaluator_epochs):
        model.train()
        totals = {name: 0.0 for name in FACTOR_NAMES}
        total_loss = 0.0
        n = 0
        for images, _shape, factors in loaders["train"]:
            images = images.to(device, non_blocking=True)
            factors = factors.to(device, non_blocking=True)
            logits = model(images)
            loss, components = _factor_losses(logits, factors)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            batch = images.shape[0]
            total_loss += float(loss.detach().item()) * batch
            for name, value in components.items():
                totals[name] += float(value.detach().item()) * batch
            n += batch
        scheduler.step()
        validation = evaluate_factor_evaluator(model, loaders["validation"], device)
        row = {
            "epoch": epoch + 1,
            "train_loss": total_loss / n,
            "train_factor_ce": {name: totals[name] / n for name in FACTOR_NAMES},
            "validation": validation,
        }
        history.append(row)
        print(
            f"Factor evaluator epoch {epoch + 1}/{args.evaluator_epochs} "
            f"loss={row['train_loss']:.4f} val_macro={validation['macro_accuracy']:.4f}",
            flush=True,
        )
        atomic_torch_save(
            {
                "schema_version": "shapes3d-factor-evaluator-training-1.0.0",
                "target_epochs": args.evaluator_epochs,
                "completed_epochs": epoch + 1,
                "state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "history": history,
                "torch_rng_state": torch.get_rng_state(),
                "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                "loader_rng_state": loaders["train"].generator.get_state(),
            },
            checkpoint,
        )
    test = evaluate_factor_evaluator(model, loaders["test"], device)
    checks = {
        name: test["accuracy"][name] >= FACTOR_EVALUATOR_THRESHOLDS[name]
        for name in FACTOR_NAMES
    }
    atomic_torch_save(
        {
            "schema_version": "shapes3d-factor-evaluator-final-1.0.0",
            "state_dict": model.state_dict(),
            "factor_names": list(FACTOR_NAMES),
            "factor_cardinalities": list(FACTOR_CARDINALITIES),
        },
        final_path,
    )
    result = {
        "schema_version": "shapes3d-factor-evaluator-result-1.0.0",
        "protocol_version": SHAPES3D_PROTOCOL_VERSION,
        "fixed_final_epoch": args.evaluator_epochs,
        "validation": history[-1]["validation"],
        "test": test,
        "competence_checks": checks,
        "passes_competence_gate": all(checks.values()),
        "checkpoint": {"path": str(final_path.resolve()), "sha256": sha256_file(final_path)},
        "split_manifest_sha256": sha256_file(directory / "split_manifest.json"),
    }
    _atomic_json(result_path, result, checksum=True)
    if not result["passes_competence_gate"]:
        raise RuntimeError(f"Shapes3D factor evaluator failed gate: {checks}")
    return final_path


def _build_fcswae() -> FCSWAE:
    config = _model_config()
    kwargs = {key: value for key, value in config.items() if key != "conditional_style_mmd_weight"}
    return FCSWAE(**kwargs)


def _configure_training(args: argparse.Namespace) -> None:
    fcfg.semantic_dim = 64
    fcfg.style_dim = 64
    fcfg.n_classes = 4
    fcfg.in_channels = 3
    fcfg.image_size = 64
    fcfg.batch_size = args.batch_size
    fcfg.total_epochs = args.epochs
    fcfg.phase_a_end = 20
    fcfg.phase_b_end = 40
    fcfg.phase_c_end = 70
    fcfg.phase_d_end = args.epochs
    fcfg.lr_scheduler_step = 50
    fcfg.delta_final = 0.0
    fcfg.style_sigma_floor = 0.0


def train_model(args: argparse.Namespace) -> Path:
    directory = _run_dir(args)
    final_path = directory / "model.pth"
    if args.skip_existing and final_path.exists():
        return final_path
    directory.mkdir(parents=True, exist_ok=True)
    _set_seed(args.seed)
    loaders, manifest = _loaders(args)
    save_shapes3d_manifest(directory / "split_manifest.json", manifest)
    run_config = _run_config(args)
    _atomic_json(directory / "run_config.json", run_config)
    _configure_training(args)
    device = torch.device(args.device)
    model = _build_fcswae()
    trainer = FCSWAETrainer(model, loaders["train"], device=device)
    checkpoint = directory / "training_checkpoint.pt"
    start_epoch, history = 0, []
    if args.auto_resume and checkpoint.exists():
        start_epoch, history = trainer.load_training_checkpoint(
            checkpoint, expected_run_config=run_config
        )
    history = trainer.train(
        epochs=args.epochs,
        start_epoch=start_epoch,
        history=history,
        checkpoint_path=checkpoint,
        checkpoint_every=args.checkpoint_every,
        run_config=run_config,
    )
    _atomic_json(directory / "training_history.json", history)
    atomic_torch_save(
        {
            "schema_version": "shapes3d-fcswae-final-1.0.0",
            "run_config": run_config,
            "state_dict": model.state_dict(),
        },
        final_path,
    )
    return final_path


def _load_models(args: argparse.Namespace, device: torch.device):
    model_path = _run_dir(args) / "model.pth"
    payload = torch.load(model_path, map_location="cpu", weights_only=False)
    if payload.get("schema_version") != "shapes3d-fcswae-final-1.0.0":
        raise ValueError(f"Not a Shapes3D F-CS-WAE checkpoint: {model_path}")
    model = _build_fcswae().to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    evaluator_path = _evaluator_dir(args) / "factor_evaluator.pth"
    evaluator_payload = torch.load(
        evaluator_path, map_location="cpu", weights_only=False
    )
    evaluator = Shapes3DFactorEvaluator().to(device)
    evaluator.load_state_dict(evaluator_payload["state_dict"])
    evaluator.eval()
    return model, evaluator, model_path, evaluator_path


def _balanced_positions(labels: torch.Tensor, n_samples: int, seed: int) -> torch.Tensor:
    n_classes = int(labels.max().item()) + 1
    per_class = n_samples // n_classes
    if per_class < 2:
        raise ValueError("Balanced subset is too small")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    selected = []
    labels_cpu = labels.cpu()
    for class_index in range(n_classes):
        positions = torch.nonzero(labels_cpu == class_index, as_tuple=False).flatten()
        if positions.numel() < per_class:
            raise ValueError(f"Factor class {class_index} has too few samples")
        order = torch.randperm(positions.numel(), generator=generator)
        selected.append(positions[order[:per_class]])
    packed = torch.cat(selected)
    return packed[torch.randperm(packed.numel(), generator=generator)]


def _circular_probe(features: torch.Tensor, labels: torch.Tensor, seed: int) -> dict:
    x = features.cpu().float().numpy()
    y = labels.cpu().long().numpy()
    split = stratified_probe_split(y, seed=seed)
    mean = x[split.train].mean(0, keepdims=True)
    scale = x[split.train].std(0, keepdims=True)
    scale[scale < 1e-8] = 1.0
    x = (x - mean) / scale
    angles_degrees = np.linspace(-30.0, 30.0, 15)[y]
    radians = np.deg2rad(angles_degrees)
    target = np.stack([np.sin(radians), np.cos(radians)], axis=1)
    model = Ridge(alpha=1.0)
    model.fit(x[split.train], target[split.train])
    prediction = model.predict(x[split.test])
    predicted_angle = np.arctan2(prediction[:, 0], prediction[:, 1])
    true_angle = radians[split.test]
    difference = np.angle(np.exp(1j * (predicted_angle - true_angle)))
    return {
        "mean_absolute_circular_error_degrees": float(
            np.mean(np.abs(np.rad2deg(difference)))
        ),
        "split": split.metadata(),
        "estimator": "ridge regression on sin/cos orientation targets",
    }


def _factor_audit(
    features: torch.Tensor,
    factors: torch.Tensor,
    *,
    latent_name: str,
    args: argparse.Namespace,
    device: torch.device,
) -> dict:
    cells = {}
    raw_p = []
    for column, name in enumerate(FACTOR_NAMES):
        labels = factors[:, column]
        seed = 40_000 + 100 * args.seed + 10 * column + (0 if latent_name == "content" else 1)
        with torch.enable_grad():
            probes = run_probe_suite(
                features,
                labels,
                seed=seed,
                probe_names=("logistic", "rbf_svm"),
                epochs=args.probe_epochs,
            )
        hsic_positions = _balanced_positions(labels, HSIC_SAMPLES, seed + 1)
        dependence = hsic_permutation_test(
            features[hsic_positions].to(device),
            labels[hsic_positions].to(device),
            seed=seed + 2,
            n_permutations=args.hsic_permutations,
        )
        raw_p.append(dependence["p_value"])
        cell = {"probes": probes, "hsic": dependence}
        if name == "orientation":
            cell["circular_probe"] = _circular_probe(features, labels, seed + 3)
        cells[name] = cell
    correction = holm_family(raw_p)
    for index, name in enumerate(FACTOR_NAMES):
        cells[name]["hsic"]["holm_adjusted_p_value"] = correction["adjusted_p_values"][index]
        cells[name]["hsic"]["holm_reject"] = correction["reject"][index]
    return {
        "latent_view": latent_name,
        "multiple_testing": correction,
        "factors": cells,
    }


@torch.no_grad()
def _score_images(
    evaluator: Shapes3DFactorEvaluator,
    images: torch.Tensor,
    targets: torch.Tensor,
    batch_size: int,
) -> dict:
    correct = {name: 0 for name in FACTOR_NAMES}
    total = images.shape[0]
    for start in range(0, total, batch_size):
        stop = min(total, start + batch_size)
        logits = evaluator(images[start:stop])
        for column, name in enumerate(FACTOR_NAMES):
            correct[name] += int(
                (logits[name].argmax(1) == targets[start:stop, column]).sum().item()
            )
    return {name: correct[name] / total for name in FACTOR_NAMES}


def _different_shape_style_donors(factors: torch.Tensor, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    groups = [
        torch.nonzero(factors[:, SHAPE_FACTOR_INDEX] == value, as_tuple=False).flatten()
        for value in range(4)
    ]
    if len({group.numel() for group in groups}) != 1:
        raise ValueError("Swap population must be exactly shape-balanced")
    donor = torch.empty(factors.shape[0], dtype=torch.long)
    for shape, positions in enumerate(groups):
        target_group = groups[(shape + 1) % 4]
        order = torch.randperm(target_group.numel(), generator=generator)
        donor[positions] = target_group[order]
    return donor


def _plot_factor_heatmap(audit: dict, output: Path) -> None:
    values = np.asarray(
        [
            [
                audit[latent]["factors"][name]["probes"]["models"]["logistic"]["test"]["accuracy"]
                for name in FACTOR_NAMES
            ]
            for latent in ("content", "style")
        ]
    )
    fig, axis = plt.subplots(figsize=(9.2, 2.8))
    image = axis.imshow(values, vmin=0.0, vmax=1.0, cmap="viridis", aspect="auto")
    axis.set_xticks(range(len(FACTOR_NAMES)), FACTOR_NAMES, rotation=25, ha="right")
    axis.set_yticks([0, 1], ["content z_c", "style z_s"])
    for row in range(2):
        for column in range(len(FACTOR_NAMES)):
            axis.text(column, row, f"{values[row, column]:.2f}", ha="center", va="center", color="white" if values[row, column] < 0.55 else "black")
    fig.colorbar(image, ax=axis, label="held-out logistic accuracy")
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_swap_grid(
    source: torch.Tensor,
    donor: torch.Tensor,
    swapped: torch.Tensor,
    output: Path,
) -> None:
    count = min(8, source.shape[0])
    grid = make_grid(
        torch.cat([source[:count], donor[:count], swapped[:count]], dim=0).cpu(),
        nrow=count,
        padding=2,
    )
    fig, axis = plt.subplots(figsize=(12, 4.8))
    axis.imshow(grid.permute(1, 2, 0).clamp(0, 1).numpy())
    axis.axis("off")
    axis.set_title("row 1: content source; row 2: style donor; row 3: decoded swap")
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def audit_model(args: argparse.Namespace) -> Path:
    directory = _run_dir(args)
    output = directory / "factor_audit.json"
    if args.skip_existing and output.exists():
        payload = json.loads(output.read_text())
        if payload.get("manifest", {}).get("protocol_version") == SHAPES3D_PROTOCOL_VERSION:
            return output
    device = torch.device(args.device)
    model, evaluator, model_path, evaluator_path = _load_models(args, device)
    loaders, split_manifest = _loaders(args)
    test_dataset = loaders["test"].dataset
    shape_labels = torch.from_numpy(test_dataset.factors[:, SHAPE_FACTOR_INDEX])
    positions = _balanced_positions(
        shape_labels, 4 * AUDIT_PER_SHAPE, 30_000 + args.seed
    )
    audit_loader = DataLoader(
        Subset(test_dataset, positions.tolist()),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
    )
    adapter = build_factorized_audit_adapter(model)
    posterior_generator = torch.Generator(device="cpu").manual_seed(31_000 + args.seed)
    prior_generator = torch.Generator(device="cpu").manual_seed(32_000 + args.seed)
    content, style, style_mean, logvar, factors_all, images_all = [], [], [], [], [], []
    reconstructions, prior_images = [], []
    reconstruction_abs = 0.0
    reconstruction_pixels = 0
    with torch.no_grad():
        for images, _shape, factors in audit_loader:
            images = images.to(device, non_blocking=True)
            factors = factors.to(device, non_blocking=True)
            views = adapter.encode_views(images, generator=posterior_generator)
            reconstruction = adapter.decode(views.content_sample, views.style_sample)
            prior_style = adapter.sample_style_prior(
                images.shape[0], generator=prior_generator, device=device,
                dtype=views.style_sample.dtype,
            )
            generated = adapter.decode(views.content_sample, prior_style)
            reconstruction_abs += float((reconstruction - images).abs().sum().item())
            reconstruction_pixels += images.numel()
            content.append(views.content_sample.cpu())
            style.append(views.style_sample.cpu())
            style_mean.append(views.style_mean.cpu())
            logvar.append(views.style_logvar.cpu())
            factors_all.append(factors.cpu())
            images_all.append(images.cpu())
            reconstructions.append(reconstruction.cpu())
            prior_images.append(generated.cpu())

    content = torch.cat(content)
    style = torch.cat(style)
    style_mean = torch.cat(style_mean)
    logvar = torch.cat(logvar)
    factors = torch.cat(factors_all)
    images_cpu = torch.cat(images_all)
    recon_cpu = torch.cat(reconstructions)
    prior_cpu = torch.cat(prior_images)
    labels_shape = factors[:, SHAPE_FACTOR_INDEX]

    factor_audit = {
        "content": _factor_audit(
            content, factors, latent_name="content", args=args, device=device
        ),
        "style": _factor_audit(
            style, factors, latent_name="style", args=args, device=device
        ),
    }
    mmd_positions = _balanced_positions(labels_shape, MMD_SAMPLES, 33_000 + args.seed)
    style_mmd = style[mmd_positions].to(device)
    reference = adapter.sample_style_prior(
        style_mmd.shape[0],
        generator=torch.Generator(device="cpu").manual_seed(34_000 + args.seed),
        device=device,
        dtype=style_mmd.dtype,
    )
    global_mmd = mmd2_permutation_test(
        style_mmd, reference, seed=35_000 + args.seed,
        n_permutations=args.mmd_permutations,
    )
    conditional_mmd = conditional_mmd_to_standard_normal(
        style.to(device), labels_shape.to(device), n_classes=4,
        seed=36_000 + args.seed,
    )

    with torch.no_grad():
        reconstruction_accuracy = _score_images(
            evaluator, recon_cpu.to(device), factors.to(device), args.batch_size
        )
        prior_accuracy = _score_images(
            evaluator, prior_cpu.to(device), factors.to(device), args.batch_size
        )
        donors = _different_shape_style_donors(factors, 37_000 + args.seed)
        swapped_chunks = []
        for start in range(0, content.shape[0], args.batch_size):
            stop = min(content.shape[0], start + args.batch_size)
            swapped_chunks.append(
                adapter.decode(
                    content[start:stop].to(device),
                    style[donors[start:stop]].to(device),
                ).cpu()
            )
        swapped = torch.cat(swapped_chunks)
        swap_targets = factors[donors].clone()
        swap_targets[:, SHAPE_FACTOR_INDEX] = factors[:, SHAPE_FACTOR_INDEX]
        swap_accuracy = _score_images(
            evaluator, swapped.to(device), swap_targets.to(device), args.batch_size
        )

    reconstruction_l1 = reconstruction_abs / reconstruction_pixels
    style_factor_names = [name for name in FACTOR_NAMES if name != "shape"]
    reconstructed_style_mean = float(
        np.mean([reconstruction_accuracy[name] for name in style_factor_names])
    )
    content_shape_accuracy = factor_audit["content"]["factors"]["shape"]["probes"]["models"]["logistic"]["test"]["accuracy"]
    evaluator_result = json.loads((_evaluator_dir(args) / "result.json").read_text())
    checks = {
        "factor_evaluator": bool(evaluator_result["passes_competence_gate"]),
        "reconstruction_l1": reconstruction_l1 <= MODEL_COMPETENCE_THRESHOLDS["reconstruction_l1_max"],
        "content_shape_probe": content_shape_accuracy >= MODEL_COMPETENCE_THRESHOLDS["content_shape_logistic_accuracy_min"],
        "reconstruction_shape": reconstruction_accuracy["shape"] >= MODEL_COMPETENCE_THRESHOLDS["reconstruction_shape_accuracy_min"],
        "reconstruction_style_mean": reconstructed_style_mean >= MODEL_COMPETENCE_THRESHOLDS["reconstruction_style_factor_accuracy_mean_min"],
    }
    source_indices = test_dataset.indices[positions.numpy()]
    results = {
        "n_audit_samples": int(content.shape[0]),
        "audit_source_indices_sha256": hashlib.sha256(
            np.asarray(source_indices, dtype=np.int64).tobytes()
        ).hexdigest(),
        "factor_audit": factor_audit,
        "sampling_contract": {
            "global_style_mmd": global_mmd,
            "conditional_shape_mmd": conditional_mmd,
            "style_posterior": summarize_style_posterior(logvar, 0.0),
        },
        "decoder": {
            "reconstruction_l1": reconstruction_l1,
            "reconstruction_factor_accuracy": reconstruction_accuracy,
            "prior_style_factor_accuracy_against_source": prior_accuracy,
            "posterior_swap_target_accuracy": swap_accuracy,
            "posterior_swap_definition": (
                "content from source; style from a donor with a different shape; "
                "shape target is source and all other factor targets are donor"
            ),
        },
        "competence_checks": checks,
        "passes_competence_gate": all(checks.values()),
    }
    manifest = {
        "schema_version": "shapes3d-factor-audit-result-1.0.0",
        "protocol_version": SHAPES3D_PROTOCOL_VERSION,
        "stage0_audit_protocol_version": AUDIT_PROTOCOL_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "checkpoint": {"path": str(model_path.resolve()), "sha256": sha256_file(model_path)},
        "factor_evaluator": {"path": str(evaluator_path.resolve()), "sha256": sha256_file(evaluator_path)},
        "acceptance_criteria_sha256": sha256_file(_root(args) / "acceptance_criteria.json"),
        "split_manifest": split_manifest,
        "repository": _git_metadata(),
        "implementation_file_sha256": _implementation_hashes(),
    }
    _atomic_json(output, {"manifest": manifest, "results": results}, checksum=True)
    _plot_factor_heatmap(factor_audit, directory / "factor_loading_heatmap.png")
    _plot_swap_grid(
        images_cpu, images_cpu[donors], swapped, directory / "posterior_swap_grid.png"
    )
    return output


def main() -> None:
    args = parse_args()
    source = _prepare(args)
    if args.stage == "verify-data":
        print(json.dumps(source, indent=2))
    elif args.stage == "evaluator":
        print(train_evaluator(args))
    elif args.stage == "train":
        print(train_model(args))
    elif args.stage == "audit":
        print(audit_model(args))
    elif args.stage == "seed0":
        if args.seed != 0:
            raise ValueError("--stage seed0 requires --seed 0")
        train_evaluator(args)
        train_model(args)
        print(audit_model(args))


if __name__ == "__main__":
    main()

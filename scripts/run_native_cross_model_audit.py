#!/usr/bin/env python3
"""Train and audit native F-CS-WAE, DRIT and DIVA on Rotated-MNIST.

This entrypoint intentionally refuses the legacy proxy families.  DRIT is
audited in each of its two domain-specific attribute spaces; DIVA is audited
on its native residual ``z_x``; F-CS-WAE is audited on ``z_s``.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config_f_cs_wae import f_cs_wae_config as fcfg
from src.datasets.rotated_mnist import (
    DomainSubset,
    build_rotated_mnist_loaders,
    save_split_manifest,
)
from src.metrics.audit_protocol import (
    AUDIT_PROTOCOL_VERSION,
    conditional_mmd_to_standard_normal,
    hsic_permutation_test,
    mmd2_permutation_test,
    run_probe_suite,
)
from src.metrics.factorized_adapter import (
    build_factorized_audit_adapter,
    summarize_style_posterior,
)
from src.models.f_cs_wae import FCSWAE
from src.models.native_factorized_baselines import NativeDIVA, NativeDRIT
from src.models.rotated_mnist_classifier import RotatedMNISTClassifier
from src.trainers.native_factorized_trainers import DIVATrainer, DRITTrainer
from src.trainers.trainer_f_cs_wae import FCSWAETrainer, atomic_torch_save
from src.utils.provenance import sha256_file


CROSS_MODEL_PROTOCOL_VERSION = "native-cross-model-1.0.0"
FAMILIES = ("fcswae", "drit", "diva")
DEFAULT_SEEDS = (0, 1, 2)
SPLIT_SEED = 2027
TRAIN_PER_CLASS = 2000
TEST_PER_CLASS = 500
EVAL_PER_CLASS_PER_DOMAIN = 100

# Frozen before the pilot is inspected.  A failed gate is retained as a null
# or incompetence result; it is never silently discarded.
COMPETENCE_THRESHOLDS = {
    "external_real_test_accuracy_min": 0.95,
    "content_logistic_accuracy_min": 0.80,
    "reconstruction_l1_max": 0.20,
    "prior_style_content_retention_min": 0.70,
}
REFERENCE_IMPLEMENTATIONS = {
    "drit": {
        "repository": "https://github.com/HsinYingLee/DRIT",
        "revision_inspected": "f19f50a8fa5f28dffbd93a0ed034da616232d769",
    },
    "diva": {
        "repository": "https://github.com/AMLab-Amsterdam/DIVA",
        "revision_inspected": "4c5282a8e54feee01626f5e8a54595ea570ac169",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("evaluator", "train-one", "audit-one", "run-one", "pilot", "full", "aggregate"),
        default="pilot",
    )
    parser.add_argument("--family", choices=FAMILIES)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--devices", nargs="+", default=["cuda:0", "cuda:1"])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--evaluator-epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--checkpoint-every", type=int, default=5)
    parser.add_argument("--mmd-permutations", type=int, default=500)
    parser.add_argument("--hsic-permutations", type=int, default=200)
    parser.add_argument("--probe-epochs", type=int, default=300)
    parser.add_argument("--output-root", default="runs_cross_model/native_v1")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--auto-resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True))
    os.replace(temporary, path)


def _atomic_result(path: Path, payload: dict) -> str:
    _atomic_json(path, payload)
    digest = sha256_file(path)
    sidecar = path.with_suffix(path.suffix + ".sha256")
    temporary = sidecar.with_name(f".{sidecar.name}.tmp")
    temporary.write_text(f"{digest}  {path.name}\n")
    os.replace(temporary, sidecar)
    return digest


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
        "src/datasets/rotated_mnist.py",
        "src/metrics/factorized_adapter.py",
        "src/models/f_cs_wae.py",
        "src/models/native_factorized_baselines.py",
        "src/trainers/native_factorized_trainers.py",
        "src/trainers/trainer_f_cs_wae.py",
        "scripts/run_native_cross_model_audit.py",
    )
    return {name: sha256_file(ROOT / name) for name in files}


def _run_dir(args: argparse.Namespace, family: str, seed: int) -> Path:
    return ROOT / args.output_root / family / f"seed_{seed}"


def _evaluator_path(args: argparse.Namespace) -> Path:
    return ROOT / args.output_root / "evaluator" / "rotated_mnist_cnn_seed0.pth"


def _acceptance_path(args: argparse.Namespace) -> Path:
    return ROOT / args.output_root / "acceptance_criteria.json"


def _write_frozen_protocol(args: argparse.Namespace) -> None:
    path = _acceptance_path(args)
    payload = {
        "cross_model_protocol_version": CROSS_MODEL_PROTOCOL_VERSION,
        "stage0_audit_protocol_version": AUDIT_PROTOCOL_VERSION,
        "families": list(FAMILIES),
        "native_style_blocks": {
            "fcswae": "Euclidean z_s",
            "drit": "domain-specific Gaussian attribute z_a, audited separately",
            "diva": "residual z_x with standard Gaussian prior",
        },
        "dataset": {
            "name": "two-domain Rotated-MNIST",
            "angles_degrees": [-30.0, 30.0],
            "split_seed": SPLIT_SEED,
            "train_sources_per_class": TRAIN_PER_CLASS,
            "test_sources_per_class": TEST_PER_CLASS,
            "audit_sources_per_class_per_domain": EVAL_PER_CLASS_PER_DOMAIN,
        },
        "training": {
            "epochs": int(args.epochs),
            "independent_evaluator_epochs": int(args.evaluator_epochs),
            "model_seeds": list(args.seeds),
            "selection_policy": "fixed final epoch; no audit-based checkpoint selection",
        },
        "evaluation": {
            "mmd_permutations": int(args.mmd_permutations),
            "hsic_permutations": int(args.hsic_permutations),
            "probe_split": "stratified 60/20/20, scaling fit on probe train only",
            "domain_policy": "audit each domain separately, then macro-average",
        },
        "competence_thresholds": COMPETENCE_THRESHOLDS,
        "legacy_proxy_artifacts_excluded": str(ROOT / "runs_diag/cross_model"),
    }
    if path.exists():
        existing = json.loads(path.read_text())
        if existing != payload:
            raise ValueError(
                f"Frozen protocol already exists with different settings: {path}"
            )
    else:
        _atomic_result(path, payload)


def _loaders(args: argparse.Namespace, seed: int):
    return build_rotated_mnist_loaders(
        root=ROOT / args.data_dir,
        seed=seed,
        split_seed=SPLIT_SEED,
        batch_size=args.batch_size,
        train_per_class=TRAIN_PER_CLASS,
        test_per_class=TEST_PER_CLASS,
        num_workers=args.num_workers,
        download=True,
    )


@torch.no_grad()
def _evaluate_classifier(model, loader, device: torch.device) -> dict:
    model.eval()
    correct = total = 0
    loss_sum = 0.0
    for batch in loader:
        images, labels = batch[:2]
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        loss_sum += float(F.cross_entropy(logits, labels, reduction="sum").item())
        correct += int((logits.argmax(1) == labels).sum().item())
        total += labels.numel()
    return {"accuracy": correct / total, "cross_entropy_nats": loss_sum / total, "n": total}


def train_evaluator(args: argparse.Namespace) -> Path:
    checkpoint = _evaluator_path(args)
    metadata_path = checkpoint.with_suffix(".json")
    if args.skip_existing and checkpoint.exists() and metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        accuracy = float(metadata["real_test"]["accuracy"])
        if accuracy < COMPETENCE_THRESHOLDS["external_real_test_accuracy_min"]:
            raise RuntimeError(
                f"Stored external evaluator failed competence gate: {accuracy:.4f}"
            )
        return checkpoint
    device = torch.device(args.device)
    _set_seed(0)
    train_loader, test_loader, split_manifest = _loaders(args, seed=0)
    model = RotatedMNISTClassifier().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    history = []
    start_epoch = 0
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    if args.auto_resume and checkpoint.exists() and not metadata_path.exists():
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if payload.get("schema_version") != "rotated-mnist-evaluator-1.0.0":
            raise ValueError(f"Unsupported evaluator checkpoint: {checkpoint}")
        if payload.get("target_epochs") != args.evaluator_epochs:
            raise ValueError(
                "Evaluator resume target changed: "
                f"{payload.get('target_epochs')} != {args.evaluator_epochs}"
            )
        model.load_state_dict(payload["state_dict"])
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        history = list(payload["history"])
        start_epoch = int(payload["completed_epochs"])
        if len(history) != start_epoch:
            raise ValueError("Evaluator checkpoint history is inconsistent")
        if "torch_rng_state" in payload:
            torch.set_rng_state(payload["torch_rng_state"])
        loader_generator = getattr(train_loader, "generator", None)
        if loader_generator is not None and payload.get("loader_rng_state") is not None:
            loader_generator.set_state(payload["loader_rng_state"])
    for epoch in range(start_epoch, args.evaluator_epochs):
        model.train()
        correct = total = 0
        loss_sum = 0.0
        for images, labels, _domains in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(images)
            loss = F.cross_entropy(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach().item()) * labels.numel()
            correct += int((logits.argmax(1) == labels).sum().item())
            total += labels.numel()
        row = {
            "epoch": epoch + 1,
            "train_loss": loss_sum / total,
            "train_accuracy": correct / total,
        }
        history.append(row)
        print(
            f"Evaluator epoch {epoch + 1}/{args.evaluator_epochs} "
            f"loss={row['train_loss']:.4f} acc={row['train_accuracy']:.4f}",
            flush=True,
        )
        atomic_torch_save(
            {
                "schema_version": "rotated-mnist-evaluator-1.0.0",
                "target_epochs": args.evaluator_epochs,
                "completed_epochs": epoch + 1,
                "state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "history": history,
                "torch_rng_state": torch.get_rng_state(),
                "loader_rng_state": (
                    train_loader.generator.get_state()
                    if getattr(train_loader, "generator", None) is not None
                    else None
                ),
            },
            checkpoint,
        )
    test_metrics = _evaluate_classifier(model, test_loader, device)
    metadata = {
        "schema_version": "rotated-mnist-evaluator-result-1.0.0",
        "cross_model_protocol_version": CROSS_MODEL_PROTOCOL_VERSION,
        "independent_from_generative_training": True,
        "fixed_final_epoch": args.evaluator_epochs,
        "real_test": test_metrics,
        "split_manifest": split_manifest,
        "checkpoint_sha256": sha256_file(checkpoint),
    }
    _atomic_result(metadata_path, metadata)
    if test_metrics["accuracy"] < COMPETENCE_THRESHOLDS["external_real_test_accuracy_min"]:
        raise RuntimeError(f"External evaluator failed competence gate: {test_metrics}")
    return checkpoint


def _model_config(family: str) -> dict:
    if family == "fcswae":
        return {
            "semantic_dim": 64,
            "style_dim": 32,
            "n_classes": 10,
            "in_channels": 1,
            "image_size": 28,
            "n_centers": 1,
            "rho_prior": 0.7,
            "ema_momentum": 0.95,
            "style_sigma_floor": 0.0,
            "conditional_style_mmd_weight": 0.0,
        }
    if family == "drit":
        return {"content_dim": 64, "style_dim": 8}
    if family == "diva":
        return {
            "domain_dim": 32,
            "style_dim": 32,
            "semantic_dim": 32,
            "n_domains": 2,
            "n_classes": 10,
        }
    raise ValueError(f"Unsupported family {family!r}; proxy models are prohibited")


def _build_model(family: str, config: dict) -> torch.nn.Module:
    if family == "fcswae":
        kwargs = {key: value for key, value in config.items() if key != "conditional_style_mmd_weight"}
        return FCSWAE(**kwargs)
    if family == "drit":
        return NativeDRIT(**config)
    if family == "diva":
        return NativeDIVA(**config)
    raise ValueError(f"Unsupported family {family!r}; proxy models are prohibited")


def _run_config(args: argparse.Namespace, family: str, seed: int) -> dict:
    return {
        "cross_model_protocol_version": CROSS_MODEL_PROTOCOL_VERSION,
        "dataset": "rotated_mnist_two_domain",
        "split_seed": SPLIT_SEED,
        "train_per_class": TRAIN_PER_CLASS,
        "test_per_class": TEST_PER_CLASS,
        "family": family,
        "seed": int(seed),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "model": _model_config(family),
        "n_centers": _model_config(family).get("n_centers"),
        "semantic_dim": _model_config(family).get("semantic_dim", _model_config(family).get("content_dim")),
        "style_dim": _model_config(family)["style_dim"],
        "delta_final": _model_config(family).get("conditional_style_mmd_weight"),
        "style_sigma_floor": _model_config(family).get("style_sigma_floor"),
        "phase_a_end": 20 if family == "fcswae" else None,
        "phase_b_end": 40 if family == "fcswae" else None,
    }


def _save_final_model(path: Path, model: torch.nn.Module, run_config: dict) -> None:
    atomic_torch_save(
        {
            "schema_version": "native-cross-model-final-model-1.0.0",
            "run_config": run_config,
            "state_dict": model.state_dict(),
        },
        path,
    )


def train_one(args: argparse.Namespace, family: str, seed: int) -> Path:
    if family not in FAMILIES:
        raise ValueError("Only native F-CS-WAE, DRIT and DIVA are permitted")
    run_dir = _run_dir(args, family, seed)
    final_model = run_dir / "model.pth"
    if args.skip_existing and final_model.exists():
        print(f"[{family}/seed_{seed}] reuse final model {final_model}", flush=True)
        return final_model
    run_dir.mkdir(parents=True, exist_ok=True)
    _set_seed(seed)
    train_loader, _test_loader, split_manifest = _loaders(args, seed=seed)
    save_split_manifest(run_dir / "split_manifest.json", split_manifest)
    run_config = _run_config(args, family, seed)
    _atomic_json(run_dir / "run_config.json", run_config)
    model = _build_model(family, run_config["model"])
    device = torch.device(args.device)
    checkpoint = run_dir / "training_checkpoint.pt"
    start_epoch, history = 0, []

    if family == "diva":
        trainer = DIVATrainer(model, train_loader, device)
        if args.auto_resume and checkpoint.exists():
            start_epoch, history = trainer.load_checkpoint(checkpoint, run_config)
        history = trainer.train(
            epochs=args.epochs,
            checkpoint_path=checkpoint,
            checkpoint_every=args.checkpoint_every,
            run_config=run_config,
            start_epoch=start_epoch,
            history=history,
        )
    elif family == "drit":
        train_dataset = train_loader.dataset
        domain_loaders = []
        for domain in (0, 1):
            generator = torch.Generator().manual_seed(seed + 1000 + domain)
            domain_loaders.append(DataLoader(
                DomainSubset(train_dataset, domain),
                batch_size=args.batch_size,
                shuffle=True,
                generator=generator,
                num_workers=args.num_workers,
                pin_memory=torch.cuda.is_available(),
                persistent_workers=args.num_workers > 0,
            ))
        trainer = DRITTrainer(model, tuple(domain_loaders), device)
        if args.auto_resume and checkpoint.exists():
            start_epoch, history = trainer.load_checkpoint(checkpoint, run_config)
        history = trainer.train(
            epochs=args.epochs,
            checkpoint_path=checkpoint,
            checkpoint_every=args.checkpoint_every,
            run_config=run_config,
            start_epoch=start_epoch,
            history=history,
        )
    else:
        # Keep the family-defining global aggregate style-MMD and disable the
        # conditional remedy so this run tests the original sampling contract.
        fcfg.semantic_dim = run_config["semantic_dim"]
        fcfg.style_dim = run_config["style_dim"]
        fcfg.n_classes = 10
        fcfg.in_channels = 1
        fcfg.image_size = 28
        fcfg.total_epochs = args.epochs
        fcfg.phase_a_end = 20
        fcfg.phase_b_end = 40
        fcfg.phase_c_end = 70
        fcfg.phase_d_end = args.epochs
        fcfg.lr_scheduler_step = 50
        fcfg.delta_final = 0.0
        fcfg.style_sigma_floor = 0.0
        trainer = FCSWAETrainer(model, train_loader, device=device)
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

    _atomic_json(run_dir / "training_history.json", history)
    _save_final_model(final_model, model, run_config)
    return final_model


def _load_trained_model(args: argparse.Namespace, family: str, seed: int, device: torch.device):
    path = _run_dir(args, family, seed) / "model.pth"
    if not path.exists():
        raise FileNotFoundError(f"Train {family}/seed_{seed} first: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema_version") != "native-cross-model-final-model-1.0.0":
        raise ValueError(f"Not a native cross-model checkpoint: {path}")
    run_config = payload["run_config"]
    if run_config.get("cross_model_protocol_version") != CROSS_MODEL_PROTOCOL_VERSION:
        raise ValueError("Checkpoint protocol does not match native audit protocol")
    model = _build_model(family, run_config["model"])
    model.load_state_dict(payload["state_dict"])
    model.to(device).eval()
    return model, run_config, path


def _load_evaluator(args: argparse.Namespace, device: torch.device):
    path = _evaluator_path(args)
    if not path.exists():
        raise FileNotFoundError(f"Train the independent evaluator first: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = RotatedMNISTClassifier().to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    metadata = json.loads(path.with_suffix(".json").read_text())
    return model, metadata, path


@torch.no_grad()
def _actual_reconstruction(model, family: str, images: torch.Tensor, domain: int, generator):
    if family == "diva":
        reconstruction, _views = model(images, generator=generator)
        return reconstruction
    adapter = build_factorized_audit_adapter(model)
    domains = torch.full((images.shape[0],), domain, device=images.device, dtype=torch.long)
    views = adapter.encode_views(images, generator=generator, domain=domains)
    return adapter.decode(views.content_sample, views.style_sample, domain=domain)


@torch.no_grad()
def _audit_domain(
    args: argparse.Namespace,
    model,
    family: str,
    seed: int,
    domain: int,
    dataset,
    evaluator,
    device: torch.device,
) -> dict:
    domain_dataset = DomainSubset(dataset, domain)
    count = 10 * EVAL_PER_CLASS_PER_DOMAIN
    loader = DataLoader(
        Subset(domain_dataset, list(range(count))),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
    )
    adapter = build_factorized_audit_adapter(model)
    posterior_generator = torch.Generator(device="cpu").manual_seed(50_000 + 100 * seed + domain)
    reconstruction_generator = torch.Generator(device="cpu").manual_seed(51_000 + 100 * seed + domain)
    prior_generator = torch.Generator(device="cpu").manual_seed(60_000 + 100 * seed + domain)
    content, style_mean, style_sample, logvar = [], [], [], []
    labels_all, source_indices = [], []
    reconstruction_abs_sum = 0.0
    reconstruction_pixels = 0
    generated_correct = generated_total = 0
    position = 0
    for images, labels, domains in loader:
        batch_count = images.shape[0]
        source_indices.extend(
            int(dataset.source_indices[position + offset]) for offset in range(batch_count)
        )
        position += batch_count
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        domains = domains.to(device, non_blocking=True)
        views = adapter.encode_views(
            images, generator=posterior_generator, domain=domains
        )
        reconstruction = _actual_reconstruction(
            model, family, images, domain, reconstruction_generator
        )
        reconstruction_abs_sum += float((reconstruction - images).abs().sum().item())
        reconstruction_pixels += images.numel()
        prior_style = adapter.sample_style_prior(
            batch_count,
            generator=prior_generator,
            device=device,
            dtype=views.style_sample.dtype,
            domain=domain,
        )
        target_domain = 1 - domain if family == "drit" else domain
        generated = adapter.decode(
            views.content_sample, prior_style, domain=target_domain
        )
        predictions = evaluator(generated).argmax(1)
        generated_correct += int((predictions == labels).sum().item())
        generated_total += labels.numel()
        content.append(views.content_sample.detach())
        style_mean.append(views.style_mean.detach())
        style_sample.append(views.style_sample.detach())
        if views.style_logvar is not None:
            logvar.append(views.style_logvar.detach())
        labels_all.append(labels.detach())

    content_tensor = torch.cat(content)
    style_mean_tensor = torch.cat(style_mean)
    style_tensor = torch.cat(style_sample)
    labels_tensor = torch.cat(labels_all)
    prior_reference = adapter.sample_style_prior(
        style_tensor.shape[0],
        generator=torch.Generator(device="cpu").manual_seed(70_000 + 100 * seed + domain),
        device=device,
        dtype=style_tensor.dtype,
        domain=domain,
    )
    global_mmd = mmd2_permutation_test(
        style_tensor,
        prior_reference,
        seed=80_000 + 100 * seed + domain,
        n_permutations=args.mmd_permutations,
    )
    conditional_mmd = conditional_mmd_to_standard_normal(
        style_tensor,
        labels_tensor,
        n_classes=10,
        seed=90_000 + 100 * seed + domain,
    )
    label_hsic = hsic_permutation_test(
        style_tensor,
        labels_tensor,
        seed=100_000 + 100 * seed + domain,
        n_permutations=args.hsic_permutations,
    )
    # The model-side collection is inference-only, but Torch probes train
    # small classifiers and therefore explicitly re-enable autograd.
    with torch.enable_grad():
        style_probe = run_probe_suite(
            style_tensor,
            labels_tensor,
            seed=110_000 + 100 * seed + domain,
            probe_names=("logistic", "mlp", "rbf_svm"),
            epochs=args.probe_epochs,
        )
        content_probe = run_probe_suite(
            content_tensor,
            labels_tensor,
            seed=120_000 + 100 * seed + domain,
            probe_names=("logistic",),
            epochs=args.probe_epochs,
        )
    posterior = (
        summarize_style_posterior(torch.cat(logvar), 0.0) if logvar else None
    )
    conditional_value = conditional_mmd["mean_mmd2_u"]
    global_value = global_mmd["statistic"]
    ratio = conditional_value / global_value if global_value > 1e-12 else None
    source_array = np.asarray(source_indices, dtype=np.int64)
    return {
        "domain": domain,
        "rotation_degrees": [-30.0, 30.0][domain],
        "n_samples": int(labels_tensor.numel()),
        "eval_source_indices_sha256": hashlib.sha256(source_array.tobytes()).hexdigest(),
        "class_counts": torch.bincount(labels_tensor, minlength=10).cpu().tolist(),
        "global_mmd": global_mmd,
        "conditional_mmd": conditional_mmd,
        "conditional_global_ratio": ratio,
        "conditional_minus_global_mmd2_u": conditional_value - global_value,
        "label_hsic": label_hsic,
        "style_probe": style_probe,
        "content_probe": content_probe,
        "style_posterior": posterior,
        "reconstruction_l1": reconstruction_abs_sum / reconstruction_pixels,
        "prior_style_content_retention": generated_correct / generated_total,
        "generation_target_domain": target_domain,
        "style_mean_shape": list(style_mean_tensor.shape),
    }


def _macro(domains: list[dict], evaluator_accuracy: float) -> dict:
    def mean(path):
        values = []
        for item in domains:
            value = item
            for key in path:
                value = value[key]
            values.append(float(value))
        return float(np.mean(values))

    ratios = [item["conditional_global_ratio"] for item in domains]
    valid_ratios = [float(value) for value in ratios if value is not None]
    metrics = {
        "global_mmd2_u": mean(("global_mmd", "statistic")),
        "global_mmd_p_value": mean(("global_mmd", "p_value")),
        "conditional_mmd2_u": mean(("conditional_mmd", "mean_mmd2_u")),
        "conditional_minus_global_mmd2_u": mean(("conditional_minus_global_mmd2_u",)),
        "conditional_global_ratio": float(np.mean(valid_ratios)) if valid_ratios else None,
        "style_logistic_accuracy": mean(("style_probe", "models", "logistic", "test", "accuracy")),
        "style_mlp_accuracy": mean(("style_probe", "models", "mlp", "test", "accuracy")),
        "style_rbf_accuracy": mean(("style_probe", "models", "rbf_svm", "test", "accuracy")),
        "label_hsic": mean(("label_hsic", "statistic")),
        "label_hsic_p_value": mean(("label_hsic", "p_value")),
        "content_logistic_accuracy": mean(("content_probe", "models", "logistic", "test", "accuracy")),
        "reconstruction_l1": mean(("reconstruction_l1",)),
        "prior_style_content_retention": mean(("prior_style_content_retention",)),
        "external_real_test_accuracy": float(evaluator_accuracy),
    }
    checks = {
        "external_evaluator": metrics["external_real_test_accuracy"] >= COMPETENCE_THRESHOLDS["external_real_test_accuracy_min"],
        "content_probe": metrics["content_logistic_accuracy"] >= COMPETENCE_THRESHOLDS["content_logistic_accuracy_min"],
        "reconstruction": metrics["reconstruction_l1"] <= COMPETENCE_THRESHOLDS["reconstruction_l1_max"],
        "content_retention": metrics["prior_style_content_retention"] >= COMPETENCE_THRESHOLDS["prior_style_content_retention_min"],
    }
    return {"metrics": metrics, "competence_checks": checks, "passes_competence_gate": all(checks.values())}


def audit_one(args: argparse.Namespace, family: str, seed: int) -> Path:
    run_dir = _run_dir(args, family, seed)
    output = run_dir / "audit.json"
    if args.skip_existing and output.exists():
        payload = json.loads(output.read_text())
        if payload.get("manifest", {}).get("cross_model_protocol_version") == CROSS_MODEL_PROTOCOL_VERSION:
            print(f"[{family}/seed_{seed}] reuse current audit {output}", flush=True)
            return output
    device = torch.device(args.device)
    model, run_config, checkpoint = _load_trained_model(args, family, seed, device)
    evaluator, evaluator_metadata, evaluator_path = _load_evaluator(args, device)
    _train_loader, test_loader, split_manifest = _loaders(args, seed=seed)
    domains = [
        _audit_domain(
            args, model, family, seed, domain, test_loader.dataset,
            evaluator, device,
        )
        for domain in (0, 1)
    ]
    aggregate = _macro(domains, evaluator_metadata["real_test"]["accuracy"])
    manifest = {
        "schema_version": "native-cross-model-audit-result-1.0.0",
        "cross_model_protocol_version": CROSS_MODEL_PROTOCOL_VERSION,
        "stage0_audit_protocol_version": AUDIT_PROTOCOL_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "family": family,
        "seed": seed,
        "native_factorization": build_factorized_audit_adapter(model).metadata(),
        "run_config": run_config,
        "checkpoint": {"path": str(checkpoint.resolve()), "sha256": sha256_file(checkpoint)},
        "external_evaluator": {"path": str(evaluator_path.resolve()), "sha256": sha256_file(evaluator_path)},
        "split_manifest": split_manifest,
        "repository": _git_metadata(),
        "implementation_file_sha256": _implementation_hashes(),
        "reference_implementations": REFERENCE_IMPLEMENTATIONS,
        "legacy_proxy_artifacts_used": False,
    }
    _atomic_result(output, {"manifest": manifest, "domains": domains, "macro": aggregate})
    return output


def aggregate(args: argparse.Namespace, seeds: list[int]) -> Path:
    rows = []
    missing = []
    for family in FAMILIES:
        for seed in seeds:
            path = _run_dir(args, family, seed) / "audit.json"
            if not path.exists():
                missing.append(str(path))
                continue
            payload = json.loads(path.read_text())
            row = {
                "family": family,
                "seed": seed,
                **payload["macro"]["metrics"],
                "passes_competence_gate": payload["macro"]["passes_competence_gate"],
                "audit_path": str(path),
                "audit_sha256": sha256_file(path),
            }
            rows.append(row)
    if missing:
        raise FileNotFoundError("Missing native audits:\n" + "\n".join(missing))
    family_summary = {}
    numeric_keys = [
        "global_mmd2_u", "conditional_mmd2_u", "conditional_minus_global_mmd2_u",
        "style_logistic_accuracy", "style_mlp_accuracy", "style_rbf_accuracy",
        "label_hsic", "content_logistic_accuracy", "reconstruction_l1",
        "prior_style_content_retention",
    ]
    for family in FAMILIES:
        family_rows = [row for row in rows if row["family"] == family]
        family_summary[family] = {
            "n_seeds": len(family_rows),
            "competent_seeds": sum(row["passes_competence_gate"] for row in family_rows),
            "metrics": {
                key: {
                    "mean": float(np.mean([row[key] for row in family_rows])),
                    "std": float(np.std([row[key] for row in family_rows], ddof=1)) if len(family_rows) > 1 else 0.0,
                }
                for key in numeric_keys
            },
        }
    output = ROOT / args.output_root / "cross_model_summary.json"
    payload = {
        "manifest": {
            "schema_version": "native-cross-model-summary-1.0.0",
            "cross_model_protocol_version": CROSS_MODEL_PROTOCOL_VERSION,
            "seeds": seeds,
            "legacy_proxy_artifacts_used": False,
            "acceptance_criteria_sha256": sha256_file(_acceptance_path(args)),
        },
        "rows": rows,
        "family_summary": family_summary,
    }
    _atomic_result(output, payload)
    return output


def _subprocess_command(args: argparse.Namespace, family: str, seed: int, device: str) -> list[str]:
    command = [
        sys.executable, str(Path(__file__).resolve()),
        "--stage", "run-one", "--family", family, "--seed", str(seed),
        "--device", device, "--epochs", str(args.epochs),
        "--evaluator-epochs", str(args.evaluator_epochs),
        "--batch-size", str(args.batch_size), "--num-workers", str(args.num_workers),
        "--checkpoint-every", str(args.checkpoint_every),
        "--mmd-permutations", str(args.mmd_permutations),
        "--hsic-permutations", str(args.hsic_permutations),
        "--probe-epochs", str(args.probe_epochs),
        "--output-root", args.output_root, "--data-dir", args.data_dir,
        "--auto-resume", "--skip-existing",
    ]
    return command


def run_matrix(args: argparse.Namespace, seeds: list[int]) -> None:
    if not args.devices:
        raise ValueError("At least one device is required")
    jobs = [(family, seed) for seed in seeds for family in FAMILIES]
    queues = [jobs[index::len(args.devices)] for index in range(len(args.devices))]

    def worker(device: str, queue: list[tuple[str, int]]):
        for family, seed in queue:
            run_dir = _run_dir(args, family, seed)
            run_dir.mkdir(parents=True, exist_ok=True)
            command = _subprocess_command(args, family, seed, device)
            print("$ " + " ".join(command), flush=True)
            if args.dry_run:
                continue
            with (run_dir / "pipeline.log").open("a") as log:
                subprocess.run(
                    command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                    check=True,
                )

    with ThreadPoolExecutor(max_workers=len(args.devices)) as executor:
        futures = [executor.submit(worker, device, queue) for device, queue in zip(args.devices, queues) if queue]
        for future in futures:
            future.result()


def main() -> None:
    args = parse_args()
    _write_frozen_protocol(args)
    if args.stage in {"train-one", "audit-one", "run-one"} and args.family is None:
        raise ValueError(f"--family is required for --stage {args.stage}")
    if args.stage == "evaluator":
        train_evaluator(args)
    elif args.stage == "train-one":
        train_one(args, args.family, args.seed)
    elif args.stage == "audit-one":
        audit_one(args, args.family, args.seed)
    elif args.stage == "run-one":
        train_one(args, args.family, args.seed)
        audit_one(args, args.family, args.seed)
    elif args.stage in {"pilot", "full"}:
        train_evaluator(args)
        seeds = [0] if args.stage == "pilot" else list(args.seeds)
        run_matrix(args, seeds)
        aggregate(args, seeds)
    elif args.stage == "aggregate":
        aggregate(args, list(args.seeds))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Train and audit native factorized baselines on Shapes3D."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import scripts.run_shapes3d_factor_audit as common
from src.datasets.shapes3d import FACTOR_NAMES, SHAPE_FACTOR_INDEX, verify_shapes3d_file
from src.metrics.audit_protocol import (
    AUDIT_PROTOCOL_VERSION,
    conditional_mmd_to_standard_normal,
    mmd2_permutation_test,
)
from src.metrics.factorized_adapter import (
    build_factorized_audit_adapter,
    summarize_style_posterior,
)
from src.models.shapes3d_factor_evaluator import Shapes3DFactorEvaluator
from src.models.shapes3d_factorized_baselines import (
    Shapes3DConditionalVAE,
    Shapes3DContentStyleVAE,
)
from src.trainers.shapes3d_factorized_baselines import (
    Shapes3DFactorizedBaselineTrainer,
)
from src.trainers.trainer_f_cs_wae import atomic_torch_save
from src.utils.provenance import sha256_file


BASELINE_PROTOCOL_VERSION = "shapes3d-factorized-baselines-1.0.0"
MODEL_NAMES = ("conditional_vae", "content_style_vae")
KL_FINAL = 0.01
KL_WARMUP_EPOCHS = 20
CLASSIFIER_WEIGHT = 0.1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODEL_NAMES, required=True)
    parser.add_argument(
        "--stage", choices=("verify-data", "train", "audit", "full"), default="full"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--data", default="data/shapes3d/3dshapes.h5")
    parser.add_argument("--output-root", default="runs_shapes3d/factor_audit_v1")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--checkpoint-every", type=int, default=5)
    parser.add_argument("--probe-epochs", type=int, default=300)
    parser.add_argument("--mmd-permutations", type=int, default=500)
    parser.add_argument("--hsic-permutations", type=int, default=200)
    parser.add_argument("--auto-resume", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--skip-source-hash", action="store_true")
    return parser.parse_args()


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _atomic_json(path: Path, payload: dict, *, checksum: bool = False) -> None:
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


def _directory(args: argparse.Namespace) -> Path:
    return ROOT / args.output_root / args.model / f"seed_{args.seed}"


def _build_model(name: str):
    kwargs = dict(
        content_dim=64,
        style_dim=64,
        n_classes=4,
        in_channels=3,
        image_size=64,
    )
    if name == "conditional_vae":
        return Shapes3DConditionalVAE(**kwargs)
    if name == "content_style_vae":
        return Shapes3DContentStyleVAE(**kwargs)
    raise ValueError(name)


def _run_config(args: argparse.Namespace) -> dict:
    return {
        "baseline_protocol_version": BASELINE_PROTOCOL_VERSION,
        "model": args.model,
        "seed": int(args.seed),
        "dataset": "shapes3d",
        "dataset_sha256": common.SHAPES3D_SHA256,
        "split_seed": common.SPLIT_SEED,
        "split_counts": dict(common.DEFAULT_SPLIT_COUNTS),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "content_dim": 64,
        "style_dim": 64,
        "n_classes": 4,
        "image_size": 64,
        "kl_final_per_coordinate": KL_FINAL,
        "kl_warmup_epochs": KL_WARMUP_EPOCHS,
        "classifier_weight": CLASSIFIER_WEIGHT if args.model == "content_style_vae" else 0.0,
        "fixed_final_epoch": True,
        "no_audit_based_checkpoint_selection": True,
    }


def _prepare(args: argparse.Namespace) -> None:
    source = verify_shapes3d_file(
        ROOT / args.data, verify_sha256=not args.skip_source_hash
    )
    evaluator_result = ROOT / args.output_root / "evaluator" / "seed_0" / "result.json"
    if not evaluator_result.exists():
        raise FileNotFoundError("Train the independent Shapes3D evaluator first")
    if not json.loads(evaluator_result.read_text())["passes_competence_gate"]:
        raise RuntimeError("Independent Shapes3D factor evaluator failed competence")
    acceptance = ROOT / args.output_root / "baseline_acceptance_criteria.json"
    payload = {
        "schema_version": "shapes3d-factorized-baseline-acceptance-1.0.0",
        "protocol_version": BASELINE_PROTOCOL_VERSION,
        "source": source,
        "models": list(MODEL_NAMES),
        "seeds": [0, 1, 2],
        "pilot_seed": 0,
        "model_dimensions": {"content": 64, "style": 64},
        "training": {
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "kl_final_per_coordinate": KL_FINAL,
            "kl_warmup_epochs": KL_WARMUP_EPOCHS,
            "content_style_classifier_weight": CLASSIFIER_WEIGHT,
            "fixed_final_epoch": True,
            "no_audit_based_checkpoint_selection": True,
        },
        "evaluation": {
            "audit_per_shape": common.AUDIT_PER_SHAPE,
            "mmd_samples": common.MMD_SAMPLES,
            "hsic_samples": common.HSIC_SAMPLES,
            "mmd_permutations": int(args.mmd_permutations),
            "hsic_permutations": int(args.hsic_permutations),
            "multiple_testing": "Holm separately within each latent view",
            "primary_view": "stochastic posterior sample",
        },
        "competence_thresholds": common.MODEL_COMPETENCE_THRESHOLDS,
        "promotion_rule": "replicate seeds 1-2 based on base competence only, never leakage direction",
    }
    if acceptance.exists():
        if json.loads(acceptance.read_text()) != payload:
            raise ValueError(f"Frozen baseline criteria differ: {acceptance}")
    else:
        _atomic_json(acceptance, payload, checksum=True)


def _loaders(args: argparse.Namespace):
    return common._loaders(args, batch_size=args.batch_size)


def train(args: argparse.Namespace) -> Path:
    directory = _directory(args)
    final = directory / "model.pth"
    if args.skip_existing and final.exists():
        return final
    directory.mkdir(parents=True, exist_ok=True)
    _set_seed(args.seed)
    loaders, manifest = _loaders(args)
    common.save_shapes3d_manifest(directory / "split_manifest.json", manifest)
    config = _run_config(args)
    _atomic_json(directory / "run_config.json", config)
    model = _build_model(args.model)
    trainer = Shapes3DFactorizedBaselineTrainer(
        model,
        loaders["train"],
        device=torch.device(args.device),
        kl_final=KL_FINAL,
        classifier_weight=CLASSIFIER_WEIGHT,
        warmup_epochs=KL_WARMUP_EPOCHS,
    )
    history = trainer.train(
        epochs=args.epochs,
        checkpoint_path=directory / "training_checkpoint.pt",
        checkpoint_every=args.checkpoint_every,
        run_config=config,
        auto_resume=args.auto_resume,
    )
    _atomic_json(directory / "training_history.json", {"epochs": history})
    atomic_torch_save(
        {
            "schema_version": "shapes3d-factorized-baseline-final-1.0.0",
            "run_config": config,
            "state_dict": model.state_dict(),
        },
        final,
    )
    return final


def _load_for_audit(args: argparse.Namespace, device: torch.device):
    directory = _directory(args)
    model_path = directory / "model.pth"
    payload = torch.load(model_path, map_location="cpu", weights_only=False)
    if payload.get("schema_version") != "shapes3d-factorized-baseline-final-1.0.0":
        raise ValueError(f"Unexpected baseline checkpoint: {model_path}")
    model = _build_model(args.model).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    evaluator_path = ROOT / args.output_root / "evaluator" / "seed_0" / "factor_evaluator.pth"
    evaluator_payload = torch.load(evaluator_path, map_location="cpu", weights_only=False)
    evaluator = Shapes3DFactorEvaluator().to(device)
    evaluator.load_state_dict(evaluator_payload["state_dict"])
    evaluator.eval()
    return model, evaluator, model_path, evaluator_path


def audit(args: argparse.Namespace) -> Path:
    directory = _directory(args)
    output = directory / "factor_audit.json"
    if args.skip_existing and output.exists():
        payload = json.loads(output.read_text())
        if payload.get("manifest", {}).get("protocol_version") == BASELINE_PROTOCOL_VERSION:
            return output
    device = torch.device(args.device)
    model, evaluator, model_path, evaluator_path = _load_for_audit(args, device)
    loaders, split_manifest = _loaders(args)
    test_dataset = loaders["test"].dataset
    shape_labels = torch.from_numpy(test_dataset.factors[:, SHAPE_FACTOR_INDEX])
    positions = common._balanced_positions(
        shape_labels, 4 * common.AUDIT_PER_SHAPE, 50_000 + args.seed
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
    posterior_generator = torch.Generator(device="cpu").manual_seed(51_000 + args.seed)
    prior_generator = torch.Generator(device="cpu").manual_seed(52_000 + args.seed)
    content, style, logvar, factors_all, images_all = [], [], [], [], []
    reconstructions, prior_images = [], []
    reconstruction_abs, reconstruction_pixels = 0.0, 0
    with torch.no_grad():
        for images, shape, factors in audit_loader:
            images = images.to(device, non_blocking=True)
            shape = shape.to(device, non_blocking=True)
            factors = factors.to(device, non_blocking=True)
            views = adapter.encode_views(
                images, labels=shape, generator=posterior_generator
            )
            reconstruction = adapter.decode(views.content_sample, views.style_sample)
            prior_style = adapter.sample_style_prior(
                images.shape[0],
                generator=prior_generator,
                device=device,
                dtype=views.style_sample.dtype,
            )
            generated = adapter.decode(views.content_sample, prior_style)
            reconstruction_abs += float((reconstruction - images).abs().sum().item())
            reconstruction_pixels += images.numel()
            content.append(views.content_sample.cpu())
            style.append(views.style_sample.cpu())
            logvar.append(views.style_logvar.cpu())
            factors_all.append(factors.cpu())
            images_all.append(images.cpu())
            reconstructions.append(reconstruction.cpu())
            prior_images.append(generated.cpu())

    content = torch.cat(content)
    style = torch.cat(style)
    logvar = torch.cat(logvar)
    factors = torch.cat(factors_all)
    images_cpu = torch.cat(images_all)
    recon_cpu = torch.cat(reconstructions)
    prior_cpu = torch.cat(prior_images)
    shape = factors[:, SHAPE_FACTOR_INDEX]
    factor_audit = {
        "content": common._factor_audit(
            content, factors, latent_name="content", args=args, device=device
        ),
        "style": common._factor_audit(
            style, factors, latent_name="style", args=args, device=device
        ),
    }
    mmd_positions = common._balanced_positions(
        shape, common.MMD_SAMPLES, 53_000 + args.seed
    )
    style_mmd = style[mmd_positions].to(device)
    reference = adapter.sample_style_prior(
        style_mmd.shape[0],
        generator=torch.Generator(device="cpu").manual_seed(54_000 + args.seed),
        device=device,
        dtype=style_mmd.dtype,
    )
    global_mmd = mmd2_permutation_test(
        style_mmd,
        reference,
        seed=55_000 + args.seed,
        n_permutations=args.mmd_permutations,
    )
    conditional_mmd = conditional_mmd_to_standard_normal(
        style.to(device), shape.to(device), n_classes=4, seed=56_000 + args.seed
    )
    with torch.no_grad():
        reconstruction_accuracy = common._score_images(
            evaluator, recon_cpu.to(device), factors.to(device), args.batch_size
        )
        prior_accuracy = common._score_images(
            evaluator, prior_cpu.to(device), factors.to(device), args.batch_size
        )
        donors = common._different_shape_style_donors(factors, 57_000 + args.seed)
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
        swap_accuracy = common._score_images(
            evaluator, swapped.to(device), swap_targets.to(device), args.batch_size
        )

    reconstruction_l1 = reconstruction_abs / reconstruction_pixels
    style_factor_names = [name for name in FACTOR_NAMES if name != "shape"]
    reconstructed_style_mean = float(
        np.mean([reconstruction_accuracy[name] for name in style_factor_names])
    )
    content_shape_accuracy = factor_audit["content"]["factors"]["shape"]["probes"]["models"]["logistic"]["test"]["accuracy"]
    checks = {
        "factor_evaluator": True,
        "reconstruction_l1": reconstruction_l1 <= common.MODEL_COMPETENCE_THRESHOLDS["reconstruction_l1_max"],
        "content_shape_probe": content_shape_accuracy >= common.MODEL_COMPETENCE_THRESHOLDS["content_shape_logistic_accuracy_min"],
        "reconstruction_shape": reconstruction_accuracy["shape"] >= common.MODEL_COMPETENCE_THRESHOLDS["reconstruction_shape_accuracy_min"],
        "reconstruction_style_mean": reconstructed_style_mean >= common.MODEL_COMPETENCE_THRESHOLDS["reconstruction_style_factor_accuracy_mean_min"],
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
        },
        "competence_checks": checks,
        "passes_competence_gate": all(checks.values()),
    }
    manifest = {
        "schema_version": "shapes3d-factorized-baseline-audit-result-1.0.0",
        "protocol_version": BASELINE_PROTOCOL_VERSION,
        "stage0_audit_protocol_version": AUDIT_PROTOCOL_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "model_metadata": adapter.metadata(),
        "seed": int(args.seed),
        "checkpoint": {"path": str(model_path.resolve()), "sha256": sha256_file(model_path)},
        "factor_evaluator": {"path": str(evaluator_path.resolve()), "sha256": sha256_file(evaluator_path)},
        "acceptance_criteria_sha256": sha256_file(ROOT / args.output_root / "baseline_acceptance_criteria.json"),
        "split_manifest": split_manifest,
        "implementation_file_sha256": {
            name: sha256_file(ROOT / name)
            for name in (
                "src/models/shapes3d_factorized_baselines.py",
                "src/trainers/shapes3d_factorized_baselines.py",
                "src/metrics/factorized_adapter.py",
                "scripts/run_shapes3d_factorized_baseline.py",
            )
        },
    }
    _atomic_json(output, {"manifest": manifest, "results": results}, checksum=True)
    common._plot_factor_heatmap(factor_audit, directory / "factor_loading_heatmap.png")
    common._plot_swap_grid(
        images_cpu, images_cpu[donors], swapped, directory / "posterior_swap_grid.png"
    )
    return output


def main() -> None:
    args = parse_args()
    _prepare(args)
    if args.stage == "verify-data":
        print(ROOT / args.output_root / "baseline_acceptance_criteria.json")
    elif args.stage == "train":
        print(train(args))
    elif args.stage == "audit":
        print(audit(args))
    else:
        train(args)
        print(audit(args))


if __name__ == "__main__":
    main()

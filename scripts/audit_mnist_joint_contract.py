#!/usr/bin/env python3
"""Held-out MNIST audit for the F-CS-WAE joint latent contract.

For every checkpoint and class, this compares independently collected banks
from q(z_c, z_s | y) and p(z_c | y)p(z_s).  Content, style, and their joint
are reported with both permutation-calibrated multiscale MMD and kernel-free
energy distance.  Within-class content/style dependence is independently
checked with HSIC and distance correlation.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.compute_leakage_diagnostics import (  # noqa: E402
    get_eval_subset_loader,
    load_fcswae_checkpoint,
)
from src.metrics.audit_protocol import (  # noqa: E402
    EUCLIDEAN_SIGMAS,
    SPHERICAL_SIGMAS,
    classwise_conditional_distance_correlation_permutation_test,
    classwise_conditional_hsic_permutation_test,
    energy_distance_permutation_test,
    multiscale_rbf_kernel,
)
from src.metrics.factorized_adapter import build_factorized_audit_adapter  # noqa: E402
from src.utils.provenance import sha256_file  # noqa: E402


PROTOCOL_VERSION = "mnist-joint-contract-1.1.0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        action="append",
        required=True,
        metavar="NAME=PATH",
        help="Repeat for each checkpoint under comparison.",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--samples-per-class", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--permutations", type=int, default=199)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output",
        default="runs_diag/mnist_joint_contract/results.json",
    )
    return parser.parse_args()


def _parse_checkpoints(specifications: list[str]) -> list[tuple[str, Path]]:
    parsed = []
    for specification in specifications:
        if "=" not in specification:
            raise ValueError(f"checkpoint must be NAME=PATH: {specification!r}")
        name, raw_path = specification.split("=", 1)
        path = Path(raw_path)
        if not name or not path.is_file():
            raise ValueError(f"invalid checkpoint specification: {specification!r}")
        parsed.append((name, path))
    if len({name for name, _ in parsed}) != len(parsed):
        raise ValueError("checkpoint names must be unique")
    return parsed


def _balanced_test_loader(samples_per_class: int, batch_size: int) -> DataLoader:
    base_loader = get_eval_subset_loader(
        "mnist", n_samples=10_000, batch_size=batch_size, seed=0, split="test"
    )
    dataset = base_loader.dataset.dataset
    labels = np.asarray(dataset.targets)
    indices = []
    for label in range(10):
        selected = np.flatnonzero(labels == label)[:samples_per_class]
        if selected.size != samples_per_class:
            raise ValueError(f"class {label} has only {selected.size} test examples")
        indices.extend(selected.tolist())
    return DataLoader(
        Subset(dataset, indices), batch_size=batch_size, shuffle=False, num_workers=0
    )


def _mmd_values_from_kernel(
    kernel: torch.Tensor, n: int, *, seed: int, permutations: int
) -> torch.Tensor:
    """Observed plus balanced-label permutation MMD values."""

    if kernel.shape != (2 * n, 2 * n):
        raise ValueError("kernel shape does not match two equal sample banks")
    kernel = kernel.clone()
    kernel.fill_diagonal_(0.0)
    total_off_diagonal = kernel.sum()
    generator = torch.Generator(device="cpu").manual_seed(seed)
    signs = [
        torch.cat(
            [torch.ones(n, dtype=kernel.dtype), -torch.ones(n, dtype=kernel.dtype)]
        )
    ]
    for _ in range(permutations):
        permutation = torch.randperm(2 * n, generator=generator)
        sign = -torch.ones(2 * n, dtype=kernel.dtype)
        sign[permutation[:n]] = 1.0
        signs.append(sign)
    assignments = torch.stack(signs, dim=1).to(kernel.device)
    quadratic = (assignments * (kernel @ assignments)).sum(dim=0)
    within_ordered = (total_off_diagonal + quadratic) / 2.0
    cross_ordered = (total_off_diagonal - quadratic) / 2.0
    return within_ordered / float(n * (n - 1)) - cross_ordered / float(n * n)


def _summary(values: torch.Tensor) -> dict:
    observed = float(values[0].item())
    null = values[1:].detach().cpu().double().numpy()
    exceedances = int(np.count_nonzero(null >= observed))
    null_std = float(null.std(ddof=1)) if null.size > 1 else None
    standardized = None
    if null_std is not None and null_std > 0:
        standardized = float((observed - null.mean()) / null_std)
    return {
        "mmd2_u": observed,
        "p_value": float((exceedances + 1) / (null.size + 1)),
        "null_mean": float(null.mean()),
        "null_std": null_std,
        "standardized_excess": standardized,
    }


def _energy_values(result: dict, *, device: torch.device) -> torch.Tensor:
    return torch.tensor(
        [result["statistic"], *result["null_values"]],
        dtype=torch.float64,
        device=device,
    )


def _energy_summary(values: torch.Tensor) -> dict:
    summary = _summary(values)
    summary["energy_u"] = summary.pop("mmd2_u")
    return summary


def _compact_permutation_result(result: dict) -> dict:
    return {key: value for key, value in result.items() if key != "null_values"}


def _agreement(mmd_rows: list[dict], energy_rows: list[dict]) -> dict:
    pairs = [
        {
            "mmd_reject_0.05": mmd["p_value"] <= 0.05,
            "energy_reject_0.05": energy["p_value"] <= 0.05,
        }
        for mmd, energy in zip(mmd_rows, energy_rows)
    ]
    for pair in pairs:
        pair["agree"] = pair["mmd_reject_0.05"] == pair["energy_reject_0.05"]
    n_agree = sum(pair["agree"] for pair in pairs)
    return {
        "n_comparisons": len(pairs),
        "n_agree": n_agree,
        "agreement_rate": n_agree / len(pairs),
        "pairs": pairs,
    }


def _sample_prior(model, label: int, n: int, seed: int, device: torch.device):
    cuda_devices = [device.index or 0] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        return model.sample_from_class_prior(label, n, device)


@torch.no_grad()
def _audit_checkpoint(
    name: str,
    path: Path,
    loader: DataLoader,
    device: torch.device,
    samples_per_class: int,
    permutations: int,
    seed: int,
) -> dict:
    model = load_fcswae_checkpoint(path, "mnist", device)
    adapter = build_factorized_audit_adapter(model)
    posterior_generator = torch.Generator(device="cpu").manual_seed(10_000 + seed)
    content, style, labels = [], [], []
    absolute_error = 0.0
    pixel_count = 0
    correct = 0
    for images, batch_labels in loader:
        images = images.to(device)
        batch_labels = batch_labels.to(device)
        views = adapter.encode_views(
            images, generator=posterior_generator, labels=batch_labels
        )
        reconstructions = adapter.decode(views.content_sample, views.style_sample)
        absolute_error += F.l1_loss(
            reconstructions, images, reduction="sum"
        ).item()
        pixel_count += images.numel()
        correct += int(
            (model.classifier(views.content_mean).argmax(dim=1) == batch_labels)
            .sum()
            .item()
        )
        content.append(views.content_sample)
        style.append(views.style_sample)
        labels.append(batch_labels)
    content = torch.cat(content)
    style = torch.cat(style)
    labels = torch.cat(labels)

    class_rows = []
    content_values, style_values, joint_values = [], [], []
    content_energy_values, style_energy_values, joint_energy_values = [], [], []
    for label in range(10):
        mask = labels == label
        posterior_content = content[mask]
        posterior_style = style[mask]
        prior_content, prior_style = _sample_prior(
            model,
            label,
            samples_per_class,
            20_000 + 100 * seed + label,
            device,
        )
        pooled_content = F.normalize(
            torch.cat([posterior_content, prior_content]), p=2, dim=1
        )
        pooled_style = torch.cat([posterior_style, prior_style])
        content_kernel = multiscale_rbf_kernel(
            pooled_content, pooled_content, SPHERICAL_SIGMAS
        )
        style_kernel = multiscale_rbf_kernel(
            pooled_style, pooled_style, EUCLIDEAN_SIGMAS
        )
        class_content = _mmd_values_from_kernel(
            content_kernel,
            samples_per_class,
            seed=30_000 + 100 * seed + label,
            permutations=permutations,
        )
        class_style = _mmd_values_from_kernel(
            style_kernel,
            samples_per_class,
            seed=40_000 + 100 * seed + label,
            permutations=permutations,
        )
        class_joint = _mmd_values_from_kernel(
            content_kernel * style_kernel,
            samples_per_class,
            seed=50_000 + 100 * seed + label,
            permutations=permutations,
        )
        joint_features = torch.cat(
            [
                pooled_content / np.sqrt(2.0),
                pooled_style / np.sqrt(2.0 * pooled_style.shape[1]),
            ],
            dim=1,
        )
        content_energy = energy_distance_permutation_test(
            pooled_content[:samples_per_class],
            pooled_content[samples_per_class:],
            seed=30_000 + 100 * seed + label,
            n_permutations=permutations,
        )
        style_energy = energy_distance_permutation_test(
            pooled_style[:samples_per_class],
            pooled_style[samples_per_class:],
            seed=40_000 + 100 * seed + label,
            n_permutations=permutations,
        )
        joint_energy = energy_distance_permutation_test(
            joint_features[:samples_per_class],
            joint_features[samples_per_class:],
            seed=50_000 + 100 * seed + label,
            n_permutations=permutations,
        )
        content_values.append(class_content)
        style_values.append(class_style)
        joint_values.append(class_joint)
        content_energy_values.append(_energy_values(content_energy, device=device))
        style_energy_values.append(_energy_values(style_energy, device=device))
        joint_energy_values.append(_energy_values(joint_energy, device=device))
        class_rows.append(
            {
                "label": label,
                "content": _summary(class_content),
                "style": _summary(class_style),
                "joint": _summary(class_joint),
                "content_energy": _compact_permutation_result(content_energy),
                "style_energy": _compact_permutation_result(style_energy),
                "joint_energy": _compact_permutation_result(joint_energy),
            }
        )

    content_summary = _summary(torch.stack(content_values).mean(dim=0))
    style_summary = _summary(torch.stack(style_values).mean(dim=0))
    joint_summary = _summary(torch.stack(joint_values).mean(dim=0))
    content_energy_summary = _energy_summary(
        torch.stack(content_energy_values).mean(dim=0)
    )
    style_energy_summary = _energy_summary(
        torch.stack(style_energy_values).mean(dim=0)
    )
    joint_energy_summary = _energy_summary(
        torch.stack(joint_energy_values).mean(dim=0)
    )
    conditional_hsic = classwise_conditional_hsic_permutation_test(
        content,
        style,
        labels,
        n_classes=10,
        seed=60_000 + seed,
        n_permutations=permutations,
    )
    conditional_dcor = classwise_conditional_distance_correlation_permutation_test(
        content,
        style,
        labels,
        n_classes=10,
        seed=60_000 + seed,
        n_permutations=permutations,
    )
    mmd_agreement_rows = [content_summary, style_summary, joint_summary]
    energy_agreement_rows = [
        content_energy_summary, style_energy_summary, joint_energy_summary
    ]
    for class_row in class_rows:
        for clause in ("content", "style", "joint"):
            mmd_agreement_rows.append(class_row[clause])
            energy_agreement_rows.append(class_row[f"{clause}_energy"])
    cross_estimator_agreement = _agreement(
        mmd_agreement_rows, energy_agreement_rows
    )
    dependence_agrees = (
        conditional_hsic["p_value"] <= 0.05
    ) == (conditional_dcor["p_value"] <= 0.05)

    result = {
        "name": name,
        "checkpoint": str(path.resolve()),
        "checkpoint_sha256": sha256_file(path),
        "n_test": int(labels.numel()),
        "reconstruction_mae": absolute_error / pixel_count,
        "semantic_head_accuracy": correct / int(labels.numel()),
        "content": content_summary,
        "style": style_summary,
        "joint": joint_summary,
        "content_energy": content_energy_summary,
        "style_energy": style_energy_summary,
        "joint_energy": joint_energy_summary,
        "within_class_dependence": {
            "hsic": _compact_permutation_result(conditional_hsic),
            "distance_correlation": _compact_permutation_result(conditional_dcor),
            "rejection_agreement": dependence_agrees,
        },
        "cross_estimator_agreement": cross_estimator_agreement,
        "classes": class_rows,
    }
    print(
        f"{name}: MAE={result['reconstruction_mae']:.5f} "
        f"ACC={result['semantic_head_accuracy']:.4f} "
        f"content={result['content']['mmd2_u']:.6g} "
        f"style={result['style']['mmd2_u']:.6g} "
        f"joint={result['joint']['mmd2_u']:.6g} "
        f"p={result['joint']['p_value']:.4g} "
        f"energy={result['joint_energy']['energy_u']:.6g} "
        f"pE={result['joint_energy']['p_value']:.4g} "
        f"agree={result['cross_estimator_agreement']['agreement_rate']:.1%}",
        flush=True,
    )
    return result


def main() -> None:
    args = parse_args()
    if args.samples_per_class < 2:
        raise ValueError("samples-per-class must be at least 2")
    if args.permutations < 1:
        raise ValueError("permutations must be positive")
    checkpoints = _parse_checkpoints(args.checkpoint)
    device = torch.device(args.device)
    loader = _balanced_test_loader(args.samples_per_class, args.batch_size)
    results = [
        _audit_checkpoint(
            name,
            path,
            loader,
            device,
            args.samples_per_class,
            args.permutations,
            args.seed,
        )
        for name, path in checkpoints
    ]
    payload = {
        "protocol": {
            "version": PROTOCOL_VERSION,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "dataset": "MNIST test (balanced, first N examples per class)",
            "samples_per_class_per_bank": args.samples_per_class,
            "permutations": args.permutations,
            "seed": args.seed,
            "content_sigmas": list(SPHERICAL_SIGMAS),
            "style_sigmas": list(EUCLIDEAN_SIGMAS),
            "joint_kernel": "content multiscale RBF * style multiscale RBF",
            "energy_geometry": {
                "content": "chordal Euclidean distance on unit-normalized content",
                "style": "Euclidean distance in the declared unit-Gaussian prior coordinates",
                "joint": (
                    "Euclidean distance on concatenated content/sqrt(2) and "
                    "style/sqrt(2*d_style), using the declared prior scale"
                ),
            },
            "dependence_robustness": (
                "mean classwise HSIC and distance correlation with matched within-class permutations"
            ),
        },
        "results": results,
    }
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True))
    os.replace(temporary, output)
    print(f"wrote {output}", flush=True)


if __name__ == "__main__":
    main()

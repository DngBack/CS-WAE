#!/usr/bin/env python3
"""Audit exported CelebA-HQ or UTKFace posterior/sampler latent pairs."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.metrics.audit_protocol import hsic_permutation_test  # noqa: E402
from src.metrics.continuous_condition import (  # noqa: E402
    CONTINUOUS_AUDIT_VERSION,
    continuous_hsic_permutation_test,
    paired_conditional_mmd_permutation_test,
    paired_factorized_joint_mmd_permutation_test,
)
from src.utils.provenance import sha256_file  # noqa: E402


RESULT_SCHEMA_VERSION = "face-contract-audit-result-1.0.0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--latents", required=True)
    parser.add_argument("--condition-kind", choices=("continuous", "categorical"))
    parser.add_argument("--permutations", type=int, default=199)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=512)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def _atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True))
    os.replace(temporary, path)


def _metadata_for_archive(path: Path) -> dict:
    metadata_path = path.with_suffix(".json")
    if metadata_path.is_file():
        return json.loads(metadata_path.read_text())
    return {}


def _continuous_probe(features: torch.Tensor, target: torch.Tensor, seed: int) -> dict | None:
    if features.shape[0] < 20:
        return None
    generator = np.random.default_rng(seed)
    order = generator.permutation(features.shape[0])
    cut = max(1, int(0.8 * order.size))
    train, test = order[:cut], order[cut:]
    x = features.numpy()
    y = target.numpy()
    scaler = StandardScaler().fit(x[train])
    predictor = Ridge(alpha=1.0).fit(scaler.transform(x[train]), y[train])
    prediction = predictor.predict(scaler.transform(x[test]))
    return {
        "mae_years": float(mean_absolute_error(y[test], prediction)),
        "r2": float(r2_score(y[test], prediction)),
        "n_train": int(train.size),
        "n_test": int(test.size),
        "split_seed": int(seed),
        "estimator": "standardized ridge regression (alpha=1), held-out 80/20 split",
    }


def _latent_descriptives(features: torch.Tensor) -> dict:
    """Report absolute scale so kernel standardization cannot hide outliers."""

    values = features.double()
    finite = torch.isfinite(values)
    row_norm = torch.linalg.vector_norm(values, dim=1)
    coordinate_std = values.std(dim=0, unbiased=False)
    quantiles = torch.quantile(row_norm, torch.tensor([0.5, 0.9, 0.95, 0.99], dtype=torch.double))
    return {
        "finite_fraction": float(finite.double().mean().item()),
        "row_l2_norm": {
            "q50": float(quantiles[0].item()),
            "q90": float(quantiles[1].item()),
            "q95": float(quantiles[2].item()),
            "q99": float(quantiles[3].item()),
            "max": float(row_norm.max().item()),
        },
        "coordinate_std": {
            "mean": float(coordinate_std.mean().item()),
            "median": float(coordinate_std.median().item()),
            "max": float(coordinate_std.max().item()),
        },
    }


def _dependence_test(
    features: torch.Tensor,
    target: torch.Tensor,
    *,
    kind: str,
    seed: int,
    permutations: int,
) -> dict:
    if kind == "continuous":
        return continuous_hsic_permutation_test(
            features,
            target,
            seed=seed,
            n_permutations=permutations,
        )
    _, remapped = torch.unique(target.long(), sorted=True, return_inverse=True)
    return hsic_permutation_test(
        features,
        remapped,
        seed=seed,
        n_permutations=permutations,
    )


def main() -> None:
    args = parse_args()
    if args.permutations < 1 or args.max_samples < 4:
        raise ValueError("permutations must be positive and max-samples must be >=4")
    latent_path = Path(args.latents)
    if not latent_path.is_file():
        raise FileNotFoundError(latent_path)
    metadata = _metadata_for_archive(latent_path)
    condition_kind = args.condition_kind or metadata.get("condition_kind")
    if condition_kind not in {"continuous", "categorical"}:
        raise ValueError("condition kind is absent from metadata; pass --condition-kind")
    with np.load(latent_path, allow_pickle=False) as archive:
        required = {"q_content", "q_style", "p_content", "p_style", "condition"}
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(f"latent archive missing arrays: {sorted(missing)}")
        available = int(archive["condition"].shape[0])
        n = min(args.max_samples, available)
        q_content = torch.from_numpy(archive["q_content"][:n]).float()
        q_style = torch.from_numpy(archive["q_style"][:n]).float()
        p_content = torch.from_numpy(archive["p_content"][:n]).float()
        p_style = torch.from_numpy(archive["p_style"][:n]).float()
        raw_condition = torch.from_numpy(archive["condition"][:n]).float()
        class_label = (
            torch.from_numpy(archive["class_label"][:n]).long()
            if "class_label" in archive.files
            else raw_condition.long()
        )
        attributes = archive["attributes"][:n] if "attributes" in archive.files else None
        attribute_names = (
            [str(value) for value in archive["attribute_names"].tolist()]
            if "attribute_names" in archive.files
            else []
        )
    condition = raw_condition if condition_kind == "continuous" else class_label

    direct = {
        "content": paired_conditional_mmd_permutation_test(
            q_content,
            p_content,
            condition,
            seed=10_000 + args.seed,
            n_permutations=args.permutations,
            condition_kind=condition_kind,
            geometry="spherical",
        ),
        "style": paired_conditional_mmd_permutation_test(
            q_style,
            p_style,
            condition,
            seed=20_000 + args.seed,
            n_permutations=args.permutations,
            condition_kind=condition_kind,
            geometry="euclidean",
        ),
        "joint": paired_factorized_joint_mmd_permutation_test(
            q_content,
            q_style,
            p_content,
            p_style,
            condition,
            seed=30_000 + args.seed,
            n_permutations=args.permutations,
            condition_kind=condition_kind,
        ),
    }
    condition_leakage = {
        "style": _dependence_test(
            q_style,
            condition,
            kind=condition_kind,
            seed=40_000 + args.seed,
            permutations=args.permutations,
        ),
        "content": _dependence_test(
            q_content,
            condition,
            kind=condition_kind,
            seed=50_000 + args.seed,
            permutations=args.permutations,
        ),
    }
    probes = {}
    if condition_kind == "continuous":
        probes = {
            "age_from_style": _continuous_probe(q_style, raw_condition, 60_000 + args.seed),
            "age_from_content": _continuous_probe(q_content, raw_condition, 70_000 + args.seed),
        }

    attribute_leakage = {}
    if attributes is not None and attributes.ndim == 2:
        for column, name in enumerate(attribute_names):
            target = torch.from_numpy(attributes[:, column]).float()
            unique = torch.unique(target)
            if unique.numel() < 2:
                continue
            is_categorical = bool(
                unique.numel() <= 10 and torch.allclose(unique, unique.round())
                and float(unique.min()) >= 0.0
            )
            target_for_test = target.long() if is_categorical else target
            target_kind = "categorical" if is_categorical else "continuous"
            attribute_leakage[name] = {
                "content": _dependence_test(
                    q_content,
                    target_for_test,
                    kind=target_kind,
                    seed=80_000 + 200 * column + args.seed,
                    permutations=args.permutations,
                ),
                "style": _dependence_test(
                    q_style,
                    target_for_test,
                    kind=target_kind,
                    seed=90_000 + 200 * column + args.seed,
                    permutations=args.permutations,
                ),
            }

    result = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "audit_protocol_version": CONTINUOUS_AUDIT_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "n_samples": n,
        "condition_kind": condition_kind,
        "source": {
            "latent_archive": str(latent_path.resolve()),
            "latent_archive_sha256": sha256_file(latent_path),
            "export_metadata": metadata,
        },
        "direct_conditional_equality": direct,
        "condition_dependence": condition_leakage,
        "latent_descriptives": {
            "posterior_content": _latent_descriptives(q_content),
            "sampler_content": _latent_descriptives(p_content),
            "posterior_style": _latent_descriptives(q_style),
            "sampler_style": _latent_descriptives(p_style),
        },
        "continuous_probes": probes,
        "content_style_attribute_dependence": attribute_leakage,
        "interpretation": {
            "equality": (
                "The direct p-values test exact integrated conditional equality; "
                "failure to reject is not evidence of practical equivalence."
            ),
            "leakage": (
                "Style-condition and style-attribute tests localize dependence but "
                "do not assign semantic causality to a latent coordinate."
            ),
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    _atomic_json(output, result)
    digest = sha256_file(output)
    output.with_suffix(output.suffix + ".sha256").write_text(f"{digest}  {output.name}\n")
    print(json.dumps({"output": str(output), "sha256": digest}, indent=2))


if __name__ == "__main__":
    main()

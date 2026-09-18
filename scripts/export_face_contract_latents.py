#!/usr/bin/env python3
"""Export aligned posterior/sampler latent pairs for face contract audits.

Examples
--------
UTKFace continuous-age contract::

    python scripts/export_face_contract_latents.py \
      --dataset utkface --checkpoint runs_f/utkface/seed_0/f_cs_wae_model.pth \
      --data-dir data --image-size 128 --output runs_diag/faces/utkface_seed0.npz

CelebA-HQ binary attribute contract (metadata.csv defaults to ``Smiling``)::

    python scripts/export_face_contract_latents.py \
      --dataset celebahq --checkpoint runs_f/celebahq/seed_0/f_cs_wae_model.pth \
      --data-dir data --metadata-csv data/CelebA-HQ/metadata.csv \
      --condition-column Smiling --output runs_diag/faces/celebahq_seed0.npz
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.datasets.face_conditions import (  # noqa: E402
    DEFAULT_AGE_BIN_EDGES,
    load_celebahq_populations,
    load_utkface_populations,
)
from src.metrics.factorized_adapter import FCSWAEAuditAdapter  # noqa: E402
from src.models.f_cs_wae import FCSWAE  # noqa: E402
from src.utils.provenance import sha256_file  # noqa: E402


EXPORT_SCHEMA_VERSION = "face-contract-latents-1.0.0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("utkface", "celebahq"), required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--metadata-csv")
    parser.add_argument("--condition-column", default="Smiling")
    parser.add_argument("--image-size", type=int, choices=(64, 128, 256), default=128)
    parser.add_argument("--split", choices=("train", "validation", "test"), default="test")
    parser.add_argument("--max-samples", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def _load_model(path: Path, image_size: int, device: torch.device) -> FCSWAE:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    state = checkpoint.get("model_state_dict", checkpoint)
    semantic_dim = int(state["encoder.fc_mu_c.weight"].shape[0])
    style_dim = int(state["encoder.fc_mu_s.weight"].shape[0])
    n_classes, n_centers = map(int, state["ema_centers"].shape[:2])
    model = FCSWAE(
        semantic_dim=semantic_dim,
        style_dim=style_dim,
        n_classes=n_classes,
        in_channels=3,
        image_size=image_size,
        n_centers=n_centers,
    ).to(device)
    model.load_state_dict(state)
    model.eval()
    return model


def _sample_discrete_prior(
    model: FCSWAE,
    labels: torch.Tensor,
    *,
    seed: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    content = torch.empty((labels.numel(), model.semantic_dim), device=device)
    style = torch.empty((labels.numel(), model.style_dim), device=device)
    cuda_devices = [device.index or 0] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        for label in torch.unique(labels).tolist():
            positions = torch.nonzero(labels == label, as_tuple=False).flatten()
            sampled_content, sampled_style = model.sample_from_class_prior(
                int(label), int(positions.numel()), device
            )
            content[positions] = sampled_content
            style[positions] = sampled_style
    return content, style


def _age_knots(device: torch.device) -> torch.Tensor:
    edges = torch.tensor(DEFAULT_AGE_BIN_EDGES, dtype=torch.float32, device=device)
    return (edges[:-1] + edges[1:]) / 2.0


def _atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True))
    os.replace(temporary, path)


@torch.no_grad()
def main() -> None:
    args = parse_args()
    if args.max_samples < 4 or args.batch_size < 1:
        raise ValueError("max-samples must be >=4 and batch-size must be positive")
    device = torch.device(args.device)
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    if args.dataset == "utkface":
        populations, dataset_metadata = load_utkface_populations(
            data_dir=args.data_dir,
            image_size=args.image_size,
        )
        condition_kind = "continuous"
    else:
        populations, dataset_metadata = load_celebahq_populations(
            data_dir=args.data_dir,
            metadata_csv=args.metadata_csv,
            condition_column=args.condition_column,
            image_size=args.image_size,
        )
        condition_kind = "categorical"
    population = populations[args.split]
    if len(population) < 4:
        raise ValueError(f"split {args.split!r} has only {len(population)} valid images")
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    indices = torch.randperm(len(population), generator=generator)[
        : min(args.max_samples, len(population))
    ].tolist()
    loader = DataLoader(
        Subset(population, indices),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    model = _load_model(checkpoint_path, args.image_size, device)
    if model.n_classes != int(dataset_metadata["n_classes"]):
        raise ValueError(
            f"checkpoint has {model.n_classes} centers but dataset declares "
            f"{dataset_metadata['n_classes']} classes"
        )
    adapter = FCSWAEAuditAdapter(model)
    posterior_generator = torch.Generator(device="cpu").manual_seed(10_000 + args.seed)
    q_content, q_style, labels, conditions, attributes = [], [], [], [], []
    for images, batch_labels, batch_conditions, batch_attributes in loader:
        images = images.to(device, non_blocking=True)
        views = adapter.encode_views(images, generator=posterior_generator)
        q_content.append(views.content_sample)
        q_style.append(views.style_sample)
        labels.append(batch_labels.to(device))
        conditions.append(batch_conditions.to(device))
        attributes.append(batch_attributes)
    q_content_tensor = torch.cat(q_content)
    q_style_tensor = torch.cat(q_style)
    labels_tensor = torch.cat(labels)
    conditions_tensor = torch.cat(conditions)
    if args.dataset == "utkface":
        prior_generator = torch.Generator(device="cpu").manual_seed(20_000 + args.seed)
        p_content_tensor, p_style_tensor = model.sample_from_continuous_prior(
            conditions_tensor,
            _age_knots(device),
            generator=prior_generator,
        )
        sampler = "normalized interpolation of age-bin EMA centers at real-valued age"
    else:
        p_content_tensor, p_style_tensor = _sample_discrete_prior(
            model,
            labels_tensor,
            seed=20_000 + args.seed,
            device=device,
        )
        sampler = "class-center spherical content prior times N(0,I) style prior"

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.stem}.tmp.npz")
    np.savez_compressed(
        temporary,
        q_content=q_content_tensor.cpu().numpy(),
        q_style=q_style_tensor.cpu().numpy(),
        p_content=p_content_tensor.cpu().numpy(),
        p_style=p_style_tensor.cpu().numpy(),
        condition=conditions_tensor.cpu().numpy(),
        class_label=labels_tensor.cpu().numpy(),
        attributes=torch.cat(attributes).numpy(),
        attribute_names=np.asarray(population.attribute_names),
    )
    os.replace(temporary, output)
    metadata = {
        "schema_version": EXPORT_SCHEMA_VERSION,
        "dataset": args.dataset,
        "dataset_metadata": dataset_metadata,
        "split": args.split,
        "ordered_subset_indices": indices,
        "n_samples": len(indices),
        "seed": args.seed,
        "condition_kind": condition_kind,
        "declared_sampler": sampler,
        "checkpoint": {
            "path": str(checkpoint_path.resolve()),
            "sha256": sha256_file(checkpoint_path),
        },
        "latent_archive": {
            "path": output.name,
            "sha256": sha256_file(output),
        },
    }
    metadata_path = output.with_suffix(".json")
    _atomic_json(metadata_path, metadata)
    print(json.dumps({"latent_archive": str(output), "metadata": str(metadata_path)}, indent=2))


if __name__ == "__main__":
    main()

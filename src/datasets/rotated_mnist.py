"""Deterministic two-domain Rotated-MNIST for native factorization audits.

Each selected MNIST source image appears once in each domain (-30 and +30
degrees).  The ordered source subset is balanced by digit.  Domain is
therefore exactly balanced and independent of the digit label in the finite
dataset, while all model families see the same source images and transforms.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF


ROTATION_ANGLES = (-30.0, 30.0)
SPLIT_SCHEMA_VERSION = "rotated-mnist-two-domain-1.0.0"


def balanced_source_indices(
    targets: torch.Tensor,
    *,
    per_class: int | None,
    seed: int,
    n_classes: int = 10,
) -> torch.Tensor:
    """Select an ordered, class-balanced source subset without replacement."""

    targets = torch.as_tensor(targets, dtype=torch.long)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    selected = []
    available_counts = []
    for class_index in range(n_classes):
        candidates = torch.nonzero(targets == class_index, as_tuple=False).flatten()
        available_counts.append(int(candidates.numel()))
        count = int(candidates.numel()) if per_class is None else int(per_class)
        if count < 1:
            raise ValueError("per_class must be positive or None")
        if count > candidates.numel():
            raise ValueError(
                f"Class {class_index} has {candidates.numel()} examples, requested {count}"
            )
        permutation = torch.randperm(candidates.numel(), generator=generator)
        selected.append(candidates[permutation[:count]])
    # Interleave classes rather than storing ten contiguous class blocks.
    stacked = torch.stack(selected, dim=1).reshape(-1)
    if per_class is None and len(set(available_counts)) != 1:
        raise ValueError(
            "Full source split is not exactly class balanced; set per_class explicitly"
        )
    return stacked


def indices_sha256(indices: torch.Tensor) -> str:
    values = torch.as_tensor(indices, dtype=torch.int64).contiguous().numpy()
    return hashlib.sha256(values.tobytes()).hexdigest()


class TwoDomainRotatedMNIST(Dataset):
    """Return ``(image, digit, domain)`` for a deterministic paired domain set."""

    def __init__(
        self,
        root: str | Path = "data",
        *,
        train: bool,
        per_class: int | None,
        subset_seed: int,
        angles: Sequence[float] = ROTATION_ANGLES,
        download: bool = True,
    ) -> None:
        super().__init__()
        if len(angles) != 2:
            raise ValueError("The canonical cross-model protocol requires two domains")
        if tuple(float(value) for value in angles) != ROTATION_ANGLES:
            raise ValueError(f"Canonical angles are fixed at {ROTATION_ANGLES}")
        self.train = bool(train)
        self.angles = ROTATION_ANGLES
        self.subset_seed = int(subset_seed)
        self.base = datasets.MNIST(
            str(root), train=self.train, download=download, transform=None
        )
        self.source_indices = balanced_source_indices(
            self.base.targets,
            per_class=per_class,
            seed=self.subset_seed,
        )
        self.per_class = int(self.source_indices.numel() // 10)

    def __len__(self) -> int:
        return int(self.source_indices.numel() * len(self.angles))

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, int]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        source_position, domain = divmod(int(index), len(self.angles))
        source_index = int(self.source_indices[source_position])
        image = self.base.data[source_index].unsqueeze(0).float().div_(255.0)
        image = TF.rotate(
            image,
            angle=self.angles[domain],
            interpolation=InterpolationMode.BILINEAR,
            fill=0.0,
        )
        label = int(self.base.targets[source_index])
        return image, label, int(domain)

    def manifest(self) -> dict:
        paired_labels = self.base.targets[self.source_indices].repeat_interleave(2)
        paired_domains = torch.arange(2).repeat(self.source_indices.numel())
        joint_counts = torch.zeros((10, 2), dtype=torch.long)
        for label, domain in zip(paired_labels, paired_domains):
            joint_counts[int(label), int(domain)] += 1
        ordered_pairs = torch.stack(
            [
                self.source_indices.repeat_interleave(2),
                paired_domains,
            ],
            dim=1,
        )
        return {
            "schema_version": SPLIT_SCHEMA_VERSION,
            "official_split": "train" if self.train else "test",
            "angles_degrees": list(self.angles),
            "subset_seed": self.subset_seed,
            "sources_per_class": self.per_class,
            "n_source_images": int(self.source_indices.numel()),
            "n_paired_examples": len(self),
            "ordered_source_indices_sha256": indices_sha256(self.source_indices),
            "ordered_source_domain_pairs_sha256": hashlib.sha256(
                ordered_pairs.contiguous().numpy().astype(np.int64).tobytes()
            ).hexdigest(),
            "digit_domain_counts": joint_counts.tolist(),
            "finite_sample_domain_digit_independence": bool(
                torch.all(joint_counts == joint_counts[0, 0])
            ),
        }


class DomainSubset(Dataset):
    """One native domain view of a :class:`TwoDomainRotatedMNIST` dataset."""

    def __init__(self, dataset: TwoDomainRotatedMNIST, domain: int):
        if domain not in (0, 1):
            raise ValueError("domain must be 0 or 1")
        self.dataset = dataset
        self.domain = int(domain)

    def __len__(self) -> int:
        return int(self.dataset.source_indices.numel())

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, int]:
        return self.dataset[2 * int(index) + self.domain]


def build_rotated_mnist_loaders(
    *,
    root: str | Path = "data",
    seed: int = 0,
    split_seed: int = 2027,
    batch_size: int = 128,
    train_per_class: int = 2000,
    test_per_class: int = 500,
    num_workers: int = 4,
    download: bool = True,
) -> tuple[DataLoader, DataLoader, dict]:
    """Build shared train/test loaders and their immutable split manifest."""

    train_dataset = TwoDomainRotatedMNIST(
        root,
        train=True,
        per_class=train_per_class,
        subset_seed=10_000 + int(split_seed),
        download=download,
    )
    test_dataset = TwoDomainRotatedMNIST(
        root,
        train=False,
        per_class=test_per_class,
        subset_seed=20_000 + int(split_seed),
        download=download,
    )
    generator = torch.Generator(device="cpu").manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )
    manifest = {
        "schema_version": SPLIT_SCHEMA_VERSION,
        "loader_seed": int(seed),
        "split_seed": int(split_seed),
        "train": train_dataset.manifest(),
        "test": test_dataset.manifest(),
    }
    return train_loader, test_loader, manifest


def save_split_manifest(path: str | Path, manifest: dict) -> None:
    """Write a split manifest atomically."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    temporary.replace(target)

"""Deterministic face datasets for sampler-compatibility audits.

The loaders deliberately do not download face data.  CelebA-HQ and UTKFace
have usage terms that must be accepted by the researcher.  UTKFace labels are
parsed from the official ``age_gender_race_timestamp`` filename convention.
CelebA-HQ uses an explicit CSV so that the image-to-CelebA mapping, identity,
and attributes remain auditable instead of being guessed from file names.

Every sample is returned as ``(image, class_label, condition_value,
attributes)``.  The existing trainer consumes the first two entries.  Audit
code additionally uses ``condition_value`` (real age for UTKFace) and the
attribute vector.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
from pathlib import Path
import re
from typing import Iterable, Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from ..config import config


FACE_DATASET_SCHEMA_VERSION = "face-condition-dataset-1.0.0"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
DEFAULT_AGE_BIN_EDGES = (0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 117)
UTKFACE_PATTERN = re.compile(
    r"^(?P<age>\d{1,3})_(?P<gender>[01])_(?P<race>[0-4])_(?P<stamp>.+)"
    r"\.(?:jpg|jpeg|png)$",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True)
class FaceRecord:
    path: Path
    class_label: int
    condition_value: float
    attributes: tuple[float, ...]
    group: str
    split: str


def _canonical_split(value: str | int) -> str:
    normalized = str(value).strip().lower()
    aliases = {
        "0": "train",
        "train": "train",
        "training": "train",
        "1": "validation",
        "val": "validation",
        "valid": "validation",
        "validation": "validation",
        "2": "test",
        "test": "test",
        "testing": "test",
    }
    if normalized not in aliases:
        raise ValueError(f"Unrecognized split value: {value!r}")
    return aliases[normalized]


def _hash_split(group: str, seed: int) -> str:
    digest = hashlib.sha256(f"{seed}:{group}".encode("utf-8")).digest()
    fraction = int.from_bytes(digest[:8], "big") / float(2**64)
    if fraction < 0.8:
        return "train"
    if fraction < 0.9:
        return "validation"
    return "test"


def _age_bin(age: int, edges: Sequence[int]) -> int:
    if len(edges) < 3 or any(a >= b for a, b in zip(edges[:-1], edges[1:])):
        raise ValueError("age bin edges must be strictly increasing")
    if age < edges[0] or age >= edges[-1]:
        raise ValueError(f"age {age} lies outside [{edges[0]}, {edges[-1]})")
    return int(np.searchsorted(np.asarray(edges[1:-1]), age, side="right"))


def _image_transform(image_size: int, train: bool):
    if image_size not in (64, 128, 256):
        raise ValueError("face image_size must be one of 64, 128, or 256")
    operations: list = [transforms.Resize((image_size, image_size), antialias=True)]
    if train:
        operations.append(transforms.RandomHorizontalFlip())
    operations.append(transforms.ToTensor())
    return transforms.Compose(operations)


class FaceConditionDataset(Dataset):
    """Image dataset with both trainer-compatible and audit-only targets."""

    def __init__(
        self,
        records: Sequence[FaceRecord],
        *,
        image_size: int,
        train: bool,
        attribute_names: Sequence[str],
        metadata: dict,
    ) -> None:
        self.records = tuple(records)
        self.transform = _image_transform(image_size, train)
        self.attribute_names = tuple(attribute_names)
        self.metadata = dict(metadata)
        self.image_size = int(image_size)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        with Image.open(record.path) as image:
            tensor = self.transform(image.convert("RGB"))
        return (
            tensor,
            torch.tensor(record.class_label, dtype=torch.long),
            torch.tensor(record.condition_value, dtype=torch.float32),
            torch.tensor(record.attributes, dtype=torch.float32),
        )

    @property
    def groups(self) -> set[str]:
        return {record.group for record in self.records}


def _candidate_root(data_dir: str | Path, names: Iterable[str]) -> Path:
    root = Path(data_dir)
    for name in names:
        candidate = root / name
        if candidate.is_dir():
            return candidate
    return root


def _image_paths(root: Path) -> list[Path]:
    return sorted(
        path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def load_utkface_populations(
    *,
    data_dir: str | Path = "./data",
    image_size: int = 128,
    split_seed: int = 2027,
    age_bin_edges: Sequence[int] = DEFAULT_AGE_BIN_EDGES,
    strict_filenames: bool = False,
) -> tuple[dict[str, FaceConditionDataset], dict]:
    """Load aligned/cropped UTKFace images without downloading them."""

    root = _candidate_root(data_dir, ("UTKFace", "utkface", "crop_part1"))
    paths = _image_paths(root) if root.is_dir() else []
    if not paths:
        raise FileNotFoundError(
            f"No UTKFace images found under {root}. Download the aligned/cropped "
            "archive from https://susanqq.github.io/UTKFace/ after accepting its "
            "non-commercial research terms."
        )
    records: list[FaceRecord] = []
    rejected: list[str] = []
    for path in paths:
        match = UTKFACE_PATTERN.match(path.name)
        if match is None:
            rejected.append(path.name)
            continue
        age = int(match.group("age"))
        gender = int(match.group("gender"))
        race = int(match.group("race"))
        try:
            label = _age_bin(age, age_bin_edges)
        except ValueError:
            rejected.append(path.name)
            continue
        relative = path.relative_to(root).as_posix()
        records.append(
            FaceRecord(
                path=path,
                class_label=label,
                condition_value=float(age),
                attributes=(float(gender), float(race)),
                group=relative,
                split=_hash_split(relative, split_seed),
            )
        )
    if strict_filenames and rejected:
        preview = ", ".join(rejected[:5])
        raise ValueError(f"Rejected {len(rejected)} malformed UTKFace names: {preview}")
    if not records:
        raise ValueError("No valid UTKFace records remained after filename validation")
    metadata = {
        "schema_version": FACE_DATASET_SCHEMA_VERSION,
        "dataset": "UTKFace aligned/cropped",
        "root": str(root.resolve()),
        "split_seed": int(split_seed),
        "split_rule": "SHA-256 80/10/10 over relative image path",
        "condition": "age in years (continuous audit target)",
        "training_proxy": "fixed age bin",
        "age_bin_edges": [int(value) for value in age_bin_edges],
        "n_classes": len(age_bin_edges) - 1,
        "attribute_names": ["gender_code", "race_code"],
        "valid_records": len(records),
        "rejected_filenames": len(rejected),
    }
    populations = {
        split: FaceConditionDataset(
            [record for record in records if record.split == split],
            image_size=image_size,
            train=split == "train",
            attribute_names=("gender_code", "race_code"),
            metadata=metadata,
        )
        for split in ("train", "validation", "test")
    }
    return populations, metadata


def _numeric(value: str, *, column: str, row_number: int) -> float:
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(
            f"CelebA-HQ metadata row {row_number}: {column}={value!r} is not numeric"
        ) from exc


def load_celebahq_populations(
    *,
    data_dir: str | Path = "./data",
    metadata_csv: str | Path | None = None,
    condition_column: str = "Smiling",
    identity_column: str = "identity",
    image_column: str = "image",
    split_column: str = "split",
    attribute_columns: Sequence[str] | None = None,
    image_size: int = 128,
    split_seed: int = 2027,
) -> tuple[dict[str, FaceConditionDataset], dict]:
    """Load CelebA-HQ using an explicit image/identity/attribute mapping CSV.

    Binary conditions encoded as ``{-1, +1}`` are converted to ``{0, 1}``.
    If no split column is present, all images from one identity are assigned to
    the same deterministic split, preventing identity overlap.
    """

    root = _candidate_root(data_dir, ("CelebA-HQ", "celebahq", "celeba_hq"))
    csv_path = Path(metadata_csv) if metadata_csv is not None else root / "metadata.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"CelebA-HQ metadata not found: {csv_path}. Provide a CSV with at least "
            f"{image_column!r} and {condition_column!r}; include identity and split "
            "columns when available."
        )
    with csv_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        columns = tuple(reader.fieldnames or ())
    required = {image_column, condition_column}
    missing = required.difference(columns)
    if missing:
        raise ValueError(f"CelebA-HQ metadata missing columns: {sorted(missing)}")
    if not rows:
        raise ValueError("CelebA-HQ metadata contains no records")

    excluded = {image_column, condition_column, identity_column, split_column}
    if attribute_columns is None:
        attribute_columns = tuple(column for column in columns if column not in excluded)
    unknown = set(attribute_columns).difference(columns)
    if unknown:
        raise ValueError(f"Unknown CelebA-HQ attribute columns: {sorted(unknown)}")

    image_root = root / "images" if (root / "images").is_dir() else root
    available = {path.name: path for path in _image_paths(image_root)}
    raw_conditions = [
        _numeric(row[condition_column], column=condition_column, row_number=index + 2)
        for index, row in enumerate(rows)
    ]
    unique_conditions = sorted(set(raw_conditions))
    if set(unique_conditions) == {-1.0, 1.0}:
        label_map = {-1.0: 0, 1.0: 1}
    else:
        label_map = {value: index for index, value in enumerate(unique_conditions)}

    identities = [
        (row.get(identity_column, "").strip() or row[image_column].strip())
        for row in rows
    ]
    identity_map = {
        identity: index for index, identity in enumerate(sorted(set(identities)))
    }
    audit_attribute_names = (
        ("identity_id", *attribute_columns)
        if identity_column in columns
        else tuple(attribute_columns)
    )

    records: list[FaceRecord] = []
    for index, (row, condition) in enumerate(zip(rows, raw_conditions), start=2):
        relative_name = row[image_column].strip()
        candidate = image_root / relative_name
        path = candidate if candidate.is_file() else available.get(Path(relative_name).name)
        if path is None:
            raise FileNotFoundError(
                f"CelebA-HQ metadata row {index} references missing image {relative_name!r}"
            )
        group = (
            row.get(identity_column, "").strip()
            if identity_column in columns
            else relative_name
        ) or relative_name
        if split_column in columns and row.get(split_column, "").strip():
            split = _canonical_split(row[split_column])
        else:
            split = _hash_split(group, split_seed)
        attributes = tuple(
            _numeric(row[column], column=column, row_number=index)
            for column in attribute_columns
        )
        # CelebA binary attributes use -1/+1.  A common 0/1 scale is easier to
        # consume without erasing non-binary continuous metadata.
        attributes = tuple(
            (value + 1.0) / 2.0 if value in (-1.0, 1.0) else value
            for value in attributes
        )
        if identity_column in columns:
            attributes = (float(identity_map[group]), *attributes)
        records.append(
            FaceRecord(
                path=path,
                class_label=label_map[condition],
                condition_value=float(condition),
                attributes=attributes,
                group=group,
                split=split,
            )
        )
    metadata = {
        "schema_version": FACE_DATASET_SCHEMA_VERSION,
        "dataset": "CelebA-HQ",
        "root": str(root.resolve()),
        "metadata_csv": str(csv_path.resolve()),
        "condition_column": condition_column,
        "condition_label_map": {str(key): value for key, value in label_map.items()},
        "identity_column": identity_column if identity_column in columns else None,
        "attribute_names": list(audit_attribute_names),
        "n_identities": len(identity_map) if identity_column in columns else None,
        "n_classes": len(label_map),
        "split_seed": int(split_seed),
        "split_rule": (
            f"metadata column {split_column}"
            if split_column in columns
            else "SHA-256 80/10/10 grouped by identity when available"
        ),
        "valid_records": len(records),
    }
    populations = {
        split: FaceConditionDataset(
            [record for record in records if record.split == split],
            image_size=image_size,
            train=split == "train",
            attribute_names=audit_attribute_names,
            metadata=metadata,
        )
        for split in ("train", "validation", "test")
    }
    return populations, metadata


def _loaders_from_populations(
    populations: dict[str, FaceConditionDataset],
    *,
    batch_size: int,
    num_workers: int,
    seed: int | None,
) -> tuple[DataLoader, DataLoader]:
    generator = None
    if seed is not None:
        generator = torch.Generator(device="cpu").manual_seed(seed)
    common = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": num_workers > 0,
    }
    train = DataLoader(
        populations["train"], shuffle=True, generator=generator, **common
    )
    test = DataLoader(populations["test"], shuffle=False, **common)
    return train, test


def get_utkface_loaders(
    data_dir: str = "./data",
    batch_size: int | None = None,
    num_workers: int | None = None,
    seed: int | None = None,
    **kwargs,
):
    populations, _ = load_utkface_populations(data_dir=data_dir, **kwargs)
    return _loaders_from_populations(
        populations,
        batch_size=batch_size or config.batch_size,
        num_workers=config.num_workers if num_workers is None else num_workers,
        seed=seed,
    )


def get_celebahq_loaders(
    data_dir: str = "./data",
    batch_size: int | None = None,
    num_workers: int | None = None,
    seed: int | None = None,
    **kwargs,
):
    populations, _ = load_celebahq_populations(data_dir=data_dir, **kwargs)
    return _loaders_from_populations(
        populations,
        batch_size=batch_size or config.batch_size,
        num_workers=config.num_workers if num_workers is None else num_workers,
        seed=seed,
    )


def get_utkface_info() -> dict:
    return {
        "name": "UTKFace",
        "input_shape": (3, 128, 128),
        "in_channels": 3,
        "image_size": 128,
        "n_classes": len(DEFAULT_AGE_BIN_EDGES) - 1,
        "num_classes": len(DEFAULT_AGE_BIN_EDGES) - 1,
        "condition_type": "continuous_age_with_binned_training_proxy",
        "color": True,
    }


def get_celebahq_info() -> dict:
    return {
        "name": "CelebA-HQ",
        "input_shape": (3, 128, 128),
        "in_channels": 3,
        "image_size": 128,
        "n_classes": 2,
        "num_classes": 2,
        "condition_type": "binary_attribute",
        "color": True,
    }


__all__ = [
    "DEFAULT_AGE_BIN_EDGES",
    "FACE_DATASET_SCHEMA_VERSION",
    "FaceConditionDataset",
    "FaceRecord",
    "get_celebahq_info",
    "get_celebahq_loaders",
    "get_utkface_info",
    "get_utkface_loaders",
    "load_celebahq_populations",
    "load_utkface_populations",
]

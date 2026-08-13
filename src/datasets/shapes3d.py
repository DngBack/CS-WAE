"""Shapes3D loader and immutable IID factor split for the ICLR 2027 audit."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Mapping

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


SHAPES3D_SCHEMA_VERSION = "shapes3d-factor-audit-split-1.0.0"
SHAPES3D_CACHE_SCHEMA_VERSION = "shapes3d-local-cache-1.0.0"
SHAPES3D_FILENAME = "3dshapes.h5"
SHAPES3D_URL = "https://storage.googleapis.com/3d-shapes/3dshapes.h5"
SHAPES3D_SIZE_BYTES = 267_573_662
SHAPES3D_MD5 = "099a2078d58cec4daad0702c55d06868"
SHAPES3D_SHA256 = "0a0f6ed98baff276a50f3a081a7434d788da63cb135a98189b2a5b5769be1785"
FACTOR_NAMES = (
    "floor_hue",
    "wall_hue",
    "object_hue",
    "scale",
    "shape",
    "orientation",
)
FACTOR_CARDINALITIES = (10, 10, 10, 8, 4, 15)
SHAPE_FACTOR_INDEX = 4
DEFAULT_SPLIT_COUNTS = {
    "train": 120_000,
    "validation": 20_000,
    "test": 20_000,
}


def default_shapes3d_cache_paths(path: str | Path) -> tuple[Path, Path, Path]:
    """Return deterministic local cache paths adjacent to the official HDF5."""

    source = Path(path)
    return (
        source.with_name("3dshapes_images_uint8.npy"),
        source.with_name("3dshapes_factor_indices.npy"),
        source.with_name("3dshapes_cache.json"),
    )


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_shapes3d_file(path: str | Path, *, verify_sha256: bool = True) -> dict:
    """Validate file size, HDF5 keys/shapes, and optionally the canonical hash."""

    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(
            f"Shapes3D file not found: {source}. Download {SHAPES3D_URL}"
        )
    size = source.stat().st_size
    if size != SHAPES3D_SIZE_BYTES:
        raise ValueError(f"Unexpected Shapes3D size: {size} != {SHAPES3D_SIZE_BYTES}")
    digest = sha256_file(source) if verify_sha256 else None
    if verify_sha256 and digest != SHAPES3D_SHA256:
        raise ValueError(f"Unexpected Shapes3D SHA-256: {digest}")
    with h5py.File(source, "r") as handle:
        image_shape = tuple(handle["images"].shape)
        label_shape = tuple(handle["labels"].shape)
        image_dtype = str(handle["images"].dtype)
        label_dtype = str(handle["labels"].dtype)
    if image_shape != (480_000, 64, 64, 3) or label_shape != (480_000, 6):
        raise ValueError(
            f"Unexpected Shapes3D schema: images={image_shape}, labels={label_shape}"
        )
    return {
        "path": str(source.resolve()),
        "size_bytes": size,
        "sha256": digest,
        "images": {"shape": list(image_shape), "dtype": image_dtype},
        "labels": {"shape": list(label_shape), "dtype": label_dtype},
        "source_url": SHAPES3D_URL,
    }


def factor_values_to_indices(labels: np.ndarray) -> np.ndarray:
    """Map official floating factor values to categorical indices."""

    labels = np.asarray(labels)
    if labels.ndim != 2 or labels.shape[1] != len(FACTOR_NAMES):
        raise ValueError(f"Expected labels [N,6], got {labels.shape}")
    result = np.empty(labels.shape, dtype=np.int64)
    for column, cardinality in enumerate(FACTOR_CARDINALITIES):
        values = np.unique(labels[:, column])
        if values.size != cardinality:
            raise ValueError(
                f"Factor {FACTOR_NAMES[column]} has {values.size} values, "
                f"expected {cardinality}"
            )
        positions = np.searchsorted(values, labels[:, column])
        if not np.array_equal(values[positions], labels[:, column]):
            raise ValueError(f"Could not map factor {FACTOR_NAMES[column]}")
        result[:, column] = positions
    return result


def _atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True))
    os.replace(temporary, path)


def _valid_shapes3d_cache(
    image_path: Path,
    factor_path: Path,
    metadata_path: Path,
) -> bool:
    if not (image_path.exists() and factor_path.exists() and metadata_path.exists()):
        return False
    try:
        metadata = json.loads(metadata_path.read_text())
        if metadata != {
            "schema_version": SHAPES3D_CACHE_SCHEMA_VERSION,
            "source_sha256": SHAPES3D_SHA256,
            "images": {
                "path": image_path.name,
                "shape": [480_000, 64, 64, 3],
                "dtype": "uint8",
            },
            "factor_indices": {
                "path": factor_path.name,
                "shape": [480_000, 6],
                "dtype": "int64",
            },
        }:
            return False
        images = np.load(image_path, mmap_mode="r")
        factors = np.load(factor_path, mmap_mode="r")
        return (
            images.shape == (480_000, 64, 64, 3)
            and images.dtype == np.uint8
            and factors.shape == (480_000, 6)
            and factors.dtype == np.int64
        )
    except (OSError, ValueError, json.JSONDecodeError):
        return False


def materialize_shapes3d_cache(path: str | Path) -> tuple[Path, Path, dict]:
    """Create a sequential, memory-mapped cache for the compressed source.

    The official file uses large gzip chunks. Random per-image HDF5 access can
    therefore spend substantially more time decompressing than training. This
    cache is a lossless uint8 copy and does not alter the population or split.
    Completion metadata is written only after both arrays are fully durable.
    """

    source = Path(path)
    image_path, factor_path, metadata_path = default_shapes3d_cache_paths(source)
    if _valid_shapes3d_cache(image_path, factor_path, metadata_path):
        return image_path, factor_path, json.loads(metadata_path.read_text())

    image_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_image = image_path.with_name(f".{image_path.name}.tmp")
    temporary_factor = factor_path.with_name(f".{factor_path.name}.tmp")
    # open_memmap requires a filename ending in .npy to preserve the NPY header.
    temporary_image = temporary_image.with_suffix(temporary_image.suffix + ".npy")
    temporary_factor = temporary_factor.with_suffix(temporary_factor.suffix + ".npy")
    images_cache = np.lib.format.open_memmap(
        temporary_image,
        mode="w+",
        dtype=np.uint8,
        shape=(480_000, 64, 64, 3),
    )
    with h5py.File(source, "r") as handle:
        source_images = handle["images"]
        # Match the leading HDF5 chunk extent so each compressed chunk is read
        # once rather than repeatedly across smaller sequential batches.
        chunk_size = int(source_images.chunks[0]) if source_images.chunks else 15_000
        for start in range(0, source_images.shape[0], chunk_size):
            stop = min(source_images.shape[0], start + chunk_size)
            images_cache[start:stop] = source_images[start:stop]
            images_cache.flush()
            print(f"Shapes3D cache: {stop}/{source_images.shape[0]} images", flush=True)
        factors = factor_values_to_indices(handle["labels"][:])
    del images_cache
    with temporary_factor.open("wb") as handle:
        np.save(handle, factors, allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary_image, image_path)
    os.replace(temporary_factor, factor_path)
    metadata = {
        "schema_version": SHAPES3D_CACHE_SCHEMA_VERSION,
        "source_sha256": SHAPES3D_SHA256,
        "images": {
            "path": image_path.name,
            "shape": [480_000, 64, 64, 3],
            "dtype": "uint8",
        },
        "factor_indices": {
            "path": factor_path.name,
            "shape": [480_000, 6],
            "dtype": "int64",
        },
    }
    _atomic_json(metadata_path, metadata)
    return image_path, factor_path, metadata


def _indices_digest(indices: np.ndarray) -> str:
    packed = np.asarray(indices, dtype=np.int64).ravel()
    return hashlib.sha256(packed.tobytes()).hexdigest()


def fixed_iid_factor_split(
    factor_indices: np.ndarray,
    *,
    seed: int = 2027,
    split_counts: Mapping[str, int] = DEFAULT_SPLIT_COUNTS,
) -> dict[str, np.ndarray]:
    """Create disjoint shape-stratified IID subsets without a holdout search."""

    factors = np.asarray(factor_indices, dtype=np.int64)
    required_names = ("train", "validation", "test")
    if tuple(split_counts.keys()) != required_names:
        raise ValueError(f"split_counts keys must be ordered as {required_names}")
    if any(int(split_counts[name]) % 4 for name in required_names):
        raise ValueError("Each split count must be divisible by four shape classes")
    generator = np.random.default_rng(seed)
    per_shape = {
        name: int(split_counts[name]) // 4 for name in required_names
    }
    result = {name: [] for name in required_names}
    for shape_index in range(4):
        candidates = np.flatnonzero(factors[:, SHAPE_FACTOR_INDEX] == shape_index)
        candidates = candidates[generator.permutation(candidates.size)]
        needed = sum(per_shape.values())
        if candidates.size < needed:
            raise ValueError(
                f"Shape {shape_index} has {candidates.size} rows; need {needed}"
            )
        offset = 0
        for name in required_names:
            count = per_shape[name]
            result[name].append(candidates[offset : offset + count])
            offset += count
    packed = {}
    for name in required_names:
        values = np.concatenate(result[name]).astype(np.int64, copy=False)
        # Deterministically mix the four shape blocks without changing members.
        local = np.random.default_rng(seed + 10_000 + required_names.index(name))
        packed[name] = values[local.permutation(values.size)]
    concatenated = np.concatenate(list(packed.values()))
    if np.unique(concatenated).size != concatenated.size:
        raise AssertionError("Shapes3D splits overlap")
    return packed


def split_manifest(
    *,
    source_metadata: dict,
    factor_indices: np.ndarray,
    splits: Mapping[str, np.ndarray],
    split_seed: int,
) -> dict:
    """Describe exact populations and factor marginals for every split."""

    result = {
        "schema_version": SHAPES3D_SCHEMA_VERSION,
        "split_seed": int(split_seed),
        "split_type": "fixed shape-stratified IID factor split",
        "source": source_metadata,
        "factor_names": list(FACTOR_NAMES),
        "factor_cardinalities": list(FACTOR_CARDINALITIES),
        "designated_semantic_factor": "shape",
        "splits": {},
    }
    for name, indices in splits.items():
        values = factor_indices[indices]
        result["splits"][name] = {
            "n_samples": int(indices.size),
            "ordered_indices_sha256": _indices_digest(indices),
            "factor_counts": {
                factor_name: np.bincount(
                    values[:, column], minlength=FACTOR_CARDINALITIES[column]
                ).astype(int).tolist()
                for column, factor_name in enumerate(FACTOR_NAMES)
            },
        }
    return result


class Shapes3DDataset(Dataset):
    """Multiprocess-safe lazy HDF5 image access for a fixed source subset."""

    def __init__(
        self,
        path: str | Path,
        indices: np.ndarray,
        factor_indices: np.ndarray,
        *,
        image_cache_path: str | Path | None = None,
    ) -> None:
        self.path = str(Path(path).resolve())
        self.indices = np.asarray(indices, dtype=np.int64)
        self.factors = np.asarray(factor_indices[self.indices], dtype=np.int64)
        self.image_cache_path = (
            str(Path(image_cache_path).resolve()) if image_cache_path is not None else None
        )
        self._handle = None
        self._images = None
        self._pid = None

    def __len__(self) -> int:
        return int(self.indices.size)

    def _ensure_open(self) -> None:
        pid = os.getpid()
        if self._handle is None or self._pid != pid:
            if self._handle is not None:
                if hasattr(self._handle, "close"):
                    self._handle.close()
            if self.image_cache_path is not None:
                self._handle = np.load(self.image_cache_path, mmap_mode="r")
                self._images = self._handle
            else:
                self._handle = h5py.File(self.path, "r")
                self._images = self._handle["images"]
            self._pid = pid

    def __getitem__(self, position: int):
        self._ensure_open()
        image = np.asarray(self._images[int(self.indices[position])])
        image_tensor = torch.from_numpy(image.copy()).permute(2, 0, 1).float().div_(255.0)
        factors = torch.from_numpy(self.factors[position].copy()).long()
        shape = int(factors[SHAPE_FACTOR_INDEX].item())
        return image_tensor, shape, factors

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_handle"] = None
        state["_images"] = None
        state["_pid"] = None
        return state

    def close(self) -> None:
        if self._handle is not None:
            if hasattr(self._handle, "close"):
                self._handle.close()
            elif isinstance(self._handle, np.memmap) and self._handle._mmap is not None:
                self._handle._mmap.close()
        self._handle = self._images = self._pid = None

    def __del__(self):
        self.close()


def load_shapes3d_populations(
    *,
    path: str | Path = "data/shapes3d/3dshapes.h5",
    split_seed: int = 2027,
    split_counts: Mapping[str, int] = DEFAULT_SPLIT_COUNTS,
    verify_sha256: bool = True,
    use_cache: bool = True,
) -> tuple[dict[str, Shapes3DDataset], dict]:
    source = verify_shapes3d_file(path, verify_sha256=verify_sha256)
    image_cache_path = None
    if use_cache:
        image_cache_path, factor_cache_path, cache_metadata = materialize_shapes3d_cache(path)
        factors = np.load(factor_cache_path, mmap_mode="r")
        source["local_cache"] = cache_metadata
    else:
        with h5py.File(path, "r") as handle:
            factors = factor_values_to_indices(handle["labels"][:])
    splits = fixed_iid_factor_split(
        factors, seed=split_seed, split_counts=split_counts
    )
    manifest = split_manifest(
        source_metadata=source,
        factor_indices=factors,
        splits=splits,
        split_seed=split_seed,
    )
    datasets = {
        name: Shapes3DDataset(
            path, indices, factors, image_cache_path=image_cache_path
        )
        for name, indices in splits.items()
    }
    return datasets, manifest


def build_shapes3d_loaders(
    *,
    path: str | Path = "data/shapes3d/3dshapes.h5",
    seed: int = 0,
    split_seed: int = 2027,
    batch_size: int = 128,
    num_workers: int = 4,
    split_counts: Mapping[str, int] = DEFAULT_SPLIT_COUNTS,
    verify_sha256: bool = True,
    use_cache: bool = True,
) -> tuple[dict[str, DataLoader], dict]:
    populations, manifest = load_shapes3d_populations(
        path=path,
        split_seed=split_seed,
        split_counts=split_counts,
        verify_sha256=verify_sha256,
        use_cache=use_cache,
    )
    train_generator = torch.Generator(device="cpu").manual_seed(seed)
    loaders = {
        "train": DataLoader(
            populations["train"],
            batch_size=batch_size,
            shuffle=True,
            generator=train_generator,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=num_workers > 0,
        ),
        "validation": DataLoader(
            populations["validation"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=num_workers > 0,
        ),
        "test": DataLoader(
            populations["test"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=num_workers > 0,
        ),
    }
    manifest["loader_seed"] = int(seed)
    return loaders, manifest


def save_shapes3d_manifest(path: str | Path, manifest: dict) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    temporary.replace(target)


__all__ = [
    "DEFAULT_SPLIT_COUNTS",
    "FACTOR_CARDINALITIES",
    "FACTOR_NAMES",
    "SHAPE_FACTOR_INDEX",
    "SHAPES3D_CACHE_SCHEMA_VERSION",
    "SHAPES3D_SCHEMA_VERSION",
    "Shapes3DDataset",
    "build_shapes3d_loaders",
    "factor_values_to_indices",
    "fixed_iid_factor_split",
    "load_shapes3d_populations",
    "materialize_shapes3d_cache",
    "save_shapes3d_manifest",
    "verify_shapes3d_file",
]

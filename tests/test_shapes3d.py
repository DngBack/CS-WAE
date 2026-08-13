from collections import OrderedDict

import h5py
import numpy as np
import torch

from src.datasets.shapes3d import (
    FACTOR_CARDINALITIES,
    Shapes3DDataset,
    factor_values_to_indices,
    fixed_iid_factor_split,
)
from src.models.f_cs_wae import FCSWAE
from src.models.shapes3d_factor_evaluator import Shapes3DFactorEvaluator
from src.models.shapes3d_factorized_baselines import (
    Shapes3DConditionalVAE,
    Shapes3DContentStyleVAE,
)
from src.metrics.factorized_adapter import build_factorized_audit_adapter


def _factor_grid(repeats: int = 2) -> np.ndarray:
    grid = np.stack(
        np.meshgrid(*[np.arange(size) for size in FACTOR_CARDINALITIES], indexing="ij"),
        axis=-1,
    ).reshape(-1, 6)
    return np.tile(grid, (repeats, 1)).astype(np.float64)


def test_factor_mapping_and_fixed_split_are_disjoint_and_shape_balanced():
    labels = _factor_grid(repeats=2)
    mapped = factor_values_to_indices(labels)
    assert np.array_equal(mapped, labels.astype(np.int64))
    counts = OrderedDict(train=40, validation=20, test=20)
    first = fixed_iid_factor_split(mapped, seed=2027, split_counts=counts)
    second = fixed_iid_factor_split(mapped, seed=2027, split_counts=counts)
    for name, count in counts.items():
        assert np.array_equal(first[name], second[name])
        assert first[name].size == count
        shape_counts = np.bincount(mapped[first[name], 4], minlength=4)
        assert np.all(shape_counts == count // 4)
    combined = np.concatenate(list(first.values()))
    assert np.unique(combined).size == combined.size


def test_lazy_hdf5_dataset_returns_image_shape_and_factor_vector(tmp_path):
    path = tmp_path / "tiny.h5"
    images = np.arange(3 * 64 * 64 * 3, dtype=np.uint8).reshape(3, 64, 64, 3)
    factors = np.zeros((3, 6), dtype=np.int64)
    factors[:, 4] = [0, 1, 2]
    with h5py.File(path, "w") as handle:
        handle.create_dataset("images", data=images)
    dataset = Shapes3DDataset(path, np.array([2, 0]), factors)
    image, shape, vector = dataset[0]
    assert image.shape == (3, 64, 64)
    assert image.dtype == torch.float32
    assert 0.0 <= float(image.min()) <= float(image.max()) <= 1.0
    assert shape == 2
    assert vector.tolist() == factors[2].tolist()


def test_memory_mapped_dataset_returns_same_image(tmp_path):
    path = tmp_path / "tiny.h5"
    path.touch()
    images = np.arange(3 * 64 * 64 * 3, dtype=np.uint8).reshape(3, 64, 64, 3)
    cache = tmp_path / "images.npy"
    np.save(cache, images, allow_pickle=False)
    factors = np.zeros((3, 6), dtype=np.int64)
    factors[:, 4] = [0, 1, 2]
    dataset = Shapes3DDataset(
        path,
        np.array([2, 0]),
        factors,
        image_cache_path=cache,
    )
    image, shape, vector = dataset[0]
    expected = torch.from_numpy(images[2].copy()).permute(2, 0, 1).float() / 255.0
    assert torch.equal(image, expected)
    assert shape == 2
    assert vector.tolist() == factors[2].tolist()


def test_fcswae_and_factor_evaluator_have_native_64_outputs():
    model = FCSWAE(
        semantic_dim=16,
        style_dim=12,
        n_classes=4,
        in_channels=3,
        image_size=64,
    )
    images = torch.rand(2, 3, 64, 64)
    reconstruction = model(images)[0]
    assert reconstruction.shape == images.shape
    assert len(model.decoder.ups) == 4

    evaluator = Shapes3DFactorEvaluator(hidden_dim=32)
    logits = evaluator(images)
    assert tuple(logits) == (
        "floor_hue", "wall_hue", "object_hue", "scale", "shape", "orientation"
    )
    assert [value.shape[1] for value in logits.values()] == list(FACTOR_CARDINALITIES)


def test_shapes3d_factorized_baselines_have_reproducible_native_views():
    images = torch.rand(4, 3, 64, 64)
    labels = torch.tensor([0, 1, 2, 3])
    for model in (
        Shapes3DConditionalVAE(content_dim=16, style_dim=12),
        Shapes3DContentStyleVAE(content_dim=16, style_dim=12),
    ):
        model.eval()
        adapter = build_factorized_audit_adapter(model)
        first = adapter.encode_views(
            images,
            labels=labels,
            generator=torch.Generator().manual_seed(9),
        )
        second = adapter.encode_views(
            images,
            labels=labels,
            generator=torch.Generator().manual_seed(9),
        )
        assert first.content_sample.shape == (4, 16)
        assert first.style_sample.shape == (4, 12)
        assert torch.equal(first.style_sample, second.style_sample)
        assert first.style_logvar is not None
        decoded = adapter.decode(first.content_sample, first.style_sample)
        assert decoded.shape == images.shape


def test_shapes3d_factorized_baseline_losses_are_finite():
    images = torch.rand(4, 3, 64, 64)
    labels = torch.tensor([0, 1, 2, 3])
    conditional = Shapes3DConditionalVAE(content_dim=8, style_dim=8)
    content_style = Shapes3DContentStyleVAE(content_dim=8, style_dim=8)
    conditional_terms = conditional.loss_terms(images, labels, kl_weight=0.01)
    content_style_terms = content_style.loss_terms(
        images, labels, kl_weight=0.01, classifier_weight=0.1
    )
    assert all(torch.isfinite(value) for value in conditional_terms.values())
    assert all(torch.isfinite(value) for value in content_style_terms.values())

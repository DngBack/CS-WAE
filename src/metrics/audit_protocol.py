"""Canonical Stage-0 audit protocol for split-latent models.

This module deliberately separates two latent views:

``z_s_sample``
    A posterior sample.  This is the primary random variable for statements
    about the sampling contract q(z_s) and q(z_s | y).

``mu_s``
    The deterministic posterior mean.  This is the primary representation
    used by probes and deterministic decoder interventions such as swaps.

Training losses are not implemented here.  In particular, the training MMD
remains the biased V-statistic in ``src/utils/loss_f_cs_wae.py``.  All MMD
functions below are evaluation-only unbiased U-statistics.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Iterable, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


AUDIT_PROTOCOL_VERSION = "stage0-1.3.0"
DEFAULT_EVAL_SAMPLES = 2048
DEFAULT_HSIC_SAMPLES = 1024
DEFAULT_HSIC_PERMUTATIONS = 200
EUCLIDEAN_SIGMAS = (0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0)
SPHERICAL_SIGMAS = (0.1, 0.3, 0.5, 1.0, 2.0)


@dataclass(frozen=True)
class ProbeSplit:
    """Stratified train/validation/test indices for a probe experiment."""

    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray

    def metadata(self) -> dict:
        packed = np.concatenate([self.train, self.validation, self.test]).astype(np.int64)
        return {
            "train_size": int(self.train.size),
            "validation_size": int(self.validation.size),
            "test_size": int(self.test.size),
            "indices_sha256": hashlib.sha256(packed.tobytes()).hexdigest(),
        }


def rbf_kernel(x: torch.Tensor, y: torch.Tensor, sigma: float) -> torch.Tensor:
    """RBF kernel using squared Euclidean distance."""

    dist_sq = torch.cdist(x, y, p=2).square()
    return torch.exp(-dist_sq / (2.0 * sigma**2 + 1e-12))


def multiscale_rbf_kernel(
    x: torch.Tensor,
    y: torch.Tensor,
    sigmas: Sequence[float] = EUCLIDEAN_SIGMAS,
) -> torch.Tensor:
    """Average RBF kernel over a fixed, reported bandwidth ladder."""

    kernel = torch.zeros((x.shape[0], y.shape[0]), device=x.device, dtype=x.dtype)
    for sigma in sigmas:
        kernel.add_(rbf_kernel(x, y, float(sigma)))
    return kernel / float(len(sigmas))


def mmd2_unbiased(
    x: torch.Tensor,
    y: torch.Tensor,
    sigmas: Sequence[float] = EUCLIDEAN_SIGMAS,
) -> float:
    """Unbiased two-sample MMD² U-statistic.

    The estimate may be slightly negative at finite sample size.  It is not
    clipped because clipping changes its null distribution and biases
    calibration.
    """

    if x.ndim != 2 or y.ndim != 2:
        raise ValueError("MMD inputs must be rank-2 tensors")
    if x.shape[0] < 2 or y.shape[0] < 2:
        raise ValueError("Unbiased MMD requires at least two samples per set")
    if x.shape[1] != y.shape[1]:
        raise ValueError("MMD inputs must have the same feature dimension")

    k_xx = multiscale_rbf_kernel(x, x, sigmas)
    k_yy = multiscale_rbf_kernel(y, y, sigmas)
    k_xy = multiscale_rbf_kernel(x, y, sigmas)
    m, n = x.shape[0], y.shape[0]
    xx = (k_xx.sum() - k_xx.diagonal().sum()) / (m * (m - 1))
    yy = (k_yy.sum() - k_yy.diagonal().sum()) / (n * (n - 1))
    xy = k_xy.mean()
    return float((xx + yy - 2.0 * xy).item())


def mmd2_permutation_test(
    x: torch.Tensor,
    y: torch.Tensor,
    seed: int = 0,
    n_permutations: int = DEFAULT_HSIC_PERMUTATIONS,
    sigmas: Sequence[float] = EUCLIDEAN_SIGMAS,
) -> dict:
    """Calibrate equal-size two-sample MMD²_U by label permutation.

    Under the equality null, pooled observations are exchangeable.  We build
    the multiscale kernel once and evaluate balanced random relabellings with
    batched quadratic forms.  This is a test of exact equality, not an
    equivalence test: a small p-value can coexist with a numerically small
    MMD when the evaluation set has enough power.
    """

    if x.ndim != 2 or y.ndim != 2:
        raise ValueError("MMD inputs must be rank-2 tensors")
    if x.shape != y.shape:
        raise ValueError("Permutation calibration currently requires equal-size, equal-dimension inputs")
    if x.shape[0] < 2:
        raise ValueError("MMD permutation calibration requires at least two samples per group")
    if n_permutations < 1:
        raise ValueError("n_permutations must be positive")

    n = x.shape[0]
    pooled = torch.cat([x, y], dim=0)
    kernel = multiscale_rbf_kernel(pooled, pooled, sigmas)
    kernel.fill_diagonal_(0.0)
    total_off_diagonal = kernel.sum()

    generator = torch.Generator(device="cpu").manual_seed(seed)
    assignments = []
    observed_sign = torch.cat([
        torch.ones(n, dtype=pooled.dtype),
        -torch.ones(n, dtype=pooled.dtype),
    ])
    assignments.append(observed_sign)
    for _ in range(n_permutations):
        permutation = torch.randperm(2 * n, generator=generator)
        sign = -torch.ones(2 * n, dtype=pooled.dtype)
        sign[permutation[:n]] = 1.0
        assignments.append(sign)
    signs = torch.stack(assignments, dim=1).to(pooled.device)

    quadratic = (signs * (kernel @ signs)).sum(dim=0)
    within_ordered = (total_off_diagonal + quadratic) / 2.0
    cross_ordered = (total_off_diagonal - quadratic) / 2.0
    values = within_ordered / float(n * (n - 1)) - cross_ordered / float(n * n)
    observed = float(values[0].item())
    null = values[1:].detach().cpu().numpy().astype(np.float64)
    exceedances = int(np.count_nonzero(null >= observed))
    null_std = float(null.std(ddof=1)) if null.size > 1 else None
    standardized_excess = None
    if null_std is not None and null_std > 0.0:
        standardized_excess = float((observed - null.mean()) / null_std)
    return {
        "statistic": observed,
        "p_value": float((exceedances + 1) / (n_permutations + 1)),
        "null_mean": float(null.mean()),
        "null_std": null_std,
        "standardized_excess": standardized_excess,
        "null_quantiles": {
            "q05": float(np.quantile(null, 0.05)),
            "q50": float(np.quantile(null, 0.50)),
            "q95": float(np.quantile(null, 0.95)),
            "q99": float(np.quantile(null, 0.99)),
        },
        "null_values": null.tolist(),
        "n_per_group": int(n),
        "n_permutations": int(n_permutations),
        "sigmas": [float(value) for value in sigmas],
        "estimator": "unbiased MMD^2 U-statistic with balanced pooled-label permutation null",
        "interpretation_warning": (
            "This tests exact distributional equality; failure to reject is not proof of equivalence."
        ),
    }


def _balanced_energy_values_from_distances(
    distances: torch.Tensor,
    n: int,
    *,
    seed: int,
    n_permutations: int,
) -> torch.Tensor:
    """Observed plus balanced-label energy statistics from a pooled distance matrix."""

    if distances.shape != (2 * n, 2 * n):
        raise ValueError("distance matrix shape does not match two equal sample banks")
    if n < 2:
        raise ValueError("energy distance requires at least two samples per bank")
    distances = distances.clone()
    distances.fill_diagonal_(0.0)
    total_off_diagonal = distances.sum()
    generator = torch.Generator(device="cpu").manual_seed(seed)
    signs = [
        torch.cat(
            [
                torch.ones(n, dtype=distances.dtype),
                -torch.ones(n, dtype=distances.dtype),
            ]
        )
    ]
    for _ in range(n_permutations):
        permutation = torch.randperm(2 * n, generator=generator)
        sign = -torch.ones(2 * n, dtype=distances.dtype)
        sign[permutation[:n]] = 1.0
        signs.append(sign)
    assignments = torch.stack(signs, dim=1).to(distances.device)
    quadratic = (assignments * (distances @ assignments)).sum(dim=0)
    within_ordered = (total_off_diagonal + quadratic) / 2.0
    cross_ordered = (total_off_diagonal - quadratic) / 2.0
    # cross_ordered contains both X->Y and Y->X distances.  This expression
    # is therefore 2 E||X-Y|| minus the two unbiased within-bank terms.
    return (
        cross_ordered / float(n * n)
        - within_ordered / float(n * (n - 1))
    )


def energy_distance_unbiased(x: torch.Tensor, y: torch.Tensor) -> float:
    """Unbiased finite-sample energy-distance statistic.

    Like an unbiased MMD estimate, this can be slightly negative under the
    equality null.  Inputs are assumed to already use a scientifically
    meaningful common coordinate scale.
    """

    if x.ndim != 2 or y.ndim != 2:
        raise ValueError("energy-distance inputs must be rank-2 tensors")
    if x.shape[1] != y.shape[1]:
        raise ValueError("energy-distance inputs must have the same feature dimension")
    if x.shape[0] < 2 or y.shape[0] < 2:
        raise ValueError("energy distance requires at least two samples per bank")
    cross = 2.0 * torch.cdist(x, y, p=2).mean()
    distance_x = torch.cdist(x, x, p=2)
    distance_y = torch.cdist(y, y, p=2)
    m, n = x.shape[0], y.shape[0]
    within_x = distance_x.sum() / float(m * (m - 1))
    within_y = distance_y.sum() / float(n * (n - 1))
    return float((cross - within_x - within_y).item())


def energy_distance_permutation_test(
    x: torch.Tensor,
    y: torch.Tensor,
    seed: int = 0,
    n_permutations: int = DEFAULT_HSIC_PERMUTATIONS,
) -> dict:
    """Balanced pooled-label permutation test using Euclidean energy distance."""

    if x.ndim != 2 or y.ndim != 2:
        raise ValueError("energy-distance inputs must be rank-2 tensors")
    if x.shape != y.shape:
        raise ValueError("permutation calibration requires equal-size, equal-dimension inputs")
    if n_permutations < 1:
        raise ValueError("n_permutations must be positive")
    n = x.shape[0]
    pooled = torch.cat([x, y], dim=0)
    values = _balanced_energy_values_from_distances(
        torch.cdist(pooled, pooled, p=2),
        n,
        seed=seed,
        n_permutations=n_permutations,
    )
    observed = float(values[0].item())
    null = values[1:].detach().cpu().numpy().astype(np.float64)
    exceedances = int(np.count_nonzero(null >= observed))
    null_std = float(null.std(ddof=1)) if null.size > 1 else None
    standardized_excess = None
    if null_std is not None and null_std > 0.0:
        standardized_excess = float((observed - null.mean()) / null_std)
    return {
        "statistic": observed,
        "p_value": float((exceedances + 1) / (n_permutations + 1)),
        "null_mean": float(null.mean()),
        "null_std": null_std,
        "standardized_excess": standardized_excess,
        "null_quantiles": {
            "q05": float(np.quantile(null, 0.05)),
            "q50": float(np.quantile(null, 0.50)),
            "q95": float(np.quantile(null, 0.95)),
            "q99": float(np.quantile(null, 0.99)),
        },
        "null_values": null.tolist(),
        "n_per_group": int(n),
        "n_permutations": int(n_permutations),
        "estimator": "unbiased energy distance with balanced pooled-label permutation null",
        "interpretation_warning": (
            "This tests exact distributional equality; failure to reject is not proof of equivalence."
        ),
    }


def global_mmd_to_standard_normal(
    z_s_sample: torch.Tensor,
    seed: int = 0,
    sigmas: Sequence[float] = EUCLIDEAN_SIGMAS,
) -> float:
    """Evaluation MMD²_U(q(z_s), N(0,I)) with a seeded reference draw."""

    generator = torch.Generator(device="cpu").manual_seed(seed)
    reference = torch.randn(
        z_s_sample.shape,
        generator=generator,
        dtype=z_s_sample.dtype,
        device="cpu",
    ).to(z_s_sample.device)
    return mmd2_unbiased(z_s_sample, reference, sigmas)


def conditional_mmd_to_standard_normal(
    z_s_sample: torch.Tensor,
    labels: torch.Tensor,
    n_classes: int,
    seed: int = 0,
    sigmas: Sequence[float] = EUCLIDEAN_SIGMAS,
) -> dict:
    """Per-class MMD²_U(q(z_s|y=k), N(0,I)) with independent references."""

    per_class: dict[str, float] = {}
    counts: dict[str, int] = {}
    for class_index in range(n_classes):
        class_values = z_s_sample[labels == class_index]
        count = int(class_values.shape[0])
        counts[str(class_index)] = count
        if count < 2:
            continue
        generator = torch.Generator(device="cpu").manual_seed(seed + class_index)
        reference = torch.randn(
            class_values.shape,
            generator=generator,
            dtype=class_values.dtype,
            device="cpu",
        ).to(class_values.device)
        per_class[str(class_index)] = mmd2_unbiased(class_values, reference, sigmas)
    if not per_class:
        raise ValueError("Conditional MMD requires a represented class with two samples")
    values = np.asarray(list(per_class.values()), dtype=np.float64)
    return {
        "mean_mmd2_u": float(values.mean()),
        "min_mmd2_u": float(values.min()),
        "max_mmd2_u": float(values.max()),
        "per_class_mmd2_u": per_class,
        "class_counts": counts,
        "reference_seed_base": int(seed),
        "sigmas": [float(value) for value in sigmas],
        "estimator": "mean per-class unbiased MMD^2 U-statistic to independent N(0,I) references",
    }


def delta_inter(z: torch.Tensor, labels: torch.Tensor, n_classes: int) -> float:
    """Mean pairwise Euclidean distance between class-conditional means."""

    means = [z[labels == k].mean(0) for k in range(n_classes) if int((labels == k).sum())]
    if len(means) < 2:
        raise ValueError("Delta_inter requires at least two represented classes")
    distances = [torch.linalg.vector_norm(a - b) for i, a in enumerate(means) for b in means[i + 1 :]]
    return float(torch.stack(distances).mean().item())


def stratified_probe_split(
    labels: torch.Tensor | np.ndarray,
    seed: int = 0,
    train_fraction: float = 0.60,
    validation_fraction: float = 0.20,
) -> ProbeSplit:
    """Create a deterministic 60/20/20 stratified split by default."""

    y = labels.detach().cpu().numpy() if isinstance(labels, torch.Tensor) else np.asarray(labels)
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be in (0, 1)")
    if not 0.0 < validation_fraction < 1.0 - train_fraction:
        raise ValueError("validation_fraction must leave a non-empty test split")

    indices = np.arange(y.shape[0])
    train_idx, remainder_idx = train_test_split(
        indices,
        train_size=train_fraction,
        random_state=seed,
        shuffle=True,
        stratify=y,
    )
    relative_validation = validation_fraction / (1.0 - train_fraction)
    val_idx, test_idx = train_test_split(
        remainder_idx,
        train_size=relative_validation,
        random_state=seed + 1,
        shuffle=True,
        stratify=y[remainder_idx],
    )
    return ProbeSplit(
        train=np.sort(train_idx),
        validation=np.sort(val_idx),
        test=np.sort(test_idx),
    )


def _fit_standardizer(x_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = x_train.mean(axis=0, keepdims=True)
    scale = x_train.std(axis=0, keepdims=True)
    scale[scale < 1e-8] = 1.0
    return mean.astype(np.float32), scale.astype(np.float32)


class StandardizedTorchProbe(nn.Module):
    """Torch probe carrying standardization fitted on the probe train split."""

    def __init__(self, mean: np.ndarray, scale: np.ndarray, network: nn.Module):
        super().__init__()
        self.register_buffer("feature_mean", torch.from_numpy(mean.squeeze(0)))
        self.register_buffer("feature_scale", torch.from_numpy(scale.squeeze(0)))
        self.network = network

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network((x - self.feature_mean) / self.feature_scale)


def _accuracy_and_f1(model, x: np.ndarray, y: np.ndarray) -> dict:
    predictions = model.predict(x)
    return {
        "accuracy": float(accuracy_score(y, predictions)),
        "macro_f1": float(f1_score(y, predictions, average="macro")),
    }


def _torch_probe_metrics(model: nn.Module, x: np.ndarray, y: np.ndarray) -> dict:
    with torch.no_grad():
        logits = model(torch.from_numpy(x).float())
        predictions = logits.argmax(1).cpu().numpy()
        cross_entropy = float(
            F.cross_entropy(logits, torch.from_numpy(y).long()).item()
        )
    return {
        "accuracy": float(accuracy_score(y, predictions)),
        "macro_f1": float(f1_score(y, predictions, average="macro")),
        "cross_entropy_nats": cross_entropy,
    }


def _train_torch_probe(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    n_classes: int,
    hidden_dims: tuple[int, ...],
    seed: int,
    epochs: int,
) -> nn.Module:
    """Fit a fixed-capacity probe with validation-only early stopping."""

    torch.manual_seed(seed)
    dims = (x_train.shape[1],) + hidden_dims + (n_classes,)
    layers: list[nn.Module] = []
    for index, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
        layers.append(nn.Linear(in_dim, out_dim))
        if index < len(dims) - 2:
            layers.append(nn.ReLU())
    model = nn.Sequential(*layers)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-4)
    xtr = torch.from_numpy(x_train).float()
    ytr = torch.from_numpy(y_train).long()
    xva = torch.from_numpy(x_validation).float()
    yva = torch.from_numpy(y_validation).long()
    best_state = None
    best_loss = math.inf
    stale = 0
    patience = 30
    for _ in range(epochs):
        model.train()
        loss = F.cross_entropy(model(xtr), ytr)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            validation_loss = float(F.cross_entropy(model(xva), yva).item())
        if validation_loss < best_loss - 1e-6:
            best_loss = validation_loss
            best_state = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model


def run_probe_suite(
    features: torch.Tensor,
    labels: torch.Tensor,
    seed: int = 0,
    probe_names: Iterable[str] = ("logistic", "mlp", "rbf_svm", "knn"),
    epochs: int = 300,
) -> dict:
    """Run fixed probes using one stratified split and train-only scaling."""

    x = features.detach().cpu().float().numpy()
    y = labels.detach().cpu().long().numpy()
    split = stratified_probe_split(y, seed=seed)
    mean, scale = _fit_standardizer(x[split.train])
    x_scaled = ((x - mean) / scale).astype(np.float32)
    n_classes = int(y.max()) + 1
    test_counts = np.bincount(y[split.test], minlength=n_classes).astype(np.float64)
    test_probabilities = test_counts[test_counts > 0] / test_counts.sum()
    test_label_entropy = float(-(test_probabilities * np.log(test_probabilities)).sum())
    results: dict[str, dict] = {}

    for name in probe_names:
        if name == "logistic":
            model = _train_torch_probe(
                x_scaled[split.train], y[split.train],
                x_scaled[split.validation], y[split.validation],
                n_classes=n_classes, hidden_dims=(), seed=seed, epochs=epochs,
            )
            metric = lambda idx: _torch_probe_metrics(model, x_scaled[idx], y[idx])
        elif name == "mlp":
            model = _train_torch_probe(
                x_scaled[split.train], y[split.train],
                x_scaled[split.validation], y[split.validation],
                n_classes=n_classes, hidden_dims=(128, 64), seed=seed, epochs=epochs,
            )
            metric = lambda idx: _torch_probe_metrics(model, x_scaled[idx], y[idx])
        elif name == "rbf_svm":
            model = SVC(C=1.0, kernel="rbf", gamma="scale", random_state=seed)
            model.fit(x_scaled[split.train], y[split.train])
            metric = lambda idx: _accuracy_and_f1(model, x_scaled[idx], y[idx])
        elif name == "knn":
            model = KNeighborsClassifier(n_neighbors=5, weights="distance")
            model.fit(x_scaled[split.train], y[split.train])
            metric = lambda idx: _accuracy_and_f1(model, x_scaled[idx], y[idx])
        else:
            raise ValueError(f"Unknown probe '{name}'")
        model_result = {
            "train": metric(split.train),
            "validation": metric(split.validation),
            "test": metric(split.test),
        }
        test_cross_entropy = model_result["test"].get("cross_entropy_nats")
        if test_cross_entropy is not None:
            model_result["test"]["information_lower_bound_nats"] = float(
                test_label_entropy - test_cross_entropy
            )
        results[name] = model_result

    return {
        "feature_standardization": "mean/std fitted on train split only",
        "test_label_entropy_nats": test_label_entropy,
        "split": split.metadata(),
        "models": results,
    }


def fit_logistic_probe(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    validation_features: torch.Tensor,
    validation_labels: torch.Tensor,
    seed: int = 0,
    epochs: int = 300,
) -> tuple[float, StandardizedTorchProbe]:
    """Compatibility helper returning a standardized callable Torch probe."""

    xtr = train_features.detach().cpu().float().numpy()
    ytr = train_labels.detach().cpu().long().numpy()
    xva = validation_features.detach().cpu().float().numpy()
    yva = validation_labels.detach().cpu().long().numpy()
    mean, scale = _fit_standardizer(xtr)
    network = _train_torch_probe(
        ((xtr - mean) / scale).astype(np.float32), ytr,
        ((xva - mean) / scale).astype(np.float32), yva,
        n_classes=int(np.concatenate([ytr, yva]).max()) + 1,
        hidden_dims=(), seed=seed, epochs=epochs,
    )
    wrapped = StandardizedTorchProbe(mean, scale, network)
    validation_metrics = _torch_probe_metrics(wrapped, xva, yva)
    return validation_metrics["accuracy"], wrapped


def _center_kernel(kernel: torch.Tensor) -> torch.Tensor:
    return kernel - kernel.mean(0, keepdim=True) - kernel.mean(1, keepdim=True) + kernel.mean()


def multiscale_hsic_statistic(
    features: torch.Tensor,
    labels: torch.Tensor,
    sigmas: Sequence[float] = EUCLIDEAN_SIGMAS,
) -> tuple[float, torch.Tensor, torch.Tensor]:
    """Biased centered HSIC statistic; significance comes from permutation."""

    if features.shape[0] != labels.shape[0] or features.shape[0] < 4:
        raise ValueError("HSIC requires matching inputs with at least four samples")
    k = _center_kernel(multiscale_rbf_kernel(features, features, sigmas))
    label_kernel = (labels[:, None] == labels[None, :]).to(features.dtype)
    l = _center_kernel(label_kernel)
    statistic = (k * l).sum() / float((features.shape[0] - 1) ** 2)
    return float(statistic.item()), k, l


def hsic_permutation_test(
    features: torch.Tensor,
    labels: torch.Tensor,
    seed: int = 0,
    n_permutations: int = DEFAULT_HSIC_PERMUTATIONS,
    sigmas: Sequence[float] = EUCLIDEAN_SIGMAS,
    permutation_batch_size: int = 32,
) -> dict:
    """Multi-scale HSIC with a plus-one calibrated permutation p-value.

    The null is evaluated in batches without materializing one ``n x n``
    label-kernel matrix per permutation.  For centered ``K``, the Frobenius
    product with a centered delta-label kernel equals the product with the
    uncentered kernel.  Writing that kernel as ``S S^T`` for a one-hot class
    assignment ``S`` gives ``trace(S^T K S)``.  One matrix multiplication can
    therefore evaluate several permutations and classes at once.  This is
    algebraically equivalent to explicitly permuting the centered label
    kernel, but makes paper-scale CPU calibration practical.
    """

    if n_permutations < 1:
        raise ValueError("n_permutations must be positive")
    if permutation_batch_size < 1:
        raise ValueError("permutation_batch_size must be positive")

    observed, k_centered, _ = multiscale_hsic_statistic(features, labels, sigmas)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    null = []
    n = features.shape[0]
    n_classes = int(labels.max().item()) + 1
    permutations = [torch.randperm(n, generator=generator) for _ in range(n_permutations)]
    denominator = float((n - 1) ** 2)
    for start in range(0, n_permutations, permutation_batch_size):
        batch_indices = permutations[start : start + permutation_batch_size]
        permuted_labels = torch.stack(
            [labels[index.to(labels.device)] for index in batch_indices], dim=0
        )
        assignments = F.one_hot(permuted_labels.long(), num_classes=n_classes).to(
            device=k_centered.device,
            dtype=k_centered.dtype,
        )
        batch_size = assignments.shape[0]
        flat_assignments = assignments.permute(1, 0, 2).reshape(
            n, batch_size * n_classes
        )
        products = k_centered @ flat_assignments
        batch_statistics = (flat_assignments * products).sum(0).reshape(
            batch_size, n_classes
        ).sum(1) / denominator
        null.extend(batch_statistics.detach().cpu().double().tolist())
    null_array = np.asarray(null, dtype=np.float64)
    exceedances = int(np.count_nonzero(null_array >= observed))
    return {
        "statistic": observed,
        "p_value": float((exceedances + 1) / (n_permutations + 1)),
        "null_mean": float(null_array.mean()) if null_array.size else None,
        "null_std": float(null_array.std(ddof=1)) if null_array.size > 1 else None,
        "null_quantiles": {
            "q90": float(np.quantile(null_array, 0.90)) if null_array.size else None,
            "q95": float(np.quantile(null_array, 0.95)) if null_array.size else None,
            "q99": float(np.quantile(null_array, 0.99)) if null_array.size else None,
        },
        "n_samples": int(n),
        "n_permutations": int(n_permutations),
        "permutation_batch_size": int(permutation_batch_size),
        "sigmas": [float(value) for value in sigmas],
        "estimator": "biased centered HSIC with permutation calibration",
    }


def _spherical_multiscale_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = F.normalize(x, p=2, dim=1)
    y = F.normalize(y, p=2, dim=1)
    chordal_sq = (2.0 - 2.0 * (x @ y.T)).clamp_min(0.0)
    result = torch.zeros_like(chordal_sq)
    for sigma in SPHERICAL_SIGMAS:
        result.add_(torch.exp(-chordal_sq / (2.0 * sigma**2 + 1e-12)))
    return result / len(SPHERICAL_SIGMAS)


def _unbiased_mmd_from_kernels(k_xx: torch.Tensor, k_yy: torch.Tensor, k_xy: torch.Tensor) -> float:
    m, n = k_xx.shape[0], k_yy.shape[0]
    xx = (k_xx.sum() - k_xx.diagonal().sum()) / (m * (m - 1))
    yy = (k_yy.sum() - k_yy.diagonal().sum()) / (n * (n - 1))
    return float((xx + yy - 2.0 * k_xy.mean()).item())


def joint_mmd_unbiased(
    mu_c: torch.Tensor,
    mu_s: torch.Tensor,
    labels: torch.Tensor,
    n_classes: int,
    seed: int = 0,
) -> float:
    """Classwise U-statistic JointMMD between joint and permuted product proxy."""

    generator = torch.Generator(device="cpu").manual_seed(seed)
    values = []
    for class_index in range(n_classes):
        mask = labels == class_index
        if int(mask.sum()) < 4:
            continue
        zc = F.normalize(mu_c[mask], p=2, dim=1)
        zs = mu_s[mask]
        zs = (zs - zs.mean(0, keepdim=True)) / zs.std(0, keepdim=True).clamp_min(1e-6)
        permutation = torch.randperm(zs.shape[0], generator=generator).to(zs.device)
        zs_permuted = zs[permutation]

        kc = _spherical_multiscale_kernel(zc, zc)
        ks_joint = multiscale_rbf_kernel(zs, zs)
        ks_product = multiscale_rbf_kernel(zs_permuted, zs_permuted)
        ks_cross = multiscale_rbf_kernel(zs, zs_permuted)
        values.append(_unbiased_mmd_from_kernels(kc * ks_joint, kc * ks_product, kc * ks_cross))
    if not values:
        raise ValueError("JointMMD requires at least one class with four samples")
    return float(np.mean(values))


def classwise_conditional_hsic_permutation_test(
    mu_c: torch.Tensor,
    mu_s: torch.Tensor,
    labels: torch.Tensor,
    n_classes: int,
    seed: int = 0,
    n_permutations: int = DEFAULT_HSIC_PERMUTATIONS,
) -> dict:
    """Classwise HSIC test of mu_c independence from mu_s conditional on y.

    The reported statistic is the unweighted mean over represented classes.
    Its permutation null shuffles style codes independently within each
    class, preserving both class-conditional marginals.
    """

    blocks = []
    per_class = {}
    for class_index in range(n_classes):
        mask = labels == class_index
        if int(mask.sum()) < 4:
            continue
        zc = F.normalize(mu_c[mask], p=2, dim=1)
        zs = mu_s[mask]
        zs = (zs - zs.mean(0, keepdim=True)) / zs.std(0, keepdim=True).clamp_min(1e-6)
        kc = _center_kernel(_spherical_multiscale_kernel(zc, zc))
        ks = _center_kernel(multiscale_rbf_kernel(zs, zs))
        denominator = float((zc.shape[0] - 1) ** 2)
        statistic = float(((kc * ks).sum() / denominator).item())
        blocks.append((kc, ks, denominator))
        per_class[str(class_index)] = statistic
    if not blocks:
        raise ValueError("Conditional HSIC requires a represented class with four samples")

    observed = float(np.mean(list(per_class.values())))
    generator = torch.Generator(device="cpu").manual_seed(seed)
    null_values = []
    for _ in range(n_permutations):
        class_null = []
        for kc, ks, denominator in blocks:
            permutation = torch.randperm(kc.shape[0], generator=generator).to(kc.device)
            ks_permuted = ks[permutation][:, permutation]
            class_null.append(float(((kc * ks_permuted).sum() / denominator).item()))
        null_values.append(float(np.mean(class_null)))
    null = np.asarray(null_values, dtype=np.float64)
    exceedances = int(np.count_nonzero(null >= observed))
    return {
        "statistic": observed,
        "per_class_statistic": per_class,
        "p_value": float((exceedances + 1) / (n_permutations + 1)),
        "null_mean": float(null.mean()) if null.size else None,
        "null_std": float(null.std(ddof=1)) if null.size > 1 else None,
        "null_quantiles": {
            "q90": float(np.quantile(null, 0.90)) if null.size else None,
            "q95": float(np.quantile(null, 0.95)) if null.size else None,
            "q99": float(np.quantile(null, 0.99)) if null.size else None,
        },
        "n_permutations": int(n_permutations),
        "estimator": "mean classwise centered HSIC; within-class permutation calibration",
    }


def _double_center_distances(values: torch.Tensor) -> torch.Tensor:
    if values.ndim != 2 or values.shape[0] < 4:
        raise ValueError("distance correlation requires a rank-2 tensor with four samples")
    distances = torch.cdist(values, values, p=2)
    return (
        distances
        - distances.mean(dim=0, keepdim=True)
        - distances.mean(dim=1, keepdim=True)
        + distances.mean()
    )


def _distance_correlation_from_centered(
    centered_x: torch.Tensor,
    centered_y: torch.Tensor,
) -> torch.Tensor:
    covariance_squared = (centered_x * centered_y).mean()
    variance_x_squared = centered_x.square().mean()
    variance_y_squared = centered_y.square().mean()
    denominator = torch.sqrt(variance_x_squared * variance_y_squared).clamp_min(1e-12)
    return covariance_squared / denominator


def distance_correlation_statistic(x: torch.Tensor, y: torch.Tensor) -> float:
    """Biased sample distance correlation for two multivariate variables."""

    if x.shape[0] != y.shape[0]:
        raise ValueError("distance-correlation inputs must contain the same number of samples")
    statistic = _distance_correlation_from_centered(
        _double_center_distances(x),
        _double_center_distances(y),
    )
    return float(statistic.item())


def classwise_conditional_distance_correlation_permutation_test(
    mu_c: torch.Tensor,
    mu_s: torch.Tensor,
    labels: torch.Tensor,
    n_classes: int,
    seed: int = 0,
    n_permutations: int = DEFAULT_HSIC_PERMUTATIONS,
) -> dict:
    """Kernel-free within-class test of content/style dependence.

    Content is projected to the unit sphere and style dimensions are
    standardized within class before Euclidean distances are computed.  The
    unweighted class mean mirrors the conditional-HSIC estimand, while the
    null independently permutes style observations inside every class.
    """

    if mu_c.shape[0] != mu_s.shape[0] or mu_c.shape[0] != labels.shape[0]:
        raise ValueError("content, style, and labels must contain the same samples")
    if n_permutations < 1:
        raise ValueError("n_permutations must be positive")
    blocks = []
    per_class = {}
    for class_index in range(n_classes):
        mask = labels == class_index
        if int(mask.sum()) < 4:
            continue
        zc = F.normalize(mu_c[mask], p=2, dim=1)
        zs = mu_s[mask]
        zs = (zs - zs.mean(0, keepdim=True)) / zs.std(
            0, keepdim=True, unbiased=False
        ).clamp_min(1e-6)
        centered_content = _double_center_distances(zc)
        centered_style = _double_center_distances(zs)
        statistic = float(
            _distance_correlation_from_centered(
                centered_content, centered_style
            ).item()
        )
        blocks.append((centered_content, centered_style))
        per_class[str(class_index)] = statistic
    if not blocks:
        raise ValueError("conditional distance correlation requires a class with four samples")

    observed = float(np.mean(list(per_class.values())))
    generator = torch.Generator(device="cpu").manual_seed(seed)
    null_values = []
    for _ in range(n_permutations):
        class_null = []
        for centered_content, centered_style in blocks:
            permutation = torch.randperm(
                centered_content.shape[0], generator=generator
            ).to(centered_content.device)
            permuted_style = centered_style[permutation][:, permutation]
            class_null.append(
                float(
                    _distance_correlation_from_centered(
                        centered_content, permuted_style
                    ).item()
                )
            )
        null_values.append(float(np.mean(class_null)))
    null = np.asarray(null_values, dtype=np.float64)
    exceedances = int(np.count_nonzero(null >= observed))
    null_std = float(null.std(ddof=1)) if null.size > 1 else None
    standardized_excess = None
    if null_std is not None and null_std > 0.0:
        standardized_excess = float((observed - null.mean()) / null_std)
    return {
        "statistic": observed,
        "per_class_statistic": per_class,
        "p_value": float((exceedances + 1) / (n_permutations + 1)),
        "null_mean": float(null.mean()),
        "null_std": null_std,
        "standardized_excess": standardized_excess,
        "null_quantiles": {
            "q90": float(np.quantile(null, 0.90)),
            "q95": float(np.quantile(null, 0.95)),
            "q99": float(np.quantile(null, 0.99)),
        },
        "null_values": null.tolist(),
        "n_permutations": int(n_permutations),
        "estimator": (
            "mean classwise biased distance correlation; within-class permutation calibration"
        ),
        "geometry": (
            "chordal distance on unit-normalized content; Euclidean distance on "
            "within-class standardized style"
        ),
    }

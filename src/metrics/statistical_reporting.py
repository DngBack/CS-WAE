"""Multiplicity-aware helpers for derived paper tables.

These functions do not alter any frozen experiment artifact.  They are used
only when deriving a presentation layer from domain-level test results.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    """Return Holm step-down family-wise adjusted p-values in input order."""

    values = np.asarray(p_values, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("Holm correction requires a non-empty one-dimensional list")
    if np.any(~np.isfinite(values)) or np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("p-values must be finite and lie in [0,1]")
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    adjusted_sorted = np.maximum.accumulate(
        (values.size - np.arange(values.size)) * sorted_values
    ).clip(max=1.0)
    adjusted = np.empty_like(adjusted_sorted)
    adjusted[order] = adjusted_sorted
    return adjusted.tolist()


def holm_family(p_values: Sequence[float], alpha: float = 0.05) -> dict:
    """Serialize raw/adjusted p-values and rejection decisions."""

    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0,1)")
    raw = [float(value) for value in p_values]
    adjusted = holm_adjust(raw)
    return {
        "method": "Holm step-down family-wise error control",
        "alpha": float(alpha),
        "raw_p_values": raw,
        "adjusted_p_values": adjusted,
        "reject": [bool(value <= alpha) for value in adjusted],
        "n_tests": len(raw),
    }


__all__ = ["holm_adjust", "holm_family"]

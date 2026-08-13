"""Independent multi-head pixel evaluator for Shapes3D factors."""

from __future__ import annotations

import torch
import torch.nn as nn

from ..datasets.shapes3d import FACTOR_CARDINALITIES, FACTOR_NAMES
from .backbone import ResNet18Body


class Shapes3DFactorEvaluator(nn.Module):
    """Predict every ground-truth factor from a real or decoded RGB image."""

    def __init__(self, hidden_dim: int = 256):
        super().__init__()
        self.features = ResNet18Body(in_channels=3)
        flat = self.features.out_channels * self.features.spatial**2
        self.shared = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
        )
        self.heads = nn.ModuleDict(
            {
                name: nn.Linear(hidden_dim, cardinality)
                for name, cardinality in zip(FACTOR_NAMES, FACTOR_CARDINALITIES)
            }
        )

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.shared(self.features(images))
        return {name: head(features) for name, head in self.heads.items()}


__all__ = ["Shapes3DFactorEvaluator"]

"""Independent pixel-space evaluator for the common Rotated-MNIST audit."""

from __future__ import annotations

import torch
import torch.nn as nn


class RotatedMNISTClassifier(nn.Module):
    def __init__(self, n_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.classifier = nn.Linear(128, n_classes)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(images))


__all__ = ["RotatedMNISTClassifier"]

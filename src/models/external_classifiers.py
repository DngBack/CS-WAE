"""Independent pixel-space classifiers used only for generative evaluation."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


class GrayscaleExternalCNN(nn.Module):
    """Four-convolution evaluator for MNIST and Fashion-MNIST."""

    def __init__(self, n_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(128 * 4 * 4, 256), nn.ReLU(), nn.Linear(256, n_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] == 3:
            x = x.mean(dim=1, keepdim=True)
        return self.classifier(self.features(x))


class _WideBasicBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int, dropout: float):
        super().__init__()
        self.equal_channels = in_channels == out_channels
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.dropout = dropout
        self.shortcut = None if self.equal_channels and stride == 1 else nn.Conv2d(
            in_channels, out_channels, 1, stride=stride, bias=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        activated = F.relu(self.bn1(x), inplace=False)
        residual = x if self.shortcut is None else self.shortcut(activated)
        out = self.conv1(activated)
        out = self.conv2(F.dropout(F.relu(self.bn2(out), inplace=False), self.dropout, self.training))
        return residual + out


class _WideNetworkBlock(nn.Module):
    def __init__(self, count: int, in_channels: int, out_channels: int, stride: int, dropout: float):
        super().__init__()
        blocks = []
        for index in range(count):
            blocks.append(
                _WideBasicBlock(
                    in_channels if index == 0 else out_channels,
                    out_channels,
                    stride if index == 0 else 1,
                    dropout,
                )
            )
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)


class WideResNet(nn.Module):
    """WideResNet-28-10 for the CIFAR-10 external evaluator."""

    def __init__(self, depth: int = 28, widen_factor: int = 10, n_classes: int = 10, dropout: float = 0.0):
        super().__init__()
        if (depth - 4) % 6 != 0:
            raise ValueError("WideResNet depth must satisfy (depth - 4) % 6 == 0")
        blocks_per_group = (depth - 4) // 6
        channels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        self.conv1 = nn.Conv2d(3, channels[0], 3, padding=1, bias=False)
        self.group1 = _WideNetworkBlock(blocks_per_group, channels[0], channels[1], 1, dropout)
        self.group2 = _WideNetworkBlock(blocks_per_group, channels[1], channels[2], 2, dropout)
        self.group3 = _WideNetworkBlock(blocks_per_group, channels[2], channels[3], 2, dropout)
        self.bn = nn.BatchNorm2d(channels[3])
        self.fc = nn.Linear(channels[3], n_classes)
        self.register_buffer("input_mean", torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1))
        self.register_buffer("input_std", torch.tensor([0.2470, 0.2435, 0.2616]).view(1, 3, 1, 1))
        self._initialize()

    def _initialize(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        x = (x - self.input_mean) / self.input_std
        out = self.group3(self.group2(self.group1(self.conv1(x))))
        out = F.relu(self.bn(out), inplace=False)
        out = F.adaptive_avg_pool2d(out, 1).flatten(1)
        return self.fc(out)


def external_classifier_name(dataset: str) -> str:
    return "wide_resnet_28_10" if dataset == "cifar10" else "grayscale_cnn_4conv"


def build_external_classifier(dataset: str, n_classes: int = 10) -> nn.Module:
    if dataset in {"mnist", "fashion_mnist"}:
        return GrayscaleExternalCNN(n_classes=n_classes)
    if dataset == "cifar10":
        return WideResNet(depth=28, widen_factor=10, n_classes=n_classes)
    raise ValueError(f"No external classifier protocol for dataset '{dataset}'")


def load_external_classifier_checkpoint(
    checkpoint_path: str | Path,
    dataset: str,
    device: torch.device,
) -> tuple[nn.Module, dict]:
    """Load a self-describing external evaluator checkpoint."""

    payload = torch.load(checkpoint_path, map_location=device)
    if not isinstance(payload, dict) or "model_state_dict" not in payload:
        raise ValueError("External classifier checkpoint must be a Stage-0 checkpoint payload")
    checkpoint_dataset = payload.get("dataset")
    if checkpoint_dataset != dataset:
        raise ValueError(f"Classifier dataset is '{checkpoint_dataset}', requested '{dataset}'")
    expected_name = external_classifier_name(dataset)
    if payload.get("architecture") != expected_name:
        raise ValueError(
            f"Classifier architecture is '{payload.get('architecture')}', expected '{expected_name}'"
        )
    model = build_external_classifier(dataset, int(payload.get("n_classes", 10))).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model, {key: value for key, value in payload.items() if key != "model_state_dict"}

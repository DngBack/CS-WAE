"""EMNIST Letters dataset loading utilities (26 classes, A-Z)."""

import os

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from ..config import config


def _emnist_orientation_fix(img):
    """Standard EMNIST orientation correction (rotate + flip)."""
    img = transforms.functional.rotate(img, -90)
    return transforms.functional.hflip(img)


class _RemapLabels(Dataset):
    """Remap EMNIST letter labels from 1-26 to 0-25."""

    def __init__(self, base: Dataset, offset: int = -1):
        self.base = base
        self.offset = offset

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        x, y = self.base[idx]
        return x, y + self.offset


def get_emnist_letters_loaders(data_dir="./data", batch_size=None, num_workers=None, seed=None):
    """Create EMNIST Letters train and test data loaders."""
    batch_size = batch_size or config.batch_size
    num_workers = num_workers or (2 if os.name == "nt" else 4)

    generator = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)

    transform = transforms.Compose(
        [
            transforms.Lambda(_emnist_orientation_fix),
            transforms.ToTensor(),
        ]
    )

    train_dataset = _RemapLabels(
        datasets.EMNIST(
            data_dir,
            split="letters",
            train=True,
            download=True,
            transform=transform,
        )
    )
    test_dataset = _RemapLabels(
        datasets.EMNIST(
            data_dir,
            split="letters",
            train=False,
            download=True,
            transform=transform,
        )
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        pin_memory=True,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=num_workers,
    )

    return train_loader, test_loader


def get_data_info():
    """Get information about the EMNIST Letters dataset."""
    return {
        "name": "EMNIST-Letters",
        "input_shape": (1, 28, 28),
        "n_classes": 26,
        "num_classes": 26,
        "train_size": 124800,
        "test_size": 20800,
    }

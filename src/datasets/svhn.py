"""SVHN dataset loading utilities (32×32 RGB street-view digits)."""

import os

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from ..config import config


def get_svhn_loaders(data_dir="./data", batch_size=None, num_workers=None, seed=None):
    """Create SVHN train and test data loaders."""
    batch_size = batch_size or config.batch_size
    num_workers = num_workers or (2 if os.name == "nt" else 4)

    generator = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)

    transform = transforms.Compose([transforms.ToTensor()])

    train_dataset = datasets.SVHN(
        data_dir, split="train", download=True, transform=transform
    )
    test_dataset = datasets.SVHN(
        data_dir, split="test", download=True, transform=transform
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
    return {
        "name": "SVHN",
        "input_shape": (3, 32, 32),
        "in_channels": 3,
        "image_size": 32,
        "n_classes": 10,
        "num_classes": 10,
        "train_size": 73257,
        "test_size": 26032,
        "color": True,
    }

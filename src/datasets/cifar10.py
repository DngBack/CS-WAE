"""CIFAR-10 dataset loading utilities (32×32 RGB)."""

import os

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from ..config import config


def get_cifar10_loaders(data_dir="./data", batch_size=None, num_workers=None, seed=None):
    """Create CIFAR-10 train and test data loaders."""
    batch_size = batch_size or config.batch_size
    num_workers = num_workers or (2 if os.name == "nt" else 4)

    generator = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)

    transform = transforms.Compose([transforms.ToTensor()])

    train_dataset = datasets.CIFAR10(
        data_dir, train=True, download=True, transform=transform
    )
    test_dataset = datasets.CIFAR10(
        data_dir, train=False, download=True, transform=transform
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
        "name": "CIFAR-10",
        "input_shape": (3, 32, 32),
        "in_channels": 3,
        "image_size": 32,
        "n_classes": 10,
        "num_classes": 10,
        "train_size": 50000,
        "test_size": 10000,
        "color": True,
    }

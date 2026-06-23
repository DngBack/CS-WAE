"""Sync global config with the active dataset."""

from ..config import config
from ..config_ablation import ablation_config


def apply_dataset_config(dataset: str, get_dataset_info) -> dict:
    """
    Update config / ablation_config from dataset metadata.

    Sets n_classes, in_channels, image_size (grayscale 28² vs RGB 32²).
    """
    info = get_dataset_info(dataset)
    config.n_classes = info["n_classes"]
    config.in_channels = info.get("in_channels", 1)
    config.image_size = info.get("image_size", 28)
    ablation_config.n_classes = info["n_classes"]
    return info

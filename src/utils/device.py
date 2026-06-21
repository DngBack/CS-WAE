"""Runtime device selection for training scripts."""

import torch

from src.config import config
from src.config_ablation import ablation_config


def set_device(device: str | None = None) -> torch.device:
    """Set global device on main and ablation configs."""
    if device is None:
        resolved = config.device
    else:
        resolved = torch.device(device)
        config.device = resolved
        ablation_config.device = resolved
    return resolved

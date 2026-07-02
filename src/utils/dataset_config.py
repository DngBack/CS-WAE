"""Sync global config with the active dataset."""

from ..config import config
from ..config_ablation import ablation_config
from ..models.backbone import normalize_backbone


def apply_dataset_config(dataset, get_dataset_info=None, backbone: str | None = None) -> dict:
    """
    Update config / ablation_config from dataset metadata.

    Supports both call styles:
    - apply_dataset_config("mnist", get_dataset_info, backbone=...)
    - apply_dataset_config(cfg_object, "mnist")
    """
    from ..datasets.loaders import get_dataset_info as default_get_dataset_info

    config_target = None
    if isinstance(dataset, str):
        dataset_name = dataset
        info_getter = default_get_dataset_info if get_dataset_info is None or not callable(get_dataset_info) else get_dataset_info
    else:
        config_target = dataset
        if isinstance(get_dataset_info, str):
            dataset_name = get_dataset_info
            info_getter = default_get_dataset_info
        else:
            raise TypeError("apply_dataset_config requires a dataset name when the first argument is a config object")

    info = info_getter(dataset_name)

    # Update shared global config
    config.n_classes = info["n_classes"]
    config.in_channels = info.get("in_channels", 1)
    config.image_size = info.get("image_size", 28)
    config.backbone = normalize_backbone(backbone or info.get("backbone", "cnn"))

    # Update a passed-in config-like object when provided
    if config_target is not None:
        if hasattr(config_target, "n_classes"):
            config_target.n_classes = info["n_classes"]
        if hasattr(config_target, "in_channels"):
            config_target.in_channels = info.get("in_channels", 1)
        if hasattr(config_target, "image_size"):
            config_target.image_size = info.get("image_size", 28)
        if hasattr(config_target, "backbone"):
            config_target.backbone = normalize_backbone(backbone or info.get("backbone", "cnn"))

    ablation_config.n_classes = info["n_classes"]
    return info

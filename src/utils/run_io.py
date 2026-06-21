"""Save run configuration and metrics to disk."""

import json
from datetime import datetime
from pathlib import Path

import torch


def save_json(path: str | Path, data: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(data, f, indent=2)


def config_to_dict(config_obj) -> dict:
    """Convert a Config instance to a plain dictionary."""
    result = {}
    for key, value in vars(config_obj).items():
        if isinstance(value, torch.device):
            result[key] = str(value)
        else:
            result[key] = value
    return result


def save_run_metadata(output_dir: str | Path, config_dict: dict, seed: int, extra: dict | None = None) -> None:
    """Persist run metadata for reproducibility."""
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "seed": seed,
        "config": config_dict,
    }
    if extra:
        metadata.update(extra)
    save_json(Path(output_dir) / "run_config.json", metadata)


def save_metrics(output_dir: str | Path, metrics: dict) -> None:
    save_json(Path(output_dir) / "metrics.json", metrics)

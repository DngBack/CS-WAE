#!/usr/bin/env python3
"""Evaluate held-out face reconstruction without an external face model."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.export_face_contract_latents import _load_model  # noqa: E402
from src.datasets.face_conditions import load_celebahq_populations, load_utkface_populations  # noqa: E402
from src.utils.provenance import sha256_file  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("utkface", "celebahq"), required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--metadata-csv")
    parser.add_argument("--condition-column", default="Smiling")
    parser.add_argument("--image-size", type=int, choices=(64, 128, 256), default=64)
    parser.add_argument("--max-samples", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def _summary(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if array.size > 1 else None,
        "standard_error": float(array.std(ddof=1) / np.sqrt(array.size)) if array.size > 1 else None,
    }


@torch.no_grad()
def main() -> None:
    args = parse_args()
    if args.max_samples < 1 or args.batch_size < 1:
        raise ValueError("max-samples and batch-size must be positive")
    device = torch.device(args.device)
    checkpoint = Path(args.checkpoint)
    model = _load_model(checkpoint, args.image_size, device)
    if args.dataset == "utkface":
        populations, dataset_metadata = load_utkface_populations(
            data_dir=args.data_dir, image_size=args.image_size
        )
    else:
        populations, dataset_metadata = load_celebahq_populations(
            data_dir=args.data_dir,
            metadata_csv=args.metadata_csv,
            condition_column=args.condition_column,
            image_size=args.image_size,
        )
    test = populations["test"]
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    indices = torch.randperm(len(test), generator=generator)[: min(len(test), args.max_samples)].tolist()
    loader = DataLoader(
        Subset(test, indices), batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory=device.type == "cuda",
    )
    center_l1, center_mse, sampled_l1, sampled_mse = [], [], [], []
    cuda_devices = [device.index or 0] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(100_000 + args.seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(100_000 + args.seed)
        for images, *_ in loader:
            images = images.to(device, non_blocking=True)
            mu_c, _rho_c, mu_s, _logvar_s = model.encode(images)
            center = model.decoder(torch.cat([mu_c, mu_s], dim=1))
            sampled = model(images)[0]
            for reconstruction, l1_bank, mse_bank in [
                (center, center_l1, center_mse), (sampled, sampled_l1, sampled_mse)
            ]:
                difference = reconstruction - images
                l1_bank.extend(difference.abs().flatten(1).mean(1).cpu().tolist())
                mse_bank.extend(difference.square().flatten(1).mean(1).cpu().tolist())
    center_mse_summary = _summary(center_mse)
    sampled_mse_summary = _summary(sampled_mse)
    result = {
        "schema_version": "face-reconstruction-eval-1.0.0",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": args.dataset,
        "dataset_metadata": dataset_metadata,
        "split": "test",
        "ordered_subset_indices": indices,
        "n_samples": len(indices),
        "seed": args.seed,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint),
        "posterior_center_reconstruction": {
            "l1_per_pixel": _summary(center_l1),
            "mse_per_pixel": center_mse_summary,
            "psnr_db_from_mean_mse": float(-10.0 * np.log10(max(center_mse_summary["mean"], 1e-12))),
        },
        "fixed_one_draw_posterior_reconstruction": {
            "l1_per_pixel": _summary(sampled_l1),
            "mse_per_pixel": sampled_mse_summary,
            "psnr_db_from_mean_mse": float(-10.0 * np.log10(max(sampled_mse_summary["mean"], 1e-12))),
        },
        "interpretation": (
            "Pixel-space fidelity only; these metrics do not measure identity, age retention, "
            "perceptual realism, or disentanglement."
        ),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True))
    temporary.replace(output)
    print(json.dumps({"output": str(output), "n_samples": len(indices)}, indent=2))


if __name__ == "__main__":
    main()

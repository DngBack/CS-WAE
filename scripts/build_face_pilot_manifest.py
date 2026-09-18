#!/usr/bin/env python3
"""Validate and checksum a completed two-arm face pilot directory."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", default="MANIFEST.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root).resolve()
    arms = [root / "control_seed0", root / "fact_transfer_seed0"]
    required = [
        "run_config.json", "training_history.json", "training_checkpoint.pt",
        "f_cs_wae_model.pth", "test_latents.npz", "test_latents.json",
        "test_reconstruction.json", "test_audit.json", "test_audit.json.sha256",
        "continuous_age_grid.png", "continuous_age_grid.json",
    ]
    validation = {}
    ordered_indices = []
    for arm in arms:
        missing = [name for name in required if not (arm / name).is_file()]
        if missing:
            raise FileNotFoundError(f"{arm.name} missing {missing}")
        history = json.loads((arm / "training_history.json").read_text())
        latent_metadata = json.loads((arm / "test_latents.json").read_text())
        audit = json.loads((arm / "test_audit.json").read_text())
        with np.load(arm / "test_latents.npz", allow_pickle=False) as archive:
            shapes = {name: list(archive[name].shape) for name in archive.files}
        latent_sha = _sha256(arm / "test_latents.npz")
        if latent_sha != latent_metadata["latent_archive"]["sha256"]:
            raise ValueError(f"latent checksum mismatch in {arm.name}")
        if audit["source"]["latent_archive_sha256"] != latent_sha:
            raise ValueError(f"audit source checksum mismatch in {arm.name}")
        indices = latent_metadata["ordered_subset_indices"]
        ordered_indices.append(indices)
        validation[arm.name] = {
            "training_epochs": len(history),
            "latent_shapes": shapes,
            "latent_n_samples": latent_metadata["n_samples"],
            "audit_n_samples": audit["n_samples"],
            "latent_chain_verified": True,
        }
    if ordered_indices[0] != ordered_indices[1]:
        raise ValueError("arms do not use the same ordered held-out subset")
    files = []
    output_path = root / args.output
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.resolve() != output_path.resolve():
            files.append({
                "path": path.relative_to(root).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            })
    manifest = {
        "schema_version": "face-pilot-manifest-1.0.0",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "validation": validation,
        "matched_ordered_subset": True,
        "matched_subset_size": len(ordered_indices[0]),
        "files": files,
    }
    temporary = output_path.with_name(f".{output_path.name}.tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    temporary.replace(output_path)
    print(json.dumps({
        "output": str(output_path), "files": len(files),
        "matched_subset_size": len(ordered_indices[0]), "validation": validation,
    }, indent=2))


if __name__ == "__main__":
    main()

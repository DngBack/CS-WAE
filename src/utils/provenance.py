"""Result manifests and content hashes for reproducible paper artifacts."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
from typing import Any

import torch

from ..metrics.audit_protocol import AUDIT_PROTOCOL_VERSION


RESULT_SCHEMA_VERSION = "audit-result-1.0.0"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_metadata(root: str | Path) -> dict[str, Any]:
    root = str(root)
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=root, text=True, stderr=subprocess.DEVNULL
            ).strip()
        )
        return {"commit": commit, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def build_manifest(
    *,
    repository_root: str | Path,
    checkpoint_path: str | Path,
    dataset: str,
    seed: int,
    evaluation_config: dict,
    model_config: dict | None = None,
    external_classifier_path: str | Path | None = None,
) -> dict:
    checkpoint = Path(checkpoint_path).resolve()
    external = Path(external_classifier_path).resolve() if external_classifier_path else None
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "audit_protocol_version": AUDIT_PROTOCOL_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": dataset,
        "seed": int(seed),
        "checkpoint": {
            "path": str(checkpoint),
            "sha256": sha256_file(checkpoint),
        },
        "external_classifier": None if external is None else {
            "path": str(external),
            "sha256": sha256_file(external),
        },
        "model_config": model_config or {},
        "evaluation_config": evaluation_config,
        "repository": _git_metadata(repository_root),
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
        },
    }


def save_result_with_manifest(path: str | Path, results: dict, manifest: dict) -> str:
    """Save one self-describing JSON result plus a SHA-256 sidecar."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {"manifest": manifest, "results": results}
    with output.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    digest = sha256_file(output)
    output.with_suffix(output.suffix + ".sha256").write_text(f"{digest}  {output.name}\n")
    return digest


def unwrap_result_payload(payload: dict) -> dict:
    """Read either a Stage-0 payload or a historical flat metrics dictionary."""

    if "manifest" in payload and isinstance(payload.get("results"), dict):
        return payload["results"]
    return payload

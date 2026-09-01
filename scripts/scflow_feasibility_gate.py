#!/usr/bin/env python3
"""Pre-download scientific and technical feasibility gate for official SCFlow."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
SCFLOW = ROOT / "external" / "SCFlow"
README = SCFLOW / "README.md"
INFERENCE = SCFLOW / "inference.py"
DATALOADER = SCFLOW / "scflow/dataloader.py"
TRAINER = SCFLOW / "scflow/trainer_module.py"
CONFIG = SCFLOW / "configs/inference.yaml"
OUTPUT = ROOT / "runs_modern" / "scflow_feasibility" / "gate.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def row(name: str, passed: bool, evidence: str) -> dict:
    return {"name": name, "passed": bool(passed), "evidence": evidence}


def main() -> None:
    sources = [README, INFERENCE, DATALOADER, TRAINER, CONFIG]
    for path in sources:
        if not path.is_file():
            raise FileNotFoundError(path)
    readme = README.read_text()
    inference = INFERENCE.read_text()
    dataloader = DATALOADER.read_text()
    trainer = TRAINER.read_text()
    config = CONFIG.read_text()
    checks = [
        row(
            "official_cross_image_inputs",
            "--image_c_path" in inference and "--image_s_path" in inference,
            "Official forward inference accepts distinct content and style image paths.",
        ),
        row(
            "explicit_product_construction",
            "torch.cat([features_c, features_s], dim=2)" in inference,
            "Official inference concatenates independently encoded donor embeddings.",
        ),
        row(
            "matching_training_recombination",
            "embed3_idx = other_group_idx" in dataloader
            and "clip_oai = torch.cat([clip_oai_c, clip_oai_s], dim=1)" in trainer,
            "Training sampler constructs a different-style content donor and a style donor before merging.",
        ),
        row(
            "invertible_endpoint_access",
            "def predict_forward" in trainer and "def predict_reverse" in trainer,
            "The released module exposes forward merge and reverse endpoint recovery.",
        ),
        row(
            "factor_metadata_available",
            "content_idx" in readme and "style_idx" in readme,
            "Official test split includes content and style identifiers for clause localization.",
        ),
        row(
            "latent_only_audit_without_unclip",
            "test_vis: bool = False" in trainer and "clip_dim: 1536" in config,
            "Contract and CLIP-space competence can run without the separate UnCLIP checkpoint.",
        ),
    ]
    all_pass = all(item["passed"] for item in checks)
    payload = {
        "protocol": "scflow-feasibility-gate-1.0.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "official_repository": "https://github.com/CompVis/SCFlow",
        "official_commit": subprocess.check_output(
            ["git", "-C", str(SCFLOW), "rev-parse", "HEAD"], text=True
        ).strip(),
        "source_sha256": {
            str(path.relative_to(SCFLOW)): sha256(path) for path in sources
        },
        "checks": checks,
        "decision": {
            "scope_and_hookability_pass": all_pass,
            "download_scflow_checkpoint": all_pass,
            "download_test_embeddings": all_pass,
            "download_unclip_checkpoint": False,
            "planned_claim_boundary": (
                "official modern cross-image recombination in CLIP space; image fidelity/diversity remains untested"
            ),
        },
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT.with_name(f".{OUTPUT.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, OUTPUT)
    print(json.dumps(payload["decision"], indent=2))
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()

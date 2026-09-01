#!/usr/bin/env python3
"""Static feasibility and scope gate for an official DEADiff contract audit.

This gate deliberately runs before downloading the 7+ GB checkpoint.  It
separates technical hookability from scientific scope: a model can expose two
branches while still lacking the independent replacement law audited by this
paper.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
DEADIFF = ROOT / "external" / "DEADiff"
MODEL_SOURCE = DEADIFF / "ldm/models/diffusion/blip_diffusion.py"
UNET_SOURCE = DEADIFF / "ldm/modules/diffusionmodules/openaimodel.py"
APP_SOURCE = DEADIFF / "scripts/app.py"
CONFIG = DEADIFF / "configs/inference_deadiff_512x512.yaml"
OUTPUT = ROOT / "runs_modern" / "deadiff_feasibility" / "gate.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def check(name: str, passed: bool, evidence: str, consequence: str) -> dict:
    return {
        "name": name,
        "passed": bool(passed),
        "evidence": evidence,
        "consequence": consequence,
    }


def main() -> None:
    required = [MODEL_SOURCE, UNET_SOURCE, APP_SOURCE, CONFIG]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing official DEADiff sources: {missing}")
    model = MODEL_SOURCE.read_text()
    unet = UNET_SOURCE.read_text()
    app = APP_SOURCE.read_text()
    config = CONFIG.read_text()

    technical = [
        check(
            "separate_qformer_branches",
            "self.style_blip = Blip2Qformer(" in model
            and "self.content_blip = Blip2Qformer(" in model,
            "DualBlipDiffusion constructs style_blip and content_blip separately.",
            "Separate branch embeddings can be observed without modifying weights.",
        ),
        check(
            "separate_projection_branches",
            "self.style_proj_layer = ProjLayer(" in model
            and "self.content_proj_layer = ProjLayer(" in model,
            "The official model exposes distinct style/content projection layers.",
            "Projected representations remain independently hookable.",
        ),
        check(
            "branch_specific_unet_routing",
            "ctx = [context[1]]" in unet
            and "ctx = [context[0]]" in unet
            and "content_layer" in config,
            "HierarchicalUNetModel routes content and style contexts to configured layer subsets.",
            "Decoder interventions can hold noise/prompt fixed while changing one branch.",
        ),
        check(
            "audit_adapter_supports_independent_donors",
            (ROOT / "src/metrics/deadiff_adapter.py").is_file(),
            "The local adapter accepts style_images and content_images separately and builds official branch order.",
            "A researcher-added cross-donor transport intervention is implementable.",
        ),
    ]

    scope = [
        check(
            "official_inference_uses_distinct_branch_images",
            False,
            (
                "Official get_learned_conditioning passes the same inp_image to both "
                "style_blip and content_blip; scripts/app.py exposes one image_input."
            ),
            "Independent branch donors would be an added intervention, not the declared inference law.",
        ),
        check(
            "declared_independent_product_sampler",
            False,
            (
                "The released stylization path conditions both learned branches on one reference; "
                "it does not sample a product of independently drawn branch laws."
            ),
            "The paper's train-generation replacement contract is not directly instantiated.",
        ),
    ]

    technical_pass = all(row["passed"] for row in technical)
    scope_pass = all(row["passed"] for row in scope)
    decision = {
        "technical_hookability_pass": technical_pass,
        "primary_contract_scope_pass": scope_pass,
        "download_full_checkpoint": technical_pass and scope_pass,
        "recommendation": (
            "Do not download/promote DEADiff as the primary external contract audit. "
            "It is technically suitable for a secondary leakage/decoder-transport stress test, "
            "but SCFlow or another model with declared cross-image recombination is better aligned."
        ),
    }
    commit = subprocess.check_output(
        ["git", "-C", str(DEADIFF), "rev-parse", "HEAD"], text=True
    ).strip()
    payload = {
        "protocol": "deadiff-feasibility-gate-1.0.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "official_repository": "https://github.com/bytedance/DEADiff",
        "official_commit": commit,
        "source_sha256": {
            str(path.relative_to(DEADIFF)): sha256(path) for path in required
        },
        "technical_checks": technical,
        "scope_checks": scope,
        "decision": decision,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT.with_name(f".{OUTPUT.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, OUTPUT)
    print(json.dumps(decision, indent=2))
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()

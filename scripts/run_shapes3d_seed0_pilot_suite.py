#!/usr/bin/env python3
"""Run the three Shapes3D seed-0 pilots sequentially and verify artifacts."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-root", default="runs_shapes3d/factor_audit_v1")
    return parser.parse_args()


def _run(command: list[str]) -> None:
    print(f"\n=== Sequential pilot: {' '.join(command)} ===", flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def _atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True))
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    python = sys.executable
    shared = [
        "--seed", "0",
        "--device", args.device,
        "--epochs", "100",
        "--batch-size", "64",
        "--num-workers", "4",
        "--checkpoint-every", "5",
        "--probe-epochs", "300",
        "--mmd-permutations", "500",
        "--hsic-permutations", "200",
        "--output-root", args.output_root,
        "--auto-resume",
        "--skip-existing",
    ]
    _run(
        [
            python,
            "scripts/run_shapes3d_factor_audit.py",
            "--stage", "seed0",
            "--evaluator-epochs", "20",
            "--evaluator-batch-size", "128",
            *shared,
        ]
    )
    for model in ("conditional_vae", "content_style_vae"):
        _run(
            [
                python,
                "scripts/run_shapes3d_factorized_baseline.py",
                "--model", model,
                "--stage", "full",
                *shared,
            ]
        )

    output_root = ROOT / args.output_root
    artifacts = {
        "fcswae": output_root / "fcswae" / "seed_0" / "factor_audit.json",
        "conditional_vae": output_root / "conditional_vae" / "seed_0" / "factor_audit.json",
        "content_style_vae": output_root / "content_style_vae" / "seed_0" / "factor_audit.json",
    }
    summary = {
        "schema_version": "shapes3d-seed0-pilot-suite-completion-1.0.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "execution_order": list(artifacts),
        "complete": all(path.exists() for path in artifacts.values()),
        "models": {},
    }
    for name, path in artifacts.items():
        payload = json.loads(path.read_text())
        summary["models"][name] = {
            "artifact": str(path.resolve()),
            "passes_competence_gate": bool(payload["results"]["passes_competence_gate"]),
            "competence_checks": payload["results"]["competence_checks"],
        }
    _atomic_json(output_root / "seed0_pilot_completion.json", summary)
    if not summary["complete"]:
        raise RuntimeError("Shapes3D seed-0 pilot suite did not produce every artifact")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

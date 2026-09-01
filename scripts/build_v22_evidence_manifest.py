#!/usr/bin/env python3
"""Build an identity-free manifest for the v22 evidence snapshot."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "fcswae" / "v22_evidence_manifest.json"
SIDECAR = OUTPUT.with_suffix(OUTPUT.suffix + ".sha256")

SOURCE_PATHS = [
    "scripts/build_v22_evidence_manifest.py",
    "scripts/build_v22_external_followup_summary.py",
    "scripts/audit_v22_anonymity.py",
    "scripts/audit_mnist_joint_contract.py",
    "scripts/summarize_kernel_robustness.py",
    "scripts/scflow_feasibility_gate.py",
    "scripts/run_scflow_contract_audit.py",
    "scripts/summarize_scflow_audit.py",
    "src/metrics/audit_protocol.py",
    "src/metrics/deadiff_adapter.py",
    "src/trainers/trainer_f_cs_wae.py",
    "src/utils/loss_f_cs_wae.py",
    "tests/test_audit_protocol.py",
    "tests/test_deadiff_adapter.py",
]

ARTIFACT_PATHS = [
    "fcswae/reported_followup_aggregates_v16.json",
    "runs_diag/kernel_robustness/summary.json",
    "runs_diag/fact/mean_hsic_mnist_40e/submission_followup_v1/summary.json",
    "runs_modern/deadiff_feasibility/gate.json",
    "runs_modern/scflow_feasibility/gate.json",
    "runs_modern/scflow_audit/main_seed2027.json",
    "runs_modern/scflow_audit/main_seed2028.json",
    "runs_modern/scflow_audit/main_seed2029.json",
    "runs_modern/scflow_audit/summary.json",
    "fcswae/reported_external_followup_v22.json",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def describe(relative_path: str) -> dict:
    path = ROOT / relative_path
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": relative_path,
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def git_output(*args: str, cwd: Path = ROOT) -> str:
    return subprocess.check_output(
        ["git", "-C", str(cwd), *args], text=True
    ).strip()


def main() -> None:
    payload = {
        "schema": "iclr2027-v22-evidence-manifest-1.0.0",
        "identity_free_paths": True,
        "provenance": {
            "repository_head": git_output("rev-parse", "HEAD"),
            "tracked_worktree_clean": not bool(
                git_output("status", "--short", "--untracked-files=no")
            ),
            "note": (
                "Historical artifacts retain their original dirty-worktree flags; "
                "this manifest binds the exact critical source files shipped with v22."
            ),
            "external_repositories": {
                "DEADiff": git_output("rev-parse", "HEAD", cwd=ROOT / "external/DEADiff"),
                "SCFlow": git_output("rev-parse", "HEAD", cwd=ROOT / "external/SCFlow"),
            },
        },
        "protocol_versions": [
            "stage0-1.1.0",
            "stage0-1.2.0",
            "stage0-1.3.0",
            "mnist-joint-contract-1.1.0",
            "scflow-contract-audit-1.1.0",
            "scflow-contract-summary-1.0.0",
        ],
        "critical_source_snapshot": [describe(path) for path in SOURCE_PATHS],
        "headline_artifacts": [describe(path) for path in ARTIFACT_PATHS],
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT.with_name(f".{OUTPUT.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, OUTPUT)
    SIDECAR.write_text(f"{sha256(OUTPUT)}  {OUTPUT.name}\n")
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()

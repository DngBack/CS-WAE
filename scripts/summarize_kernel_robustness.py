#!/usr/bin/env python3
"""Summarize the frozen MMD/energy and HSIC/dCor robustness audit."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import statistics


ROOT = Path(__file__).resolve().parents[1]
AUDIT_ROOT = ROOT / "runs_diag" / "kernel_robustness"
MATCHED = AUDIT_ROOT / "fact_matched_seed0.json"
POSTFREEZE = AUDIT_ROOT / "fact_postfreeze_seed0.json"
OUTPUT = AUDIT_ROOT / "summary.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def mean_sd(values: list[float]) -> dict:
    return {
        "mean": statistics.mean(values),
        "sample_sd": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
        "values": values,
    }


def reduction(candidate: float, reference: float) -> float:
    return 1.0 - candidate / reference


def records(path: Path) -> dict[str, dict]:
    payload = json.loads(path.read_text())
    if payload["protocol"]["version"] != "mnist-joint-contract-1.1.0":
        raise RuntimeError(f"unexpected protocol in {path}")
    return {row["name"]: row for row in payload["results"]}


def pair_row(seed: int, control: dict, fact: dict) -> dict:
    return {
        "training_seed": seed,
        "joint_mmd_reduction": reduction(
            fact["joint"]["mmd2_u"], control["joint"]["mmd2_u"]
        ),
        "joint_energy_reduction": reduction(
            fact["joint_energy"]["energy_u"],
            control["joint_energy"]["energy_u"],
        ),
        "content_energy_reduction": reduction(
            fact["content_energy"]["energy_u"],
            control["content_energy"]["energy_u"],
        ),
        "style_energy_reduction": reduction(
            fact["style_energy"]["energy_u"],
            control["style_energy"]["energy_u"],
        ),
        "conditional_hsic_reduction": reduction(
            fact["within_class_dependence"]["hsic"]["statistic"],
            control["within_class_dependence"]["hsic"]["statistic"],
        ),
        "conditional_dcor_reduction": reduction(
            fact["within_class_dependence"]["distance_correlation"]["statistic"],
            control["within_class_dependence"]["distance_correlation"]["statistic"],
        ),
        "mmd_energy_decision_agreement": fact[
            "cross_estimator_agreement"
        ]["agreement_rate"],
        "hsic_dcor_decision_agreement": fact[
            "within_class_dependence"
        ]["rejection_agreement"],
    }


def aggregate(rows: list[dict]) -> dict:
    metrics = [key for key in rows[0] if key != "training_seed"]
    output = {}
    for metric in metrics:
        values = [row[metric] for row in rows]
        if isinstance(values[0], bool):
            output[metric] = {
                "pass_count": sum(values),
                "n": len(values),
                "all_pass": all(values),
            }
        else:
            output[metric] = mean_sd([float(value) for value in values])
    return output


def main() -> None:
    matched = records(MATCHED)
    postfreeze = records(POSTFREEZE)
    paired = [
        pair_row(seed, matched[f"control_seed{seed}"], matched[f"fact_seed{seed}"])
        for seed in (0, 1, 2)
    ]
    reference = postfreeze["control_seed0"]
    postfreeze_rows = [
        pair_row(seed, reference, postfreeze[f"fact_seed{seed}"])
        for seed in (3, 4, 5)
    ]
    payload = {
        "protocol": "kernel-robustness-summary-1.0.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "interpretation": {
            "decision_agreement": (
                "fraction of aggregate and class-level equality decisions shared by MMD and energy"
            ),
            "effect_reduction": (
                "1 - FACT/control; positive values favor FACT; all training seeds are retained"
            ),
            "scope": (
                "post-specified robustness analysis; not an optimization or model-selection criterion"
            ),
        },
        "input_sha256": {
            str(MATCHED.relative_to(ROOT)): sha256(MATCHED),
            str(POSTFREEZE.relative_to(ROOT)): sha256(POSTFREEZE),
        },
        "matched_pairs": paired,
        "matched_aggregate": aggregate(paired),
        "postfreeze_vs_frozen_control_seed0": postfreeze_rows,
        "postfreeze_aggregate": aggregate(postfreeze_rows),
        "headline": {
            "matched_joint_mmd_improved_count": sum(
                row["joint_mmd_reduction"] > 0 for row in paired
            ),
            "matched_joint_energy_improved_count": sum(
                row["joint_energy_reduction"] > 0 for row in paired
            ),
            "matched_conditional_dcor_improved_count": sum(
                row["conditional_dcor_reduction"] > 0 for row in paired
            ),
            "postfreeze_joint_energy_improved_count": sum(
                row["joint_energy_reduction"] > 0 for row in postfreeze_rows
            ),
            "postfreeze_conditional_dcor_improved_count": sum(
                row["conditional_dcor_reduction"] > 0 for row in postfreeze_rows
            ),
        },
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT.with_name(f".{OUTPUT.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, OUTPUT)
    digest = sha256(OUTPUT)
    OUTPUT.with_suffix(".json.sha256").write_text(f"{digest}  {OUTPUT.name}\n")
    print(json.dumps(payload["headline"], indent=2))
    print(f"wrote {OUTPUT} ({digest})")


if __name__ == "__main__":
    main()

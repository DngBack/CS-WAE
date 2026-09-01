#!/usr/bin/env python3
"""Aggregate the frozen three-grid SCFlow contract audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "inputs",
        nargs="*",
        default=[
            "runs_modern/scflow_audit/main_seed2027.json",
            "runs_modern/scflow_audit/main_seed2028.json",
            "runs_modern/scflow_audit/main_seed2029.json",
        ],
    )
    parser.add_argument(
        "--output", default="runs_modern/scflow_audit/summary.json"
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def aggregate(values: list[float]) -> dict:
    return {
        "mean": statistics.fmean(values),
        "sample_sd": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
        "values": values,
    }


def nested(record: dict, path: str):
    value = record
    for key in path.split("."):
        value = value[key]
    return value


def holm_rejections(p_values: list[float], alpha: float = 0.05) -> list[bool]:
    order = sorted(range(len(p_values)), key=p_values.__getitem__)
    rejected = [False] * len(p_values)
    for rank, index in enumerate(order):
        if p_values[index] > alpha / (len(p_values) - rank):
            break
        rejected[index] = True
    return rejected


def main() -> None:
    args = parse_args()
    paths = [ROOT / value for value in args.inputs]
    records = [json.loads(path.read_text()) for path in paths]
    if len(records) < 2:
        raise ValueError("summary requires at least two evaluation-subset seeds")
    protocols = {record["protocol"]["version"] for record in records}
    checkpoints = {
        record["protocol"]["checkpoint"]["sha256"] for record in records
    }
    test_splits = {record["protocol"]["test_h5"]["sha256"] for record in records}
    if protocols != {"scflow-contract-audit-1.1.0"}:
        raise ValueError(f"unexpected protocol versions: {sorted(protocols)}")
    if len(checkpoints) != 1 or len(test_splits) != 1:
        raise ValueError("all evaluation seeds must use the same checkpoint and split")

    equality_paths = [
        "results.sampler_compatibility.direct_joint_mmd.p_value",
        "results.sampler_compatibility.direct_joint_energy.p_value",
        "results.sampler_compatibility.content_marginal_mmd.p_value",
        "results.sampler_compatibility.content_marginal_energy.p_value",
        "results.sampler_compatibility.style_marginal_mmd.p_value",
        "results.sampler_compatibility.style_marginal_energy.p_value",
    ]
    equality_counts = {
        path.split(".")[-2]: sum(nested(record, path) <= 0.05 for record in records)
        for path in equality_paths
    }
    hsic_paths = [
        "results.sampler_compatibility.reverse_content_style_hsic.p_value",
        "results.sampler_compatibility.reverse_style_content_hsic.p_value",
    ]
    hsic_p_values = [
        float(nested(record, path)) for record in records for path in hsic_paths
    ]

    metric_paths = {
        "target_cosine": "results.decoder_transport.target_cosine.mean",
        "source_mean_target_cosine": (
            "results.decoder_transport.source_mean_target_cosine.mean"
        ),
        "cycle_content_cosine": (
            "results.decoder_transport.cycle_content_cosine.mean"
        ),
        "cycle_style_cosine": "results.decoder_transport.cycle_style_cosine.mean",
        "forward_content_retrieval": (
            "results.clip_space_competence.forward_target_retrieval.content_accuracy"
        ),
        "forward_style_retrieval": (
            "results.clip_space_competence.forward_target_retrieval.style_accuracy"
        ),
        "forward_joint_retrieval": (
            "results.clip_space_competence.forward_target_retrieval.joint_accuracy"
        ),
        "baseline_content_retrieval": (
            "results.clip_space_competence.source_mean_baseline_retrieval.content_accuracy"
        ),
        "baseline_style_retrieval": (
            "results.clip_space_competence.source_mean_baseline_retrieval.style_accuracy"
        ),
        "baseline_joint_retrieval": (
            "results.clip_space_competence.source_mean_baseline_retrieval.joint_accuracy"
        ),
    }
    aggregates = {
        name: aggregate([float(nested(record, path)) for record in records])
        for name, path in metric_paths.items()
    }
    cosine_gains = [
        float(nested(record, metric_paths["target_cosine"]))
        - float(nested(record, metric_paths["source_mean_target_cosine"]))
        for record in records
    ]
    aggregates["target_cosine_gain"] = aggregate(cosine_gains)

    payload = {
        "protocol": {
            "version": "scflow-contract-summary-1.0.0",
            "source_protocol": next(iter(protocols)),
            "post_specified": True,
            "evaluation_subset_seeds": [
                record["protocol"]["seed"] for record in records
            ],
            "n_combinations_per_seed": [
                record["protocol"]["n_combinations"] for record in records
            ],
            "checkpoint_sha256": next(iter(checkpoints)),
            "test_h5_sha256": next(iter(test_splits)),
            "claim_boundary": records[0]["protocol"]["claim_boundary"],
            "inputs": [
                {
                    "path": str(path.relative_to(ROOT)),
                    "sha256": sha256(path),
                }
                for path in paths
            ],
        },
        "decisions": {
            "alpha": 0.05,
            "equality_rejections_out_of_3": equality_counts,
            "hsic_raw_rejections_out_of_6": sum(value <= 0.05 for value in hsic_p_values),
            "hsic_holm_rejections_out_of_6": sum(holm_rejections(hsic_p_values)),
            "hsic_p_values": hsic_p_values,
        },
        "aggregates": aggregates,
    }
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, output)
    print(json.dumps(payload["decisions"], indent=2))
    for name, result in aggregates.items():
        print(f"{name}: {result['mean']:.6f} +/- {result['sample_sd']:.6f}")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()

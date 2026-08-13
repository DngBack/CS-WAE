#!/usr/bin/env python3
"""Create a derived cross-model paper table without mutating frozen artifacts."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.metrics.statistical_reporting import holm_family
from src.utils.provenance import sha256_file


FAMILIES = ("fcswae", "drit", "diva")
BASE_CHECKS = ("external_evaluator", "content_probe", "reconstruction")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", default="runs_cross_model/native_v1")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--out", default=None)
    return parser.parse_args()


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True))
    temporary.replace(path)
    digest = sha256_file(path)
    sidecar = path.with_suffix(path.suffix + ".sha256")
    temporary_sidecar = sidecar.with_name(f".{sidecar.name}.tmp")
    temporary_sidecar.write_text(f"{digest}  {path.name}\n")
    temporary_sidecar.replace(sidecar)


def _mean_std(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if array.size > 1 else 0.0,
    }


def derive(input_root: Path, seeds: list[int]) -> dict:
    rows = []
    sources = []
    for family in FAMILIES:
        for seed in seeds:
            path = input_root / family / f"seed_{seed}" / "audit.json"
            payload = json.loads(path.read_text())
            domains = payload["domains"]
            macro = payload["macro"]
            frozen_checks = macro["competence_checks"]
            base_checks = {name: bool(frozen_checks[name]) for name in BASE_CHECKS}
            row = {
                "family": family,
                "seed": seed,
                "metrics": {
                    key: value
                    for key, value in macro["metrics"].items()
                    if key not in {"global_mmd_p_value", "label_hsic_p_value"}
                },
                "domain_tests": [
                    {
                        "domain": int(domain["domain"]),
                        "global_mmd_p_value": float(domain["global_mmd"]["p_value"]),
                        "label_hsic_p_value": float(domain["label_hsic"]["p_value"]),
                    }
                    for domain in domains
                ],
                "base_competence_checks": base_checks,
                "passes_base_competence": all(base_checks.values()),
                "sampling_outcome": {
                    "prior_style_content_retention": float(
                        macro["metrics"]["prior_style_content_retention"]
                    ),
                    "passes_frozen_retention_threshold": bool(
                        frozen_checks["content_retention"]
                    ),
                },
                "frozen_all_check_gate": {
                    "checks": frozen_checks,
                    "passes": bool(macro["passes_competence_gate"]),
                },
                "source_audit": str(path),
                "source_audit_sha256": sha256_file(path),
            }
            rows.append(row)
            sources.append({"path": str(path), "sha256": sha256_file(path)})

    family_summary = {}
    metric_keys = (
        "global_mmd2_u",
        "conditional_mmd2_u",
        "conditional_minus_global_mmd2_u",
        "style_logistic_accuracy",
        "style_mlp_accuracy",
        "style_rbf_accuracy",
        "label_hsic",
        "content_logistic_accuracy",
        "reconstruction_l1",
        "prior_style_content_retention",
    )
    for family in FAMILIES:
        selected = [row for row in rows if row["family"] == family]
        global_p = [
            test["global_mmd_p_value"]
            for row in selected
            for test in row["domain_tests"]
        ]
        hsic_p = [
            test["label_hsic_p_value"]
            for row in selected
            for test in row["domain_tests"]
        ]
        family_summary[family] = {
            "n_seeds": len(selected),
            "base_competent_seeds": sum(row["passes_base_competence"] for row in selected),
            "frozen_all_check_passing_seeds": sum(
                row["frozen_all_check_gate"]["passes"] for row in selected
            ),
            "metrics": {
                key: _mean_std([float(row["metrics"][key]) for row in selected])
                for key in metric_keys
            },
            "global_mmd_domain_tests_holm": holm_family(global_p),
            "label_hsic_domain_tests_holm": holm_family(hsic_p),
        }
    return {
        "manifest": {
            "schema_version": "native-cross-model-paper-summary-1.0.0",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_protocol": "native-cross-model-1.0.0",
            "derived_only_no_source_artifacts_mutated": True,
            "seeds": seeds,
            "p_value_policy": (
                "No arithmetic mean p-value is used for inference; raw domain tests "
                "and within-family Holm correction are reported."
            ),
            "competence_policy": (
                "Base competence is reported separately from the frozen all-check gate; "
                "the frozen gate and its original retention threshold are preserved."
            ),
            "source_artifacts": sources,
        },
        "rows": rows,
        "family_summary": family_summary,
    }


def main() -> None:
    args = parse_args()
    input_root = ROOT / args.input_root
    output = Path(args.out) if args.out else input_root / "cross_model_paper_summary.json"
    if not output.is_absolute():
        output = ROOT / output
    _atomic_json(output, derive(input_root, list(args.seeds)))
    print(output)


if __name__ == "__main__":
    main()

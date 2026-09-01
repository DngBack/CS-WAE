#!/usr/bin/env python3
"""Normalize the externally supplied v22 FACT follow-up summary.

This script does not reconstruct absent seed rows. It records which quantities
were supplied, recomputes the valid n=5 Student-t intervals, and makes missing
provenance explicit so the manuscript cannot silently overstate verification.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "fcswae" / "reported_external_followup_v22.json"
SIDECAR = OUTPUT.with_suffix(OUTPUT.suffix + ".sha256")
T_975_DF4 = 2.7764451051977987


def two_sided_t_interval(mean: float, sample_sd: float, n: int) -> list[float]:
    if n != 5:
        raise ValueError("this frozen summary only specifies the df=4 critical value")
    margin = T_975_DF4 * sample_sd / math.sqrt(n)
    return [round(mean - margin, 4), round(mean + margin, 4)]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    payload = {
        "schema": "iclr2027-v22-external-followup-summary-1.0.0",
        "source": "external run summary supplied to the manuscript workspace on 2026-08-31",
        "verification_status": "summary_only_raw_seed_rows_and_checkpoints_not_present",
        "fresh_matched_pairs": {
            "training_seed_count": 5,
            "hyperparameters": "reported frozen",
            "joint_mmd_reduction_percent": {
                "mean": 28.5,
                "sample_sd": 4.2,
                "two_sided_95pct_t_ci": two_sided_t_interval(28.5, 4.2, 5),
                "positive_seed_count": 5,
            },
            "conditional_mmd_reduction_percent": {
                "mean": 23.2,
                "sample_sd": 5.1,
                "two_sided_95pct_t_ci": two_sided_t_interval(23.2, 5.1, 5),
            },
            "mlp_leakage_accuracy_percent": {
                "mean": 11.6,
                "sample_sd": 0.9,
                "two_sided_95pct_t_ci": two_sided_t_interval(11.6, 0.9, 5),
            },
            "all_gate_pass_count": 5,
            "all_gate_total": 5,
            "supplied_joint_ci_lower_bound_percent": 18.2,
            "supplied_joint_ci_status": (
                "not_used: inconsistent with mean=28.5, sample_sd=4.2, n=5 "
                "under a standard Student-t interval"
            ),
        },
        "leave_one_component_out": {
            "training_seed_count": None,
            "uncertainty_available": False,
            "without_adversary_mlp_increase_pp": 24.1,
            "without_mean_hsic_hsic_ratio_to_full": 1.85,
            "without_dependence_joint_reduction_lost_pp": 10.2,
            "interpretation": "component-specific degradation; not a factorial causal proof",
        },
        "mean_hsic_weight_sweep": {
            "weights": [2, 3, 4, 5],
            "reported_jointly_passing_adjacent_weights": [3, 4],
            "criteria": {
                "joint_mmd_reduction_percent_min": 20.0,
                "conditional_mmd_reduction_percent_min": 20.0,
                "mlp_leakage_percent_max": 20.0,
                "posterior_mean_hsic_ratio_max": 1.0,
                "reconstruction_degradation_percent_max_exclusive": 10.0,
            },
            "per_weight_rows_available": False,
        },
        "checkpoint_regime_trajectory": {
            "sampling_unit_count": None,
            "bootstrap_protocol_available": False,
            "absolute_spearman_style_leakage_vs_style_donor": 0.78,
            "reported_95pct_ci": [0.62, 0.88],
            "spearman_joint_discrepancy_vs_gen_acc": 0.15,
            "reported_joint_gen_acc_95pct_ci": [-0.12, 0.38],
        },
        "fact_lite": {
            "training_seed_count": None,
            "uncertainty_available": False,
            "joint_mmd_reduction_percent": 22.4,
            "mlp_leakage_accuracy_percent": 18.5,
            "gen_acc_change_pp": -1.9,
            "reconstruction_degradation_percent": 4.2,
            "reported_target": "at least two of three seeds satisfy all constraints",
            "achieved_joint_pass_count": None,
        },
        "submission_boundary": (
            "Only the fresh-pair aggregate has n and sample standard deviations. "
            "Seed identifiers, raw rows, checkpoint hashes, audit RNGs, trajectory "
            "sampling units, and bootstrap code must be added before these follow-up "
            "results can be described as artifact-verified."
        ),
    }
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    SIDECAR.write_text(f"{sha256(OUTPUT)}  {OUTPUT.name}\n")
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Run the frozen mean-HSIC=4 bridge experiment and three-seed audit."""

from pathlib import Path

import run_fact_mean_hsic_sweep as pipeline


ROOT = Path(__file__).resolve().parents[1]

# Reuse the tested sweep orchestration while isolating this final interpolation
# run from the completed {2, 5, 20} development sweep.
pipeline.RUNS = [{"name": "mean_hsic_w4", "weight": 4.0}]
pipeline.AUDIT_ROOT = (
    ROOT / "runs_diag" / "fact" / "mean_hsic_mnist_40e" / "bridge_w4"
)
pipeline.LOG_ROOT = ROOT / "logs" / "fact_mean_hsic_mnist_40e" / "bridge_w4"
pipeline.MANIFEST = (
    ROOT / "runs_fact" / "mean_hsic_mnist_40e" / "bridge_w4_manifest.json"
)


if __name__ == "__main__":
    pipeline.main()

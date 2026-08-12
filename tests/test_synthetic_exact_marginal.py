"""CPU-only tests for the durable paper-scale synthetic runner."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

from scripts.run_synthetic_exact_marginal import (
    FROZEN_ACCEPTANCE,
    _load_partial_records,
    _save_partial_records,
    evaluate_frozen_acceptance,
)


KAPPAS = [0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]


def minimal_config() -> dict:
    return {
        "audit_protocol_version": "stage0-1.2.0",
        "construction": "test construction",
        "dimension": 4,
        "n_classes": 2,
        "n_samples": 64,
        "replications": 2,
        "kappas": [0.0, 4.0],
        "seed": 7,
        "mmd_permutations": 9,
        "hsic_permutations": 9,
        "probe_epochs": 5,
    }


class SyntheticExactMarginalTests(unittest.TestCase):
    def test_partial_checkpoint_roundtrip_and_config_guard(self):
        config = minimal_config()
        records = [
            {"replication": 0, "kappa": 0.0},
            {"replication": 0, "kappa": 4.0},
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "partial_results.json"
            _save_partial_records(path, config, records)
            loaded = _load_partial_records(path, config, resume=True)
            self.assertEqual(loaded, records)
            self.assertFalse(any(path.parent.glob(".*.tmp")))

            changed = dict(config, n_samples=128)
            with self.assertRaisesRegex(ValueError, "Resume configuration mismatch"):
                _load_partial_records(path, changed, resume=True)
            with self.assertRaises(FileExistsError):
                _load_partial_records(path, config, resume=False)

    def test_partial_checkpoint_rejects_incomplete_replication(self):
        config = minimal_config()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "partial_results.json"
            with self.assertRaisesRegex(ValueError, "incomplete replication"):
                _save_partial_records(
                    path,
                    config,
                    [{"replication": 0, "kappa": 0.0}],
                )

    def test_frozen_acceptance_passes_calibrated_pedagogical_result(self):
        records = []
        for replication in range(50):
            p_value = 0.02 if replication < 2 else 0.5
            global_mmd = 1e-5 * (replication - 25)
            for kappa in KAPPAS:
                records.append(
                    {
                        "replication": replication,
                        "kappa": kappa,
                        "global_mmd2_u": global_mmd,
                        "global_mmd_p_value": p_value,
                    }
                )
        summaries = []
        for index, kappa in enumerate(KAPPAS):
            summaries.append(
                {
                    "kappa": kappa,
                    "probe_accuracy_mean": 0.5 + 0.45 * min(1.0, kappa / 4.0),
                    "hsic_mean": 0.001 + index * 0.001,
                    "conditional_mmd2_u_mean_mean": -0.001 + index * 0.001,
                }
            )
        result = evaluate_frozen_acceptance(
            {"records": records, "summary_by_kappa": summaries},
            SimpleNamespace(),
        )
        self.assertTrue(result["passes_all"])
        self.assertEqual(result["observed"]["global_mmd_rejection_count"], 2)
        self.assertEqual(
            FROZEN_ACCEPTANCE["global_mmd_rejection_count_interval"], [0, 6]
        )


if __name__ == "__main__":
    unittest.main()

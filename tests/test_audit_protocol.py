"""Unit tests for the Stage-0 audit protocol (no dataset download or GPU)."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from src.metrics.audit_protocol import (
    AUDIT_PROTOCOL_VERSION,
    fit_logistic_probe,
    hsic_permutation_test,
    mmd2_unbiased,
    run_probe_suite,
    stratified_probe_split,
)
from src.models.external_classifiers import GrayscaleExternalCNN
from src.utils.provenance import build_manifest, save_result_with_manifest, sha256_file
from scripts.compute_leakage_diagnostics import run_diagnostics


class _ToyFactorizedModel(nn.Module):
    semantic_dim = 3
    style_dim = 4
    n_classes = 3

    def encode(self, images):
        flat = images.flatten(1)
        mu_s = flat[:, :4]
        mu_c = F.normalize(flat[:, :3] + 1e-3, dim=1)
        rho_c = torch.full((images.shape[0],), 0.5, device=images.device)
        logvar_s = torch.full_like(mu_s, -4.0)
        return mu_c, rho_c, mu_s, logvar_s


class AuditProtocolTests(unittest.TestCase):
    def test_probe_split_is_stratified_disjoint_and_complete(self):
        labels = torch.arange(5).repeat_interleave(40)
        split = stratified_probe_split(labels, seed=7)
        sets = [set(split.train), set(split.validation), set(split.test)]
        self.assertFalse(sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
        self.assertEqual(sets[0] | sets[1] | sets[2], set(range(labels.numel())))
        for indices in (split.train, split.validation, split.test):
            counts = torch.bincount(labels[indices], minlength=5)
            self.assertEqual(int(counts.min()), int(counts.max()))

    def test_probe_scaler_is_fit_on_train_only(self):
        train_x = torch.tensor([[0.0, 0.0], [2.0, 4.0], [4.0, 8.0], [6.0, 12.0]])
        train_y = torch.tensor([0, 0, 1, 1])
        validation_x = torch.tensor([[100.0, 200.0], [110.0, 220.0]])
        validation_y = torch.tensor([0, 1])
        _, probe = fit_logistic_probe(
            train_x, train_y, validation_x, validation_y, seed=0, epochs=5
        )
        self.assertTrue(torch.allclose(probe.feature_mean, train_x.mean(0)))
        self.assertFalse(torch.allclose(probe.feature_mean, torch.cat([train_x, validation_x]).mean(0)))

    def test_probe_suite_reports_separate_test_metrics(self):
        generator = torch.Generator().manual_seed(0)
        labels = torch.arange(3).repeat_interleave(40)
        features = torch.randn((120, 6), generator=generator) + labels[:, None] * 2.0
        result = run_probe_suite(
            features, labels, seed=3, probe_names=("logistic", "mlp", "knn"), epochs=40
        )
        self.assertEqual(result["split"]["train_size"], 72)
        self.assertEqual(result["split"]["validation_size"], 24)
        self.assertEqual(result["split"]["test_size"], 24)
        for model_result in result["models"].values():
            self.assertLessEqual(model_result["test"]["accuracy"], 1.0)
            self.assertGreaterEqual(model_result["test"]["accuracy"], 0.0)

    def test_unbiased_mmd_detects_shift_without_clipping(self):
        generator = torch.Generator().manual_seed(11)
        x = torch.randn((256, 3), generator=generator)
        same = torch.randn((256, 3), generator=generator)
        shifted = torch.randn((256, 3), generator=generator) + 3.0
        null_value = mmd2_unbiased(x, same)
        shifted_value = mmd2_unbiased(x, shifted)
        self.assertGreater(shifted_value, null_value + 0.1)
        self.assertIsInstance(null_value, float)

    def test_multiscale_hsic_permutation_detects_dependence(self):
        generator = torch.Generator().manual_seed(5)
        labels = torch.arange(4).repeat_interleave(48)
        dependent = torch.randn((labels.numel(), 5), generator=generator) + labels[:, None] * 1.5
        calibrated = hsic_permutation_test(dependent, labels, seed=5, n_permutations=49)
        self.assertLessEqual(calibrated["p_value"], 0.05)
        self.assertEqual(calibrated["n_permutations"], 49)
        self.assertGreater(len(calibrated["sigmas"]), 1)

    def test_external_grayscale_classifier_accepts_28_and_32_pixels(self):
        classifier = GrayscaleExternalCNN()
        self.assertEqual(classifier(torch.rand(2, 1, 28, 28)).shape, (2, 10))
        self.assertEqual(classifier(torch.rand(2, 1, 32, 32)).shape, (2, 10))

    def test_end_to_end_diagnostics_names_both_latent_views(self):
        generator = torch.Generator().manual_seed(13)
        labels = torch.arange(3).repeat_interleave(40)
        images = torch.randn((120, 1, 2, 2), generator=generator) + labels[:, None, None, None]
        loader = DataLoader(TensorDataset(images, labels), batch_size=24, shuffle=False)
        results = run_diagnostics(
            _ToyFactorizedModel(), loader, torch.device("cpu"), n_classes=3,
            seed=4, probe_epochs=20, hsic_permutations=9, verbose=False,
        )
        self.assertIn("global_mmd2_u_z_s_sample", results["sampling_contract"])
        self.assertIn("probe_suite_mu_s", results["representation"])
        self.assertIn("joint_mmd2_u_mu_c_mu_s", results["within_class_dependence"])
        self.assertEqual(results["protocol"]["n_evaluation_samples"], 120)

    def test_result_manifest_hashes_checkpoint_and_output(self):
        with tempfile.TemporaryDirectory() as directory:
            temp_path = Path(directory)
            checkpoint = temp_path / "model.pth"
            torch.save({"weight": torch.tensor([1.0])}, checkpoint)
            manifest = build_manifest(
                repository_root=temp_path,
                checkpoint_path=checkpoint,
                dataset="mnist",
                seed=2,
                evaluation_config={"n_samples": 32},
                model_config={"style_dim": 4},
            )
            self.assertEqual(manifest["audit_protocol_version"], AUDIT_PROTOCOL_VERSION)
            self.assertEqual(manifest["checkpoint"]["sha256"], sha256_file(checkpoint))
            output = temp_path / "result.json"
            digest = save_result_with_manifest(output, {"metric": 1.0}, manifest)
            self.assertEqual(digest, sha256_file(output))
            self.assertTrue(output.with_suffix(".json.sha256").exists())
            payload = json.loads(output.read_text())
            self.assertEqual(payload["manifest"]["seed"], 2)
            self.assertEqual(payload["results"]["metric"], 1.0)


if __name__ == "__main__":
    unittest.main()

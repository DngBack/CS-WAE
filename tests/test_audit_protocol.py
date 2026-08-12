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
    conditional_mmd_to_standard_normal,
    fit_logistic_probe,
    hsic_permutation_test,
    mmd2_unbiased,
    mmd2_permutation_test,
    run_probe_suite,
    stratified_probe_split,
)
from src.metrics.factorized_adapter import build_factorized_audit_adapter
from src.models.external_classifiers import GrayscaleExternalCNN
from src.models.f_cs_wae import (
    effective_style_variance,
    sample_style_posterior,
    style_posterior_entropy,
)
from src.utils.provenance import build_manifest, save_result_with_manifest, sha256_file
from scripts.compute_leakage_diagnostics import run_diagnostics


class _ToyFactorizedModel(nn.Module):
    semantic_dim = 3
    style_dim = 4
    n_classes = 3
    style_sigma_floor = 0.0

    def __init__(self):
        super().__init__()
        self.decoder = nn.Identity()

    def encode(self, images):
        flat = images.flatten(1)
        mu_s = flat[:, :4]
        mu_c = F.normalize(flat[:, :3] + 1e-3, dim=1)
        rho_c = torch.full((images.shape[0],), 0.5, device=images.device)
        logvar_s = torch.full_like(mu_s, -4.0)
        return mu_c, rho_c, mu_s, logvar_s

    def sample_style(self, mu_s, logvar_s, generator=None):
        return sample_style_posterior(mu_s, logvar_s, generator=generator)


class AuditProtocolTests(unittest.TestCase):
    def test_protocol_version_is_bumped_for_entropy_contract(self):
        self.assertEqual(AUDIT_PROTOCOL_VERSION, "stage0-1.2.0")

    def test_unified_style_sampler_is_seeded_and_applies_variance_floor(self):
        mu = torch.zeros((3, 4))
        logvar = torch.full_like(mu, -20.0)
        first_generator = torch.Generator().manual_seed(17)
        second_generator = torch.Generator().manual_seed(17)
        sample, std = sample_style_posterior(
            mu, logvar, sigma_floor=0.35, generator=first_generator
        )
        expected_noise = torch.randn(mu.shape, generator=second_generator)
        expected_std = effective_style_variance(logvar, 0.35).sqrt()
        self.assertTrue(torch.allclose(std, expected_std))
        self.assertTrue(torch.allclose(sample, expected_noise * expected_std))
        self.assertGreaterEqual(float(std.min()), 0.35)

    def test_style_entropy_uses_effective_variance(self):
        logvar = torch.zeros((2, 5))
        entropy = style_posterior_entropy(logvar, sigma_floor=0.0)
        expected_per_dimension = 0.5 * np.log(2.0 * np.pi * np.e)
        self.assertTrue(
            torch.allclose(
                entropy,
                torch.full((2,), 5 * expected_per_dimension, dtype=entropy.dtype),
            )
        )

    def test_adapter_rejects_unknown_latent_split(self):
        with self.assertRaisesRegex(TypeError, "Do not infer style"):
            build_factorized_audit_adapter(nn.Linear(4, 4))

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
        logistic_test = result["models"]["logistic"]["test"]
        self.assertIn("cross_entropy_nats", logistic_test)
        self.assertIn("information_lower_bound_nats", logistic_test)

    def test_unbiased_mmd_detects_shift_without_clipping(self):
        generator = torch.Generator().manual_seed(11)
        x = torch.randn((256, 3), generator=generator)
        same = torch.randn((256, 3), generator=generator)
        shifted = torch.randn((256, 3), generator=generator) + 3.0
        null_value = mmd2_unbiased(x, same)
        shifted_value = mmd2_unbiased(x, shifted)
        self.assertGreater(shifted_value, null_value + 0.1)
        self.assertIsInstance(null_value, float)

    def test_conditional_mmd_detects_dependence_under_gaussian_marginal(self):
        generator = torch.Generator().manual_seed(31)
        z = torch.randn((600, 4), generator=generator)
        labels = (z[:, 0] > 0).long()
        conditional = conditional_mmd_to_standard_normal(
            z, labels, n_classes=2, seed=31
        )
        marginal_reference = torch.randn((600, 4), generator=generator)
        marginal = mmd2_unbiased(z, marginal_reference)
        self.assertGreater(conditional["mean_mmd2_u"], marginal + 0.01)
        self.assertEqual(set(conditional["per_class_mmd2_u"]), {"0", "1"})

    def test_mmd_permutation_calibration_detects_large_shift(self):
        generator = torch.Generator().manual_seed(19)
        x = torch.randn((96, 4), generator=generator)
        y = torch.randn((96, 4), generator=generator) + 2.5
        calibrated = mmd2_permutation_test(x, y, seed=19, n_permutations=49)
        self.assertLessEqual(calibrated["p_value"], 0.05)
        self.assertEqual(calibrated["n_per_group"], 96)
        self.assertEqual(len(calibrated["null_values"]), 49)
        self.assertAlmostEqual(calibrated["statistic"], mmd2_unbiased(x, y), places=5)

    def test_multiscale_hsic_permutation_detects_dependence(self):
        generator = torch.Generator().manual_seed(5)
        labels = torch.arange(4).repeat_interleave(48)
        dependent = torch.randn((labels.numel(), 5), generator=generator) + labels[:, None] * 1.5
        calibrated = hsic_permutation_test(dependent, labels, seed=5, n_permutations=49)
        self.assertLessEqual(calibrated["p_value"], 0.05)
        self.assertEqual(calibrated["n_permutations"], 49)
        self.assertGreater(len(calibrated["sigmas"]), 1)

    def test_hsic_permutation_batching_is_result_invariant(self):
        generator = torch.Generator().manual_seed(29)
        labels = torch.arange(3).repeat_interleave(32)
        features = torch.randn((labels.numel(), 6), generator=generator)
        one_at_a_time = hsic_permutation_test(
            features,
            labels,
            seed=41,
            n_permutations=23,
            permutation_batch_size=1,
        )
        batched = hsic_permutation_test(
            features,
            labels,
            seed=41,
            n_permutations=23,
            permutation_batch_size=8,
        )
        self.assertEqual(one_at_a_time["p_value"], batched["p_value"])
        self.assertAlmostEqual(
            one_at_a_time["null_mean"], batched["null_mean"], places=7
        )
        self.assertAlmostEqual(
            one_at_a_time["null_quantiles"]["q95"],
            batched["null_quantiles"]["q95"],
            places=7,
        )

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
        self.assertEqual(results["protocol"]["global_mmd_reference_seed"], 10004)
        self.assertEqual(results["protocol"]["factorized_adapter"]["model_family"], "F-CS-WAE")
        self.assertIn(
            "effective_noise_rms_norm_mean",
            results["sampling_contract"]["style_posterior"],
        )

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

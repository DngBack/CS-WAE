"""Unit tests for clause-aligned FACT constraints."""

from __future__ import annotations

import unittest

import torch
import torch.nn.functional as F

from src.utils.loss_f_cs_wae import (
    audit_aligned_mean_hsic,
    conditional_product_hsic,
    fact_contract_constraints,
    mean_conditional_hsic_constraint,
    style_label_hsic,
)


class _PriorModel:
    n_classes = 2
    semantic_dim = 4

    def _sample_prior_z_c(self, class_idx, n, device):
        center = torch.zeros(self.semantic_dim, device=device)
        center[class_idx] = 1.0
        samples = 3.0 * center.unsqueeze(0) + 0.25 * torch.randn(
            n, self.semantic_dim, device=device
        )
        return F.normalize(samples, p=2, dim=1)


class FactContractLossTests(unittest.TestCase):
    def test_conditional_hsic_detects_changed_pairing(self):
        generator = torch.Generator().manual_seed(31)
        signal = torch.linspace(-3.0, 3.0, 96).unsqueeze(1)
        content = torch.cat(
            [signal, torch.randn(96, 3, generator=generator) * 0.05], dim=1
        )
        style = torch.cat(
            [signal, torch.randn(96, 5, generator=generator) * 0.05], dim=1
        )
        dependent = conditional_product_hsic(content, style)
        permutation = torch.randperm(96, generator=generator)
        independent = conditional_product_hsic(content, style[permutation])
        self.assertGreater(float(dependent), float(independent) + 1e-4)

    def test_style_only_dependence_blocks_content_gradient(self):
        torch.manual_seed(17)
        n = 48
        labels = torch.cat(
            [torch.zeros(n // 2, dtype=torch.long), torch.ones(n // 2, dtype=torch.long)]
        )
        shared = torch.linspace(-2.0, 2.0, n).unsqueeze(1)
        content = F.normalize(
            torch.cat([shared, torch.randn(n, 3) * 0.1], dim=1), p=2, dim=1
        ).detach().requires_grad_(True)
        style = torch.cat([shared, torch.randn(n, 5) * 0.1], dim=1)
        style = style.detach().requires_grad_(True)
        values = fact_contract_constraints(
            content,
            style,
            labels,
            _PriorModel(),
            style_only_dependence=True,
        )
        values.dependence.backward()
        self.assertIsNone(content.grad)
        self.assertIsNotNone(style.grad)
        self.assertTrue(torch.isfinite(style.grad).all())
        self.assertGreater(float(style.grad.norm()), 0.0)

    def test_fact_constraints_are_finite_and_nonnegative(self):
        torch.manual_seed(9)
        labels = torch.arange(40) % 2
        content = F.normalize(torch.randn(40, 4), p=2, dim=1).requires_grad_(True)
        style = torch.randn(40, 6, requires_grad=True)
        values = fact_contract_constraints(
            content, style, labels, _PriorModel(), style_only_dependence=False
        )
        self.assertEqual(values.represented_classes, 2)
        for value in (
            values.style,
            values.content,
            values.dependence,
            values.style_raw,
            values.content_raw,
            values.dependence_raw,
            values.style_null,
            values.content_null,
            values.dependence_null,
        ):
            self.assertTrue(torch.isfinite(value))
        self.assertGreaterEqual(float(values.style), 0.0)
        self.assertGreaterEqual(float(values.content), 0.0)
        self.assertGreaterEqual(float(values.dependence), 0.0)

    def test_multiple_matched_null_draws_are_supported(self):
        torch.manual_seed(23)
        labels = torch.arange(64) % 2
        content = F.normalize(torch.randn(64, 4), p=2, dim=1).requires_grad_(True)
        style = torch.randn(64, 6, requires_grad=True)
        values = fact_contract_constraints(
            content, style, labels, _PriorModel(), null_draws=4
        )
        total = values.style + values.content + values.dependence
        total.backward()
        self.assertTrue(torch.isfinite(total))
        self.assertIsNotNone(content.grad)
        self.assertIsNotNone(style.grad)

    def test_null_draw_count_must_be_positive(self):
        labels = torch.arange(8) % 2
        content = F.normalize(torch.randn(8, 4), p=2, dim=1)
        style = torch.randn(8, 6)
        with self.assertRaisesRegex(ValueError, "null_draws must be positive"):
            fact_contract_constraints(
                content, style, labels, _PriorModel(), null_draws=0
            )

    def test_style_label_hsic_detects_nonlinear_label_structure(self):
        torch.manual_seed(37)
        labels = torch.arange(160) % 2
        angle = torch.rand(160) * (2.0 * torch.pi)
        radius = 1.0 + 2.0 * labels.float()
        style = torch.stack([radius * angle.cos(), radius * angle.sin()], dim=1)
        observed = style_label_hsic(style, labels)
        permuted = style_label_hsic(style, labels[torch.randperm(labels.numel())])
        self.assertGreater(float(observed), float(permuted) + 1e-4)

    def test_label_dependence_constraint_backpropagates_to_style(self):
        torch.manual_seed(41)
        labels = torch.arange(64) % 2
        content = F.normalize(torch.randn(64, 4), p=2, dim=1)
        style = torch.randn(64, 6, requires_grad=True)
        values = fact_contract_constraints(
            content,
            style,
            labels,
            _PriorModel(),
            null_draws=4,
            label_dependence=True,
        )
        values.label_dependence.backward()
        self.assertIsNotNone(style.grad)
        self.assertTrue(torch.isfinite(style.grad).all())

    def test_audit_aligned_mean_hsic_is_style_scale_invariant(self):
        torch.manual_seed(47)
        content = F.normalize(torch.randn(32, 4), p=2, dim=1)
        style = torch.randn(32, 6)
        base = audit_aligned_mean_hsic(content, style)
        transformed = audit_aligned_mean_hsic(content, 3.0 * style + 2.0)
        self.assertTrue(torch.allclose(base, transformed, atol=1e-6, rtol=1e-5))

    def test_mean_dependence_constraint_updates_style_only(self):
        torch.manual_seed(53)
        labels = torch.arange(80) % 2
        content = F.normalize(torch.randn(80, 4), p=2, dim=1).requires_grad_()
        projection = torch.randn(4, 6)
        style = content.detach() @ projection + 0.05 * torch.randn(80, 6)
        style.requires_grad_()
        value, raw, null, represented = mean_conditional_hsic_constraint(
            content, style, labels, 2, null_draws=4, style_only=True
        )
        self.assertEqual(represented, 2)
        self.assertGreater(float(raw), float(null))
        value.backward()
        self.assertIsNone(content.grad)
        self.assertIsNotNone(style.grad)
        self.assertGreater(float(style.grad.abs().sum()), 0.0)

    def test_fact_mean_dependence_requires_posterior_means(self):
        labels = torch.arange(16) % 2
        content = F.normalize(torch.randn(16, 4), p=2, dim=1)
        style = torch.randn(16, 6)
        with self.assertRaisesRegex(ValueError, "requires mean_content"):
            fact_contract_constraints(
                content,
                style,
                labels,
                _PriorModel(),
                mean_dependence=True,
            )


if __name__ == "__main__":
    unittest.main()

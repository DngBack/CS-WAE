"""Unit tests for the factorized joint-contract training loss."""

from __future__ import annotations

import unittest

import torch

from src.utils.loss_f_cs_wae import joint_product_mmd


class JointContractLossTests(unittest.TestCase):
    def test_identical_banks_have_zero_biased_mmd(self):
        generator = torch.Generator().manual_seed(4)
        content = torch.randn(24, 5, generator=generator)
        style = torch.randn(24, 7, generator=generator)
        value = joint_product_mmd(content, style, content, style)
        self.assertAlmostEqual(float(value), 0.0, places=6)

    def test_product_kernel_detects_changed_pairing_with_fixed_marginals(self):
        values = torch.linspace(-2.5, 2.5, 64).unsqueeze(1)
        content = torch.cat([values, torch.ones_like(values)], dim=1)
        style = values.clone()
        permuted_style = style.flip(0)
        same = joint_product_mmd(content, style, content, style)
        changed = joint_product_mmd(content, style, content, permuted_style)
        self.assertGreater(float(changed), float(same) + 1e-3)

    def test_loss_backpropagates_to_both_posterior_blocks(self):
        generator = torch.Generator().manual_seed(8)
        q_content = torch.randn(16, 4, generator=generator, requires_grad=True)
        q_style = torch.randn(16, 6, generator=generator, requires_grad=True)
        p_content = torch.randn(16, 4, generator=generator)
        p_style = torch.randn(16, 6, generator=generator)
        value = joint_product_mmd(q_content, q_style, p_content, p_style)
        value.backward()
        self.assertTrue(torch.isfinite(q_content.grad).all())
        self.assertTrue(torch.isfinite(q_style.grad).all())
        self.assertGreater(float(q_content.grad.norm()), 0.0)
        self.assertGreater(float(q_style.grad.norm()), 0.0)


if __name__ == "__main__":
    unittest.main()

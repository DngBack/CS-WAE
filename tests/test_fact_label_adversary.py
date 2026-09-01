"""CPU tests for nonlinear style-label adversarial removal."""

from __future__ import annotations

import unittest

import torch
import torch.nn.functional as F

from src.trainers.trainer_f_cs_wae import (
    StyleLabelAdversary,
    uniform_label_confusion_loss,
)


class FactLabelAdversaryTests(unittest.TestCase):
    def test_adversary_detects_nonlinear_radius_labels(self):
        torch.manual_seed(4)
        style = torch.randn(256, 2)
        radius = style.square().sum(dim=1)
        labels = (radius > radius.median()).long()
        adversary = StyleLabelAdversary(2, 2, hidden_dims=(32, 16))
        optimizer = torch.optim.Adam(adversary.parameters(), lr=2e-2)
        for _ in range(120):
            loss = F.cross_entropy(adversary(style), labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        accuracy = (adversary(style).argmax(dim=1) == labels).float().mean()
        self.assertGreater(float(accuracy), 0.90)

    def test_confusion_moves_style_without_updating_frozen_adversary(self):
        torch.manual_seed(7)
        style = torch.randn(32, 5, requires_grad=True)
        adversary = StyleLabelAdversary(5, 3, hidden_dims=(8,))
        for parameter in adversary.parameters():
            parameter.requires_grad_(False)
        loss = uniform_label_confusion_loss(adversary(style))
        loss.backward()
        self.assertGreater(float(style.grad.abs().sum()), 0.0)
        self.assertTrue(all(parameter.grad is None for parameter in adversary.parameters()))

    def test_uniform_predictions_have_zero_confusion(self):
        logits = torch.zeros(12, 10)
        self.assertAlmostEqual(float(uniform_label_confusion_loss(logits)), 0.0, places=7)


if __name__ == "__main__":
    unittest.main()

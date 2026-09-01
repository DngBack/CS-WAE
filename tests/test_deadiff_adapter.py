"""Dependency-free tests for the DEADiff audit boundary adapter."""

from __future__ import annotations

from types import SimpleNamespace
import unittest

import torch
import torch.nn as nn

from src.metrics.deadiff_adapter import DEADiffAuditAdapter


class _FakeBlip(nn.Module):
    def __init__(self, offset: float):
        super().__init__()
        self.offset = offset

    def extract_features(self, batch, mode):
        assert mode == "multimodal"
        images = batch["image"]
        pooled = images.mean(dim=(1, 2, 3), keepdim=False)
        embeds = pooled[:, None, None].expand(-1, 2, 3) + self.offset
        return SimpleNamespace(multimodal_embeds=embeds)


class _FakeTextEncoder:
    def encode(self, text):
        return torch.arange(len(text) * 12, dtype=torch.float32).reshape(len(text), 4, 3)


class _FakeDEADiff:
    def __init__(self):
        self.style_blip = _FakeBlip(10.0)
        self.content_blip = _FakeBlip(20.0)
        self.style_proj_layer = nn.Identity()
        self.content_proj_layer = nn.Identity()
        self.cond_stage_model = _FakeTextEncoder()
        self.clip_mean = (0.0, 0.0, 0.0)
        self.clip_std = (1.0, 1.0, 1.0)


class DEADiffAdapterTests(unittest.TestCase):
    def test_two_donor_images_use_separate_branches(self):
        adapter = DEADiffAuditAdapter(_FakeDEADiff())
        style_images = torch.full((2, 3, 16, 16), -1.0)
        content_images = torch.full((2, 3, 16, 16), 1.0)
        views = adapter.encode_views(
            style_images=style_images, content_images=content_images
        )
        self.assertEqual(views.style.shape, (2, 2, 3))
        self.assertTrue(torch.allclose(views.style, torch.full_like(views.style, 10.0)))
        self.assertTrue(torch.allclose(views.content, torch.full_like(views.content, 21.0)))

    def test_recombination_and_conditioning_preserve_branch_order(self):
        adapter = DEADiffAuditAdapter(_FakeDEADiff())
        style_images = torch.stack(
            [torch.full((3, 8, 8), -1.0), torch.full((3, 8, 8), 1.0)]
        )
        content_images = torch.stack(
            [torch.full((3, 8, 8), -0.5), torch.full((3, 8, 8), 0.5)]
        )
        views = adapter.encode_views(
            style_images=style_images, content_images=content_images
        )
        swapped = adapter.recombine(
            views,
            style_indices=torch.tensor([1, 0]),
            content_indices=torch.tensor([0, 1]),
        )
        conditioning = adapter.conditioning(swapped, ["first", "second"])
        self.assertTrue(torch.equal(conditioning[0][0], views.style[[1, 0]]))
        self.assertTrue(torch.equal(conditioning[1][0], views.content))
        self.assertTrue(torch.equal(conditioning[0][1], conditioning[1][1]))

    def test_unknown_model_is_rejected_explicitly(self):
        with self.assertRaisesRegex(TypeError, "missing"):
            DEADiffAuditAdapter(nn.Linear(3, 3))


if __name__ == "__main__":
    unittest.main()

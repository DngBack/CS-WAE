"""Audit adapter for the official DEADiff dual-branch conditioner.

The adapter intentionally uses duck typing and does not import DEADiff.  This
keeps the core audit environment independent of the legacy inference stack
while making the exact boundary explicit: separate donor images are an audit
intervention, because official DEADiff inference feeds one ``inp_image`` to
both Q-Former branches.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class DEADiffBranchEmbeddings:
    style: torch.Tensor
    content: torch.Tensor

    def validate(self) -> None:
        if self.style.ndim != 3 or self.content.ndim != 3:
            raise ValueError("DEADiff branch embeddings must have shape [batch, tokens, width]")
        if self.style.shape != self.content.shape:
            raise ValueError("DEADiff style/content branch shapes must match")


class DEADiffAuditAdapter:
    """Expose DEADiff style/content embeddings and controlled recombinations."""

    REQUIRED_ATTRIBUTES = (
        "style_blip",
        "content_blip",
        "style_proj_layer",
        "content_proj_layer",
        "cond_stage_model",
        "clip_mean",
        "clip_std",
    )

    def __init__(self, model) -> None:
        missing = [name for name in self.REQUIRED_ATTRIBUTES if not hasattr(model, name)]
        if missing:
            raise TypeError(f"not a compatible DEADiff model; missing {missing}")
        self.model = model

    def preprocess_reference(self, images: torch.Tensor) -> torch.Tensor:
        """Match official 224px CLIP preprocessing for inputs in [-1, 1]."""

        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("reference images must have shape [batch, 3, height, width]")
        images = F.interpolate(
            (images + 1.0) / 2.0,
            size=(224, 224),
            mode="bicubic",
            align_corners=True,
            antialias=True,
        )
        mean = torch.as_tensor(
            self.model.clip_mean, device=images.device, dtype=images.dtype
        ).reshape(1, 3, 1, 1)
        std = torch.as_tensor(
            self.model.clip_std, device=images.device, dtype=images.dtype
        ).reshape(1, 3, 1, 1)
        return (images - mean) / std

    @staticmethod
    def _encode_branch(images, text: Sequence[str], blip, projection) -> torch.Tensor:
        features = blip.extract_features(
            {"image": images, "text_input": list(text)}, mode="multimodal"
        ).multimodal_embeds
        return projection(features)

    @torch.no_grad()
    def encode_views(
        self,
        *,
        style_images: torch.Tensor,
        content_images: torch.Tensor,
        style_subject: Sequence[str] | str = "style",
        content_subject: Sequence[str] | str = "content",
    ) -> DEADiffBranchEmbeddings:
        """Encode two donor banks independently through the official branches."""

        if style_images.shape[0] != content_images.shape[0]:
            raise ValueError("style and content donor batches must have equal size")
        batch = style_images.shape[0]
        if isinstance(style_subject, str):
            style_subject = [style_subject] * batch
        if isinstance(content_subject, str):
            content_subject = [content_subject] * batch
        if len(style_subject) != batch or len(content_subject) != batch:
            raise ValueError("one subject instruction is required per donor")
        style = self._encode_branch(
            self.preprocess_reference(style_images),
            style_subject,
            self.model.style_blip,
            self.model.style_proj_layer,
        )
        content = self._encode_branch(
            self.preprocess_reference(content_images),
            content_subject,
            self.model.content_blip,
            self.model.content_proj_layer,
        )
        result = DEADiffBranchEmbeddings(style=style, content=content)
        result.validate()
        return result

    @staticmethod
    def recombine(
        embeddings: DEADiffBranchEmbeddings,
        *,
        style_indices: torch.Tensor | None = None,
        content_indices: torch.Tensor | None = None,
    ) -> DEADiffBranchEmbeddings:
        """Create a deterministic donor recombination without re-encoding."""

        embeddings.validate()
        batch = embeddings.style.shape[0]
        device = embeddings.style.device
        if style_indices is None:
            style_indices = torch.arange(batch, device=device)
        if content_indices is None:
            content_indices = torch.arange(batch, device=device)
        if style_indices.shape != (batch,) or content_indices.shape != (batch,):
            raise ValueError("donor indices must have shape [batch]")
        result = DEADiffBranchEmbeddings(
            style=embeddings.style[style_indices],
            content=embeddings.content[content_indices],
        )
        result.validate()
        return result

    def conditioning(
        self,
        embeddings: DEADiffBranchEmbeddings,
        target_text: Sequence[str],
    ) -> list[list[torch.Tensor]]:
        """Build the two-entry conditioning object routed by DEADiff's UNet."""

        embeddings.validate()
        if len(target_text) != embeddings.style.shape[0]:
            raise ValueError("one target prompt is required per embedding pair")
        text_states = self.model.cond_stage_model.encode(list(target_text))
        return [
            [embeddings.style, text_states],
            [embeddings.content, text_states],
        ]

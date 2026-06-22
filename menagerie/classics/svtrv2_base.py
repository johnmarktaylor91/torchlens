"""SVTRv2-Base: CTC scene-text recognition with FRM and semantic guidance.

Paper: Du et al. 2024/2025, "SVTRv2: CTC Beats Encoder-Decoder Models in Scene
Text Recognition".  The compact model keeps patch embedding, feature
rearrangement for CTC alignment, transformer visual modeling, and an SGM branch.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SVTRv2Base(nn.Module):
    """Compact SVTRv2 text recognizer."""

    def __init__(self, classes: int = 32, dim: int = 48) -> None:
        """Initialize visual transformer, FRM, SGM, and CTC classifier.

        Parameters
        ----------
        classes:
            CTC vocabulary size.
        dim:
            Hidden feature width.
        """

        super().__init__()
        self.patch = nn.Conv2d(1, dim, kernel_size=(4, 4), stride=(4, 4))
        self.frm_gate = nn.Conv2d(dim, 2, kernel_size=1)
        layer = nn.TransformerEncoderLayer(dim, nhead=4, dim_feedforward=96, batch_first=True)
        self.visual = nn.TransformerEncoder(layer, num_layers=2)
        self.semantic = nn.GRU(dim, dim // 2, batch_first=True, bidirectional=True)
        self.ctc = nn.Linear(dim, classes)

    def _feature_rearrange(self, feat: torch.Tensor) -> torch.Tensor:
        """Rearrange visual features into CTC-friendly reading order.

        Parameters
        ----------
        feat:
            Patch feature map.

        Returns
        -------
        torch.Tensor
            Feature sequence aligned for CTC.
        """

        horizontal = feat.mean(dim=2).transpose(1, 2)
        vertical = feat.mean(dim=3).transpose(1, 2)
        vertical = torch.nn.functional.interpolate(
            vertical.transpose(1, 2),
            size=horizontal.shape[1],
            mode="linear",
            align_corners=False,
        ).transpose(1, 2)
        gate = torch.softmax(self.frm_gate(feat).mean(dim=(2, 3)), dim=-1)
        return gate[:, 0:1, None] * horizontal + gate[:, 1:2, None] * vertical

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Recognize a text-line image as CTC logits.

        Parameters
        ----------
        image:
            Grayscale text-line image.

        Returns
        -------
        torch.Tensor
            CTC logits with shape ``(batch, width_steps, classes)``.
        """

        msr = torch.nn.functional.interpolate(
            image, size=(32, 96), mode="bilinear", align_corners=False
        )
        feat = self.patch(msr)
        seq = self._feature_rearrange(feat)
        seq = self.visual(seq)
        sem, _ = self.semantic(seq)
        return self.ctc(seq + sem)


def build() -> nn.Module:
    """Build compact SVTRv2-Base.

    Returns
    -------
    nn.Module
        Random-initialized SVTRv2 module.
    """

    return SVTRv2Base()


def example_input() -> torch.Tensor:
    """Create a small grayscale text-line image.

    Returns
    -------
    torch.Tensor
        Image tensor with shape ``(1, 1, 32, 96)``.
    """

    return torch.randn(1, 1, 32, 96)


MENAGERIE_ENTRIES = [("SVTRv2-Base", "build", "example_input", "2024", "E6")]

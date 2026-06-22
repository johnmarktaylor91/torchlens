"""PPSeg BiSeNet alias: bilateral real-time semantic segmentation.

This compact `ppseg` spelling mirrors PaddleSeg's BiSeNet target: spatial path,
context path, attention refinement, and feature-fusion prediction.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from menagerie.classics.paddleseg_bisenet import BiSeNet


def build() -> nn.Module:
    """Build compact PPSeg BiSeNet.

    Returns
    -------
    nn.Module
        Random-initialized BiSeNet.
    """

    return BiSeNet()


def example_input() -> torch.Tensor:
    """Create a small RGB segmentation image.

    Returns
    -------
    torch.Tensor
        Image tensor with shape ``(1, 3, 32, 32)``.
    """

    return torch.randn(1, 3, 32, 32)


MENAGERIE_ENTRIES = [("ppseg_bisenet", "build", "example_input", "2018", "E5")]

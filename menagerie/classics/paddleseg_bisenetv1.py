"""PaddleSeg BiSeNetV1: spatial/context-path bilateral segmentation.

This target corresponds to the original BiSeNet v1 architecture in PaddleSeg,
with a spatial path, a fast-downsampled context path, attention refinement, and
feature fusion.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from menagerie.classics.paddleseg_bisenet import BiSeNet


def build() -> nn.Module:
    """Build compact PaddleSeg BiSeNetV1.

    Returns
    -------
    nn.Module
        Random-initialized BiSeNetV1.
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


MENAGERIE_ENTRIES = [("paddleseg_bisenetv1", "build", "example_input", "2018", "E5")]

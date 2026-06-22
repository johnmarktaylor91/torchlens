"""PPSeg SegFormer alias: hierarchical transformer semantic segmentation."""

from __future__ import annotations

import torch
import torch.nn as nn

from menagerie.classics.paddleseg_segformer import SegFormer


def build() -> nn.Module:
    """Build compact PPSeg SegFormer.

    Returns
    -------
    nn.Module
        Random-initialized SegFormer.
    """

    return SegFormer()


def example_input() -> torch.Tensor:
    """Create a small RGB segmentation image.

    Returns
    -------
    torch.Tensor
        Image tensor with shape ``(1, 3, 32, 32)``.
    """

    return torch.randn(1, 3, 32, 32)


MENAGERIE_ENTRIES = [("ppseg_segformer", "build", "example_input", "2021", "E5")]

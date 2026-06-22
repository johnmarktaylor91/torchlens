"""PPSeg PP-LiteSeg alias: real-time SPPM/UAFM segmentation."""

from __future__ import annotations

import torch
import torch.nn as nn

from menagerie.classics.paddleseg_pp_liteseg import PPLiteSeg


def build() -> nn.Module:
    """Build compact PPSeg PP-LiteSeg.

    Returns
    -------
    nn.Module
        Random-initialized PP-LiteSeg.
    """

    return PPLiteSeg()


def example_input() -> torch.Tensor:
    """Create a small RGB segmentation image.

    Returns
    -------
    torch.Tensor
        Image tensor with shape ``(1, 3, 32, 32)``.
    """

    return torch.randn(1, 3, 32, 32)


MENAGERIE_ENTRIES = [("ppseg_pp_liteseg", "build", "example_input", "2022", "E5")]

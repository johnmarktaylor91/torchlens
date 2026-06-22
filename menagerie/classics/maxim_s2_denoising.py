"""MAXIM compact denoising variant.

This module reuses the compact MAXIM restoration backbone for the denoising
target; MAXIM uses the same multi-axis MLP U-Net family across denoising and
deblurring tasks, with task-specific training/checkpoints rather than a distinct
inference primitive.
"""

from __future__ import annotations

from torch import Tensor, nn

from .maxim_s1_deblurring import MAXIMRestorer
from .maxim_s1_deblurring import example_input as _example_input


def build() -> nn.Module:
    """Build compact MAXIM denoising model.

    Returns
    -------
    nn.Module
        Random-initialized MAXIM restorer.
    """
    return MAXIMRestorer().eval()


def example_input() -> Tensor:
    """Return a small noisy RGB image.

    Returns
    -------
    Tensor
        Tensor with shape ``(1, 3, 32, 32)``.
    """
    return _example_input()


MENAGERIE_ENTRIES = [
    ("maxim_s2_denoising", "build", "example_input", "2022", "DC"),
]

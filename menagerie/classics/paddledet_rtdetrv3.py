"""RT-DETRv3 compact PaddleDetection-style detector."""

from __future__ import annotations

from torch import Tensor, nn

from .paddledet_deim import PaddleDEIM
from .paddledet_deim import example_input as _example_input


def build() -> nn.Module:
    """Build compact RT-DETRv3.

    Returns
    -------
    nn.Module
        Random-initialized RT-DETRv3-style detector.
    """
    return PaddleDEIM(dim=56, queries=24, classes=20).eval()


def example_input() -> Tensor:
    """Return a small RGB image.

    Returns
    -------
    Tensor
        Tensor with shape ``(1, 3, 64, 64)``.
    """
    return _example_input()


MENAGERIE_ENTRIES = [
    ("paddledet_rtdetrv3", "build", "example_input", "2025", "DC"),
]

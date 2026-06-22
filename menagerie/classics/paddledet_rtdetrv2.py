"""RT-DETRv2 compact PaddleDetection-style detector."""

from __future__ import annotations

from torch import Tensor, nn

from .paddledet_deim import PaddleDEIM
from .paddledet_deim import example_input as _example_input


def build() -> nn.Module:
    """Build compact RT-DETRv2.

    Returns
    -------
    nn.Module
        Random-initialized RT-DETRv2-style detector.
    """
    return PaddleDEIM(dim=48, queries=28, classes=20).eval()


def example_input() -> Tensor:
    """Return a small RGB image.

    Returns
    -------
    Tensor
        Tensor with shape ``(1, 3, 64, 64)``.
    """
    return _example_input()


MENAGERIE_ENTRIES = [
    ("paddledet_rtdetrv2", "build", "example_input", "2024", "DC"),
]

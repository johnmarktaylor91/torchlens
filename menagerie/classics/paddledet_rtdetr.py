"""RT-DETR compact PaddleDetection-style real-time DETR.

RT-DETR uses efficient hybrid encoder features and a transformer decoder with
object queries for real-time end-to-end detection.  This compact reconstruction
shares the same inference primitive as the DEIM deployment graph.
"""

from __future__ import annotations

from torch import Tensor, nn

from .paddledet_deim import PaddleDEIM
from .paddledet_deim import example_input as _example_input


def build() -> nn.Module:
    """Build compact RT-DETR.

    Returns
    -------
    nn.Module
        Random-initialized RT-DETR-style detector.
    """
    return PaddleDEIM().eval()


def example_input() -> Tensor:
    """Return a small RGB image.

    Returns
    -------
    Tensor
        Tensor with shape ``(1, 3, 64, 64)``.
    """
    return _example_input()


MENAGERIE_ENTRIES = [
    ("paddledet_rtdetr", "build", "example_input", "2023", "DC"),
]

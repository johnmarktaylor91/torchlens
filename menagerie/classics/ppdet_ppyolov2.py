"""ppdet_ppyolov2 exact-name classic.

Paper: PP-YOLOv2: A Practical Object Detector.

This exact-name module exposes the existing compact PP-YOLOv2 reconstruction:
R50vd-DCN-style features, Mish/DropBlock-like regularization, PAN aggregation,
and IoU-aware YOLO heads.
"""

from __future__ import annotations

from torch import Tensor, nn

from menagerie.classics.ppyolov2_r50vd_dcn import build as _build
from menagerie.classics.ppyolov2_r50vd_dcn import example_input as _example_input


def build() -> nn.Module:
    """Build a compact random-init PP-YOLOv2 detector.

    Returns
    -------
    nn.Module
        Dependency-free PP-YOLOv2-style detector.
    """

    return _build()


def example_input() -> Tensor:
    """Return a small RGB image.

    Returns
    -------
    Tensor
        Input tensor for a compact detection trace.
    """

    return _example_input()


MENAGERIE_ENTRIES = [("ppdet_ppyolov2", "build", "example_input", "2021", "DET")]

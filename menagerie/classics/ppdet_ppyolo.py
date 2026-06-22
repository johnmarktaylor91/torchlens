"""ppdet_ppyolo exact-name classic.

Paper: PP-YOLO: An Effective and Efficient Implementation of Object Detector.

The compact implementation is the existing dependency-free PP-YOLO
R50vd-DCN reconstruction: vd-style stem, deformable late-stage sampling,
CoordConv, SPP/FPN fusion, and IoU-aware YOLO prediction heads.
"""

from __future__ import annotations

from torch import Tensor, nn

from menagerie.classics.ppyolo_r50vd_dcn import build as _build
from menagerie.classics.ppyolo_r50vd_dcn import example_input as _example_input


def build() -> nn.Module:
    """Build a compact random-init PP-YOLO detector.

    Returns
    -------
    nn.Module
        Dependency-free PP-YOLO-style detector.
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


MENAGERIE_ENTRIES = [("ppdet_ppyolo", "build", "example_input", "2020", "DET")]

"""ppdet_fcos exact-name classic.

Paper: FCOS: Fully Convolutional One-Stage Object Detection.

The imported compact model keeps anchor-free per-pixel class, box-distance, and
centerness prediction heads.
"""

from __future__ import annotations

from menagerie.classics.paddledet_fcos import build as build
from menagerie.classics.paddledet_fcos import example_input as example_input

__all__ = ["MENAGERIE_ENTRIES", "build", "example_input"]

MENAGERIE_ENTRIES = [("ppdet_fcos", "build", "example_input", "2019", "DET")]

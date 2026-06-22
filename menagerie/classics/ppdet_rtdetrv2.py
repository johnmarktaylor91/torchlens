"""ppdet_rtdetrv2 exact-name classic.

Paper: RT-DETRv2.

The imported compact model keeps the RT-DETR real-time transformer detector
structure with improved encoder/decoder selection primitives.
"""

from __future__ import annotations

from menagerie.classics.paddledet_rtdetrv2 import build as build
from menagerie.classics.paddledet_rtdetrv2 import example_input as example_input

__all__ = ["MENAGERIE_ENTRIES", "build", "example_input"]

MENAGERIE_ENTRIES = [("ppdet_rtdetrv2", "build", "example_input", "2024", "DET")]

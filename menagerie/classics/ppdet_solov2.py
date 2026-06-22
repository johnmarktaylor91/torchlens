"""ppdet_solov2 exact-name classic.

Paper: SOLOv2: Dynamic and Fast Instance Segmentation.

The imported compact model keeps grid-based category prediction and dynamic
mask-kernel generation.
"""

from __future__ import annotations

from menagerie.classics.paddledet_solov2 import build_paddledet_solov2 as build
from menagerie.classics.paddledet_solov2 import example_input as example_input

__all__ = ["MENAGERIE_ENTRIES", "build", "example_input"]

MENAGERIE_ENTRIES = [("ppdet_solov2", "build", "example_input", "2020", "DET")]

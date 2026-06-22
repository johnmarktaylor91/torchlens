"""ppdet_clrnet exact-name classic.

Paper: CLRNet: Cross Layer Refinement Network for Lane Detection.

The imported compact model keeps lane priors, cross-layer feature refinement,
and lane-coordinate heads from the PaddleDetection CLRNet reconstruction.
"""

from __future__ import annotations

from menagerie.classics.paddledet_clrnet import build as build
from menagerie.classics.paddledet_clrnet import example_input as example_input

__all__ = ["MENAGERIE_ENTRIES", "build", "example_input"]

MENAGERIE_ENTRIES = [("ppdet_clrnet", "build", "example_input", "2022", "DET")]

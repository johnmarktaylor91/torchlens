"""ppdet_gfl exact-name classic.

Paper: Generalized Focal Loss.

The imported compact model keeps quality focal classification and distribution
box regression from the existing PaddleDetection GFL reconstruction.
"""

from __future__ import annotations

from menagerie.classics.paddledet_gfl import build as build
from menagerie.classics.paddledet_gfl import example_input as example_input

__all__ = ["MENAGERIE_ENTRIES", "build", "example_input"]

MENAGERIE_ENTRIES = [("ppdet_gfl", "build", "example_input", "2020", "DET")]

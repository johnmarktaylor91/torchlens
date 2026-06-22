"""ppdet_retinanet exact-name classic.

Paper: Focal Loss for Dense Object Detection.

The imported compact model keeps FPN features, anchor heads, and focal-loss
style dense classification/regression primitives.
"""

from __future__ import annotations

from menagerie.classics.paddledet_retinanet import build as build
from menagerie.classics.paddledet_retinanet import example_input as example_input

__all__ = ["MENAGERIE_ENTRIES", "build", "example_input"]

MENAGERIE_ENTRIES = [("ppdet_retinanet", "build", "example_input", "2017", "DET")]

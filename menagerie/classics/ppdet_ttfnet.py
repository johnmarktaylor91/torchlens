"""ppdet_ttfnet exact-name classic.

Paper: Training-Time-Friendly Network for Real-Time Object Detection.

The imported compact model keeps center heatmap and box regression heads used by
TTFNet's anchor-free detector.
"""

from __future__ import annotations

from menagerie.classics.paddledet_ttfnet import build_paddledet_ttfnet as build
from menagerie.classics.paddledet_ttfnet import example_input as example_input

__all__ = ["MENAGERIE_ENTRIES", "build", "example_input"]

MENAGERIE_ENTRIES = [("ppdet_ttfnet", "build", "example_input", "2019", "DET")]

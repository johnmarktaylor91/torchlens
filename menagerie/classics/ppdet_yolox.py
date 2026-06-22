"""ppdet_yolox exact-name classic.

Paper: YOLOX: Exceeding YOLO Series in 2021.

The imported compact model keeps YOLOX's decoupled anchor-free head and PAN
feature fusion.
"""

from __future__ import annotations

from menagerie.classics.paddledet_yolox import build as build
from menagerie.classics.paddledet_yolox import example_input as example_input

__all__ = ["MENAGERIE_ENTRIES", "build", "example_input"]

MENAGERIE_ENTRIES = [("ppdet_yolox", "build", "example_input", "2021", "DET")]

"""ppdet_rtdetr exact-name classic.

Paper: RT-DETR: DETRs Beat YOLOs on Real-time Object Detection.

The imported compact model keeps hybrid encoder features, IoU-aware query
selection, and transformer decoder detection heads.
"""

from __future__ import annotations

from menagerie.classics.paddledet_rtdetr import build as build
from menagerie.classics.paddledet_rtdetr import example_input as example_input

__all__ = ["MENAGERIE_ENTRIES", "build", "example_input"]

MENAGERIE_ENTRIES = [("ppdet_rtdetr", "build", "example_input", "2023", "DET")]

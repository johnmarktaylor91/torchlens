"""ppdet_picodet exact-name classic.

Paper: PP-PicoDet: A Better Real-Time Object Detector on Mobile Devices.

The imported compact model keeps ESNet-style lightweight depthwise blocks,
CSP-PAN fusion, and GFL-style distribution regression.
"""

from __future__ import annotations

from menagerie.classics.paddledet_picodet import build as build
from menagerie.classics.paddledet_picodet import example_input as example_input

__all__ = ["MENAGERIE_ENTRIES", "build", "example_input"]

MENAGERIE_ENTRIES = [("ppdet_picodet", "build", "example_input", "2021", "DET")]

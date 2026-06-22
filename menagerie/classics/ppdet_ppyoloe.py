"""ppdet_ppyoloe exact-name classic.

Paper: PP-YOLOE: An Evolved Version of YOLO.

The imported compact model keeps CSPRep blocks, PAN fusion, and anchor-free
decoupled class/box/objectness heads.
"""

from __future__ import annotations

from menagerie.classics.paddledet_ppyoloe import build as build
from menagerie.classics.paddledet_ppyoloe import example_input as example_input

__all__ = ["MENAGERIE_ENTRIES", "build", "example_input"]

MENAGERIE_ENTRIES = [("ppdet_ppyoloe", "build", "example_input", "2022", "DET")]

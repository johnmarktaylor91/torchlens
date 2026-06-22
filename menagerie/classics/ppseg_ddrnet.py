"""PaddleSeg DDRNet alias module.

This module exposes the compact DDRNet reconstruction under the ``ppseg``
catalog name used by older PaddleSeg recipes.
"""

from __future__ import annotations

from menagerie.classics.paddleseg_ddrnet import build, example_input


MENAGERIE_ENTRIES = [
    ("ppseg_ddrnet", "build", "example_input", "2022", "DC"),
]

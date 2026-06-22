"""PaddleSeg CCNet alias module.

This module exposes the same compact CCNet reconstruction under the ``ppseg``
catalog name used by older PaddleSeg recipes.
"""

from __future__ import annotations

from menagerie.classics.paddleseg_ccnet import build, example_input


MENAGERIE_ENTRIES = [
    ("ppseg_ccnet", "build", "example_input", "2019", "DC"),
]

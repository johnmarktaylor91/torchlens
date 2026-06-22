"""Compatibility module for Allegro Module R6."""

from __future__ import annotations

from torch import nn

from .allegro import build_module_r6 as build
from .allegro import example_input


MENAGERIE_ENTRIES = [
    ("allegro_Allegro_Module_R6", "build", "example_input", "2024", "DE"),
]

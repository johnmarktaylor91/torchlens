"""Alias module for OpenPI π0-FAST.

Paper: Physical Intelligence 2025, "FAST: Efficient Robot Action
Tokenization" / OpenPI π0-FAST.
"""

from __future__ import annotations

from menagerie.classics.openpi_pi0_torchvla import build_fast as build
from menagerie.classics.openpi_pi0_torchvla import example_fast_input as example_input

MENAGERIE_ENTRIES = [("openpi_pi0_fast_jax", "build", "example_input", "2025", "E7")]

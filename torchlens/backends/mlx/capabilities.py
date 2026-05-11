"""Capability flags for the technical-preview MLX backend."""

from __future__ import annotations

supports_backward_capture = False
supports_fastlog = False
supports_intervention = False
supports_rng_replay = False
supports_compile_capture = False

__all__ = [
    "supports_backward_capture",
    "supports_compile_capture",
    "supports_fastlog",
    "supports_intervention",
    "supports_rng_replay",
]

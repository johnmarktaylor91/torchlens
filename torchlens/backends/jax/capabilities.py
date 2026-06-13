"""Capability flags for the jaxpr-first JAX backend preview."""

from __future__ import annotations

supports_backward_capture = False
supports_fastlog = False
supports_intervention = False
supports_rng_replay = False
supports_validation_replay = True

__all__ = [
    "supports_backward_capture",
    "supports_fastlog",
    "supports_intervention",
    "supports_rng_replay",
    "supports_validation_replay",
]

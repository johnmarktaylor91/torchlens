"""Compatibility shim for the relocated Bundle implementation."""

from __future__ import annotations

from torchlens.bundle import AmbiguousLabelError

from ..bundle import Bundle

__all__ = ["AmbiguousLabelError", "Bundle"]

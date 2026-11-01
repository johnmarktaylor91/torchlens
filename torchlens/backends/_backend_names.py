"""The canonical default-backend name, split out of :mod:`torchlens.backends.registry`.

A leaf module so modules that only compare against the torch backend name do not
import the registry; the registry re-exports the constant.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from .registry import BackendName

TORCH_BACKEND_NAME: Final[BackendName] = "torch"
"""Canonical registry name of the default torch backend.

Public (non-``backends``) code that branches on backend identity must compare
against this constant, never a hard-coded literal — the backend-literal gate
in ``tests/test_backend_registry.py`` enforces it.
"""

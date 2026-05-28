"""Input auto-routing registry namespace."""

from __future__ import annotations

from torchlens.autoroute._registry import Detector, Registry

_registry = Registry(kind="input")

register = _registry.register
unregister = _registry.unregister
iter_by_priority = _registry.iter_by_priority
list = _registry.list
info = _registry.info
snapshot = _registry.snapshot

from torchlens.autoroute import _builtin_input as _builtin_input  # noqa: E402,F401

__all__ = [
    "Detector",
    "Registry",
    "info",
    "iter_by_priority",
    "list",
    "register",
    "snapshot",
    "unregister",
]

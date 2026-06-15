"""Output auto-routing registry namespace."""

from __future__ import annotations

from torchlens.autoroute._registry import Detector, Registry

_registry = Registry(kind="output")

register = _registry.register
unregister = _registry.unregister
iter_by_priority = _registry.iter_by_priority
list = _registry.list
info = _registry.info
snapshot = _registry.snapshot

from torchlens.autoroute import _builtin_output as _builtin_output  # noqa: E402,F401

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

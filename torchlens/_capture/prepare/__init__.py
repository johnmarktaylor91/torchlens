"""PREPARE stage of torch capture: request resolution, cache and dispatch.

The stage entry points are re-exported lazily: ``run_entry`` imports
``prepare.resolve`` at import time and ``door`` imports ``run_entry``, so an
eager ``door`` import from this package module would form an import cycle.
"""

from __future__ import annotations

import importlib
from typing import Any

__tl_layer__ = "L4"

#: Attribute name -> (home module, attribute) resolved on first access.
_LAZY_ATTRS: dict[str, tuple[str, str]] = {
    "prepare_capture": ("torchlens._capture.prepare.door", "prepare_capture"),
    "_trace_torch_model": ("torchlens._capture.prepare.door", "_trace_torch_model"),
    "_dispatch_trace": ("torchlens._capture.prepare.dispatch", "_dispatch_trace"),
    "_refuse_trace_entry_shapes": (
        "torchlens._capture.prepare.dispatch",
        "_refuse_trace_entry_shapes",
    ),
    "_trace_mlx_model_from_public_kwargs": (
        "torchlens._capture.prepare.dispatch",
        "_trace_mlx_model_from_public_kwargs",
    ),
    "CAPTURE_CACHE_KEY_CURATED": ("torchlens._capture.prepare.cache", "CAPTURE_CACHE_KEY_CURATED"),
    "CAPTURE_CACHE_KEY_NEUTRAL": ("torchlens._capture.prepare.cache", "CAPTURE_CACHE_KEY_NEUTRAL"),
}


def __getattr__(name: str) -> Any:
    """Resolve a stage entry point from its home module on first access."""

    try:
        module_name, attribute = _LAZY_ATTRS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    value = getattr(importlib.import_module(module_name), attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Advertise the lazy names alongside the module globals."""

    return sorted(set(globals()) | set(_LAZY_ATTRS))

"""Leaf registry for the torch-only providers ``BackwardPass`` reads.

``BackwardPass.aten_ops`` reads the ATen calls the torch backward census
recorded, and ``flops_attribution`` attributes them; both providers live in
modules that import torch and the capture stack. Importing them from
``backward_pass.py`` tied every backward module into the package's largest
import cycle, so each provider registers itself here when its own module is
imported, and ``BackwardPass`` looks it up by name. This module imports
nothing from torchlens.

Fill guarantees: ``recorded_ops`` registers with the census module that is the
only writer of the records it reads, so an absent provider means nothing was
recorded. The attribution functions register with
``report._backward_attribution``, which the census window imports as it opens
(a deferred import: the report package imports the capture stack) and loading
a trace imports (its executed-FLOPs record check), so a pass that has recorded
ops or a saved record always finds them; a lookup that still misses raises
``KeyError`` naming the provider rather than returning nothing.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

_PROVIDERS: dict[str, Callable[..., Any]] = {}


def register_backward_provider(name: str, provider: Callable[..., Any]) -> None:
    """Register ``provider`` under ``name`` (called once at its module's import)."""

    _PROVIDERS[name] = provider


def backward_provider(name: str) -> Callable[..., Any]:
    """Return the provider registered under ``name``; ``KeyError`` if none is."""

    return _PROVIDERS[name]


def optional_backward_provider(name: str) -> Callable[..., Any] | None:
    """Return the provider registered under ``name``, or ``None``."""

    return _PROVIDERS.get(name)

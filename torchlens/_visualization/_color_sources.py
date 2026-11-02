"""Resolve the non-string ``color_by=`` sources of ``Trace.draw``.

``color_by`` takes a field name or callable (resolved by ``_encoding``), a
mapping of layer label or module address to value (an "overlay"; this module
turns it into a named callable), or an attribution result or an iterable of
them (resolved by ``torchlens.attribution._viz``).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ._encoding_request import EncodingChannelRequest


def _attribution_kind(value: Any) -> str | None:
    """Return the ``color_by`` kind an attribution result declares, else ``None``.

    ``Attribution`` declares ``"attribution"`` and ``AttributionGraph``
    declares ``"graph"`` (with a ``_color_by_source`` method beside it), so
    this module reads them without importing ``torchlens.attribution``: that
    package imports the core, and an import from here would pull all of it
    into the core's import cycle.
    """

    kind = getattr(type(value), "_color_by_kind", None)
    return kind if isinstance(kind, str) else None


def _is_attribution_source(source: Any) -> bool:
    """Return whether ``source`` is an attribution result or an iterable of them."""

    if _attribution_kind(source) is not None:
        return True
    if isinstance(source, (str, bytes, Mapping)) or callable(source):
        return False
    if isinstance(source, (list, tuple, set, frozenset)):
        return bool(source) and all(_attribution_kind(item) == "attribution" for item in source)
    return False


def resolve_color_source(trace: Any, color_by: Any) -> Any:
    """Turn a mapping or attribution ``color_by`` into a named callable.

    Field names, builtin tokens, callables and channel requests pass through
    unchanged; the encoding resolver validates them.
    """

    if color_by is None or isinstance(color_by, (str, EncodingChannelRequest)):
        return color_by
    if callable(color_by):
        return color_by
    if isinstance(color_by, Mapping):
        return _overlay_callable(trace, color_by)
    if _is_attribution_source(color_by):
        result = color_by if _attribution_kind(color_by) is not None else next(iter(color_by))
        return EncodingChannelRequest(
            source=result._color_by_source(trace, color_by), display_name="attribution"
        )
    return color_by


def _overlay_callable(trace: Any, scores: Mapping[str, Any]) -> EncodingChannelRequest:
    """Build the ``color_by`` callable for a label or module-address mapping."""

    values = dict(scores)
    module_keys = set(values) & _module_addresses(trace)

    def overlay(node: Any) -> Any:
        """Return the mapped value for one rendered node, else ``None``."""

        label = getattr(node, "layer_label", None)
        if label in values:
            return values[label]
        base = str(label).split(":", 1)[0] if label is not None else None
        if base in values:
            return values[base]
        for address in getattr(node, "output_of_modules", ()) or ():
            if address in module_keys:
                return values[address]
        return None

    return EncodingChannelRequest(source=overlay, display_name="overlay")


def _module_addresses(trace: Any) -> set[str]:
    """Return every module address that some captured op is the output of."""

    addresses: set[str] = set()
    for label in getattr(trace, "layer_labels", ()) or ():
        addresses.update(getattr(trace[label], "output_of_modules", ()) or ())
    return addresses


__all__ = ["resolve_color_source"]

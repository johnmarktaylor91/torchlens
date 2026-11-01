"""Resolve the non-string ``color_by=`` sources of ``Trace.draw``.

``color_by`` takes a field name or callable (resolved by ``_encoding``), a
mapping of layer label or module address to value (an "overlay"; this module
turns it into a named callable), or an attribution result or an iterable of
them (resolved by ``torchlens.attribution._viz``).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ._encoding import EncodingChannelRequest


def _is_attribution_source(source: Any) -> bool:
    """Return whether ``source`` is an attribution result or an iterable of them."""

    from ..attribution._core import Attribution

    if isinstance(source, Attribution):
        return True
    if isinstance(source, (str, bytes, Mapping)) or callable(source):
        return False
    if isinstance(source, (list, tuple, set, frozenset)):
        return bool(source) and all(isinstance(item, Attribution) for item in source)
    try:
        from ..attribution.graph import AttributionGraph
    except ImportError:
        return False
    return isinstance(source, AttributionGraph)


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
        from ..attribution._viz import attribution_color_by

        return EncodingChannelRequest(
            source=attribution_color_by(trace, color_by), display_name="attribution"
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

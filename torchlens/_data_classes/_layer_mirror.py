"""Layer mirror reads: the representative op behind a Layer and its mirrored values."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ._layer_spec import _LAYER_MIRROR_SPEC

if TYPE_CHECKING:
    from .layer import Layer
    from .op import Op


def _layer_rep_op(layer: Layer) -> Op | None:
    """Return the representative (first-pass) op backing one Layer's mirrors.

    Reads raw ``__dict__`` storage so descriptor bodies never re-enter the
    attribute protocol. ``OpAccessor._list`` is pass-index sorted, so index 0
    is the first pass — the op the former ``__init__`` copied from.
    """

    ops = layer.__dict__.get("ops")
    if ops is None:
        return None
    item_list = ops.__dict__.get("_list")
    return item_list[0] if item_list else None


def _layer_mirror_read(layer: Layer, name: str) -> Any:
    """Return the mirror value for ``name`` (no shadow consulted).

    Raises ``AttributeError`` when the layer has no representative op yet or
    the op's source cell is unset — the caller (descriptor or state
    enumeration) translates that into the historical missing-field behavior.
    """

    rep = _layer_rep_op(layer)
    if rep is None:
        raise AttributeError(name)
    source, normalize = _LAYER_MIRROR_SPEC[name]
    value = getattr(rep, source)
    return normalize(value) if normalize is not None else value

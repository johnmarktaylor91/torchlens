"""Aligned Super views for TorchLens intervention bundles.

Internal surface:

* :class:`SuperOp` -- per-node view returned by ``Bundle.node(...)``.
* :class:`SuperLayer` -- per-layer-label view returned by ``Bundle.layers[...]``.
* :class:`TraceAccessor` -- dict-like accessor for ``Bundle.traces``.

This subpackage is intentionally not part of the top-level TorchLens public API.
"""

from __future__ import annotations

from .super_op import SuperLayer, SuperLayerAccessor, SuperOp, SuperOpAccessor, TraceAccessor


__all__ = [
    "SuperLayer",
    "SuperLayerAccessor",
    "SuperOp",
    "SuperOpAccessor",
    "TraceAccessor",
]

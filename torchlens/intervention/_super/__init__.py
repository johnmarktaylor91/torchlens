"""Aligned Super views for TorchLens intervention bundles.

Internal surface:

* :class:`SuperOp` -- per-site view returned by ``Bundle.at(site)``.
* :class:`SuperLayer` -- per-layer-label view returned by ``Bundle.layers[...]``.
* :class:`TraceAccessor` -- dict-like accessor over member traces.
* :class:`SuperModule`, :class:`SuperBuffer`, :class:`SuperParam`,
  :class:`SuperGradFn`, :class:`SuperModuleCall`, and
  :class:`SuperGradFnCall` -- aligned views for the remaining Trace log
  families.

This subpackage is intentionally not part of the top-level TorchLens public API.
"""

from __future__ import annotations

from .super_logs import (
    SuperBuffer,
    SuperBufferAccessor,
    SuperGradFn,
    SuperGradFnAccessor,
    SuperGradFnCall,
    SuperGradFnCallAccessor,
    SuperModule,
    SuperModuleAccessor,
    SuperModuleCall,
    SuperModuleCallAccessor,
    SuperParam,
    SuperParamAccessor,
)
from .super_op import (
    SuperAtenOp,
    SuperLayer,
    SuperLayerAccessor,
    SuperOp,
    SuperOpAccessor,
    TraceAccessor,
)

__all__ = [
    "SuperAtenOp",
    "SuperBuffer",
    "SuperBufferAccessor",
    "SuperGradFn",
    "SuperGradFnAccessor",
    "SuperGradFnCall",
    "SuperGradFnCallAccessor",
    "SuperLayer",
    "SuperLayerAccessor",
    "SuperModule",
    "SuperModuleAccessor",
    "SuperModuleCall",
    "SuperModuleCallAccessor",
    "SuperOp",
    "SuperOpAccessor",
    "SuperParam",
    "SuperParamAccessor",
    "TraceAccessor",
]
